#!/usr/bin/env python3
"""Generate a GOAL1 CSV whose first q row is anchored to current/manual q0.

This script is offline by default. Manual q0 mode does not import ROS2. State
mode only subscribes once to /state_parameter; it does not publish topics,
launch controllers, send effort commands, enable GP, or move the robot.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

sys.dont_write_bytecode = True

from generate_goal1_joint_trajectory import (  # noqa: E402
    ACCELERATION_LIMITS,
    DEFAULT_OUTPUT_DIR,
    JOINT_COUNT,
    POSITION_LOWER_LIMITS,
    POSITION_UPPER_LIMITS,
    PROFILES,
    VELOCITY_LIMITS,
    axis_stats,
    columns_for_csv,
    fmt_float,
    generate_time,
    generate_trajectory,
    profile_terms,
    require_numpy,
    validate_profile,
    write_csv,
    write_plots,
)


DEFAULT_PREFIX = "goal1_current_q_anchored_3s_50hz"
DEFAULT_STATE_TOPIC = "/state_parameter"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an offline GOAL1 current-q anchored joint-space CSV. "
            "Manual mode is pure offline; --from-state only subscribes once."
        ),
    )
    q0_source = parser.add_mutually_exclusive_group(required=True)
    q0_source.add_argument("--q0", help="Comma-separated 7-element q0 in rad.")
    q0_source.add_argument(
        "--from-state",
        action="store_true",
        help="Subscribe once to /state_parameter and use current position as q0.",
    )
    parser.add_argument("--state-topic", default=DEFAULT_STATE_TOPIC, help=f"Default: {DEFAULT_STATE_TOPIC}")
    parser.add_argument("--state-timeout-sec", type=float, default=1.0, help="Default: 1.0")
    parser.add_argument("--duration", type=float, default=3.0, help="Default: 3.0")
    parser.add_argument("--sample-rate", type=float, default=50.0, help="Default: 50.0")
    parser.add_argument("--profile", default="spatial_rich", choices=sorted(PROFILES), help="Default: spatial_rich")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help=f"Default: {DEFAULT_PREFIX}")
    parser.add_argument("--include-jerk", action="store_true", help="Include optional joint_jerk_1..7 columns.")
    plots = parser.add_mutually_exclusive_group()
    plots.add_argument("--plots", dest="plots", action="store_true", help="Generate PNG plots.")
    plots.add_argument("--no-plots", dest="plots", action="store_false", help="Skip PNG plots.")
    parser.set_defaults(plots=False)
    safety = parser.add_mutually_exclusive_group()
    safety.add_argument("--fail-on-unsafe", dest="fail_on_unsafe", action="store_true", help="Exit nonzero if unsafe.")
    safety.add_argument(
        "--no-fail-on-unsafe",
        dest="fail_on_unsafe",
        action="store_false",
        help="Write outputs even if safety checks fail.",
    )
    parser.set_defaults(fail_on_unsafe=True)
    parser.add_argument(
        "--joint-limit-margin-rad",
        type=float,
        default=0.05,
        help="Margin applied inside conservative position limits. Default: 0.05",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print the summary paths and key status only; output files are still written.",
    )
    return parser.parse_args()


def parse_q0(text: str) -> list[float]:
    values = [item.strip() for item in text.split(",")]
    if len(values) != JOINT_COUNT or any(not item for item in values):
        raise ValueError("--q0 must contain exactly 7 comma-separated values")
    try:
        q0 = [float(item) for item in values]
    except ValueError as exc:
        raise ValueError("--q0 must contain numeric values") from exc
    validate_finite_vector(q0, "q0")
    return q0


def validate_finite_vector(values: Sequence[float], name: str) -> None:
    if len(values) != JOINT_COUNT:
        raise ValueError(f"{name} must contain exactly 7 values")
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError(f"{name} must contain finite values")


def validate_args(args: argparse.Namespace) -> None:
    if args.duration <= 0.0 or not math.isfinite(args.duration):
        raise ValueError("--duration must be finite and positive")
    if args.sample_rate <= 0.0 or not math.isfinite(args.sample_rate):
        raise ValueError("--sample-rate must be finite and positive")
    if args.state_timeout_sec <= 0.0 or not math.isfinite(args.state_timeout_sec):
        raise ValueError("--state-timeout-sec must be finite and positive")
    if args.joint_limit_margin_rad < 0.0 or not math.isfinite(args.joint_limit_margin_rad):
        raise ValueError("--joint-limit-margin-rad must be finite and nonnegative")
    for lower, upper in zip(POSITION_LOWER_LIMITS, POSITION_UPPER_LIMITS):
        if lower + args.joint_limit_margin_rad >= upper - args.joint_limit_margin_rad:
            raise ValueError("--joint-limit-margin-rad leaves no valid position range")
    if not args.prefix:
        raise ValueError("--prefix must be non-empty")
    if not str(args.state_topic).strip():
        raise ValueError("--state-topic must be non-empty")


def read_q0_from_state(state_topic: str, timeout_sec: float) -> tuple[list[float], list[float]]:
    import rclpy
    from rclpy.node import Node
    from custom_msgs.msg import StateParameter

    class SingleStateReader(Node):
        def __init__(self) -> None:
            super().__init__("goal1_current_q_anchor_reader")
            self.latest_state: tuple[list[float], list[float]] | None = None
            self.error = ""
            self.subscription = self.create_subscription(StateParameter, state_topic, self.callback, 10)
            self.get_logger().warn(
                "GOAL1 current-q anchored generator: subscribing to state only; "
                "no publishers, no controller launch, no effort command."
            )

        def callback(self, msg: Any) -> None:
            if len(msg.position) != JOINT_COUNT or len(msg.velocity) != JOINT_COUNT:
                self.error = "state_parameter position/velocity must each contain 7 values"
                return
            q = [float(value) for value in msg.position]
            dq = [float(value) for value in msg.velocity]
            if not all(math.isfinite(value) for value in q + dq):
                self.error = "state_parameter contains non-finite q or dq"
                return
            self.latest_state = (q, dq)

    rclpy.init(args=[sys.argv[0]])
    node = SingleStateReader()
    try:
        deadline = node.get_clock().now().nanoseconds + int(timeout_sec * 1e9)
        while rclpy.ok() and node.get_clock().now().nanoseconds <= deadline:
            rclpy.spin_once(node, timeout_sec=0.02)
            if node.error:
                raise ValueError(node.error)
            if node.latest_state is not None:
                return node.latest_state
        raise TimeoutError(f"no state received on {state_topic} within {timeout_sec:.3f}s; CSV was not generated")
    finally:
        node.destroy_node()
        rclpy.shutdown()


def anchor_trajectory(np: Any, trajectory: dict[str, Any], q0: Sequence[float]) -> dict[str, Any]:
    anchored = dict(trajectory)
    q_profile = trajectory["q"]
    delta_q = q_profile - q_profile[0, :]
    anchored["q"] = np.asarray(q0, dtype=float).reshape(1, JOINT_COUNT) + delta_q
    return anchored


def validate_expected_columns(include_jerk: bool) -> None:
    expected = (
        ["time"]
        + [f"joint_pos_{index}" for index in range(1, JOINT_COUNT + 1)]
        + [f"joint_vel_{index}" for index in range(1, JOINT_COUNT + 1)]
        + [f"joint_acc_{index}" for index in range(1, JOINT_COUNT + 1)]
    )
    actual = columns_for_csv(include_jerk)
    missing = [column for column in expected if column not in actual]
    if missing:
        raise ValueError(f"CSV columns are missing replay-required fields: {missing}")


def check_finite_array(np: Any, values: Any, name: str) -> bool:
    return bool(np.all(np.isfinite(values)))


def build_safety_summary(
    np: Any,
    trajectory: dict[str, Any],
    include_jerk: bool,
    joint_limit_margin_rad: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    q_stats = axis_stats(trajectory["q"], np)
    dq_stats = axis_stats(trajectory["dq"], np)
    ddq_stats = axis_stats(trajectory["ddq"], np)
    jerk_stats = axis_stats(trajectory["jerk"], np) if include_jerk else None

    finite_q = check_finite_array(np, trajectory["q"], "q")
    finite_dq = check_finite_array(np, trajectory["dq"], "dq")
    finite_ddq = check_finite_array(np, trajectory["ddq"], "ddq")
    finite_jerk = check_finite_array(np, trajectory["jerk"], "jerk") if include_jerk else True

    joint_summaries = []
    overall_safe = finite_q and finite_dq and finite_ddq and finite_jerk
    for joint_index in range(JOINT_COUNT):
        lower_limit = POSITION_LOWER_LIMITS[joint_index] + joint_limit_margin_rad
        upper_limit = POSITION_UPPER_LIMITS[joint_index] - joint_limit_margin_rad
        position_ok = q_stats[joint_index]["min"] >= lower_limit - 1e-12 and q_stats[joint_index]["max"] <= upper_limit + 1e-12
        velocity_ok = dq_stats[joint_index]["max_abs"] <= VELOCITY_LIMITS[joint_index] + 1e-12
        acceleration_ok = ddq_stats[joint_index]["max_abs"] <= ACCELERATION_LIMITS[joint_index] + 1e-12
        checks = [position_ok, velocity_ok, acceleration_ok]

        item = {
            "joint": joint_index + 1,
            "position_min": q_stats[joint_index]["min"],
            "position_max": q_stats[joint_index]["max"],
            "position_range": q_stats[joint_index]["range"],
            "max_abs_velocity": dq_stats[joint_index]["max_abs"],
            "max_abs_acceleration": ddq_stats[joint_index]["max_abs"],
            "position_lower_limit_with_margin": lower_limit,
            "position_upper_limit_with_margin": upper_limit,
            "velocity_limit": VELOCITY_LIMITS[joint_index],
            "acceleration_limit": ACCELERATION_LIMITS[joint_index],
            "position_ok": position_ok,
            "velocity_ok": velocity_ok,
            "acceleration_ok": acceleration_ok,
        }
        if include_jerk and jerk_stats is not None:
            jerk_ok = math.isfinite(jerk_stats[joint_index]["max_abs"])
            item.update({"max_abs_jerk": jerk_stats[joint_index]["max_abs"], "jerk_finite": jerk_ok})
            checks.append(jerk_ok)
        joint_safe = all(checks)
        item["joint_safe"] = joint_safe
        joint_summaries.append(item)
        overall_safe = overall_safe and joint_safe

    safety_summary = {
        "finite_q": finite_q,
        "finite_dq": finite_dq,
        "finite_ddq": finite_ddq,
        "finite_jerk": finite_jerk if include_jerk else "not_requested",
        "position_check": all(item["position_ok"] for item in joint_summaries),
        "velocity_check": all(item["velocity_ok"] for item in joint_summaries),
        "acceleration_check": all(item["acceleration_ok"] for item in joint_summaries),
        "jerk_check": all(item.get("jerk_finite", True) for item in joint_summaries) if include_jerk else "not_requested",
        "joint_limit_margin_rad": joint_limit_margin_rad,
        "overall_safety_status": "safe" if overall_safe else "unsafe",
    }
    return joint_summaries, safety_summary


def build_summary(
    args: argparse.Namespace,
    dt: float,
    time: Any,
    profile: Any,
    q0_source: str,
    q0: Sequence[float],
    state_dq: Sequence[float] | None,
    trajectory: dict[str, Any],
    joint_summaries: list[dict[str, Any]],
    safety_summary: dict[str, Any],
    output_paths: dict[str, str],
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "q0_source": q0_source,
        "state_topic": args.state_topic if q0_source == "state" else None,
        "state_timeout_sec": args.state_timeout_sec if q0_source == "state" else None,
        "duration": args.duration,
        "sample_rate": args.sample_rate,
        "dt": dt,
        "profile": args.profile,
        "row_count": int(len(time)),
        "q0": [float(value) for value in q0],
        "state_dq": [float(value) for value in state_dq] if state_dq is not None else None,
        "q_first_row": [float(value) for value in trajectory["q"][0, :]],
        "q_final_row": [float(value) for value in trajectory["q"][-1, :]],
        "per_joint_q_min_max": [
            {
                "joint": item["joint"],
                "min": item["position_min"],
                "max": item["position_max"],
            }
            for item in joint_summaries
        ],
        "per_joint_dq_max_abs": [
            {"joint": item["joint"], "max_abs": item["max_abs_velocity"]} for item in joint_summaries
        ],
        "per_joint_ddq_max_abs": [
            {"joint": item["joint"], "max_abs": item["max_abs_acceleration"]} for item in joint_summaries
        ],
        "generation_config": {
            "include_jerk": args.include_jerk,
            "joint_limit_margin_rad": args.joint_limit_margin_rad,
            "fail_on_unsafe": args.fail_on_unsafe,
        },
        "profile_terms": profile_terms(profile),
        "joint_summaries": joint_summaries,
        "safety_summary": safety_summary,
        "output_paths": output_paths,
        "explicit_statement": "offline only, no ROS replay, no controller integration, no robot motion",
    }


def write_json_summary(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown_summary(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# GOAL1 Current-q Anchored Trajectory Summary",
        "",
        "## Config",
        "",
        f"- q0_source: `{summary['q0_source']}`",
        f"- state_topic: `{summary['state_topic']}`",
        f"- duration: `{summary['duration']}`",
        f"- sample_rate: `{summary['sample_rate']}`",
        f"- profile: `{summary['profile']}`",
        f"- row_count: `{summary['row_count']}`",
        f"- joint_limit_margin_rad: `{summary['generation_config']['joint_limit_margin_rad']}`",
        "",
        "## Anchor",
        "",
        "- q0: " + ", ".join(fmt_float(value) for value in summary["q0"]),
        "- q_first_row: " + ", ".join(fmt_float(value) for value in summary["q_first_row"]),
        "- q_final_row: " + ", ".join(fmt_float(value) for value in summary["q_final_row"]),
        "",
        "## Safety Status",
        "",
        f"- overall_safety_status: `{summary['safety_summary']['overall_safety_status']}`",
        f"- position_check: `{str(summary['safety_summary']['position_check']).lower()}`",
        f"- velocity_check: `{str(summary['safety_summary']['velocity_check']).lower()}`",
        f"- acceleration_check: `{str(summary['safety_summary']['acceleration_check']).lower()}`",
        f"- jerk_check: `{summary['safety_summary']['jerk_check']}`",
        "",
        "## Per-joint Summary",
        "",
        "| joint | q min | q max | max abs dq | max abs ddq | safe |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in summary["joint_summaries"]:
        lines.append(
            "| q{joint} | {q_min} | {q_max} | {dq} | {ddq} | {safe} |".format(
                joint=item["joint"],
                q_min=fmt_float(item["position_min"]),
                q_max=fmt_float(item["position_max"]),
                dq=fmt_float(item["max_abs_velocity"]),
                ddq=fmt_float(item["max_abs_acceleration"]),
                safe=str(item["joint_safe"]).lower(),
            )
        )
    lines.extend(
        [
            "",
            "## Output Paths",
            "",
        ]
    )
    lines.extend(f"- {key}: `{value}`" for key, value in summary["output_paths"].items())
    lines.extend(
        [
            "",
            "## Explicit Scope",
            "",
            summary["explicit_statement"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def print_console_summary(summary: dict[str, Any], summary_only: bool) -> None:
    print(f"CSV: {summary['output_paths']['csv']}")
    print(f"JSON summary: {summary['output_paths']['json_summary']}")
    print(f"Markdown summary: {summary['output_paths']['markdown_summary']}")
    print(f"q0_source: {summary['q0_source']}")
    print(f"q_first_row: {[round(value, 9) for value in summary['q_first_row']]}")
    print(f"overall_safety_status: {summary['safety_summary']['overall_safety_status']}")
    print(summary["explicit_statement"])
    if not summary_only:
        print(f"row_count: {summary['row_count']}")
        print(f"q_final_row: {[round(value, 9) for value in summary['q_final_row']]}")


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        q0_source = "state" if args.from_state else "manual"
        if args.from_state:
            q0, state_dq = read_q0_from_state(args.state_topic, args.state_timeout_sec)
            validate_finite_vector(q0, "state q")
            validate_finite_vector(state_dq, "state dq")
        else:
            q0 = parse_q0(args.q0)
            state_dq = None

        np = require_numpy()
        profile = PROFILES[args.profile]
        validate_profile(profile)
        validate_expected_columns(args.include_jerk)
        time, dt = generate_time(np, args.duration, args.sample_rate)
        base_trajectory = generate_trajectory(np, time, profile, args.include_jerk)
        trajectory = anchor_trajectory(np, base_trajectory, q0)
        joint_summaries, safety_summary = build_safety_summary(
            np, trajectory, args.include_jerk, args.joint_limit_margin_rad
        )
    except (ModuleNotFoundError, TimeoutError, ValueError) as exc:
        print(f"Failed to generate anchored CSV: {exc}", file=sys.stderr)
        return 1 if isinstance(exc, ModuleNotFoundError) else 2

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{args.prefix}.csv"
    json_path = output_dir / f"{args.prefix}_summary.json"
    md_path = output_dir / f"{args.prefix}_summary.md"

    output_paths = {
        "csv": str(csv_path),
        "json_summary": str(json_path),
        "markdown_summary": str(md_path),
    }
    if args.plots:
        try:
            output_paths.update(write_plots(output_dir, args.prefix, time, trajectory, args.include_jerk))
        except ModuleNotFoundError as exc:
            print(f"Failed to generate plots: {exc}", file=sys.stderr)
            return 1

    write_csv(csv_path, time, trajectory, args.include_jerk)
    summary = build_summary(
        args,
        dt,
        time,
        profile,
        q0_source,
        q0,
        state_dq,
        trajectory,
        joint_summaries,
        safety_summary,
        output_paths,
    )
    write_json_summary(json_path, summary)
    write_markdown_summary(md_path, summary)
    print_console_summary(summary, args.summary_only)

    if args.fail_on_unsafe and safety_summary["overall_safety_status"] != "safe":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
