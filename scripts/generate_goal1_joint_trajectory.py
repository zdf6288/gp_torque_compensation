#!/usr/bin/env python3
"""Generate an offline GOAL1 all-q joint-space trajectory.

This script is intentionally offline-only. It does not import ROS2, launch a
controller, connect to Franka, publish commands, enable GP, or modify runtime
control behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


sys.dont_write_bytecode = True

DEFAULT_OUTPUT_DIR = Path("outputs/goal1_joint_trajectory")
DEFAULT_PREFIX = "goal1_allq_conservative"
JOINT_COUNT = 7


@dataclass(frozen=True)
class MultiSineProfile:
    name: str
    nominal_q: tuple[float, ...]
    amplitudes: tuple[tuple[float, ...], ...]
    frequencies: tuple[tuple[float, ...], ...]
    phases: tuple[tuple[float, ...], ...]


# 这是第一版 conservative profile：所有 q1..q7 都运动，但幅度和频率都很低。
# q7 也参与运动，不过幅度明显小于主要 joints，避免一开始把腕部姿态变化做得太激进。
CONSERVATIVE_PROFILE = MultiSineProfile(
    name="conservative",
    nominal_q=(0.0, -0.785398163, 0.0, -2.35619449, 0.0, 1.570796327, 0.785398163),
    amplitudes=(
        (0.18, 0.04),
        (0.14, 0.035),
        (0.16, 0.03),
        (0.12, 0.025),
        (0.14, 0.03),
        (0.10, 0.025),
        (0.05, 0.015),
    ),
    frequencies=(
        (0.060, 0.115),
        (0.075, 0.100),
        (0.055, 0.095),
        (0.065, 0.120),
        (0.085, 0.105),
        (0.070, 0.130),
        (0.090, 0.145),
    ),
    phases=(
        (0.0, 1.1),
        (0.7, 1.8),
        (1.4, 2.5),
        (2.1, 3.2),
        (2.8, 3.9),
        (3.5, 4.6),
        (4.2, 5.3),
    ),
)

SPATIAL_RICH_PROFILE = MultiSineProfile(
    name="spatial_rich",
    nominal_q=(0.0, -0.785398163, 0.0, -2.35619449, 0.0, 1.570796327, 0.785398163),
    amplitudes=(
        (0.24, 0.070, 0.025),
        (0.17, 0.055, 0.025),
        (0.22, 0.060, 0.025),
        (0.13, 0.045, 0.020),
        (0.18, 0.055, 0.025),
        (0.12, 0.040, 0.020),
        (0.07, 0.025, 0.012),
    ),
    frequencies=(
        (0.044, 0.110, 0.175),
        (0.052, 0.128, 0.168),
        (0.039, 0.116, 0.182),
        (0.047, 0.104, 0.158),
        (0.061, 0.121, 0.171),
        (0.055, 0.137, 0.164),
        (0.073, 0.143, 0.188),
    ),
    phases=(
        (0.0, 1.4, 3.1),
        (0.9, 2.6, 4.4),
        (1.8, 3.5, 5.2),
        (2.7, 4.3, 0.7),
        (3.6, 5.1, 1.6),
        (4.5, 0.8, 2.5),
        (5.4, 1.7, 3.4),
    ),
)

PROFILES = {
    "conservative": CONSERVATIVE_PROFILE,
    "spatial_rich": SPATIAL_RICH_PROFILE,
}

# 这些 limits 只用于 offline preliminary screening。正式仿真或真机前，必须用
# 项目配置和机器人实际 limits 重新确认。本脚本通过 safety check 只说明 CSV
# 自身在这些保守阈值下没有越界，不证明 controller tracking 或硬件安全。
POSITION_LOWER_LIMITS = (-2.80, -1.70, -2.80, -3.00, -2.80, 0.00, -2.80)
POSITION_UPPER_LIMITS = (2.80, 1.70, 2.80, -0.10, 2.80, 3.70, 2.80)
VELOCITY_LIMITS = (0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70)
ACCELERATION_LIMITS = (1.50, 1.50, 1.50, 1.50, 1.50, 1.50, 1.50)
JERK_LIMITS = (8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0)

SAFETY_CAVEATS = [
    "offline only",
    "no ROS replay",
    "no FK",
    "no controller integration",
    "no real robot validation",
    "not safe for direct Franka execution without later simulation and separate safety review",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate offline GOAL1 conservative all-q joint-space CSV, plots, and safety summary.",
    )
    parser.add_argument("--duration", type=float, default=20.0, help="Trajectory duration in seconds. Default: 20.0")
    parser.add_argument("--sample-rate", type=float, default=100.0, help="Sample rate in Hz. Default: 100.0")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help=f"Output filename prefix. Default: {DEFAULT_PREFIX}")
    parser.add_argument("--include-jerk", action="store_true", help="Include analytic jerk columns, plot, and checks.")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG plot generation.")
    parser.add_argument("--fail-on-unsafe", action="store_true", help="Exit nonzero if any safety check fails.")
    parser.add_argument("--profile", default="conservative", choices=sorted(PROFILES), help="Trajectory profile.")
    return parser.parse_args()


def require_numpy() -> Any:
    try:
        import numpy as np
    except ModuleNotFoundError:
        print("Missing Python dependency: numpy", file=sys.stderr)
        print("Suggested install command: python3 -m pip install numpy", file=sys.stderr)
        raise
    return np


def require_matplotlib(output_dir: Path) -> Any:
    if "MPLCONFIGDIR" not in os.environ:
        mpl_config_dir = output_dir / ".matplotlib"
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("Missing Python dependency: matplotlib", file=sys.stderr)
        print("Suggested install command: python3 -m pip install matplotlib", file=sys.stderr)
        raise
    return plt


def validate_args(args: argparse.Namespace) -> None:
    if args.duration <= 0:
        raise ValueError("--duration must be positive")
    if args.sample_rate <= 0:
        raise ValueError("--sample-rate must be positive")
    if not args.prefix:
        raise ValueError("--prefix must be non-empty")


def validate_profile(profile: MultiSineProfile) -> None:
    for field_name in ("nominal_q", "amplitudes", "frequencies", "phases"):
        if len(getattr(profile, field_name)) != JOINT_COUNT:
            raise ValueError(f"profile {profile.name} has invalid {field_name} length")
    for joint_index in range(JOINT_COUNT):
        term_count = len(profile.amplitudes[joint_index])
        if term_count == 0:
            raise ValueError(f"profile {profile.name} q{joint_index + 1} has no sine terms")
        if len(profile.frequencies[joint_index]) != term_count or len(profile.phases[joint_index]) != term_count:
            raise ValueError(f"profile {profile.name} q{joint_index + 1} has inconsistent sine terms")


def generate_time(np: Any, duration: float, sample_rate: float) -> tuple[Any, float]:
    dt = 1.0 / sample_rate
    sample_count = int(math.floor(duration * sample_rate)) + 1
    return np.arange(sample_count, dtype=float) * dt, dt


def generate_trajectory(np: Any, time: Any, profile: MultiSineProfile, include_jerk: bool) -> dict[str, Any]:
    q = np.zeros((len(time), JOINT_COUNT), dtype=float)
    dq = np.zeros_like(q)
    ddq = np.zeros_like(q)
    jerk = np.zeros_like(q) if include_jerk else None

    for joint_index in range(JOINT_COUNT):
        q[:, joint_index] = profile.nominal_q[joint_index]
        for term_index, amplitude in enumerate(profile.amplitudes[joint_index]):
            frequency = profile.frequencies[joint_index][term_index]
            phase = profile.phases[joint_index][term_index]
            omega = 2.0 * math.pi * frequency
            angle = omega * time + phase

            q[:, joint_index] += amplitude * np.sin(angle)
            dq[:, joint_index] += amplitude * omega * np.cos(angle)
            ddq[:, joint_index] += -amplitude * omega**2 * np.sin(angle)
            if jerk is not None:
                jerk[:, joint_index] += -amplitude * omega**3 * np.cos(angle)

    return {"q": q, "dq": dq, "ddq": ddq, "jerk": jerk}


def columns_for_csv(include_jerk: bool) -> list[str]:
    columns = ["time"]
    for group in ("joint_pos", "joint_vel", "joint_acc"):
        columns.extend(f"{group}_{index}" for index in range(1, JOINT_COUNT + 1))
    if include_jerk:
        columns.extend(f"joint_jerk_{index}" for index in range(1, JOINT_COUNT + 1))
    return columns


def write_csv(path: Path, time: Any, trajectory: dict[str, Any], include_jerk: bool) -> None:
    columns = columns_for_csv(include_jerk)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for row_index, t_value in enumerate(time):
            row = [f"{float(t_value):.9f}"]
            for key in ("q", "dq", "ddq"):
                row.extend(f"{float(value):.12f}" for value in trajectory[key][row_index, :])
            if include_jerk:
                row.extend(f"{float(value):.12f}" for value in trajectory["jerk"][row_index, :])
            writer.writerow(row)


def axis_stats(values: Any, np: Any) -> list[dict[str, float]]:
    result = []
    for joint_index in range(JOINT_COUNT):
        series = values[:, joint_index]
        min_value = float(np.min(series))
        max_value = float(np.max(series))
        result.append(
            {
                "min": min_value,
                "max": max_value,
                "range": max_value - min_value,
                "max_abs": float(np.max(np.abs(series))),
            }
        )
    return result


def check_limit(value: float, limit: float, tolerance: float = 1e-12) -> bool:
    return value <= limit + tolerance


def build_safety_summary(np: Any, trajectory: dict[str, Any], include_jerk: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    q_stats = axis_stats(trajectory["q"], np)
    dq_stats = axis_stats(trajectory["dq"], np)
    ddq_stats = axis_stats(trajectory["ddq"], np)
    jerk_stats = axis_stats(trajectory["jerk"], np) if include_jerk else None

    joint_summaries = []
    overall_safe = True
    for joint_index in range(JOINT_COUNT):
        position_ok = (
            q_stats[joint_index]["min"] >= POSITION_LOWER_LIMITS[joint_index] - 1e-12
            and q_stats[joint_index]["max"] <= POSITION_UPPER_LIMITS[joint_index] + 1e-12
        )
        velocity_ok = check_limit(dq_stats[joint_index]["max_abs"], VELOCITY_LIMITS[joint_index])
        acceleration_ok = check_limit(ddq_stats[joint_index]["max_abs"], ACCELERATION_LIMITS[joint_index])
        jerk_ok = None
        if include_jerk and jerk_stats is not None:
            jerk_ok = check_limit(jerk_stats[joint_index]["max_abs"], JERK_LIMITS[joint_index])

        checks = [position_ok, velocity_ok, acceleration_ok]
        if jerk_ok is not None:
            checks.append(jerk_ok)
        joint_safe = all(checks)
        overall_safe = overall_safe and joint_safe

        summary = {
            "joint": joint_index + 1,
            "position_min": q_stats[joint_index]["min"],
            "position_max": q_stats[joint_index]["max"],
            "position_range": q_stats[joint_index]["range"],
            "max_abs_velocity": dq_stats[joint_index]["max_abs"],
            "max_abs_acceleration": ddq_stats[joint_index]["max_abs"],
            "position_lower_limit": POSITION_LOWER_LIMITS[joint_index],
            "position_upper_limit": POSITION_UPPER_LIMITS[joint_index],
            "velocity_limit": VELOCITY_LIMITS[joint_index],
            "acceleration_limit": ACCELERATION_LIMITS[joint_index],
            "position_ok": position_ok,
            "velocity_ok": velocity_ok,
            "acceleration_ok": acceleration_ok,
            "joint_safe": joint_safe,
        }
        if include_jerk and jerk_stats is not None:
            summary.update(
                {
                    "max_abs_jerk": jerk_stats[joint_index]["max_abs"],
                    "jerk_limit": JERK_LIMITS[joint_index],
                    "jerk_ok": jerk_ok,
                }
            )
        joint_summaries.append(summary)

    safety_summary = {
        "position_check": all(item["position_ok"] for item in joint_summaries),
        "velocity_check": all(item["velocity_ok"] for item in joint_summaries),
        "acceleration_check": all(item["acceleration_ok"] for item in joint_summaries),
        "jerk_check": all(item.get("jerk_ok", True) for item in joint_summaries) if include_jerk else "not_requested",
        "overall_safety_status": "safe" if overall_safe else "unsafe",
    }
    return joint_summaries, safety_summary


def profile_terms(profile: MultiSineProfile) -> list[dict[str, Any]]:
    terms = []
    for joint_index in range(JOINT_COUNT):
        terms.append(
            {
                "joint": joint_index + 1,
                "nominal_q": profile.nominal_q[joint_index],
                "amplitudes": list(profile.amplitudes[joint_index]),
                "frequencies": list(profile.frequencies[joint_index]),
                "phases": list(profile.phases[joint_index]),
            }
        )
    return terms


def build_summary(
    args: argparse.Namespace,
    dt: float,
    sample_count: int,
    profile: MultiSineProfile,
    joint_summaries: list[dict[str, Any]],
    safety_summary: dict[str, Any],
    output_paths: dict[str, str],
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generation_config": {
            "duration": args.duration,
            "sample_rate": args.sample_rate,
            "dt": dt,
            "number_of_samples": sample_count,
            "profile": args.profile,
            "include_jerk": args.include_jerk,
        },
        "nominal_q": list(profile.nominal_q),
        "profile_terms": profile_terms(profile),
        "joint_summaries": joint_summaries,
        "safety_summary": safety_summary,
        "caveat": SAFETY_CAVEATS,
        "output_paths": output_paths,
    }


def fmt_float(value: float) -> str:
    return f"{value:.6g}"


def write_markdown_summary(path: Path, summary: dict[str, Any]) -> None:
    config = summary["generation_config"]
    safety = summary["safety_summary"]
    lines = [
        "# GOAL1 All-q Joint Trajectory Summary",
        "",
        "## Generation Config",
        "",
        f"- duration: `{config['duration']}`",
        f"- sample_rate: `{config['sample_rate']}`",
        f"- dt: `{config['dt']}`",
        f"- number_of_samples: `{config['number_of_samples']}`",
        f"- profile: `{config['profile']}`",
        f"- include_jerk: `{str(config['include_jerk']).lower()}`",
        "",
        "## Safety Status",
        "",
        f"- overall_safety_status: `{safety['overall_safety_status']}`",
        f"- position_check: `{str(safety['position_check']).lower()}`",
        f"- velocity_check: `{str(safety['velocity_check']).lower()}`",
        f"- acceleration_check: `{str(safety['acceleration_check']).lower()}`",
        f"- jerk_check: `{safety['jerk_check']}`",
        "",
        "## Nominal q",
        "",
        "- " + ", ".join(f"q{index + 1}={fmt_float(value)}" for index, value in enumerate(summary["nominal_q"])),
        "",
        "## Per-joint Profile",
        "",
        "| joint | nominal_q | amplitudes | frequencies | phases |",
        "| --- | ---: | --- | --- | --- |",
    ]

    for term in summary["profile_terms"]:
        lines.append(
            "| q{joint} | {nominal_q} | {amplitudes} | {frequencies} | {phases} |".format(
                joint=term["joint"],
                nominal_q=fmt_float(term["nominal_q"]),
                amplitudes=", ".join(fmt_float(value) for value in term["amplitudes"]),
                frequencies=", ".join(fmt_float(value) for value in term["frequencies"]),
                phases=", ".join(fmt_float(value) for value in term["phases"]),
            )
        )

    lines.extend(
        [
            "",
            "## Per-joint Safety Summary",
            "",
            "| joint | q min | q max | q range | max abs dq | max abs ddq | max abs jerk | safe |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for item in summary["joint_summaries"]:
        jerk_value = item.get("max_abs_jerk")
        lines.append(
            "| q{joint} | {q_min} | {q_max} | {q_range} | {dq} | {ddq} | {jerk} | {safe} |".format(
                joint=item["joint"],
                q_min=fmt_float(item["position_min"]),
                q_max=fmt_float(item["position_max"]),
                q_range=fmt_float(item["position_range"]),
                dq=fmt_float(item["max_abs_velocity"]),
                ddq=fmt_float(item["max_abs_acceleration"]),
                jerk=fmt_float(jerk_value) if jerk_value is not None else "not_requested",
                safe=str(item["joint_safe"]).lower(),
            )
        )

    lines.extend(
        [
            "",
            "## Caveat",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in summary["caveat"])
    lines.extend(
        [
            "",
            "This script is only an offline CSV generator and preliminary checker. It deliberately leaves FK to a later MuJoCo / Isaac Lab stage and does not authorize ROS replay or real Franka execution.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json_summary(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def plot_group(plt: Any, path: Path, time: Any, values: Any, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    for joint_index in range(JOINT_COUNT):
        ax.plot(time, values[:, joint_index], label=f"q{joint_index + 1}")
    ax.set_title(title)
    ax.set_xlabel("time [s]")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_plots(output_dir: Path, prefix: str, time: Any, trajectory: dict[str, Any], include_jerk: bool) -> dict[str, str]:
    plt = require_matplotlib(output_dir)
    plot_paths = {
        "q_plot": output_dir / f"{prefix}_q.png",
        "dq_plot": output_dir / f"{prefix}_dq.png",
        "ddq_plot": output_dir / f"{prefix}_ddq.png",
    }
    plot_group(plt, plot_paths["q_plot"], time, trajectory["q"], "GOAL1 all-q joint positions", "position [rad]")
    plot_group(plt, plot_paths["dq_plot"], time, trajectory["dq"], "GOAL1 all-q joint velocities", "velocity [rad/s]")
    plot_group(plt, plot_paths["ddq_plot"], time, trajectory["ddq"], "GOAL1 all-q joint accelerations", "acceleration [rad/s^2]")
    if include_jerk:
        plot_paths["jerk_plot"] = output_dir / f"{prefix}_jerk.png"
        plot_group(plt, plot_paths["jerk_plot"], time, trajectory["jerk"], "GOAL1 all-q joint jerk", "jerk [rad/s^3]")
    return {key: str(value) for key, value in plot_paths.items()}


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
    except ValueError as exc:
        print(f"Invalid arguments: {exc}", file=sys.stderr)
        return 2

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        np = require_numpy()
        profile = PROFILES[args.profile]
        validate_profile(profile)
        time, dt = generate_time(np, args.duration, args.sample_rate)
        trajectory = generate_trajectory(np, time, profile, args.include_jerk)
        joint_summaries, safety_summary = build_safety_summary(np, trajectory, args.include_jerk)
    except ModuleNotFoundError:
        return 1

    csv_path = output_dir / f"{args.prefix}.csv"
    json_path = output_dir / f"{args.prefix}_summary.json"
    md_path = output_dir / f"{args.prefix}_summary.md"

    write_csv(csv_path, time, trajectory, args.include_jerk)

    output_paths = {
        "csv": str(csv_path),
        "json_summary": str(json_path),
        "markdown_summary": str(md_path),
    }
    if not args.no_plots:
        try:
            output_paths.update(write_plots(output_dir, args.prefix, time, trajectory, args.include_jerk))
        except ModuleNotFoundError:
            return 1

    summary = build_summary(args, dt, len(time), profile, joint_summaries, safety_summary, output_paths)
    write_json_summary(json_path, summary)
    write_markdown_summary(md_path, summary)

    print(f"CSV: {csv_path}")
    print(f"JSON summary: {json_path}")
    print(f"Markdown summary: {md_path}")
    print(f"Overall safety status: {safety_summary['overall_safety_status']}")
    print("Offline only: no ROS replay, no FK, no controller integration, no real robot validation.")

    if args.fail_on_unsafe and safety_summary["overall_safety_status"] != "safe":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
