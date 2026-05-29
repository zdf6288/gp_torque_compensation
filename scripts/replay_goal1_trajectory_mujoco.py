#!/usr/bin/env python3
"""Replay GOAL1 all-q joint positions in MuJoCo as a kinematic check.

This script is standalone and offline-only. It does not import ROS2, publish
commands, use actuator control, run torque control, enable GP, or validate real
robot safety.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


sys.dont_write_bytecode = True

DEFAULT_CSV = Path("outputs/goal1_joint_trajectory/goal1_allq_conservative.csv")
DEFAULT_MODEL = Path("/home/dummd/mujoco_models/mujoco_menagerie/franka_emika_panda/panda.xml")
DEFAULT_OUTPUT_DIR = Path("outputs/goal1_mujoco_replay")
DEFAULT_PREFIX = "goal1_allq_mujoco_replay"
DEFAULT_EE_BODY = "hand"
DEFAULT_JOINT_NAMES = "joint1,joint2,joint3,joint4,joint5,joint6,joint7"
JOINT_COUNT = 7

CAVEATS = [
    "MuJoCo standalone kinematic replay only",
    "no torque control",
    "no actuator control",
    "no ROS2 integration",
    "no real robot validation",
    "no GP-on",
    "no guarantee of controller tracking",
    "no guarantee of hardware safety",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay GOAL1 B all-q joint positions in MuJoCo and record EE xyz.",
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help=f"Default: {DEFAULT_CSV}")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help=f"Default: {DEFAULT_MODEL}")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help=f"Output filename prefix. Default: {DEFAULT_PREFIX}")
    parser.add_argument("--ee-body", default=DEFAULT_EE_BODY, help=f"End-effector body name. Default: {DEFAULT_EE_BODY}")
    parser.add_argument("--ee-site", default=None, help="End-effector site name. If provided, site has priority.")
    parser.add_argument("--joint-names", default=DEFAULT_JOINT_NAMES, help=f"Comma-separated 7 arm joints. Default: {DEFAULT_JOINT_NAMES}")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG plot generation.")
    parser.add_argument("--list-model-names", action="store_true", help="Print available bodies/sites/joints and exit.")
    return parser.parse_args()


def require_numpy() -> Any:
    try:
        import numpy as np
    except ModuleNotFoundError:
        print("Missing Python dependency: numpy", file=sys.stderr)
        print("Suggested install command: .venv/bin/python -m pip install numpy", file=sys.stderr)
        raise
    return np


def require_mujoco() -> Any:
    try:
        import mujoco
    except ModuleNotFoundError:
        print("Missing Python dependency: mujoco", file=sys.stderr)
        print("Suggested install command: .venv/bin/python -m pip install mujoco", file=sys.stderr)
        raise
    return mujoco


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
        print("Warning: matplotlib is not available; skipping plots.", file=sys.stderr)
        return None
    return plt


def load_model(mujoco: Any, model_path: Path) -> Any:
    if not model_path.exists():
        raise FileNotFoundError(f"MuJoCo model not found: {model_path}")
    return mujoco.MjModel.from_xml_path(str(model_path))


def model_names(mujoco: Any, model: Any, object_type: Any, count: int) -> list[str]:
    names = []
    for index in range(count):
        name = mujoco.mj_id2name(model, object_type, index)
        if name is not None:
            names.append(name)
    return names


def available_names(mujoco: Any, model: Any) -> dict[str, list[str]]:
    return {
        "bodies": model_names(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, model.nbody),
        "sites": model_names(mujoco, model, mujoco.mjtObj.mjOBJ_SITE, model.nsite),
        "joints": model_names(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt),
    }


def print_model_names(mujoco: Any, model: Any) -> None:
    names = available_names(mujoco, model)
    print(f"nq: {model.nq}")
    print(f"nv: {model.nv}")
    print(f"nu: {model.nu}")
    print(f"nbody: {model.nbody}")
    print(f"nsite: {model.nsite}")
    print("Bodies:")
    print_names(names["bodies"])
    print("Sites:")
    print_names(names["sites"])
    print("Joints:")
    print_names(names["joints"])


def print_names(names: list[str]) -> None:
    if not names:
        print("  (none)")
        return
    for name in names:
        print(f"  {name}")


def parse_joint_names(raw_joint_names: str) -> list[str]:
    joint_names = [name.strip() for name in raw_joint_names.split(",") if name.strip()]
    if len(joint_names) != JOINT_COUNT:
        raise ValueError(f"--joint-names must contain exactly {JOINT_COUNT} names, got {len(joint_names)}")
    return joint_names


def validate_csv_path(csv_path: Path) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"GOAL1 B CSV not found: {csv_path}. Run the GOAL1 B generator first before MuJoCo replay."
        )


def read_goal1_csv(csv_path: Path, np: Any) -> tuple[Any, Any]:
    validate_csv_path(csv_path)
    required_columns = ["time"] + [f"joint_pos_{index}" for index in range(1, JOINT_COUNT + 1)]

    time_values: list[float] = []
    q_values: list[list[float]] = []
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        missing = [column for column in required_columns if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"CSV missing required columns: {', '.join(missing)}")

        for row_index, row in enumerate(reader, start=2):
            try:
                time_values.append(float(row["time"]))
                q_values.append([float(row[f"joint_pos_{index}"]) for index in range(1, JOINT_COUNT + 1)])
            except ValueError as exc:
                raise ValueError(f"Invalid numeric value in CSV row {row_index}: {exc}") from exc

    if not time_values:
        raise ValueError(f"CSV contains no samples: {csv_path}")
    return np.asarray(time_values, dtype=float), np.asarray(q_values, dtype=float)


def resolve_joint_qpos_addresses(mujoco: Any, model: Any, joint_names: list[str]) -> list[int]:
    addresses = []
    missing = []
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            missing.append(joint_name)
            continue
        qposadr = int(model.jnt_qposadr[joint_id])
        if qposadr < 0 or qposadr >= model.nq:
            raise ValueError(f"Joint {joint_name} has invalid qpos address: {qposadr}")
        addresses.append(qposadr)

    if missing:
        print(f"Missing joint name(s): {', '.join(missing)}", file=sys.stderr)
        print("Available joints:", file=sys.stderr)
        print_names_to_stderr(available_names(mujoco, model)["joints"])
        raise ValueError("Joint name validation failed")
    return addresses


def resolve_ee_selection(mujoco: Any, model: Any, ee_site: str | None, ee_body: str | None) -> tuple[str, int, str]:
    names = available_names(mujoco, model)
    if ee_site:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site)
        if site_id < 0:
            print(f"Missing ee site: {ee_site}", file=sys.stderr)
            print("Available sites:", file=sys.stderr)
            print_names_to_stderr(names["sites"])
            print("Available bodies:", file=sys.stderr)
            print_names_to_stderr(names["bodies"])
            raise ValueError("EE site validation failed")
        return "site", int(site_id), ee_site

    if not ee_body:
        raise ValueError("Either --ee-site or --ee-body must be provided")

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_body)
    if body_id < 0:
        print(f"Missing ee body: {ee_body}", file=sys.stderr)
        print("Available bodies:", file=sys.stderr)
        print_names_to_stderr(names["bodies"])
        raise ValueError("EE body validation failed")
    return "body", int(body_id), ee_body


def print_names_to_stderr(names: list[str]) -> None:
    if not names:
        print("  (none)", file=sys.stderr)
        return
    for name in names:
        print(f"  {name}", file=sys.stderr)


def replay_kinematic(
    mujoco: Any,
    np: Any,
    model: Any,
    time_values: Any,
    q_values: Any,
    qpos_addresses: list[int],
    ee_kind: str,
    ee_id: int,
) -> Any:
    data = mujoco.MjData(model)
    ee_positions = np.zeros((len(time_values), 3), dtype=float)

    for row_index in range(len(time_values)):
        data.qpos[:] = model.qpos0
        for joint_index, qposadr in enumerate(qpos_addresses):
            data.qpos[qposadr] = q_values[row_index, joint_index]
        mujoco.mj_forward(model, data)

        if ee_kind == "site":
            ee_positions[row_index, :] = data.site_xpos[ee_id]
        else:
            ee_positions[row_index, :] = data.xpos[ee_id]

    return ee_positions


def write_ee_path_csv(path: Path, time_values: Any, q_values: Any, ee_positions: Any) -> None:
    columns = ["time", "ee_x", "ee_y", "ee_z"]
    columns.extend(f"joint_pos_{index}" for index in range(1, JOINT_COUNT + 1))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for row_index, time_value in enumerate(time_values):
            row = [
                f"{float(time_value):.9f}",
                f"{float(ee_positions[row_index, 0]):.12f}",
                f"{float(ee_positions[row_index, 1]):.12f}",
                f"{float(ee_positions[row_index, 2]):.12f}",
            ]
            row.extend(f"{float(value):.12f}" for value in q_values[row_index, :])
            writer.writerow(row)


def make_summary(
    np: Any,
    args: argparse.Namespace,
    model: Any,
    joint_names: list[str],
    ee_kind: str,
    ee_name: str,
    time_values: Any,
    ee_positions: Any,
    output_paths: dict[str, Path],
) -> dict[str, Any]:
    ee_min = np.min(ee_positions, axis=0)
    ee_max = np.max(ee_positions, axis=0)
    ee_range = ee_max - ee_min
    dt_median = float(np.median(np.diff(time_values))) if len(time_values) > 1 else 0.0
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "csv_path": str(args.csv),
        "model_path": str(args.model),
        "model": {
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(model.nu),
            "nbody": int(model.nbody),
            "nsite": int(model.nsite),
        },
        "selected_joint_names": joint_names,
        "selected_ee": {
            "kind": ee_kind,
            "name": ee_name,
        },
        "number_of_samples": int(len(time_values)),
        "time_start": float(time_values[0]),
        "time_end": float(time_values[-1]),
        "dt_median": dt_median,
        "ee_xyz": {
            "min": axis_dict(ee_min),
            "max": axis_dict(ee_max),
            "range": axis_dict(ee_range),
        },
        "output_paths": {key: str(value) for key, value in output_paths.items()},
        "caveats": CAVEATS,
    }


def axis_dict(values: Any) -> dict[str, float]:
    return {
        "x": float(values[0]),
        "y": float(values[1]),
        "z": float(values[2]),
    }


def write_summary_json(path: Path, summary: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    selected_ee = summary["selected_ee"]
    ee_range = summary["ee_xyz"]["range"]
    with path.open("w", encoding="utf-8") as stream:
        stream.write("# GOAL1 MuJoCo Kinematic Replay Summary\n\n")
        stream.write("## Inputs\n\n")
        stream.write(f"- csv_path: `{summary['csv_path']}`\n")
        stream.write(f"- model_path: `{summary['model_path']}`\n")
        stream.write(f"- selected_ee: `{selected_ee['kind']}:{selected_ee['name']}`\n")
        stream.write(f"- selected_joint_names: `{', '.join(summary['selected_joint_names'])}`\n\n")
        stream.write("## Model\n\n")
        for key, value in summary["model"].items():
            stream.write(f"- {key}: `{value}`\n")
        stream.write("\n## Replay\n\n")
        stream.write(f"- number_of_samples: `{summary['number_of_samples']}`\n")
        stream.write(f"- time_start: `{summary['time_start']}`\n")
        stream.write(f"- time_end: `{summary['time_end']}`\n")
        stream.write(f"- dt_median: `{summary['dt_median']}`\n\n")
        stream.write("## EE xyz range\n\n")
        stream.write(f"- x_range: `{ee_range['x']}`\n")
        stream.write(f"- y_range: `{ee_range['y']}`\n")
        stream.write(f"- z_range: `{ee_range['z']}`\n\n")
        stream.write("## Caveats\n\n")
        for caveat in summary["caveats"]:
            stream.write(f"- {caveat}\n")


def make_plots(output_dir: Path, prefix: str, time_values: Any, ee_positions: Any) -> dict[str, Path]:
    plt = require_matplotlib(output_dir)
    if plt is None:
        return {}

    output_paths: dict[str, Path] = {}
    xyz_path = output_dir / f"{prefix}_ee_xyz.png"
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    labels = ("x", "y", "z")
    for axis_index, axis in enumerate(axes):
        axis.plot(time_values, ee_positions[:, axis_index], linewidth=1.5)
        axis.set_ylabel(f"ee_{labels[axis_index]} [m]")
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("GOAL1 MuJoCo EE xyz vs time")
    fig.tight_layout()
    fig.savefig(xyz_path, dpi=160)
    plt.close(fig)
    output_paths["ee_xyz_plot"] = xyz_path

    path_3d = output_dir / f"{prefix}_ee_path_3d.png"
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(8, 7))
        axis = fig.add_subplot(111, projection="3d")
        axis.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2], linewidth=1.5)
        axis.scatter(ee_positions[0, 0], ee_positions[0, 1], ee_positions[0, 2], s=30, label="start")
        axis.scatter(ee_positions[-1, 0], ee_positions[-1, 1], ee_positions[-1, 2], s=30, label="end")
        axis.set_xlabel("ee_x [m]")
        axis.set_ylabel("ee_y [m]")
        axis.set_zlabel("ee_z [m]")
        axis.set_title("GOAL1 MuJoCo EE 3D path")
        axis.legend()
        fig.tight_layout()
        fig.savefig(path_3d, dpi=160)
        plt.close(fig)
        output_paths["ee_path_3d_plot"] = path_3d
    except Exception as exc:
        print(f"Warning: failed to generate 3D plot, continuing without it: {exc}", file=sys.stderr)

    return output_paths


def run(args: argparse.Namespace) -> int:
    np = require_numpy()
    mujoco = require_mujoco()
    model = load_model(mujoco, args.model)

    if args.list_model_names:
        print_model_names(mujoco, model)
        return 0

    joint_names = parse_joint_names(args.joint_names)
    time_values, q_values = read_goal1_csv(args.csv, np)
    qpos_addresses = resolve_joint_qpos_addresses(mujoco, model, joint_names)
    ee_kind, ee_id, ee_name = resolve_ee_selection(mujoco, model, args.ee_site, args.ee_body)

    ee_positions = replay_kinematic(mujoco, np, model, time_values, q_values, qpos_addresses, ee_kind, ee_id)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "ee_path_csv": args.output_dir / f"{args.prefix}_ee_path.csv",
        "summary_json": args.output_dir / f"{args.prefix}_summary.json",
        "summary_md": args.output_dir / f"{args.prefix}_summary.md",
    }
    write_ee_path_csv(output_paths["ee_path_csv"], time_values, q_values, ee_positions)

    if not args.no_plots:
        output_paths.update(make_plots(args.output_dir, args.prefix, time_values, ee_positions))

    summary = make_summary(np, args, model, joint_names, ee_kind, ee_name, time_values, ee_positions, output_paths)
    write_summary_json(output_paths["summary_json"], summary)
    write_summary_md(output_paths["summary_md"], summary)

    ee_range = summary["ee_xyz"]["range"]
    print("GOAL1 MuJoCo kinematic replay completed.")
    print(f"Selected EE: {ee_kind}:{ee_name}")
    print(f"Samples: {summary['number_of_samples']}")
    print(f"Time: {summary['time_start']} to {summary['time_end']} s")
    print(f"EE range [m]: x={ee_range['x']:.6f}, y={ee_range['y']:.6f}, z={ee_range['z']:.6f}")
    print(f"Output dir: {args.output_dir}")
    return 0


def main() -> int:
    try:
        return run(parse_args())
    except (FileNotFoundError, ModuleNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
