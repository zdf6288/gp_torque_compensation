#!/usr/bin/env python3
"""Render GOAL1 all-q FR3 MuJoCo kinematic replay to a video file.

This script is standalone and offline-only. It does not import ROS2, publish
commands, use actuator control, run torque control, enable GP, or validate real
robot safety.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


sys.dont_write_bytecode = True

DEFAULT_CSV = Path("outputs/goal1_joint_trajectory/goal1_allq_conservative.csv")
DEFAULT_MODEL = Path("/home/dummd/mujoco_models/mujoco_menagerie/franka_fr3/fr3.xml")
DEFAULT_OUTPUT_DIR = Path("outputs/goal1_mujoco_video")
DEFAULT_PREFIX = "goal1_allq_fr3_mujoco_video"
DEFAULT_JOINT_NAMES = "fr3_joint1,fr3_joint2,fr3_joint3,fr3_joint4,fr3_joint5,fr3_joint6,fr3_joint7"
JOINT_COUNT = 7

CAVEATS = [
    "FR3 MuJoCo kinematic replay video only",
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
        description="Render GOAL1 B all-q joint positions in an FR3 MuJoCo model.",
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help=f"Default: {DEFAULT_CSV}")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help=f"Default: {DEFAULT_MODEL}")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help=f"Output filename prefix. Default: {DEFAULT_PREFIX}")
    parser.add_argument("--joint-names", default=DEFAULT_JOINT_NAMES, help=f"Comma-separated 7 arm joints. Default: {DEFAULT_JOINT_NAMES}")
    parser.add_argument("--width", type=positive_int, default=1280, help="Rendered frame width in pixels. Default: 1280")
    parser.add_argument("--height", type=positive_int, default=720, help="Rendered frame height in pixels. Default: 720")
    parser.add_argument("--fps", type=positive_int, default=30, help="Output video FPS. Default: 30")
    parser.add_argument("--playback-speed", type=positive_float, default=1.0, help="Source-time playback multiplier. Default: 1.0")
    parser.add_argument("--max-frames", type=positive_int, default=None, help="Optional cap for quick test renders.")
    parser.add_argument("--frame-stride", type=positive_int, default=1, help="Render every Nth planned video frame. Default: 1")
    parser.add_argument("--camera", default=None, help="Optional MuJoCo camera name. Uses default camera when omitted.")
    parser.add_argument("--output-format", choices=("mp4", "gif"), default="mp4", help="Video format. Default: mp4")
    parser.add_argument("--list-model-names", action="store_true", help="Print available bodies/sites/joints/cameras and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and frame plan without rendering.")
    return parser.parse_args()


def positive_int(raw_value: str) -> int:
    value = int(raw_value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def positive_float(raw_value: str) -> float:
    value = float(raw_value)
    if value <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


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


def package_available(package_name: str) -> bool:
    return importlib.util.find_spec(package_name) is not None


def writer_availability() -> dict[str, bool]:
    return {
        "imageio": package_available("imageio"),
        "imageio_ffmpeg": package_available("imageio_ffmpeg"),
        "cv2": package_available("cv2"),
    }


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
        "cameras": model_names(mujoco, model, mujoco.mjtObj.mjOBJ_CAMERA, model.ncam),
    }


def print_model_names(mujoco: Any, model: Any) -> None:
    names = available_names(mujoco, model)
    print(f"nq: {model.nq}")
    print(f"nv: {model.nv}")
    print(f"nu: {model.nu}")
    print(f"nbody: {model.nbody}")
    print(f"nsite: {model.nsite}")
    print(f"ncam: {model.ncam}")
    print("Bodies:")
    print_names(names["bodies"])
    print("Sites:")
    print_names(names["sites"])
    print("Joints:")
    print_names(names["joints"])
    print("Cameras:")
    print_names(names["cameras"])


def print_names(names: list[str]) -> None:
    if not names:
        print("  (none)")
        return
    for name in names:
        print(f"  {name}")


def print_names_to_stderr(names: list[str]) -> None:
    if not names:
        print("  (none)", file=sys.stderr)
        return
    for name in names:
        print(f"  {name}", file=sys.stderr)


def parse_joint_names(raw_joint_names: str) -> list[str]:
    joint_names = [name.strip() for name in raw_joint_names.split(",") if name.strip()]
    if len(joint_names) != JOINT_COUNT:
        raise ValueError(f"--joint-names must contain exactly {JOINT_COUNT} names, got {len(joint_names)}")
    return joint_names


def validate_csv_path(csv_path: Path) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"GOAL1 B CSV not found: {csv_path}. Run the GOAL1 B generator first before MuJoCo video rendering."
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


def validate_camera(mujoco: Any, model: Any, camera: str | None) -> None:
    if camera is None:
        return
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if camera_id < 0:
        print(f"Missing camera: {camera}", file=sys.stderr)
        print("Available cameras:", file=sys.stderr)
        print_names_to_stderr(available_names(mujoco, model)["cameras"])
        raise ValueError("Camera validation failed")


def build_frame_plan(
    np: Any,
    time_values: Any,
    fps: int,
    playback_speed: float,
    frame_stride: int,
    max_frames: int | None,
) -> dict[str, Any]:
    source_start = float(time_values[0])
    source_end = float(time_values[-1])
    source_duration = max(0.0, source_end - source_start)
    source_step = playback_speed * frame_stride / float(fps)

    if source_duration == 0.0:
        target_times = np.asarray([source_start], dtype=float)
    else:
        frame_count = int(np.floor(source_duration / source_step)) + 1
        target_times = source_start + np.arange(frame_count, dtype=float) * source_step
        target_times = np.minimum(target_times, source_end)

    if max_frames is not None:
        target_times = target_times[:max_frames]

    indices = nearest_sample_indices(np, time_values, target_times)
    return {
        "indices": indices,
        "target_times": target_times,
        "source_step": float(source_step),
        "planned_frame_count": int(len(indices)),
    }


def nearest_sample_indices(np: Any, time_values: Any, target_times: Any) -> Any:
    right = np.searchsorted(time_values, target_times, side="left")
    right = np.clip(right, 0, len(time_values) - 1)
    left = np.clip(right - 1, 0, len(time_values) - 1)
    use_right = np.abs(time_values[right] - target_times) <= np.abs(target_times - time_values[left])
    return np.where(use_right, right, left).astype(int)


class ImageioVideoWriter:
    def __init__(self, path: Path, fps: int, output_format: str) -> None:
        import imageio.v2 as imageio

        self._writer = imageio.get_writer(str(path), fps=fps, format=output_format)
        self.backend = f"imageio-{output_format}"

    def append(self, frame: Any) -> None:
        self._writer.append_data(frame)

    def close(self) -> None:
        self._writer.close()


class Cv2VideoWriter:
    def __init__(self, path: Path, fps: int, width: int, height: int) -> None:
        import cv2

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._cv2 = cv2
        self._writer = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))
        if not self._writer.isOpened():
            raise RuntimeError(f"cv2 failed to open video writer: {path}")
        self.backend = "cv2-mp4v"

    def append(self, frame: Any) -> None:
        self._writer.write(self._cv2.cvtColor(frame, self._cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        self._writer.release()


def open_video_writer(path: Path, fps: int, width: int, height: int, output_format: str) -> Any:
    availability = writer_availability()
    if output_format == "gif" and availability["imageio"]:
        return ImageioVideoWriter(path, fps, output_format)
    if output_format == "mp4" and availability["imageio"] and availability["imageio_ffmpeg"]:
        return ImageioVideoWriter(path, fps, output_format)
    if output_format == "mp4" and availability["cv2"]:
        return Cv2VideoWriter(path, fps, width, height)

    missing = []
    if output_format == "mp4":
        missing.append("imageio + imageio_ffmpeg, or opencv-python")
    else:
        missing.append("imageio")
    raise RuntimeError(
        "No usable video writer package is installed. "
        f"Missing: {', '.join(missing)}. "
        "Suggested install command: .venv/bin/python -m pip install imageio imageio-ffmpeg "
        "or .venv/bin/python -m pip install opencv-python. "
        "No package was installed automatically."
    )


def render_video(
    mujoco: Any,
    model: Any,
    q_values: Any,
    qpos_addresses: list[int],
    frame_indices: Any,
    args: argparse.Namespace,
    video_path: Path,
) -> tuple[int, str]:
    writer = open_video_writer(video_path, args.fps, args.width, args.height, args.output_format)
    renderer = None
    frame_count = 0
    try:
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        for source_index in frame_indices:
            data.qpos[:] = model.qpos0
            for joint_index, qposadr in enumerate(qpos_addresses):
                data.qpos[qposadr] = q_values[int(source_index), joint_index]
            mujoco.mj_forward(model, data)
            if args.camera is None:
                renderer.update_scene(data)
            else:
                renderer.update_scene(data, camera=args.camera)
            writer.append(renderer.render())
            frame_count += 1
    except Exception as exc:
        raise RuntimeError(
            f"MuJoCo rendering or video writing failed: {exc}. "
            "If this is a headless/OpenGL issue, try running with MUJOCO_GL=egl, "
            "MUJOCO_GL=osmesa, or on a machine with a working display environment. "
            "Do not modify shell profile or system OpenGL configuration without review."
        ) from exc
    finally:
        if renderer is not None:
            renderer.close()
        writer.close()
    return frame_count, writer.backend


def make_summary(
    args: argparse.Namespace,
    model: Any,
    joint_names: list[str],
    time_values: Any,
    frame_plan: dict[str, Any],
    video_path: Path,
    rendering_succeeded: bool,
    writer_backend: str | None,
    rendered_frame_count: int,
    error: str | None,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "csv_path": str(args.csv),
        "model_path": str(args.model),
        "selected_joint_names": joint_names,
        "model": {
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(model.nu),
            "nbody": int(model.nbody),
            "nsite": int(model.nsite),
        },
        "video_path": str(video_path),
        "output_format": args.output_format,
        "width": int(args.width),
        "height": int(args.height),
        "fps": int(args.fps),
        "playback_speed": float(args.playback_speed),
        "frame_stride": int(args.frame_stride),
        "source_time_step_per_rendered_frame": frame_plan["source_step"],
        "rendered_frame_count": int(rendered_frame_count),
        "planned_frame_count": int(frame_plan["planned_frame_count"]),
        "source_sample_count": int(len(time_values)),
        "source_time_start": float(time_values[0]),
        "source_time_end": float(time_values[-1]),
        "rendering_succeeded": bool(rendering_succeeded),
        "writer_backend_used": writer_backend,
        "writer_availability": writer_availability(),
        "error": error,
        "caveats": CAVEATS,
    }


def write_summary_json(path: Path, summary: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        stream.write("# GOAL1 FR3 MuJoCo Video Export Summary\n\n")
        stream.write("## Inputs\n\n")
        stream.write(f"- csv_path: `{summary['csv_path']}`\n")
        stream.write(f"- model_path: `{summary['model_path']}`\n")
        stream.write(f"- selected_joint_names: `{', '.join(summary['selected_joint_names'])}`\n\n")
        stream.write("## Model\n\n")
        for key, value in summary["model"].items():
            stream.write(f"- {key}: `{value}`\n")
        stream.write("\n## Video\n\n")
        video_keys = [
            "video_path",
            "output_format",
            "width",
            "height",
            "fps",
            "playback_speed",
            "frame_stride",
            "source_time_step_per_rendered_frame",
            "rendered_frame_count",
            "planned_frame_count",
            "source_sample_count",
            "source_time_start",
            "source_time_end",
            "rendering_succeeded",
            "writer_backend_used",
        ]
        for key in video_keys:
            stream.write(f"- {key}: `{summary[key]}`\n")
        stream.write("\n## Writer Availability\n\n")
        for key, value in summary["writer_availability"].items():
            stream.write(f"- {key}: `{value}`\n")
        if summary["error"]:
            stream.write("\n## Error\n\n")
            stream.write(f"`{summary['error']}`\n")
        stream.write("\n## Caveats\n\n")
        for caveat in summary["caveats"]:
            stream.write(f"- {caveat}\n")


def print_dry_run(args: argparse.Namespace, model: Any, joint_names: list[str], time_values: Any, frame_plan: dict[str, Any]) -> None:
    print("GOAL1 FR3 MuJoCo video dry-run completed.")
    print(f"CSV: {args.csv}")
    print(f"Model: {args.model}")
    print(f"Joints: {', '.join(joint_names)}")
    print(f"Source samples: {len(time_values)}")
    print(f"Source time: {float(time_values[0])} to {float(time_values[-1])} s")
    print(f"Model sizes: nq={model.nq}, nv={model.nv}, nu={model.nu}, nbody={model.nbody}, nsite={model.nsite}")
    print(f"Frame plan: {frame_plan['planned_frame_count']} frames at {args.fps} fps")
    print(f"Playback speed: {args.playback_speed}")
    print(f"Frame stride: {args.frame_stride}")
    print(f"Writer availability: {writer_availability()}")
    print("No video was rendered because --dry-run was used.")


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
    validate_camera(mujoco, model, args.camera)
    frame_plan = build_frame_plan(np, time_values, args.fps, args.playback_speed, args.frame_stride, args.max_frames)

    if args.dry_run:
        print_dry_run(args, model, joint_names, time_values, frame_plan)
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.output_dir / f"{args.prefix}.{args.output_format}"
    summary_json = args.output_dir / f"{args.prefix}_summary.json"
    summary_md = args.output_dir / f"{args.prefix}_summary.md"

    rendered_frame_count = 0
    writer_backend = None
    error = None
    try:
        rendered_frame_count, writer_backend = render_video(
            mujoco,
            model,
            q_values,
            qpos_addresses,
            frame_plan["indices"],
            args,
            video_path,
        )
    except RuntimeError as exc:
        error = str(exc)
        summary = make_summary(
            args,
            model,
            joint_names,
            time_values,
            frame_plan,
            video_path,
            rendering_succeeded=False,
            writer_backend=writer_backend,
            rendered_frame_count=rendered_frame_count,
            error=error,
        )
        write_summary_json(summary_json, summary)
        write_summary_md(summary_md, summary)
        print(f"Error: {error}", file=sys.stderr)
        print(f"Wrote failure summary: {summary_json}")
        print(f"Wrote failure summary: {summary_md}")
        return 1

    summary = make_summary(
        args,
        model,
        joint_names,
        time_values,
        frame_plan,
        video_path,
        rendering_succeeded=True,
        writer_backend=writer_backend,
        rendered_frame_count=rendered_frame_count,
        error=None,
    )
    write_summary_json(summary_json, summary)
    write_summary_md(summary_md, summary)

    print("GOAL1 FR3 MuJoCo kinematic video export completed.")
    print(f"Video: {video_path}")
    print(f"Rendered frames: {rendered_frame_count}")
    print(f"Writer backend: {writer_backend}")
    print(f"Summary JSON: {summary_json}")
    print(f"Summary MD: {summary_md}")
    return 0


def main() -> int:
    try:
        return run(parse_args())
    except (FileNotFoundError, ModuleNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
