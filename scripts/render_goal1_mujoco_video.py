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
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


sys.dont_write_bytecode = True

DEFAULT_CSV = Path("outputs/goal1_joint_trajectory/goal1_allq_conservative.csv")
DEFAULT_MODEL = Path("/home/dummd/mujoco_models/mujoco_menagerie/franka_fr3/fr3.xml")
DEFAULT_OUTPUT_DIR = Path("outputs/goal1_mujoco_video")
DEFAULT_TEMP_MODEL_DIR = DEFAULT_OUTPUT_DIR / "temp_models"
DEFAULT_PREFIX = "goal1_allq_fr3_mujoco_video"
DEFAULT_JOINT_NAMES = "fr3_joint1,fr3_joint2,fr3_joint3,fr3_joint4,fr3_joint5,fr3_joint6,fr3_joint7"
DEFAULT_EE_SITE = "attachment_site"
DEFAULT_TRACE_COLOR = "0.0,1.0,0.1,1.0"
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
    parser.add_argument("--width", type=positive_int, default=640, help="Rendered frame width in pixels. Default: 640")
    parser.add_argument("--height", type=positive_int, default=480, help="Rendered frame height in pixels. Default: 480")
    parser.add_argument("--fps", type=positive_int, default=30, help="Output video FPS. Default: 30")
    parser.add_argument("--playback-speed", type=positive_float, default=1.0, help="Source-time playback multiplier. Default: 1.0")
    parser.add_argument("--start-time", type=nonnegative_float, default=0.0, help="Source time to start rendering from, in seconds. Default: 0.0")
    parser.add_argument("--max-frames", type=positive_int, default=None, help="Optional cap for quick test renders.")
    parser.add_argument("--frame-stride", type=positive_int, default=1, help="Render every Nth planned video frame. Default: 1")
    parser.add_argument("--auto-offscreen-xml", action="store_true", help="Generate an ignored temporary MJCF with visual/global offwidth/offheight.")
    parser.add_argument("--offscreen-width", type=positive_int, default=None, help="Temporary MJCF offscreen width. Default: --width")
    parser.add_argument("--offscreen-height", type=positive_int, default=None, help="Temporary MJCF offscreen height. Default: --height")
    parser.add_argument("--temp-model-dir", type=Path, default=DEFAULT_TEMP_MODEL_DIR, help=f"Temporary MJCF output directory. Default: {DEFAULT_TEMP_MODEL_DIR}")
    parser.add_argument("--camera", default=None, help="Optional MuJoCo model-defined camera name. Overrides --camera-preset when set.")
    parser.add_argument(
        "--camera-preset",
        choices=("default", "front", "side", "top", "iso", "close_iso"),
        default="iso",
        help="Free-camera preset used when --camera is omitted. Default: iso",
    )
    parser.add_argument("--show-timestamp", action="store_true", help="Draw replay title and time on each rendered frame.")
    parser.add_argument("--timestamp-format", default="t={time:.2f}s", help="Timestamp format string. Default: t={time:.2f}s")
    parser.add_argument("--ee-site", default=DEFAULT_EE_SITE, help=f"End-effector site used for trace output. Default: {DEFAULT_EE_SITE}")
    parser.add_argument("--show-ee-trace", action="store_true", help="Draw EE history markers in the rendered MuJoCo scene.")
    parser.add_argument("--trace-length", type=nonnegative_int, default=0, help="Trace history in rendered frames. 0 keeps all history. Default: 0")
    parser.add_argument("--trace-radius", type=positive_float, default=0.02, help="EE trace marker radius in meters. Default: 0.02")
    parser.add_argument("--trace-color", type=parse_trace_color, default=parse_trace_color(DEFAULT_TRACE_COLOR), help=f"RGBA trace color, comma-separated. Default: {DEFAULT_TRACE_COLOR}")
    parser.add_argument("--output-format", choices=("mp4", "gif"), default="mp4", help="Video format. Default: mp4")
    parser.add_argument("--list-model-names", action="store_true", help="Print available bodies/sites/joints/cameras and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and frame plan without rendering.")
    return parser.parse_args()


def positive_int(raw_value: str) -> int:
    value = int(raw_value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def nonnegative_int(raw_value: str) -> int:
    value = int(raw_value)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return value


def positive_float(raw_value: str) -> float:
    value = float(raw_value)
    if value <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def nonnegative_float(raw_value: str) -> float:
    value = float(raw_value)
    if value < 0.0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return value


def parse_trace_color(raw_value: str) -> tuple[float, float, float, float]:
    try:
        values = [float(part.strip()) for part in raw_value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--trace-color values must be numeric") from exc
    if len(values) not in (3, 4):
        raise argparse.ArgumentTypeError("--trace-color must contain 3 RGB or 4 RGBA values")
    if any(value < 0.0 or value > 1.0 for value in values):
        raise argparse.ArgumentTypeError("--trace-color values must be between 0.0 and 1.0")
    if len(values) == 3:
        values.append(1.0)
    return tuple(values)


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


def prepare_effective_model_path(args: argparse.Namespace) -> dict[str, Any]:
    offscreen_width = args.offscreen_width or args.width
    offscreen_height = args.offscreen_height or args.height
    info = {
        "original_model_path": str(args.model),
        "effective_model_path": str(args.model),
        "auto_offscreen_xml": bool(args.auto_offscreen_xml),
        "offscreen_width": int(offscreen_width),
        "offscreen_height": int(offscreen_height),
        "patched_xml_created": False,
        "patched_xml_path": None,
    }
    if not args.auto_offscreen_xml:
        return info

    model_path = args.model
    if not model_path.exists():
        raise FileNotFoundError(f"MuJoCo model not found: {model_path}")

    source_dir = model_path.parent.resolve()
    tree = ET.parse(model_path)
    root = tree.getroot()

    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    absolutize_compiler_path(compiler, "meshdir", source_dir)
    absolutize_compiler_path(compiler, "texturedir", source_dir)

    for include in root.findall(".//include"):
        file_value = include.get("file")
        if file_value:
            include.set("file", str(resolve_xml_relative_path(file_value, source_dir)))

    visual = root.find("visual")
    if visual is None:
        visual = ET.Element("visual")
        insert_index = list(root).index(compiler) + 1 if compiler in list(root) else 0
        root.insert(insert_index, visual)
    global_visual = visual.find("global")
    if global_visual is None:
        global_visual = ET.SubElement(visual, "global")
    global_visual.set("offwidth", str(offscreen_width))
    global_visual.set("offheight", str(offscreen_height))

    args.temp_model_dir.mkdir(parents=True, exist_ok=True)
    patched_path = args.temp_model_dir / f"{model_path.stem}_offscreen_{offscreen_width}x{offscreen_height}.xml"
    ET.indent(tree, space="  ")
    tree.write(patched_path, encoding="utf-8", xml_declaration=False)

    info["effective_model_path"] = str(patched_path)
    info["patched_xml_created"] = True
    info["patched_xml_path"] = str(patched_path)
    return info


def absolutize_compiler_path(compiler: Any, attribute: str, source_dir: Path) -> None:
    raw_value = compiler.get(attribute)
    if raw_value:
        compiler.set(attribute, str(resolve_xml_relative_path(raw_value, source_dir)))
    elif attribute == "meshdir":
        compiler.set(attribute, str(source_dir))


def resolve_xml_relative_path(raw_value: str, source_dir: Path) -> Path:
    path = Path(raw_value)
    if path.is_absolute():
        return path
    return (source_dir / path).resolve()


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


def validate_site(mujoco: Any, model: Any, site_name: str) -> int:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        print(f"Missing site: {site_name}", file=sys.stderr)
        print("Available sites:", file=sys.stderr)
        print_names_to_stderr(available_names(mujoco, model)["sites"])
        raise ValueError("EE site validation failed")
    return int(site_id)


def build_camera(mujoco: Any, model: Any, camera_preset: str) -> Any | None:
    if camera_preset == "default":
        return None

    center = [float(value) for value in model.stat.center]
    if len(center) >= 3:
        center[2] = max(center[2], 0.45)
    extent = max(float(model.stat.extent), 1.0)

    presets = {
        "front": {"azimuth": 180.0, "elevation": -20.0, "distance": 1.45 * extent},
        "side": {"azimuth": 90.0, "elevation": -20.0, "distance": 1.45 * extent},
        "top": {"azimuth": 180.0, "elevation": -75.0, "distance": 2.0 * extent},
        "iso": {"azimuth": 135.0, "elevation": -25.0, "distance": 1.45 * extent},
        "close_iso": {"azimuth": 135.0, "elevation": -22.0, "distance": 1.15 * extent},
    }
    if camera_preset not in presets:
        raise ValueError(f"Unsupported camera preset: {camera_preset}")

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = center[:3]
    camera.distance = presets[camera_preset]["distance"]
    camera.azimuth = presets[camera_preset]["azimuth"]
    camera.elevation = presets[camera_preset]["elevation"]
    return camera


def build_frame_plan(
    np: Any,
    time_values: Any,
    fps: int,
    playback_speed: float,
    frame_stride: int,
    start_time: float,
    max_frames: int | None,
) -> dict[str, Any]:
    csv_start = float(time_values[0])
    source_end = float(time_values[-1])
    if start_time > source_end:
        raise ValueError(f"--start-time {start_time} is after CSV end time {source_end}")
    source_start = max(csv_start, float(start_time))
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


def draw_text_overlay(np: Any, frame: Any, lines: list[str]) -> Any:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()

    image_width, image_height = image.size
    padding = 8
    line_spacing = 4
    text_sizes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    text_width = max(box[2] - box[0] for box in text_sizes)
    text_height = sum(box[3] - box[1] for box in text_sizes) + line_spacing * (len(lines) - 1)

    x = min(10, max(0, image_width - text_width - 2 * padding - 1))
    y = min(10, max(0, image_height - text_height - 2 * padding - 1))
    box_right = min(image_width - 1, x + text_width + 2 * padding)
    box_bottom = min(image_height - 1, y + text_height + 2 * padding)
    draw.rounded_rectangle(
        (x, y, box_right, box_bottom),
        radius=3,
        fill=(0, 0, 0, 175),
    )

    cursor_y = y + padding
    for line, box in zip(lines, text_sizes):
        draw.text((x + padding, cursor_y), line, fill=(255, 255, 255, 255), font=font)
        cursor_y += (box[3] - box[1]) + line_spacing
    return np.asarray(image)


def draw_ee_trace_overlay(np: Any, frame: Any, trace_rows: list[dict[str, float | int]], trace_length: int) -> Any:
    if len(trace_rows) < 2:
        return frame

    from PIL import Image, ImageDraw, ImageFont

    visible_rows = trace_rows[-trace_length:] if trace_length > 0 else trace_rows
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    image_width, image_height = image.size

    panel_width = min(260, max(180, image_width // 5))
    panel_height = min(180, max(130, image_height // 4))
    margin = 14
    x0 = max(0, image_width - panel_width - margin)
    y0 = max(0, image_height - panel_height - margin)
    x1 = image_width - margin
    y1 = image_height - margin

    draw.rounded_rectangle((x0, y0, x1, y1), radius=4, fill=(0, 0, 0, 150), outline=(0, 255, 80, 210), width=2)
    draw.text((x0 + 10, y0 + 8), "EE trace", fill=(255, 255, 255, 255), font=font)

    plot_x0 = x0 + 16
    plot_y0 = y0 + 28
    plot_x1 = x1 - 14
    plot_y1 = y1 - 14
    x_values = [float(row["x"]) for row in visible_rows]
    y_values = [float(row["y"]) for row in visible_rows]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)

    points = []
    for x_value, y_value in zip(x_values, y_values):
        px = plot_x0 + (x_value - x_min) / x_span * (plot_x1 - plot_x0)
        py = plot_y1 - (y_value - y_min) / y_span * (plot_y1 - plot_y0)
        points.append((px, py))

    draw.rectangle((plot_x0, plot_y0, plot_x1, plot_y1), outline=(255, 255, 255, 90), width=1)
    if len(points) >= 2:
        draw.line(points, fill=(0, 255, 80, 255), width=3)
    current_x, current_y = points[-1]
    draw.ellipse((current_x - 5, current_y - 5, current_x + 5, current_y + 5), fill=(255, 220, 0, 255))
    return np.asarray(image)


def add_trace_markers(
    mujoco: Any,
    np: Any,
    scene: Any,
    trace_positions: list[Any],
    trace_length: int,
    trace_radius: float,
    trace_color: tuple[float, float, float, float],
) -> tuple[int, str | None]:
    if trace_length > 0:
        visible_positions = trace_positions[-trace_length:]
    else:
        visible_positions = trace_positions

    marker_count = 0
    size = np.asarray([trace_radius, trace_radius, trace_radius], dtype=float)
    mat = np.eye(3, dtype=float).reshape(9)
    rgba = np.asarray(trace_color, dtype=np.float32)

    for position in visible_positions:
        if scene.ngeom >= scene.maxgeom:
            return marker_count, "trace markers truncated because MuJoCo scene maxgeom was reached"
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            size,
            np.asarray(position, dtype=float),
            mat,
            rgba,
        )
        geom.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
        geom.emission = 0.6
        scene.ngeom += 1
        marker_count += 1
    return marker_count, None


def format_timestamp_line(timestamp_format: str, time_value: float, frame_index: int) -> tuple[str, str | None]:
    try:
        return timestamp_format.format(time=time_value, frame=frame_index), None
    except (KeyError, IndexError, ValueError) as exc:
        return f"t={time_value:.2f}s", f"timestamp format fallback used: {exc}"


def render_video(
    mujoco: Any,
    np: Any,
    model: Any,
    time_values: Any,
    q_values: Any,
    qpos_addresses: list[int],
    frame_indices: Any,
    ee_site_id: int | None,
    args: argparse.Namespace,
    video_path: Path,
) -> dict[str, Any]:
    writer = open_video_writer(video_path, args.fps, args.width, args.height, args.output_format)
    renderer = None
    frame_count = 0
    trace_rows: list[dict[str, float | int]] = []
    trace_positions: list[Any] = []
    timestamp_status = "disabled"
    timestamp_warning_printed = False
    timestamp_format_warning: str | None = None
    trace_overlay_status = "disabled"
    trace_marker_count = 0
    try:
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        preset_camera = build_camera(mujoco, model, args.camera_preset)
        for source_index in frame_indices:
            data.qpos[:] = model.qpos0
            for joint_index, qposadr in enumerate(qpos_addresses):
                data.qpos[qposadr] = q_values[int(source_index), joint_index]
            mujoco.mj_forward(model, data)

            if args.camera is None:
                if preset_camera is None:
                    renderer.update_scene(data)
                else:
                    renderer.update_scene(data, camera=preset_camera)
            else:
                renderer.update_scene(data, camera=args.camera)

            source_time = float(time_values[int(source_index)])
            if ee_site_id is not None:
                ee_position = data.site_xpos[ee_site_id].copy()
                trace_positions.append(ee_position)
                trace_rows.append(
                    {
                        "frame_index": frame_count,
                        "source_index": int(source_index),
                        "time": source_time,
                        "x": float(ee_position[0]),
                        "y": float(ee_position[1]),
                        "z": float(ee_position[2]),
                    }
                )
                markers_added, trace_warning = add_trace_markers(
                    mujoco,
                    np,
                    renderer.scene,
                    trace_positions,
                    args.trace_length,
                    args.trace_radius,
                    args.trace_color,
                )
                trace_marker_count += markers_added
                trace_overlay_status = trace_warning or "applied: scene markers"

            frame = renderer.render()
            if ee_site_id is not None:
                try:
                    frame = draw_ee_trace_overlay(np, frame, trace_rows, args.trace_length)
                    if trace_overlay_status == "applied: scene markers":
                        trace_overlay_status = "applied: scene markers and image overlay"
                    elif trace_overlay_status == "disabled":
                        trace_overlay_status = "applied: image overlay"
                except Exception as exc:
                    trace_overlay_status = f"scene marker status kept; image overlay failed: {exc}"
            if args.show_timestamp:
                timestamp_line, format_warning = format_timestamp_line(args.timestamp_format, source_time, frame_count)
                timestamp_format_warning = timestamp_format_warning or format_warning
                try:
                    frame = draw_text_overlay(np, frame, ["GOAL1 FR3 replay", timestamp_line])
                    timestamp_status = timestamp_format_warning or "applied"
                except Exception as exc:
                    timestamp_status = f"disabled after overlay failure: {exc}"
                    if not timestamp_warning_printed:
                        print(f"Timestamp overlay failed; continuing without timestamp: {exc}", file=sys.stderr)
                        timestamp_warning_printed = True

            writer.append(frame)
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
    return {
        "rendered_frame_count": frame_count,
        "writer_backend": writer.backend,
        "timestamp_overlay_status": timestamp_status,
        "ee_trace_video_overlay_status": trace_overlay_status,
        "ee_trace_marker_count": trace_marker_count,
        "ee_trace_rows": trace_rows,
    }


def write_ee_trace_csv(path: Path, trace_rows: list[dict[str, float | int]]) -> str:
    if not trace_rows:
        return "skipped: no EE trace rows"
    fieldnames = ["frame_index", "source_index", "time", "x", "y", "z"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trace_rows)
    return "written"


def write_ee_trace_plot(path: Path, trace_rows: list[dict[str, float | int]]) -> str:
    if not trace_rows:
        return "skipped: no EE trace rows"
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return f"skipped: matplotlib unavailable ({exc})"

    x_values = [float(row["x"]) for row in trace_rows]
    y_values = [float(row["y"]) for row in trace_rows]
    z_values = [float(row["z"]) for row in trace_rows]

    fig = plt.figure(figsize=(6, 4.5))
    axis = fig.add_subplot(111, projection="3d")
    axis.plot(x_values, y_values, z_values, color="tab:orange", linewidth=1.5)
    axis.scatter([x_values[0]], [y_values[0]], [z_values[0]], color="tab:green", s=20, label="start")
    axis.scatter([x_values[-1]], [y_values[-1]], [z_values[-1]], color="tab:red", s=20, label="end")
    axis.set_title("GOAL1 FR3 EE trace")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_zlabel("z [m]")
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return "written"


def make_summary(
    args: argparse.Namespace,
    model_info: dict[str, Any],
    model: Any,
    joint_names: list[str],
    time_values: Any,
    frame_plan: dict[str, Any],
    video_path: Path,
    trace_csv_path: Path | None,
    trace_png_path: Path | None,
    render_info: dict[str, Any],
    trace_csv_status: str,
    trace_png_status: str,
    rendering_succeeded: bool,
    writer_backend: str | None,
    rendered_frame_count: int,
    error: str | None,
) -> dict[str, Any]:
    if args.camera is not None:
        camera_mode = "model-defined camera"
    elif args.camera_preset == "default":
        camera_mode = "default MuJoCo camera"
    else:
        camera_mode = "free camera preset"

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "csv_path": str(args.csv),
        "model_path": str(args.model),
        "original_model_path": model_info["original_model_path"],
        "effective_model_path": model_info["effective_model_path"],
        "auto_offscreen_xml": model_info["auto_offscreen_xml"],
        "offscreen_width": model_info["offscreen_width"],
        "offscreen_height": model_info["offscreen_height"],
        "patched_xml_created": model_info["patched_xml_created"],
        "patched_xml_path": model_info["patched_xml_path"],
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
        "start_time": float(args.start_time),
        "frame_stride": int(args.frame_stride),
        "camera": args.camera,
        "camera_preset": args.camera_preset,
        "camera_mode": camera_mode,
        "timestamp_overlay_enabled": bool(args.show_timestamp),
        "timestamp_format": args.timestamp_format,
        "timestamp_overlay_status": render_info.get("timestamp_overlay_status", "disabled"),
        "ee_site": args.ee_site,
        "ee_trace_enabled": bool(args.show_ee_trace),
        "ee_trace_length": int(args.trace_length),
        "ee_trace_radius": float(args.trace_radius),
        "ee_trace_color": list(args.trace_color),
        "ee_trace_video_overlay_status": render_info.get("ee_trace_video_overlay_status", "disabled"),
        "ee_trace_marker_count": int(render_info.get("ee_trace_marker_count", 0)),
        "ee_trace_csv_path": str(trace_csv_path) if trace_csv_path is not None else None,
        "ee_trace_png_path": str(trace_png_path) if trace_png_path is not None else None,
        "ee_trace_csv_status": trace_csv_status,
        "ee_trace_png_status": trace_png_status,
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
            "offscreen_width",
            "offscreen_height",
            "auto_offscreen_xml",
            "effective_model_path",
            "patched_xml_created",
            "patched_xml_path",
            "fps",
            "playback_speed",
            "start_time",
            "frame_stride",
            "camera",
            "camera_preset",
            "camera_mode",
            "timestamp_overlay_enabled",
            "timestamp_format",
            "timestamp_overlay_status",
            "ee_site",
            "ee_trace_enabled",
            "ee_trace_length",
            "ee_trace_radius",
            "ee_trace_color",
            "ee_trace_video_overlay_status",
            "ee_trace_marker_count",
            "ee_trace_csv_path",
            "ee_trace_png_path",
            "ee_trace_csv_status",
            "ee_trace_png_status",
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


def print_dry_run(args: argparse.Namespace, model_info: dict[str, Any], model: Any, joint_names: list[str], time_values: Any, frame_plan: dict[str, Any]) -> None:
    print("GOAL1 FR3 MuJoCo video dry-run completed.")
    print(f"CSV: {args.csv}")
    print(f"Original model: {model_info['original_model_path']}")
    print(f"Effective model: {model_info['effective_model_path']}")
    print(f"Auto offscreen XML: {model_info['auto_offscreen_xml']}")
    print(f"Offscreen size: {model_info['offscreen_width']}x{model_info['offscreen_height']}")
    print(f"Joints: {', '.join(joint_names)}")
    print(f"Source samples: {len(time_values)}")
    print(f"Source time: {float(time_values[0])} to {float(time_values[-1])} s")
    print(f"Model sizes: nq={model.nq}, nv={model.nv}, nu={model.nu}, nbody={model.nbody}, nsite={model.nsite}")
    print(f"Frame plan: {frame_plan['planned_frame_count']} frames at {args.fps} fps")
    print(f"Playback speed: {args.playback_speed}")
    print(f"Start time: {args.start_time}")
    print(f"Frame stride: {args.frame_stride}")
    print(f"Camera preset: {args.camera_preset}")
    print(f"Camera: {args.camera}")
    print(f"Timestamp overlay: {args.show_timestamp}")
    print(f"EE trace overlay: {args.show_ee_trace}")
    print(f"EE site: {args.ee_site}")
    print(f"Writer availability: {writer_availability()}")
    print("No video was rendered because --dry-run was used.")


def run(args: argparse.Namespace) -> int:
    np = require_numpy()
    mujoco = require_mujoco()
    model_info = prepare_effective_model_path(args)
    model = load_model(mujoco, Path(model_info["effective_model_path"]))

    if args.list_model_names:
        print_model_names(mujoco, model)
        return 0

    joint_names = parse_joint_names(args.joint_names)
    time_values, q_values = read_goal1_csv(args.csv, np)
    qpos_addresses = resolve_joint_qpos_addresses(mujoco, model, joint_names)
    validate_camera(mujoco, model, args.camera)
    ee_site_id = validate_site(mujoco, model, args.ee_site) if args.show_ee_trace else None
    frame_plan = build_frame_plan(
        np,
        time_values,
        args.fps,
        args.playback_speed,
        args.frame_stride,
        args.start_time,
        args.max_frames,
    )

    if args.dry_run:
        print_dry_run(args, model_info, model, joint_names, time_values, frame_plan)
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.output_dir / f"{args.prefix}.{args.output_format}"
    summary_json = args.output_dir / f"{args.prefix}_summary.json"
    summary_md = args.output_dir / f"{args.prefix}_summary.md"
    trace_csv_path = args.output_dir / f"{args.prefix}_ee_trace.csv" if args.show_ee_trace else None
    trace_png_path = args.output_dir / f"{args.prefix}_ee_trace_3d.png" if args.show_ee_trace else None

    rendered_frame_count = 0
    writer_backend = None
    render_info: dict[str, Any] = {
        "timestamp_overlay_status": "disabled",
        "ee_trace_video_overlay_status": "disabled",
        "ee_trace_marker_count": 0,
        "ee_trace_rows": [],
    }
    trace_csv_status = "disabled"
    trace_png_status = "disabled"
    error = None
    try:
        render_info = render_video(
            mujoco,
            np,
            model,
            time_values,
            q_values,
            qpos_addresses,
            frame_plan["indices"],
            ee_site_id,
            args,
            video_path,
        )
        rendered_frame_count = int(render_info["rendered_frame_count"])
        writer_backend = str(render_info["writer_backend"])
        if args.show_ee_trace:
            trace_csv_status = write_ee_trace_csv(trace_csv_path, render_info["ee_trace_rows"])
            trace_png_status = write_ee_trace_plot(trace_png_path, render_info["ee_trace_rows"])
    except RuntimeError as exc:
        error = str(exc)
        summary = make_summary(
            args,
            model_info,
            model,
            joint_names,
            time_values,
            frame_plan,
            video_path,
            trace_csv_path,
            trace_png_path,
            render_info,
            trace_csv_status,
            trace_png_status,
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
        model_info,
        model,
        joint_names,
        time_values,
        frame_plan,
        video_path,
        trace_csv_path,
        trace_png_path,
        render_info,
        trace_csv_status,
        trace_png_status,
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
    print(f"Effective model: {model_info['effective_model_path']}")
    print(f"Auto offscreen XML: {model_info['auto_offscreen_xml']}")
    print(f"Camera preset: {args.camera_preset}")
    print(f"Timestamp overlay: {render_info['timestamp_overlay_status']}")
    print(f"EE trace overlay: {render_info['ee_trace_video_overlay_status']}")
    if args.show_ee_trace:
        print(f"EE trace CSV: {trace_csv_path} ({trace_csv_status})")
        print(f"EE trace PNG: {trace_png_path} ({trace_png_status})")
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
