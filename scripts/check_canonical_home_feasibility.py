#!/usr/bin/env python3
"""Read-only canonical/session-home feasibility checker; no ROS imports."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np


PACKAGE_ROOT = (
    Path(__file__).resolve().parents[1] / "new_structure" / "py_controllers"
)
sys.path.insert(0, str(PACKAGE_ROOT))

from py_controllers.session_anchor_utils import (  # noqa: E402
    load_session_home_payload,
    read_optional_q_at_capture,
)
from py_controllers.session_home_feasibility import (  # noqa: E402
    classify_joint_home,
    compute_joint_home_metrics,
    format_joint_home_report,
    to_jsonable_joint_home_result,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check current q/dq against session-home q_at_capture."
    )
    parser.add_argument("--session-home", required=True)
    parser.add_argument("--current-q", default="")
    parser.add_argument("--current-dq", default="")
    parser.add_argument("--current-ee", default="")
    parser.add_argument("--max-abs-warn-rad", type=float, default=0.10)
    parser.add_argument("--max-abs-refuse-rad", type=float, default=0.30)
    parser.add_argument("--l2-warn-rad", type=float, default=0.20)
    parser.add_argument("--l2-refuse-rad", type=float, default=0.50)
    parser.add_argument("--dq-warn-rad-s", type=float, default=0.02)
    parser.add_argument("--dq-refuse-rad-s", type=float, default=0.05)
    parser.add_argument("--allow-missing-q-at-capture", action="store_true")
    parser.add_argument("--json-out", default="")
    return parser.parse_args()


def parse_optional_vector(text, length, name):
    if not text:
        return None
    try:
        vector = np.asarray([float(value) for value in text.split(",")])
    except ValueError as exc:
        raise ValueError(
            f"{name} must contain comma-separated numbers"
        ) from exc
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain {length} finite values")
    return vector


def thresholds_from_args(args):
    return {
        "max_abs_warn_rad": args.max_abs_warn_rad,
        "max_abs_refuse_rad": args.max_abs_refuse_rad,
        "l2_warn_rad": args.l2_warn_rad,
        "l2_refuse_rad": args.l2_refuse_rad,
        "dq_warn_rad_s": args.dq_warn_rad_s,
        "dq_refuse_rad_s": args.dq_refuse_rad_s,
    }


def main():
    args = parse_args()
    path = Path(args.session_home).expanduser()
    payload = load_session_home_payload(path)
    q_home = read_optional_q_at_capture(
        payload, f"[SessionHome] '{path}': "
    )
    q_current = parse_optional_vector(args.current_q, 7, "--current-q")
    dq_current = parse_optional_vector(args.current_dq, 7, "--current-dq")
    current_ee = parse_optional_vector(args.current_ee, 3, "--current-ee")
    metrics = compute_joint_home_metrics(q_current, dq_current, q_home)
    classification = classify_joint_home(
        metrics,
        thresholds_from_args(args),
        enabled=True,
        require_q_home=not args.allow_missing_q_at_capture,
    )
    report = to_jsonable_joint_home_result(metrics, classification)
    ee_home = payload.get("ee_pose_xyz")
    if current_ee is not None and ee_home is not None:
        ee_home = np.asarray(ee_home, dtype=float)
        if ee_home.shape == (3,) and np.all(np.isfinite(ee_home)):
            report["ee_distance_m"] = float(
                np.linalg.norm(current_ee - ee_home)
            )

    print("===== SESSION HOME FEASIBILITY =====")
    if "ee_distance_m" in report:
        print(f"ee_distance_m={report['ee_distance_m']:.6f}")
    print(format_joint_home_report(metrics, classification))
    if report.get("per_joint_delta_rad") is not None:
        print(
            "per_joint_delta_rad="
            + json.dumps(report["per_joint_delta_rad"])
        )
    if args.json_out:
        output = Path(args.json_out).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(f"json_out={output}")
    if not classification["allowed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
