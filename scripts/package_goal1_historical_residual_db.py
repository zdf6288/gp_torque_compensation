#!/usr/bin/env python3
"""Package GOAL1 historical residual database artifacts into a tar.gz archive.

Offline-only:
- no ROS
- no robot
- no controller modification
- no active compensation
- no tau_final modification

This packages generated ignored outputs so the historical DB can be transferred
or archived without committing large binary files to git.
"""

from __future__ import annotations

import argparse
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_DB_DIR = "outputs/goal1_historical_residual_db_20260604"
DEFAULT_EVAL_DIR = "outputs/goal1_historical_residual_db_eval_20260604"
DEFAULT_GATING_DIR = "outputs/goal1_historical_db_gating_policy_20260604"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Package GOAL1 historical residual DB artifacts.")
    p.add_argument("--db-dir", default=DEFAULT_DB_DIR)
    p.add_argument("--eval-dir", default=DEFAULT_EVAL_DIR)
    p.add_argument("--gating-dir", default=DEFAULT_GATING_DIR)
    p.add_argument(
        "--output",
        default="",
        help="Output .tar.gz path. Default: ~/gp_torque_data_backups/goal1_historical_residual_db_package_<timestamp>.tar.gz",
    )
    return p.parse_args()


def collect_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in paths:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if p.is_file():
                files.append(p)
    return files


def main() -> None:
    args = parse_args()

    roots = [Path(args.db_dir), Path(args.eval_dir), Path(args.gating_dir)]
    files = collect_files(roots)

    if not files:
        raise SystemExit("No files found to package.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")

    if args.output:
        output = Path(args.output).expanduser()
    else:
        output = Path.home() / "gp_torque_data_backups" / f"goal1_historical_residual_db_package_{timestamp}.tar.gz"

    output.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "package_type": "goal1_historical_residual_db_package",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "offline_only": True,
        "active_compensation": False,
        "notes": [
            "This archive contains ignored/generated offline artifacts.",
            "It is not loaded by the controller.",
            "It does not enter tau_final.",
            "Use only for offline historical retrieval analysis unless separately reviewed.",
        ],
        "roots": [str(r) for r in roots],
        "files": [{"path": str(p), "size_bytes": p.stat().st_size} for p in files],
    }

    manifest_path = output.parent / f"goal1_historical_residual_db_package_manifest_{timestamp}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with tarfile.open(output, "w:gz") as tar:
        for p in files:
            tar.add(p, arcname=str(p))
        tar.add(manifest_path, arcname=manifest_path.name)

    print("===== GOAL1 historical residual DB package built =====")
    print("archive:", output)
    print("archive_size_bytes:", output.stat().st_size)
    print("manifest:", manifest_path)
    print("file_count:", len(files))


if __name__ == "__main__":
    main()
