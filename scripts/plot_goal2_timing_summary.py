#!/usr/bin/env python3
"""
GOAL2 timing summary plotting script.

Offline-only script:
- reads the generated summary CSV from ~/gp_torque_data_backups
- writes plots to ~/gp_torque_data_backups
- does not use ROS
- does not touch Franka / controllers / launch files
"""

import csv
import math
from pathlib import Path
import matplotlib.pyplot as plt

INPUT_CSV = Path.home() / "gp_torque_data_backups/goal2_20260531_analysis/goal2_timing_summary_20260531.csv"
OUT_DIR = Path.home() / "gp_torque_data_backups/goal2_20260531_analysis/plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODE_LABELS = {
    "no_gp": "No GP",
    "pred_on": "Prediction only",
    "comp_on": "Compensation on",
    "online_update_on_comp_off": "Online update\ncomp off",
    "online_update_on_comp_on": "Online update\ncomp on",
}

MODE_ORDER = {
    "no_gp": 0,
    "pred_on": 1,
    "comp_on": 2,
    "online_update_on_comp_off": 3,
    "online_update_on_comp_on": 4,
    "unknown": 9,
}

RUN_LABELS = {
    "goal2_r1_real_no_gp_50hz_full3000": "R1\nNo GP",
    "goal2_r2_real_pred_on_delay0_50hz_full3000": "R2\nPred d0",
    "goal2_r3_real_pred_on_delay1_50hz_full3000": "R3\nPred d1",
    "goal2_r4_real_pred_on_delay2_50hz_full3000": "R4\nPred d2",
    "goal2_r5_real_pred_on_delay5_50hz_full3000": "R5\nPred d5",
    "goal2_c2_real_comp_on_delay0_50hz_full3000": "C2\nComp d0",
    "goal2_c3_real_comp_on_delay1_50hz_full3000": "C3\nComp d1",
    "goal2_c1_real_comp_on_delay2_50hz_full3000": "C1\nComp d2",
    "goal2_c4_real_comp_on_delay5_50hz_full3000": "C4\nComp d5",
    "goal2_ou1_real_online_update_on_comp_off_delay2_50hz_full3000": "OU1\nUpd off d2",
    "goal2_ou3_real_online_update_on_comp_on_delay0_50hz_full3000": "OU3\nUpd on d0",
    "goal2_ou4_real_online_update_on_comp_on_delay1_50hz_full3000": "OU4\nUpd on d1",
    "goal2_ou2_real_online_update_on_comp_on_delay2_50hz_full3000": "OU2\nUpd on d2",
    "goal2_ou5_real_online_update_on_comp_on_delay5_50hz_full3000": "OU5\nUpd on d5",
}

def to_float(x):
    if x is None:
        return math.nan
    s = str(x).strip()
    if s == "":
        return math.nan
    try:
        return float(s)
    except Exception:
        return math.nan

def fmt_value(v):
    if math.isnan(v):
        return ""
    if abs(v) >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"

def delay_key(v):
    s = str(v).strip()
    if s == "":
        return 999
    try:
        return int(s)
    except Exception:
        return 999

def load_rows():
    rows = []
    with INPUT_CSV.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(r)

    rows.sort(key=lambda r: (
        MODE_ORDER.get(r["mode"], 9),
        delay_key(r["delay_steps"]),
        r["run"],
    ))
    return rows

ROWS = load_rows()

def mode_means(metric_key):
    grouped = {}
    for r in ROWS:
        mode = r["mode"]
        v = to_float(r.get(metric_key, ""))
        if math.isnan(v):
            continue
        grouped.setdefault(mode, []).append(v)

    ordered_modes = sorted(grouped.keys(), key=lambda m: MODE_ORDER.get(m, 9))
    labels = [MODE_LABELS.get(m, m) for m in ordered_modes]
    values = [sum(grouped[m]) / len(grouped[m]) for m in ordered_modes]
    return labels, values

def save_mode_bar(metric_key, ylabel, title, filename, reference_line=None, reference_label=None):
    labels, values = mode_means(metric_key)

    plt.figure(figsize=(10, 5.5))
    x = list(range(len(labels)))
    plt.bar(x, values)
    plt.xticks(x, labels)
    plt.ylabel(ylabel)
    plt.title(title)

    if reference_line is not None:
        plt.axhline(reference_line, linestyle="--", linewidth=1)
        if reference_label:
            plt.text(
                len(labels) - 0.6,
                reference_line,
                reference_label,
                va="bottom",
                ha="right",
            )

    for i, v in enumerate(values):
        plt.text(i, v, fmt_value(v), ha="center", va="bottom")

    plt.tight_layout()
    out = OUT_DIR / filename
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"Wrote: {out}")

def save_run_bar(metric_key, ylabel, title, filename, reference_line=None, reference_label=None):
    labels = [RUN_LABELS.get(r["run"], r["run"]) for r in ROWS]
    values = [to_float(r.get(metric_key, "")) for r in ROWS]

    plt.figure(figsize=(14, 6))
    x = list(range(len(labels)))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=0)
    plt.ylabel(ylabel)
    plt.title(title)

    if reference_line is not None:
        plt.axhline(reference_line, linestyle="--", linewidth=1)
        if reference_label:
            plt.text(
                len(labels) - 0.4,
                reference_line,
                reference_label,
                va="bottom",
                ha="right",
            )

    plt.tight_layout()
    out = OUT_DIR / filename
    plt.savefig(out, dpi=180)
    plt.close()
    print(f"Wrote: {out}")

# Presentation-level mode plots.
save_mode_bar(
    "callback_wall_p95_ms",
    "Callback wall time p95 [ms]",
    "GOAL2 50 Hz: callback p95 by mode",
    "presentation_goal2_callback_p95_by_mode.png",
    reference_line=20.0,
    reference_label="20 ms deadline",
)

save_mode_bar(
    "deadline_ratio_p95",
    "Deadline usage p95",
    "GOAL2 50 Hz: deadline usage p95 by mode",
    "presentation_goal2_deadline_ratio_p95_by_mode.png",
    reference_line=1.0,
    reference_label="deadline",
)

save_mode_bar(
    "gp_total_p95_ms",
    "GP computation p95 [ms]",
    "GOAL2 50 Hz: GP computation p95 by mode",
    "presentation_goal2_gp_total_p95_by_mode.png",
    reference_line=20.0,
    reference_label="20 ms deadline",
)

# Run-level plots for detailed backup.
save_run_bar(
    "callback_wall_p95_ms",
    "Callback wall time p95 [ms]",
    "GOAL2 50 Hz: callback p95 by run",
    "presentation_goal2_callback_p95_by_run.png",
    reference_line=20.0,
    reference_label="20 ms deadline",
)

save_run_bar(
    "deadline_ratio_p95",
    "Deadline usage p95",
    "GOAL2 50 Hz: deadline usage p95 by run",
    "presentation_goal2_deadline_ratio_p95_by_run.png",
    reference_line=1.0,
    reference_label="deadline",
)

save_run_bar(
    "gp_total_p95_ms",
    "GP computation p95 [ms]",
    "GOAL2 50 Hz: GP computation p95 by run",
    "presentation_goal2_gp_total_p95_by_run.png",
    reference_line=20.0,
    reference_label="20 ms deadline",
)

print()
print("Input summary:")
print(INPUT_CSV)
print("Plots directory:")
print(OUT_DIR)
