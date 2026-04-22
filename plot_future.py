#!/usr/bin/env python3
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt


def main():
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = "cartesian_impedance_controller_data.csv"

    if not os.path.isfile(csv_file):
        print(f"[ERROR] CSV file not found: {csv_file}")
        sys.exit(1)

    df = pd.read_csv(csv_file)

    if "Time(s)" not in df.columns:
        print("[ERROR] Column 'Time(s)' not found in CSV.")
        sys.exit(1)

    t = df["Time(s)"].values

    out_dir = "future_prediction_plots"
    os.makedirs(out_dir, exist_ok=True)

    for j in range(1, 8):
        q_pred_col = f"q_pred_{j}"
        dq_pred_col = f"dq_pred_{j}"
        q_act_col = f"q_future_actual_{j}"
        dq_act_col = f"dq_future_actual_{j}"
        q_err_col = f"q_pred_err_{j}"
        dq_err_col = f"dq_pred_err_{j}"

        needed = [q_pred_col, dq_pred_col, q_act_col, dq_act_col]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            print(f"[WARN] Joint {j}: missing columns {missing}, skip.")
            continue

        # ---------- Plot 1: q pred vs actual ----------
        plt.figure(figsize=(10, 4))
        plt.plot(t, df[q_pred_col].values, label="q_pred_future")
        plt.plot(t, df[q_act_col].values, label="q_future_actual")
        plt.xlabel("Time [s]")
        plt.ylabel("q [rad]")
        plt.title(f"Joint {j}: predicted future q vs actual future q")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"joint_{j}_q_pred_vs_actual.png"), dpi=150)
        plt.close()

        # ---------- Plot 2: dq pred vs actual ----------
        plt.figure(figsize=(10, 4))
        plt.plot(t, df[dq_pred_col].values, label="dq_pred_future")
        plt.plot(t, df[dq_act_col].values, label="dq_future_actual")
        plt.xlabel("Time [s]")
        plt.ylabel("dq [rad/s]")
        plt.title(f"Joint {j}: predicted future dq vs actual future dq")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"joint_{j}_dq_pred_vs_actual.png"), dpi=150)
        plt.close()

        # ---------- Plot 3: q error ----------
        if q_err_col in df.columns:
            plt.figure(figsize=(10, 4))
            plt.plot(t, df[q_err_col].values, label="q_pred_error")
            plt.axhline(0.0, linestyle="--")
            plt.xlabel("Time [s]")
            plt.ylabel("q error [rad]")
            plt.title(f"Joint {j}: future q prediction error")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"joint_{j}_q_pred_error.png"), dpi=150)
            plt.close()

        # ---------- Plot 4: dq error ----------
        if dq_err_col in df.columns:
            plt.figure(figsize=(10, 4))
            plt.plot(t, df[dq_err_col].values, label="dq_pred_error")
            plt.axhline(0.0, linestyle="--")
            plt.xlabel("Time [s]")
            plt.ylabel("dq error [rad/s]")
            plt.title(f"Joint {j}: future dq prediction error")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"joint_{j}_dq_pred_error.png"), dpi=150)
            plt.close()

    print(f"[INFO] All plots saved to: {out_dir}")


if __name__ == "__main__":
    main()