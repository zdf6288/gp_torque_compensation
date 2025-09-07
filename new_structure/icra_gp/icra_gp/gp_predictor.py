#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gaussian Process (Mixture-of-Experts) trajectory predictor.
- Translated all Chinese comments to English.
- Tidied formatting and removed duplicate helper definitions.
- Kept function signatures and behavior intact where possible.
"""

import traceback
from datetime import datetime  # kept in case of future logging/usage
import csv

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

from icra_gp.skygp_online import SkyGP_MOE as SaoGP_MOE

# ==============================
# Global Configuration
# ==============================
SEED = 0
SAMPLE_HZ = 100                  # sampling frequency for equal-time resampling
TRAIN_RATIO = 1.0                # demo: use all data for training
MAX_EXPERTS = 80
MIN_POINTS_OFFLINE = 1
WINDOW_SIZE = None
METHOD_ID = 1                    # 1=polar->delta; 5=polar+delta->delta
DEFAULT_SPEED = 0.01             # speed used to convert polyline length to time for equal-time resampling
STOP_TAIL_ZERO_STEPS = 0         # set deltas to zero for the last N seeds near the end
np.random.seed(SEED)
torch.manual_seed(SEED)

# Tunable parameters
ANCHOR_ANGLE = np.radians(40)  # angle (rad) for anchor correspondence
K_HIST = 10                      # seed length (history window)
NEAREST_K = 2
MAX_DATA_PER_EXPERT = 50

# ==============================
# Method Hyperparameters
# ==============================
METHOD_HPARAM = {
    1: {'adam_lr': 0.001, 'adam_steps': 200}
}

# ==============================
# GP Utilities
# ==============================
def torch_to_np(x: torch.Tensor) -> np.ndarray:
    """Detach a torch tensor and move to CPU as a NumPy array."""
    return x.detach().cpu().numpy()


class Standardizer:
    """Simple feature standardizer compatible with previous interface."""

    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> "Standardizer":
        self.X_mean = X.mean(0)
        self.X_std = X.std(0).clamp_min(1e-8)
        self.Y_mean = Y.mean(0)
        self.Y_std = Y.std(0).clamp_min(1e-8)
        return self

    def x_transform(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self.X_mean) / self.X_std

    def y_transform(self, Y: torch.Tensor) -> torch.Tensor:
        return (Y - self.Y_mean) / self.Y_std

    def y_inverse_transform(self, Yn: torch.Tensor) -> torch.Tensor:
        assert Yn.shape[-1] == self.Y_std.shape[0], (
            f"Dimension mismatch: Yn.shape={Yn.shape}, std={self.Y_std.shape}"
        )
        return Yn * self.Y_std + self.Y_mean

    # Backward compatible alias
    def y_inverse(self, Yn: torch.Tensor) -> torch.Tensor:
        return self.y_inverse_transform(Yn)


def rotate_to_fixed_frame(vectors: torch.Tensor, base_dir: torch.Tensor) -> torch.Tensor:
    """
    Rotate 2D vectors into a fixed frame defined by base_dir (x-axis).
    y-axis is perpendicular to base_dir.
    """
    base = base_dir / base_dir.norm()
    x_axis = base
    y_axis = torch.tensor([-base[1], base[0]], dtype=torch.float32)
    R = torch.stack([x_axis, y_axis], dim=1)
    return vectors @ R


def polar_feat_from_xy_torch(xy: torch.Tensor, origin: torch.Tensor) -> torch.Tensor:
    """Compute polar-like features [r, cos(theta), sin(theta)] relative to origin for each (x, y)."""
    xy = xy.float()
    origin = origin.to(xy)
    shifted = xy - origin
    r = torch.sqrt(shifted[..., 0] ** 2 + shifted[..., 1] ** 2)
    theta = torch.atan2(shifted[..., 1], shifted[..., 0])
    return torch.stack([r, torch.cos(theta), torch.sin(theta)], dim=-1)


def build_dataset(traj: torch.Tensor, k: int):
    """
    Build (X, Y) for GP:
      - X: polar features of the last k positions w.r.t. global origin (traj[0])
      - Y: next-step delta rotated into the averaged initial direction frame
    Optionally zero out the last STOP_TAIL_ZERO_STEPS labels.
    """
    deltas = traj[1:] - traj[:-1]
    T = traj.shape[0]
    Xs, Ys = [], []
    global_origin = traj[0]

    # Global base direction: average of first up-to-10 segment vectors from start
    end_idx = min(10, traj.shape[0] - 1)
    dirs = traj[1 : end_idx + 1] - traj[0]
    global_base_dir = dirs.mean(dim=0)

    # Number of tail steps to zero out
    stop_tail = int(globals().get("STOP_TAIL_ZERO_STEPS", 10))

    for t in range(k, T - 1):
        feats = []
        seed_pos = traj[t - k + 1 : t + 1]

        feats.append(polar_feat_from_xy_torch(seed_pos, global_origin).reshape(-1))
        Xs.append(torch.cat(feats))

        # Force label to zero near the end (in fixed base frame)
        if t >= (T - 1 - stop_tail):
            Ys.append(torch.zeros(2, dtype=torch.float32))
        else:
            y_delta = traj[t + 1] - traj[t]
            Ys.append(rotate_to_fixed_frame(y_delta.unsqueeze(0), global_base_dir)[0])

    print(
        f"Dataset built: input dim {Xs[0].shape[0]}, samples {len(Xs)} "
        f"(last {stop_tail} seeds have zero delta)."
    )
    return torch.stack(Xs), torch.stack(Ys)


def time_split(X: torch.Tensor, Y: torch.Tensor, train_ratio: float):
    """Temporal split for (X, Y)."""
    N = X.shape[0]
    ntr = int(N * train_ratio)
    return (X[:ntr], Y[:ntr]), (X[ntr:], Y[ntr:]), ntr


def train_moe(dataset, method_id: int = METHOD_ID):
    """
    Train a Mixture-of-Experts GP with streaming point additions.
    Returns the trained moe and the standardizer.
    """
    Xtr = dataset["X_train"]
    Ytr = dataset["Y_train"]
    Din = Xtr.shape[1]
    Dout = Ytr.shape[1]

    scaler = Standardizer().fit(Xtr, Ytr)
    Xn = torch_to_np(scaler.x_transform(Xtr))
    Yn = torch_to_np(scaler.y_transform(Ytr))

    moe = SaoGP_MOE(
        x_dim=Din,
        y_dim=Dout,
        max_data_per_expert=MAX_DATA_PER_EXPERT,
        nearest_k=NEAREST_K,
        max_experts=MAX_EXPERTS,
        replacement=False,
        min_points=10**9,
        batch_step=10**9,
        window_size=256,
        light_maxiter=60,
    )
    for i in range(Xn.shape[0]):
        moe.add_point(Xn[i], Yn[i])

    params = METHOD_HPARAM.get(method_id, {"adam_lr": 0.001, "adam_steps": 200})
    if hasattr(moe, "optimize_hyperparams") and params["adam_steps"] > 0:
        for e in range(len(moe.X_list)):
            if moe.localCount[e] >= MIN_POINTS_OFFLINE:
                for p in range(2):
                    moe.optimize_hyperparams(
                        e, p, params["adam_steps"], WINDOW_SIZE, False, params["adam_lr"]
                    )

    return {"moe": moe, "scaler": scaler, "input_dim": Din}


def moe_predict(info, feat_1xD: torch.Tensor):
    """Predict (mu, var) in original label space given a 1xD feature tensor."""
    moe, scaler = info["moe"], info["scaler"]
    x = torch_to_np(feat_1xD.squeeze(0).float())  # shape: (D,)
    mu, var = moe.predict(torch_to_np(scaler.x_transform(torch.tensor(x))))
    mu = np.array(mu).reshape(1, -1)  # ensure (1, 2)
    y = torch_to_np(scaler.y_inverse(torch.tensor(mu)))  # -> (1, 2)
    return y, var


def rollout_reference(model_info, traj: torch.Tensor, start_t: int, h: int, k: int, scaler=None):
    """
    Roll out predictions in reference frame starting from index start_t using history length k.
    Returns (pred_positions, ground_truth, horizon_used).
    """
    assert start_t >= (k - 1), f"start_t={start_t} too small; requires at least {k - 1}"
    T = traj.shape[0]
    h = max(0, h)

    # Use global origin and base direction consistent with training
    global_origin = traj[0]
    if traj.shape[0] > 1:
        print("✅ Compute probe global base direction as average of first 10 segments")
        end_idx = min(10, traj.shape[0] - 1)
        dirs = traj[1 : end_idx + 1] - traj[0]
        global_base_dir = dirs.mean(dim=0)
        print(f"   global_base_dir = {global_base_dir.numpy()}")
    else:
        print("⚠️ Not enough points to compute global direction; using default [1, 0]")
        global_base_dir = torch.tensor([1.0, 0.0])

    # Initialize history
    hist_pos = [traj[start_t - (k - 1) + i].clone() for i in range(k)]
    hist_del = []
    for i in range(k):
        idx = start_t - (k - 1) + i
        prev = traj[idx - 1] if idx - 1 >= 0 else traj[0]
        hist_del.append(traj[idx] - prev)

    cur_pos = hist_pos[-1].clone()
    preds_pos = []

    for _ in range(h):
        feats = []

        # Use global_origin for consistency with training
        polar_feat = polar_feat_from_xy_torch(torch.stack(hist_pos[-k:]), global_origin)
        feats.append(polar_feat.reshape(1, -1))  # (1, 3K)

        x = torch.cat(feats, dim=1)  # (1, D)

        # GP predict in fixed frame
        y_pred, _ = moe_predict(model_info, x)  # (1, 2)
        y_pred = torch.tensor(y_pred, dtype=torch.float32)

        # Rotate step back to world frame
        gb = global_base_dir / global_base_dir.norm()
        R = torch.stack([gb, torch.tensor([-gb[1], gb[0]])], dim=1)
        step_world = y_pred @ R.T  # (1, 2)
        next_pos = cur_pos + step_world[0]
        next_del = step_world[0]

        # Update history
        hist_pos.append(next_pos)
        hist_del.append(next_del)
        cur_pos = next_pos
        preds_pos.append(next_pos)

    preds = torch.stack(preds_pos, dim=0) if preds_pos else torch.zeros((0, 2))

    # Ground truth slice (optional; may be empty near the end)
    gt = traj[start_t + 1 : start_t + 1 + h]
    return preds, gt, h


# ==============================
# Resampling & Geometry Utilities
# ==============================
def resample_polyline_equal_dt(points_xy, sample_hz: float, speed: float) -> np.ndarray:
    """
    Equal-time resample a polyline given a nominal speed.
    """
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.shape[0] < 2:
        return pts
    seg = pts[1:] - pts[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    L = float(np.sum(seg_len))
    if L <= 1e-8:
        return pts[:1]
    T_total = L / float(speed)
    dt = 1.0 / float(sample_hz)
    t_samples = np.arange(0.0, T_total + 1e-9, dt)
    s_samples = (t_samples / T_total) * L
    cum_s = np.concatenate([[0.0], np.cumsum(seg_len)])
    out = []
    j = 0
    for s in s_samples:
        while j < len(seg_len) - 1 and s > cum_s[j + 1]:
            j += 1
        ds = s - cum_s[j]
        r = 0.0 if seg_len[j] < 1e-9 else ds / seg_len[j]
        p = pts[j] + r * seg[j]
        out.append(p)
    return np.asarray(out, dtype=np.float32)


def resample_to_k(points_xy, k: int) -> np.ndarray:
    """
    Resample a polyline to exactly k points by arc-length parameterization.
    """
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.shape[0] < 2:
        return (
            np.repeat(pts[:1], k, axis=0)
            if pts.size
            else np.zeros((k, 2), dtype=np.float64)
        )
    seg = pts[1:] - pts[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    L = cum[-1]
    if L < 1e-9:
        return np.tile(pts[:1], (k, 1))
    s = np.linspace(0.0, L, k)
    out = []
    j = 0
    for si in s:
        while j < len(seg_len) - 1 and si > cum[j + 1]:
            j += 1
        ds = si - cum[j]
        r = 0.0 if seg_len[j] < 1e-9 else ds / seg_len[j]
        p = pts[j] + r * seg[j]
        out.append(p)
    return np.asarray(out, dtype=np.float64)


# ==============================
# Angle Helpers (relative to start tangent)
# ==============================
def _wrap_pi(a: np.ndarray) -> np.ndarray:
    """Wrap angle to (-pi, pi]."""
    return ((a + np.pi) % (2 * np.pi)) - np.pi


def estimate_start_tangent(xy, k: int = 5) -> float:
    """
    Estimate initial tangent direction (angle) using the first k segments.
    """
    xy = np.asarray(xy, dtype=np.float64)
    if len(xy) < 2:
        return 0.0
    k = int(max(2, min(k, len(xy) - 1)))
    v = np.diff(xy[: k + 1], axis=0)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    u = v / n
    m = u.mean(axis=0)
    if np.linalg.norm(m) < 1e-12:
        m = xy[1] - xy[0]
    return float(np.arctan2(m[1], m[0]))


def angles_relative_to_start_tangent(points, k_hist: int, min_r: float = 1e-3):
    """
    Compute relative angles w.r.t. the initial tangent and a validity mask (r > min_r).
    """
    P = np.asarray(points, dtype=np.float64)
    if len(P) == 0:
        return np.array([]), np.zeros(0, dtype=bool)
    o = P[0]
    phi0 = estimate_start_tangent(P, k=k_hist)
    v = P - o
    r = np.linalg.norm(v, axis=1)
    th = np.arctan2(v[:, 1], v[:, 0])
    th_rel = _wrap_pi(th - phi0)
    mask = r > min_r
    return th_rel, mask


def _angles_with_phi0(points, k_hist: int, min_r: float):
    """
    Compatibility wrapper.
    - If your angles_relative_to_start_tangent returns (angles, mask, phi0), return as-is.
    - Else compute phi0 here and return (angles, mask, phi0).
    """
    out = angles_relative_to_start_tangent(points, k_hist=k_hist, min_r=min_r)
    if isinstance(out, tuple) and len(out) == 3:
        return out  # (angles, mask, phi0)
    elif isinstance(out, tuple) and len(out) == 2:
        angles, mask = out
        phi0 = estimate_start_tangent(points, k=k_hist)
        return angles, mask, phi0
    else:
        raise RuntimeError("Unexpected return format from angles_relative_to_start_tangent")


def build_relative_angles(xy, origin_idx: int = 0, min_r: float = 1e-6) -> np.ndarray:
    """
    Build a relative angle array aligned to origin_idx; preceding entries are NaN.
    """
    P = np.asarray(xy, dtype=np.float64)
    N = len(P)
    if N == 0:
        return np.array([], dtype=np.float64)
    sub = P[origin_idx:]
    th_rel_sub, _ = angles_relative_to_start_tangent(sub, k_hist=K_HIST, min_r=min_r)
    out = np.full(N, np.nan, dtype=np.float64)
    out[origin_idx : origin_idx + len(th_rel_sub)] = th_rel_sub
    return out


def angle_diff(a: float, b: float) -> float:
    """Compute signed difference between angles a and b (wrapped to (-pi, pi])."""
    return _wrap_pi(a - b)


def angle_diff_mod_pi(a: float, b: float) -> float:
    """Compute minimal signed difference between a and b in (-pi, pi]."""
    return ((a - b + np.pi) % (2 * np.pi)) - np.pi


def first_index_reach_threshold(angles, mask, target, *, inclusive: bool = True, use_abs: bool = False) -> int:
    """
    Find the first index where the angle crosses the target threshold.
      - use_abs=False: directional threshold (>= target if target>=0 else <= target)
      - use_abs=True : magnitude threshold (|angle| >= |target|)
      - inclusive=True counts equality as crossing
    If never crosses, return index of the closest point (fallback).
    """
    idxs = np.where(mask)[0]
    if idxs.size == 0:
        return 0

    if use_abs:
        thr = abs(target)
        for i in idxs:
            if (abs(angles[i]) >= thr) if inclusive else (abs(angles[i]) > thr):
                return int(i)
    else:
        if target >= 0:
            for i in idxs:
                if (angles[i] >= target) if inclusive else (angles[i] > target):
                    return int(i)
        else:
            for i in idxs:
                if (angles[i] <= target) if inclusive else (angles[i] < target):
                    return int(i)

    # Fallback: nearest to target
    return int(idxs[np.argmin(np.abs(angles[idxs] - target))])


def last_window_rel_angles(points, W: int, min_r: float = 1e-3):
    """
    Return the last window of relative angles and mask using window size W.
    """
    P = np.asarray(points, dtype=np.float64)
    if P.shape[0] < 2:
        return None, None
    W = int(max(2, min(W if W is not None else 10, P.shape[0])))
    th, m = angles_relative_to_start_tangent(P, k_hist=W, min_r=min_r)
    end = len(th) - 1
    start = max(0, end - (W - 1))
    return th[start : end + 1], m[start : end + 1]


# ==============================
# Anchor Visualization & Correspondence
# ==============================
def _compute_base_unit_vec(points, n_segments: int = 10) -> np.ndarray:
    """
    Compute a base unit vector as the mean of the first n_segments tangents.
    """
    pts = np.asarray(points, dtype=np.float64)
    m = min(n_segments, pts.shape[0] - 1)
    if m < 1:
        return np.array([1.0, 0.0], dtype=np.float64)
    seg = np.diff(pts[: m + 1], axis=0)  # first m segments
    n = np.linalg.norm(seg, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    u = seg / n  # unit tangents
    v = u.mean(axis=0)  # average direction
    if np.linalg.norm(v) < 1e-12:
        v = seg[0]
    return v / max(np.linalg.norm(v), 1e-12)


def get_anchor_correspondence(
    ref_pts,
    probe_pts,
    angle_target,
    *,
    k_hist: int = 10,
    min_r: float = 1e-3,
    n_segments_base: int = 10,
):
    """
    Compute anchor correspondence on reference and probe by selecting the first index
    whose relative angle crosses `angle_target`. Return indices, vectors, points,
    and base unit vectors for both ref and probe.
    """
    ref_pts = np.asarray(ref_pts, dtype=np.float64)
    probe_pts = np.asarray(probe_pts, dtype=np.float64)
    assert ref_pts.shape[0] >= 2 and probe_pts.shape[0] >= 2, "ref/probe require at least 2 points"

    ref_ang, ref_mask, _ = _angles_with_phi0(ref_pts, k_hist=k_hist, min_r=min_r)
    pro_ang, pro_mask, _ = _angles_with_phi0(probe_pts, k_hist=k_hist, min_r=min_r)

    i_ref = first_index_reach_threshold(ref_ang, ref_mask, angle_target, inclusive=True, use_abs=False)
    i_pro = first_index_reach_threshold(pro_ang, pro_mask, angle_target, inclusive=True, use_abs=False)
    print(
        f"🎯 Target angle {angle_target:.2f} rad | "
        f"ref idx={i_ref}, angle={ref_ang[i_ref]:.2f} | "
        f"probe idx={i_pro}, angle={pro_ang[i_pro]:.2f}"
    )

    # Vectors from origin to anchor points
    o_ref, p_ref = ref_pts[0], ref_pts[i_ref]
    v_ref = p_ref - o_ref
    o_pro, p_pro = probe_pts[0], probe_pts[i_pro]
    v_pro = p_pro - o_pro

    # Base unit vectors (mean of first n_segments directions)
    u_ref = _compute_base_unit_vec(ref_pts, n_segments=n_segments_base)
    u_pro = _compute_base_unit_vec(probe_pts, n_segments=n_segments_base)

    return {
        "ref_index": i_ref,
        "ref_vector": v_ref,
        "ref_point": p_ref,
        "ref_base_unit": u_ref,
        "probe_index": i_pro,
        "probe_vector": v_pro,
        "probe_point": p_pro,
        "probe_base_unit": u_pro,
    }


def plot_anchor_vectors_from_gp(gp):
    """
    Plot, in a single figure:
      - Reference (resampled) trajectory
      - Probe (resampled) trajectory
      - Two anchor vectors: origin → anchor (for ref and for probe)
    Depends on gp.seed_end and gp.probe_end being set in predict_from_probe().
    """
    if gp.sampled is None or not gp.probe_pts or gp.seed_end is None or gp.probe_end is None:
        print("❗No anchors to plot (run train_gp / predict_from_probe first).")
        return

    ref = gp.sampled.detach().cpu().numpy()  # resampled reference trajectory
    probe = np.asarray(gp.probe_pts, dtype=np.float64)  # resampled probe trajectory
    i_ref = int(gp.seed_end)
    i_pro = int(gp.probe_end)

    # Bounds check
    if not (0 <= i_ref < len(ref)) or not (0 <= i_pro < len(probe)):
        print("❗Anchor indices out of range; cannot plot.")
        return

    o_ref, p_ref = ref[0], ref[i_ref]
    o_pro, p_pro = probe[0], probe[i_pro]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(ref[:, 0], ref[:, 1], "-", label="Reference (resampled)")
    ax.plot(probe[:, 0], probe[:, 1], "-", label="Probe (resampled)")

    # Two origin→anchor vectors
    ax.plot([o_ref[0], p_ref[0]], [o_ref[1], p_ref[1]], "--", linewidth=2, label=f"Ref anchor vec (idx={i_ref})")
    ax.plot([o_pro[0], p_pro[0]], [o_pro[1], p_pro[1]], "--", linewidth=2, label=f"Probe anchor vec (idx={i_pro})")

    # Markers
    ax.scatter([o_ref[0], p_ref[0]], [o_ref[1], p_ref[1]], s=40, marker="o")
    ax.scatter([o_pro[0], p_pro[0]], [o_pro[1], p_pro[1]], s=40, marker="x")

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("Anchor Corresponding Vectors (origin → anchor)")
    plt.tight_layout()
    plt.savefig("anchor_vectors_main.png", dpi=150, bbox_inches="tight")


# ==============================
# Predictor Class
# ==============================
class GP_predictor:
    def __init__(self):
        # State
        self.ref_pts = []
        self.sampled = None
        self.model_info = None
        self.seed_end = None

        # Probe end index
        self.probe_end = None
        self.dtheta_manual = 0.0
        self.scale_manual = 1.0

        self.probe_pts = []

        # Termination (in probe frame)
        self.probe_goal = None         # predicted finish point in probe coordinates
        self.goal_stop_eps = 0.03      # Euclidean distance threshold for stopping

        # Multiple reference trajectories
        self.refs = []

    def predict_on_transformed_probe(self):
        """
        Use anchor vectors to estimate Δθ and scale, map probe into the reference frame,
        use its last K_HIST points as the GP seed to roll out in the ref frame, then map
        predictions back to the probe frame. Supports standardization.
        """
        if not hasattr(self, "best_ref") or self.best_ref is None:
            print("❗ No best reference found (draw a probe first).")
            return

        if len(self.probe_pts) < K_HIST:
            print("❗ Probe is too short.")
            return

        # --- Step 0: Prepare data ---
        ref_np = self.best_ref["sampled"].numpy()
        model_info = self.best_ref["model_info"]
        probe_np = np.asarray(self.probe_pts, dtype=np.float64)

        # --- Step 1: Δθ and scale (manual or estimated externally) ---
        print("Estimating using the first pair of anchor vectors...")
        dtheta = self.dtheta_manual
        spatial_scale = self.scale_manual
        print(f"📐 Manual settings: Δθ={np.degrees(dtheta):.2f}°, scale={spatial_scale:.3f}")

        # --- Step 2: Map probe → reference frame ---
        c, s = np.cos(-dtheta), np.sin(-dtheta)
        R_inv = np.array([[c, -s], [s, c]])
        probe_origin = probe_np[0]
        probe_in_ref_frame = ((probe_np - probe_origin) @ R_inv.T) / spatial_scale

        # Target endpoint (for optional truncation)
        c_f, s_f = np.cos(dtheta), np.sin(dtheta)
        R_fwd = np.array([[c_f, -s_f], [s_f, c_f]], dtype=np.float64)
        ref_vec_total = ref_np[-1] - ref_np[0]
        probe_goal = probe_origin + spatial_scale * (R_fwd @ ref_vec_total)
        self.probe_goal = probe_goal
        print(f"🎯 Target end (Probe frame): {probe_goal}")

        # --- Step 3: GP seed ---
        if len(probe_in_ref_frame) < K_HIST:
            print(f"❗ probe_in_ref_frame length < {K_HIST}; cannot seed GP.")
            return
        start_t = probe_in_ref_frame.shape[0] - 1

        # --- Step 4: GP rollout (ref frame) ---
        h = 2000
        try:
            preds_ref, gt_ref, h_used = rollout_reference(
                model_info,
                torch.tensor(probe_in_ref_frame, dtype=torch.float32),
                start_t=start_t,
                h=h,
                k=K_HIST,
            )
        except Exception as e:
            print(f"❗ GP rollout failed: {e}")
            traceback.print_exc()
            return

        preds_ref_np = (
            preds_ref.numpy() if preds_ref is not None and preds_ref.numel() > 0 else np.zeros((0, 2), dtype=np.float32)
        )

        # --- Step 5: Map predictions back to probe frame ---
        c2, s2 = np.cos(dtheta), np.sin(dtheta)
        R = np.array([[c2, -s2], [s2, c2]])
        preds_world = (preds_ref_np * spatial_scale) @ R.T + probe_origin

        # Optional: truncate when reaching the target endpoint
        if self.probe_goal is not None and preds_world.shape[0] > 0:
            dists = np.linalg.norm(preds_world - self.probe_goal[None, :], axis=1)
            hit = np.where(dists <= self.goal_stop_eps)[0]
            if hit.size > 0:
                cut = int(hit[0]) + 1
                print(
                    f"✂️ Prediction enters threshold at idx={hit[0]} (d={dists[hit[0]]:.3f} ≤ {self.goal_stop_eps:.3f}); "
                    f"truncate to {cut} points."
                )
                preds_world = preds_world[:cut]

        print(f"✅ Prediction complete | Δθ={np.degrees(dtheta):.1f}°, scale={spatial_scale:.3f}")
        return preds_world

    # --- Public API (batch-style) ---
    def train_gp(self, ref_traj):
        """
        Train once with a reference trajectory.
        ref_traj: (N,2) numpy array or list[[x, y], ...]
        """
        ref_traj = np.asarray(ref_traj, dtype=np.float32)
        if ref_traj.ndim != 2 or ref_traj.shape[1] != 2:
            raise ValueError("ref_traj must be shaped (N, 2)")
        self.ref_pts = ref_traj.tolist()
        self.handle_train()
        # default to the last trained reference
        if hasattr(self, "refs") and self.refs:
            self.best_ref = self.refs[-1]
        return self.model_info

    def predict_from_probe(self, probe_traj):
        """
        Predict once with a probe trajectory (array-like of shape (M,2)).
        Returns the predicted path in the probe frame as a (K,2) ndarray or None.
        """
        if self.model_info is None:
            raise RuntimeError("Call train_gp(ref_traj) before prediction.")
        if not hasattr(self, "refs") or not self.refs:
            raise RuntimeError("No available reference; please train first.")

        probe_traj = np.asarray(probe_traj, dtype=np.float64)
        if probe_traj.ndim != 2 or probe_traj.shape[1] != 2:
            raise ValueError("probe_traj must be shaped (M, 2)")
        if probe_traj.shape[0] < 2:
            raise ValueError("probe_traj requires at least two points")

        # Resample probe to equal dt
        self.probe_pts = probe_traj.tolist()
        probe_raw = np.asarray(self.probe_pts, dtype=np.float32)
        probe_eq = resample_polyline_equal_dt(probe_raw, SAMPLE_HZ, DEFAULT_SPEED)
        if probe_eq.shape[0] >= 2:
            self.probe_pts = probe_eq.tolist()

        # Choose best reference (if only one, use it)
        self.best_ref = self.refs[-1]

        # Angle-based alignment: pick a target angle and compute indices & manual Δθ/scale
        if self.sampled is not None and len(self.probe_pts) > 1:
            ref_np = self.sampled.detach().cpu().numpy()
            probe_np = np.asarray(self.probe_pts, dtype=np.float64)

            angle_target = ANCHOR_ANGLE
            out = get_anchor_correspondence(
                ref_np,
                probe_np,
                angle_target=angle_target,
                k_hist=K_HIST,
                n_segments_base=10,
            )
            self.seed_end = out["ref_index"]
            self.probe_end = out["probe_index"]
            v_ref = out["ref_vector"]
            v_pro = out["probe_vector"]
            self.dtheta_manual = float(np.arctan2(v_pro[1], v_pro[0]) - np.arctan2(v_ref[1], v_ref[0]))
            self.scale_manual = float(np.linalg.norm(v_pro) / max(np.linalg.norm(v_ref), 1e-6))
            print(out)
        else:
            print("❗ Insufficient reference or probe points; cannot align angles/vectors.")

        preds_world = self.predict_on_transformed_probe()
        print("🧼 Probe state cleared; ready for next prediction.")
        return preds_world

    # --- Internal training routine (reference frame) ---
    def handle_train(self):
        if len(self.ref_pts) < 2:
            print("❗ Draw the reference trajectory first (at least 2 points).")
            return

        sampled = resample_polyline_equal_dt(self.ref_pts, SAMPLE_HZ, DEFAULT_SPEED)
        if sampled.shape[0] < K_HIST + 2:
            print(f"❗ Too few samples {sampled.shape[0]} < {K_HIST + 2}")
            return

        NOISE_STD = 0.000
        sampled_noisy = sampled + np.random.normal(0, NOISE_STD, size=sampled.shape)

        self.sampled = torch.tensor(sampled_noisy, dtype=torch.float32)

        X, Y = build_dataset(self.sampled, K_HIST)
        (Xtr, Ytr), (Xte, Yte), ntr = time_split(X, Y, TRAIN_RATIO)
        ds = {"X_train": Xtr, "Y_train": Ytr, "X_test": Xte, "Y_test": Yte, "n_train": ntr}
        self.model_info = train_moe(ds, METHOD_ID)

        # Record as one reference model
        self.refs.append(dict(sampled=self.sampled, model_info=self.model_info))


# ==============================
# Main (batch run)
# ==============================
if __name__ == "__main__":
    import time

    csv_path = "training_data.csv"
    rows = []
    with open(csv_path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    ref_xy = np.array([[float(r["x_actual"]), float(r["y_actual"])] for r in rows], dtype=np.float32)

    # Load probe
    csv_path = "trajectory_full.csv"
    rows = []
    with open(csv_path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    probe_xy = np.array([[float(r["x"]), float(r["y"])] for r in rows], dtype=np.float32)

    print(f"Loaded ref: {ref_xy.shape}, probe: {probe_xy.shape}")

    gp = GP_predictor()
    gp.train_gp(ref_xy)

    t0 = time.perf_counter()
    preds = gp.predict_from_probe(probe_xy)
    dt = time.perf_counter() - t0
    print(f"[predict_from_probe] elapsed: {dt*1000:.2f} ms ({dt:.4f} s)")

    # Plot ref / probe / prediction
    plt.figure(figsize=(8, 6))
    plt.plot(ref_xy[:, 0], ref_xy[:, 1], label="Reference")
    plt.plot(probe_xy[:, 0], probe_xy[:, 1], label="Probe")
    if preds is not None and len(preds) > 0:
        plt.plot(preds[:, 0], preds[:, 1], label="Prediction")

    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title("Ref / Probe / Prediction (main)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()

    # Save figure
    plt.savefig("ref_probe_prediction_main.png", dpi=150, bbox_inches="tight")

    # Save predictions to CSV
    if preds is not None and len(preds) > 0:
        with open("preds.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x_pred", "y_pred"])
            writer.writerows(preds.tolist())
        print(f"Saved predictions to preds.csv, shape={preds.shape}")