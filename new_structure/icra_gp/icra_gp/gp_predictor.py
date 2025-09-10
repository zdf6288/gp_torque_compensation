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
DEFAULT_SPEED = 0.1             # speed used to convert polyline length to time for equal-time resampling
STOP_TAIL_ZERO_STEPS = 0         # set deltas to zero for the last N seeds near the end
np.random.seed(SEED)
torch.manual_seed(SEED)

# Tunable parameters
VERBOSE = False
ANCHOR_ANGLE = np.radians(15)  # angle (rad) for anchor correspondence
K_HIST = 10                      # seed length (history window)
NEAREST_K = 1
MAX_DATA_PER_EXPERT = 400
BASE_SCALE = 100                  # initial guess for scale
ROLLOUT_STEPS = 500               # steps to rollout in demo
PHI0_K_PROBE = 500                        # segments to average for phi0 in anchor finding
PHI0_K_REF = 500                        # segments to average for phi0 in anchor finding
SELECT_HORIZON = 500
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
        # 新（共享超参一次性训练）
        moe.optimize_hyperparams_global(
            max_iter=params["adam_steps"],
            verbose=False,
            window_size=WINDOW_SIZE,
            adam_lr=params["adam_lr"],
        )
    return {"moe": moe, "scaler": scaler, "input_dim": Din}


def moe_predict(info, feat_1xD: torch.Tensor):
    """Predict (mu, var) in original label space given a 1xD feature tensor."""
    moe, scaler = info["moe"], info["scaler"]
    x_n = torch_to_np(scaler.x_transform(feat_1xD.float()))  # (1, D)
    mu_n, var_n = moe.predict(x_n.squeeze(0))                # var_n 是标准化空间
    mu_n = np.array(mu_n).reshape(1, -1)
    y = torch_to_np(scaler.y_inverse(torch.tensor(mu_n)))    # 预测均值 -> 原始空间

    # 方差也映射回原始空间：Var[Y] = Var[Y_n] * (Y_std^2)
    Y_std = torch_to_np(scaler.Y_std).reshape(-1)            # (Dout,)
    var = np.array(var_n).reshape(-1) * (Y_std**2)           # (Dout,)
    return y, var


def rollout_reference_with_var(model_info, traj: torch.Tensor, start_t: int, h: int, k: int):
    """
    与 rollout_reference 类似，但额外返回每步的预测方差（在参考坐标系，输出为 var_x,var_y 的和）。
    返回: (pred_positions, var_trace_per_step, horizon_used)
    """
    assert start_t >= (k - 1), f"start_t={start_t} too small; requires at least {k - 1}"
    T = traj.shape[0]
    h = max(0, h)

    global_origin = traj[0]
    if traj.shape[0] > 1:
        end_idx = min(10, traj.shape[0] - 1)
        dirs = traj[1 : end_idx + 1] - traj[0]
        global_base_dir = dirs.mean(dim=0)
    else:
        global_base_dir = torch.tensor([1.0, 0.0])

    hist_pos = [traj[start_t - (k - 1) + i].clone() for i in range(k)]
    cur_pos = hist_pos[-1].clone()

    preds_pos = []
    var_trace_list = []

    for _ in range(h):
        polar_feat = polar_feat_from_xy_torch(torch.stack(hist_pos[-k:]), global_origin)
        x = polar_feat.reshape(1, -1)  # (1, 3k)

        # 参考坐标系下的 delta 与方差
        y_pred, var_ref = moe_predict(model_info, x)   # var_ref: (2,) 是参考系里 x,y 的方差
        y_pred = torch.tensor(y_pred, dtype=torch.float32)

        # 旋回到“世界/参考几何”坐标（只用于累加点位；方差的 trace 旋转不变）
        gb = global_base_dir / global_base_dir.norm()
        R = torch.stack([gb, torch.tensor([-gb[1], gb[0]])], dim=1)
        step_world = y_pred @ R.T  # (1, 2)

        next_pos = cur_pos + step_world[0]
        hist_pos.append(next_pos)
        cur_pos = next_pos
        preds_pos.append(next_pos)

        # 记录该步的 trace 方差（x,y 方差之和，旋转不变；跨参考时再乘以 scale^2）
        var_trace_list.append(float(np.sum(var_ref)))

    preds = torch.stack(preds_pos, dim=0) if preds_pos else torch.zeros((0, 2))
    return preds, var_trace_list, h


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
    Initial heading φ0 by chord-average from the origin:
    mean( P[i] - P[0] ), i=1..k (clamped by available points).
    This matches the 'global_base_dir' used in training/rollout.
    Returns radians in (-pi, pi].
    """
    P = np.asarray(xy, dtype=np.float64)
    if P.shape[0] < 2:
        return 0.0
    # use up to k chords from the origin
    m = int(min(max(1, P.shape[0] - 1), max(1, k)))
    dirs = P[1:m+1] - P[0]
    v = dirs.mean(axis=0)
    if not np.isfinite(v).all() or np.linalg.norm(v) < 1e-12:
        v = P[1] - P[0]
    return float(np.arctan2(v[1], v[0]))


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
        print("⚠️ Warning: angles_relative_to_start_tangent already returns phi0; redundant computation.")
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


def first_index_reach_threshold(
    angles,
    mask,
    target,
    *,
    inclusive: bool = True
) -> int:
    """
    对相邻有效点 (i_prev -> i_curr) 做区间判定：如果 target 或 -target
    任意一个在从 angles[i_prev] 到 angles[i_curr] 的最小有符号旋转区间内，
    就返回 i_prev。否则回退到二者中“更接近”的点的前一个索引。

    - 角度单位：弧度
    - 使用 (-pi, pi] 的最小有符号差处理环绕
    """
    idxs = np.where(mask)[0]
    if idxs.size == 0:
        return 0
    if idxs.size == 1:
        return int(idxs[0])

    def _diff(a, b):
        # 最小有符号差 in (-pi, pi]
        return angle_diff_mod_pi(a, b)

    def _bracket_hit(a, b, t) -> bool:
        d  = _diff(b, a)   # a -> b 的最小旋转
        dt = _diff(t, a)   # a -> t 的最小旋转
        if d > 0:
            return (0.0 <= dt <= d) if inclusive else (0.0 < dt < d)
        elif d < 0:
            return (d <= dt <= 0.0) if inclusive else (d < dt < 0.0)
        else:
            # d == 0：a 与 b 重合，仅当 t == a 时命中（考虑数值容差）
            return inclusive and (abs(dt) <= 1e-12)

    ang = np.asarray(angles, dtype=float)
    t_pos = float(target)
    t_neg = -t_pos

    # 扫描相邻有效索引对
    for j in range(1, len(idxs)):
        i_prev = idxs[j - 1]
        i_curr = idxs[j]
        a = float(ang[i_prev])
        b = float(ang[i_curr])

        if _bracket_hit(a, b, t_pos) or _bracket_hit(a, b, t_neg):
            return int(i_prev)

    # 回退：选择对 {+target, -target} 中更接近的那个
    diffs_pos = np.array([abs(_diff(float(ang[i]), t_pos)) for i in idxs])
    diffs_neg = np.array([abs(_diff(float(ang[i]), t_neg)) for i in idxs])
    diffs = np.minimum(diffs_pos, diffs_neg)

    k = int(idxs[int(np.argmin(diffs))])
    pos = int(np.where(idxs == k)[0][0])
    return int(idxs[pos - 1]) if pos > 0 else k


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


def get_anchor_correspondence(ref_pts, probe_pts, angle_target, *, min_r: float = 1e-3, n_segments_base: int = 10):
    ref_pts = np.asarray(ref_pts, dtype=np.float64)
    probe_pts = np.asarray(probe_pts, dtype=np.float64)
    assert ref_pts.shape[0] >= 2 and probe_pts.shape[0] >= 2, "ref/probe require at least 2 points"

    # 用“原始轨迹上的弦向量平均”定义的 φ0
    ref_ang, ref_mask, _ = _angles_with_phi0(ref_pts,  k_hist=PHI0_K_REF, min_r=min_r)
    pro_ang, pro_mask, _ = _angles_with_phi0(probe_pts, k_hist=PHI0_K_PROBE, min_r=min_r)

    i_ref = first_index_reach_threshold(ref_ang, ref_mask, angle_target, inclusive=True)
    i_pro = first_index_reach_threshold(pro_ang, pro_mask, angle_target, inclusive=True)
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


def _phi0_unit(points, k_hist=K_HIST):
    """
    Compute the unit vector of the initial tangent (phi0) at the origin.
    Returns (unit_vector, phi0_radians).
    """
    pts = np.asarray(points, dtype=np.float64)
    phi0 = estimate_start_tangent(pts, k=k_hist)
    u = np.array([np.cos(phi0), np.sin(phi0)], dtype=np.float64)
    return u, phi0


def plot_relative_angle_bases_from_gp(gp, *, k_hist=K_HIST, vec_len=None, filename="relative_angle_bases.png"):
    """
    Plot the two base vectors used by the relative-angle computation (phi0 of ref & probe).
    Uses the current resampled reference (gp.sampled) and probe (gp.probe_pts).
    """
    if gp.sampled is None or not gp.probe_pts:
        print("❗No data to plot base vectors (run train_gp and predict_from_probe first).")
        return

    # Resampled reference & probe used in your pipeline
    ref = gp.sampled.detach().cpu().numpy()
    probe = np.asarray(gp.probe_pts, dtype=np.float64)

    # Compute unit base vectors (phi0) at each origin
    ref_raw_for_plot = gp.best_ref.get("raw", gp.ref_pts_raw) if hasattr(gp, "best_ref") and gp.best_ref else gp.ref_pts_raw
    u_ref, phi_ref = _phi0_unit(ref_raw_for_plot, k_hist=PHI0_K_REF)
    u_pro, phi_pro = _phi0_unit(gp.probe_pts_raw, k_hist=PHI0_K_PROBE)

    # Choose a nice arrow length if not provided
    if vec_len is None:
        all_pts = np.vstack([ref, probe])
        span = float(np.max(np.ptp(all_pts, axis=0)))
        vec_len = max(1e-9, 0.15 * span)

    o_ref = ref[0]
    o_pro = probe[0]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(ref[:, 0], ref[:, 1], "-", label="Reference (resampled)")
    ax.plot(probe[:, 0], probe[:, 1], "-", label="Probe (resampled)")

    # Draw phi0 vectors from each origin
    ax.quiver(o_ref[0], o_ref[1], vec_len * u_ref[0], vec_len * u_ref[1],
              angles="xy", scale_units="xy", scale=1, width=0.004,
              label=f"Ref φ0 (k={k_hist})")
    ax.quiver(o_pro[0], o_pro[1], vec_len * u_pro[0], vec_len * u_pro[1],
              angles="xy", scale_units="xy", scale=1, width=0.004,
              label=f"Probe φ0 (k={k_hist})")

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("Base vectors for relative-angle computation (φ0 at origin)")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved base vectors plot to {filename} | "
          f"ref φ0={np.degrees(phi_ref):.2f}°, probe φ0={np.degrees(phi_pro):.2f}°")

def _closest_index(pt, arr):
    arr = np.asarray(arr, dtype=np.float64)
    d = np.linalg.norm(arr - pt[None, :], axis=1)
    return int(np.argmin(d))

# ==============================
# Predictor Class
# ==============================
class GP_predictor:
    def __init__(self):
        # State
        self.ref_pts = []
        self.sampled = None
        
        self.ref_pts_raw = None      # 原始 ref
        self.probe_pts_raw = None    # 原始 probe
        
        self.model_info = None
        self.seed_end = None

        # Probe end index
        self.probe_end = None
        self.dtheta_manual = 0.0
        self.scale_manual = 1.0

        self.probe_pts = []

        # Termination (in probe frame)
        self.probe_goal = None         # predicted finish point in probe coordinates
        self.goal_stop_eps = 0.01      # Euclidean distance threshold for stopping

        # Multiple reference trajectories
        self.refs = []
        
        self.anchor = None

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
        h = ROLLOUT_STEPS
        try:
            preds_ref, gt_ref, h_used = rollout_reference_with_var(
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
        ref_traj = np.asarray(ref_traj, dtype=np.float32)
        if ref_traj.ndim != 2 or ref_traj.shape[1] != 2:
            raise ValueError("ref_traj must be shaped (N, 2)")
        self.ref_pts_raw = ref_traj.tolist()     # <<< 保存原始
        self.ref_pts = self.ref_pts_raw[:]       # 保持同值（handle_train 内会重采样但不覆盖 raw）
        self.handle_train()
        if hasattr(self, "refs") and self.refs:
            self.best_ref = self.refs[-1]
        return self.model_info

    def predict_from_probe(self, probe_traj):
        if self.model_info is None and not self.refs:
            raise RuntimeError("Call train_gp(ref_traj) before prediction.")

        probe_traj = np.asarray(probe_traj, dtype=np.float64)
        self.probe_pts_raw = probe_traj.tolist()

        if probe_traj.ndim != 2 or probe_traj.shape[1] != 2:
            raise ValueError("probe_traj must be shaped (M, 2)")
        if probe_traj.shape[0] < 2:
            raise ValueError("probe_traj requires at least two points")

        # 等时重采样 probe（用于 GP 特征构造/rollout）
        probe_eq = resample_polyline_equal_dt(probe_traj.astype(np.float32), SAMPLE_HZ, DEFAULT_SPEED)
        if probe_eq.shape[0] >= 2:
            self.probe_pts = probe_eq.tolist()
            print(f"🔄 Probe resampled to {len(self.probe_pts)} points.")
        probe_eq_np  = np.asarray(self.probe_pts, dtype=np.float64)
        probe_raw_np = np.asarray(self.probe_pts_raw, dtype=np.float64)

        # === 关键：在多参考中选平均预测方差最低者（用10步） ===
        if len(self.refs) == 0:
            raise RuntimeError("No available reference; please train first.")

        # === 用 MSE 选最匹配的参考（默认对齐到锚点，比较前 100 点） ===
        best_idx, best_pack, best_mse = self._choose_best_ref_by_mse(
            probe_eq_np, probe_raw_np, horizon=100, align_on_anchor=True
        )
        if best_idx is None:
            print("❗ Failed to choose a best reference (insufficient data).")
            return None

        out, dtheta, scale = best_pack
        self.best_ref = self.refs[best_idx]
        print(f"✅ Selected reference #{best_idx} for prediction.")
        self.sampled  = self.best_ref["sampled"]          # 让图里/后续函数使用“最佳参考”的采样轨迹
        self.anchor = out
        self.dtheta_manual = dtheta
        self.scale_manual  = scale
        # 将原始 anchor 点映射到“重采样后的”索引，供 plot_anchor_vectors_from_gp 使用
        ref_resampled = self.sampled.detach().cpu().numpy()           # (Nr,2)
        probe_resampled = np.asarray(self.probe_pts, dtype=np.float64) # (Np,2)

        self.seed_end  = _closest_index(self.anchor["ref_point"],   ref_resampled)
        self.probe_end = _closest_index(self.anchor["probe_point"], probe_resampled)

        # 最终用“最佳参考”的 Δθ/scale 做完整 rollout
        preds_world = self.predict_on_transformed_probe()
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
        self.refs.append(dict(
            sampled=self.sampled,
            model_info=self.model_info,
            raw=np.array(self.ref_pts_raw, dtype=np.float32)  # 保存该参考的原始轨迹
        ))

    def _choose_best_ref_by_mse(
        self,
        probe_eq_np: np.ndarray,
        probe_raw_np: np.ndarray,
        *,
        horizon: int | None = 100,     # 比如只看前 100 个对齐后的点；None 表示用全部重叠段
        align_on_anchor: bool = True   # 是否用锚点对齐两个序列的索引（更稳）
    ):
        """
        对每条参考：
        1) 用原始 ref/probe 计算锚点 -> 得到 Δθ 与 scale
        2) 将“参考的等时重采样轨迹”旋转/缩放到 probe 坐标系
        3) 按索引一一对应计算平方距离并取均值（MSE）
            - 若 align_on_anchor=True：让 ref 的锚点索引与 probe 的锚点索引对齐后再比
            - horizon 限制只比较前 horizon 个重叠点
        返回: (best_idx, (anchor_out, dtheta, scale), best_mse)
        """
        best_idx, best_mse, best_pack = None, float("inf"), None

        for ridx, ref in enumerate(self.refs):
            ref_raw = ref.get("raw", None)
            if ref_raw is None or ref["model_info"] is None:
                continue

            # 1) 锚点/尺度（用原始轨迹做角度基与锚点）
            out = get_anchor_correspondence(
                ref_raw, probe_raw_np, angle_target=ANCHOR_ANGLE, n_segments_base=10
            )
            v_ref, v_pro = out["ref_vector"], out["probe_vector"]
            dtheta = float(np.arctan2(v_pro[1], v_pro[0]) - np.arctan2(v_ref[1], v_ref[0]))
            scale  = float(np.linalg.norm(v_pro) / max(np.linalg.norm(v_ref), 1e-6))

            # 2) 参考（等时重采样）→ probe 坐标系
            ref_samp = ref["sampled"].detach().cpu().numpy()     # (Nr,2)
            c, s = np.cos(dtheta), np.sin(dtheta)
            R = np.array([[c, -s], [s,  c]], dtype=np.float64)
            ref_in_probe = (ref_samp - ref_samp[0]) @ R.T * scale + probe_eq_np[0]

            # 3) 选择对齐的重叠段并计算 MSE
            if align_on_anchor:
                # 用锚点在“重采样序列”中的最近索引做对齐
                i_ref_res = _closest_index(out["ref_point"],   ref_samp)
                i_pro_res = _closest_index(out["probe_point"], probe_eq_np)
                offset = int(i_pro_res - i_ref_res)  # ref 序列需要向右移多少才能对齐 probe

                start_ref = max(0, -offset)
                start_pro = max(0,  offset)
                n_overlap = min(ref_in_probe.shape[0] - start_ref,
                                probe_eq_np.shape[0] - start_pro)
                if n_overlap <= 0:
                    continue
                if horizon is not None:
                    n_overlap = min(n_overlap, int(horizon))

                A = ref_in_probe[start_ref : start_ref + n_overlap]
                B = probe_eq_np[start_pro : start_pro + n_overlap]
            else:
                n_overlap = min(ref_in_probe.shape[0], probe_eq_np.shape[0])
                if n_overlap <= 0:
                    continue
                if horizon is not None:
                    n_overlap = min(n_overlap, int(horizon))
                A = ref_in_probe[:n_overlap]
                B = probe_eq_np[:n_overlap]

            mse = float(np.mean(np.sum((A - B) ** 2, axis=1)))
            print(f"Ref #{ridx}: MSE@{n_overlap} = {mse:.6f}")

            if mse < best_mse:
                best_mse  = mse
                best_idx  = ridx
                best_pack = (out, dtheta, scale)

        return best_idx, best_pack, best_mse

# ==============================
# Main (batch run)
# ==============================
if __name__ == "__main__":
    import time, glob

    # 1) 多条训练轨迹
    train_files = sorted(glob.glob("hand_train_*.csv"))  # 比如 hand_train_1.csv, hand_train_2.csv, ...
    gp = GP_predictor()
    for fp in train_files:
        rows = []
        with open(fp, "r") as f:
            r = csv.DictReader(f)
            for row in r:
                rows.append(row)
        ref_xy = np.array([[float(r["x_actual"]), float(r["y_actual"])] for r in rows], dtype=np.float32)
        print(f"Training on {fp}, shape={ref_xy.shape}")
        gp.train_gp(ref_xy)

    # 2) 加载一条 probe
    rows = []
    with open("hand_probe_c.csv", "r") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    probe_xy = np.array([[float(r["x_actual"]), float(r["y_actual"])] for r in rows], dtype=np.float32)

    # 3) 预测（内部会先用每个参考 GP 预测10步选最优，再完整 rollout）
    preds = gp.predict_from_probe(probe_xy)

    # >>> add transformed reference overlay <<<
    # 从 best_ref 取参考曲线（优先原始 raw，若没有就用 sampled）
    ref_sel = gp.best_ref.get("raw")
    if ref_sel is None:
        ref_sel = gp.best_ref["sampled"].detach().cpu().numpy()

    # 用挑选出的参考来做“参考→probe”的叠加
    dtheta = gp.dtheta_manual
    scale  = gp.scale_manual
    R = np.array([[np.cos(dtheta), -np.sin(dtheta)],
                [np.sin(dtheta),  np.cos(dtheta)]], dtype=np.float64)
    ref0, probe0 = ref_sel[0], probe_xy[0]
    ref_xy_in_probe = (ref_sel - ref0) @ R.T * scale + probe0

    # --- 绘图 ---
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=False)

    def _every_k_indices(n, k):
        idxs = np.arange(0, n, k, dtype=int)
        if len(idxs) == 0 or idxs[-1] != n - 1:
            idxs = np.r_[idxs, n - 1]
        return idxs

    kstep = int(MAX_DATA_PER_EXPERT)

    # ✅ 用被选中的参考 ref_sel，而不是 ref_xy
    ref_idx   = _every_k_indices(ref_sel.shape[0],   kstep)
    probe_idx = _every_k_indices(probe_xy.shape[0], kstep)

    # --- 画对应的两个锚向量（origin -> anchor） ---
    if getattr(gp, "anchor", None) is not None:
        # ✅ ref 的原点也要来自 ref_sel
        o_ref = ref_sel[0]
        p_ref = gp.anchor["ref_point"]
        o_pro = probe_xy[0]
        p_pro = gp.anchor["probe_point"]

        ax.plot([o_ref[0], p_ref[0]], [o_ref[1], p_ref[1]],
                '--', linewidth=2, color='C3',
                label=f"Ref anchor vec (idx={gp.anchor['ref_index']})", zorder=4)
        ax.plot([o_pro[0], p_pro[0]], [o_pro[1], p_pro[1]],
                '--', linewidth=2, color='C4',
                label=f"Probe anchor vec (idx={gp.anchor['probe_index']})", zorder=4)

        ax.scatter([o_ref[0], p_ref[0]], [o_ref[1], p_ref[1]],
                s=32, marker='o', color='C3', zorder=5)
        ax.scatter([o_pro[0], p_pro[0]], [o_pro[1], p_pro[1]],
                s=32, marker='x', color='C4', zorder=5)

    # ✅ 这里用 ref_sel 画“被选中的参考”
    ax.plot(ref_sel[:, 0],  ref_sel[:, 1],  label="Reference (selected)",  zorder=3)
    ax.plot(probe_xy[:, 0], probe_xy[:, 1], label="Probe", zorder=3)
    if preds is not None and len(preds) > 0:
        ax.plot(preds[:, 0], preds[:, 1], label="Prediction", zorder=3)

    # 叠加“参考→Probe”的对齐曲线（你前面已经用 ref_sel 算好了 ref_xy_in_probe）
    ax.plot(ref_xy_in_probe[:, 0], ref_xy_in_probe[:, 1], '--', linewidth=2,
            label=f"Reference→Probe (θ={np.degrees(gp.dtheta_manual):.1f}°, s={gp.scale_manual:.3f})",
            zorder=3)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0., frameon=True, framealpha=0.85)
    plt.subplots_adjust(right=0.78)
    ax.set_title("Ref / Probe / Prediction (main)")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.tight_layout()
    plt.savefig("ref_probe_prediction_main.png", dpi=150, bbox_inches="tight")

    plot_anchor_vectors_from_gp(gp)
    plot_relative_angle_bases_from_gp(gp, k_hist=K_HIST, filename="relative_angle_bases.png")
    # Save predictions to CSV
    if preds is not None and len(preds) > 0:
        with open("preds.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x_pred", "y_pred"])
            writer.writerows(preds.tolist())
        print(f"Saved predictions to preds.csv, shape={preds.shape}")