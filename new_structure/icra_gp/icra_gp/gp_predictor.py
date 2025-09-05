#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import csv
from icra_gp.skygp_online import SkyGP_MOE as SaoGP_MOE

# ==============================
# 配置
# ==============================
SEED           = 0
SAMPLE_HZ      = 20            # 参考轨迹等时采样频率
K_HIST         = 10            # seed长度
TRAIN_RATIO    = 1.0           # 演示：全量训练
MAX_EXPERTS    = 40
NEAREST_K      = 1
MAX_DATA_PER_EXPERT = 1000
MIN_POINTS_OFFLINE  = 1
WINDOW_SIZE    = None
METHOD_ID      = 1             # 1=polar->delta; 5=polar+delta->delta
DOMAIN = dict(xmin=-2, xmax=2, ymin=-2, ymax=2)
DEFAULT_SPEED  = 0.2           # 把折线长度转时间，用于等时采样（不影响形状）
LINE_WIDTHS    = dict(draw=2.0, sampled=1.0, gt=1.0, pred=1.0, seed=1.5, probe=2.0, pred_scaled=1.0)
MATCH_MODE     = 'angle'  # 可在 similarity / affine / angle 之间切换（按 M 键）

np.random.seed(SEED)
torch.manual_seed(SEED)

# ==============================
# 方法配置
# ==============================
METHOD_CONFIGS = [
    ('polar',       'delta'),
    ('polar',       'absolute'),
    ('delta',       'delta'),
    ('delta',       'absolute'),
    ('polar+delta', 'delta'),
    ('polar+delta', 'absolute')
]
METHOD_HPARAM = {
    1: {'adam_lr': 0.001, 'adam_steps': 200},
    3: {'adam_lr': 0.003, 'adam_steps': 250},
    5: {'adam_lr': 0.001, 'adam_steps': 0},
}

# ==============================
# GP 工具
# ==============================
def torch_to_np(x): return x.detach().cpu().numpy()

class Standardizer:
    def fit(self, X, Y):
        self.X_mean = X.mean(0); self.X_std = X.std(0).clamp_min(1e-8)
        self.Y_mean = Y.mean(0); self.Y_std = Y.std(0).clamp_min(1e-8)
        return self
    def x_transform(self, X): return (X - self.X_mean) / self.X_std
    def y_transform(self, Y): return (Y - self.Y_mean) / self.Y_std
    def y_inverse(self, Yn):  return Yn * self.Y_std + self.Y_mean

def rotate_to_fixed_frame(vectors, base_dir):
    base = base_dir / base_dir.norm()
    x_axis = base
    y_axis = torch.tensor([-base[1], base[0]], dtype=torch.float32)
    R = torch.stack([x_axis, y_axis], dim=1)
    return vectors @ R

def polar_feat_from_xy_torch(xy, origin):
    xy = xy.float(); origin = origin.to(xy)
    shifted = xy - origin
    r = torch.sqrt(shifted[...,0]**2 + shifted[...,1]**2)
    theta = torch.atan2(shifted[...,1], shifted[...,0])
    return torch.stack([r, torch.cos(theta), torch.sin(theta)], dim=-1)

def build_dataset(traj, k, input_type='polar+delta', output_type='delta'):
    deltas = traj[1:] - traj[:-1]
    T = traj.shape[0]
    Xs, Ys = [], []
    global_origin = traj[0]
    global_base_dir = traj[1] - traj[0]
    for t in range(k, T-1):
        feats = []
        seed_pos = traj[t-k+1:t+1]
        delta_seq = deltas[t-k+1:t+1]
        if 'polar' in input_type:
            feats.append(polar_feat_from_xy_torch(seed_pos, global_origin).reshape(-1))
        if 'delta' in input_type:
            feats.append(rotate_to_fixed_frame(delta_seq, global_base_dir).reshape(-1))
        Xs.append(torch.cat(feats))
        if output_type == 'delta':
            y_delta = traj[t+1] - traj[t]
            Ys.append(rotate_to_fixed_frame(y_delta.unsqueeze(0), global_base_dir)[0])
        elif output_type == 'absolute':
            Ys.append(traj[t+1].reshape(-1))
        else:
            raise ValueError("Unsupported output_type")
    return torch.stack(Xs), torch.stack(Ys)

def time_split(X, Y, train_ratio):
    N = X.shape[0]; ntr = int(N * train_ratio)
    return (X[:ntr], Y[:ntr]), (X[ntr:], Y[ntr:]), ntr

def train_moe(dataset, method_id=METHOD_ID):
    Xtr = dataset['X_train']; Ytr = dataset['Y_train']
    Din = Xtr.shape[1]
    scaler = Standardizer().fit(Xtr, Ytr)
    Xn = torch_to_np(scaler.x_transform(Xtr))
    Yn = torch_to_np(scaler.y_transform(Ytr))
    moe = SaoGP_MOE(
        x_dim=Din, y_dim=2, max_data_per_expert=MAX_DATA_PER_EXPERT,
        nearest_k=NEAREST_K, max_experts=MAX_EXPERTS,
        replacement=False, min_points=10**9, batch_step=10**9,
        window_size=256, light_maxiter=60
    )
    for i in range(Xn.shape[0]):
        moe.add_point(Xn[i], Yn[i])
    params = METHOD_HPARAM.get(method_id, {'adam_lr':0.001,'adam_steps':0})
    if hasattr(moe,"optimize_hyperparams") and params['adam_steps']>0:
        for e in range(len(moe.X_list)):
            if moe.localCount[e] >= MIN_POINTS_OFFLINE:
                for p in range(2):
                    moe.optimize_hyperparams(e, p, params['adam_steps'], WINDOW_SIZE, False, params['adam_lr'])
    return {'moe': moe, 'scaler': scaler, 'input_dim': Din}

def moe_predict(info, feat_1xD):
    moe, scaler = info['moe'], info['scaler']
    x = torch_to_np(feat_1xD.squeeze(0).float())
    mu, var = moe.predict(torch_to_np(scaler.x_transform(torch.tensor(x))))
    y = torch_to_np(scaler.y_inverse(torch.tensor(mu)))
    return y, var

def rollout_from_probe_std(model_info, probe_std, K_hist, input_type, output_type):
    """
    在“已标准化”的 probe 上做 GP rollout。
    probe_std: numpy array (N,2), 已完成旋转对齐 + 几何尺度归一（不要再缩放）。
    返回: (preds_numpy, n_steps)
    """
    P = np.asarray(probe_std, dtype=np.float32)
    if P.shape[0] < K_hist + 1:
        return np.zeros((0, 2), dtype=np.float32), 0

    origin  = torch.tensor(P[0], dtype=torch.float32)
    base_dir = torch.tensor(P[1] - P[0], dtype=torch.float32)

    # seed 历史
    hist_pos = [torch.tensor(p, dtype=torch.float32) for p in P[:K_hist]]
    hist_del = [hist_pos[i] - hist_pos[i-1] for i in range(1, K_hist)]
    hist_del.insert(0, torch.zeros_like(hist_pos[0]))

    cur_pos = hist_pos[-1]
    preds = []

    # 预测步数（你可以改成需要的上限）
    for _ in range(1000):
        feats = []
        if 'polar' in input_type:
            feats.append(
                polar_feat_from_xy_torch(torch.stack(hist_pos[-K_hist:]), origin).reshape(1, -1)
            )
        if 'delta' in input_type:
            feats.append(
                rotate_to_fixed_frame(torch.stack(hist_del[-K_hist:]), base_dir).reshape(1, -1)
            )
        x = torch.cat(feats, dim=1)

        y_pred, _ = moe_predict(model_info, x)  # numpy -> to torch
        y_pred = torch.tensor(y_pred, dtype=torch.float32)

        # 将局部帧 Δ 变回世界系
        gb = base_dir / base_dir.norm()
        R = torch.stack([gb, torch.tensor([-gb[1], gb[0]], dtype=torch.float32)], dim=1)
        step_world = (y_pred @ R.T)[0]  # (2,)

        next_pos = cur_pos + step_world
        next_del = step_world

        preds.append(next_pos)
        hist_pos.append(next_pos)
        hist_del.append(next_del)
        cur_pos = next_pos

    if preds:
        preds_t = torch.stack(preds, dim=0)
        return preds_t.detach().cpu().numpy(), preds_t.shape[0]
    else:
        return np.zeros((0, 2), dtype=np.float32), 0

def rollout_reference(model_info, traj, start_t, h, k, input_type, output_type):
    assert start_t >= (k-1)
    T = traj.shape[0]
    h = max(0, min(h, T - (start_t+1)))
    origin = traj[0]; base_dir = traj[1]-traj[0]
    seed_pos = [traj[start_t-k+1+i].clone() for i in range(k)]
    seed_del = []
    for i in range(k):
        idx = start_t - (k-1) + i
        seed_del.append(traj[idx] - (traj[idx-1] if idx-1>=0 else traj[0]))
    hist_pos = seed_pos[:]; hist_del = seed_del[:]; cur_pos = seed_pos[-1].clone()
    preds=[]
    for _ in range(h):
        feats=[]
        if 'polar' in input_type:
            feats.append(polar_feat_from_xy_torch(torch.stack(hist_pos[-k:]), origin).reshape(1,-1))
        if 'delta' in input_type:
            feats.append(rotate_to_fixed_frame(torch.stack(hist_del[-k:]), base_dir).reshape(1,-1))
        x = torch.cat(feats, dim=1)
        y_pred,_=moe_predict(model_info, x)
        y_pred=torch.tensor(y_pred,dtype=torch.float32)
        if output_type=='delta':
            gb=base_dir/base_dir.norm()
            R=torch.stack([gb, torch.tensor([-gb[1], gb[0]])], dim=1)
            step_world=y_pred@R.T
            next_pos=cur_pos+step_world
            next_del=step_world
        else:
            next_pos=y_pred; next_del=next_pos-cur_pos
        preds.append(next_pos); hist_pos.append(next_pos); hist_del.append(next_del); cur_pos=next_pos
    preds=torch.stack(preds,dim=0) if preds else torch.empty(0,2)
    gt=traj[start_t+1:start_t+1+h]
    return preds, gt, h

# ==============================
# 采样 & 变换
# ==============================
def resample_polyline_equal_dt(points_xy, sample_hz, speed):
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.shape[0] < 2: return pts
    seg = pts[1:]-pts[:-1]; seg_len=np.linalg.norm(seg,axis=1)
    L=float(np.sum(seg_len))
    if L<=1e-8: return pts[:1]
    T_total=L/float(speed); dt=1.0/float(sample_hz)
    t_samples=np.arange(0.0, T_total+1e-9, dt)
    s_samples=(t_samples/T_total)*L
    cum_s=np.concatenate([[0.0], np.cumsum(seg_len)])
    out=[]; j=0
    for s in s_samples:
        while j < len(seg_len)-1 and s > cum_s[j+1]: j+=1
        ds=s-cum_s[j]; r=0.0 if seg_len[j]<1e-9 else ds/seg_len[j]
        p=pts[j]+r*seg[j]; out.append(p)
    return np.asarray(out, dtype=np.float32)

def resample_to_k(points_xy, k):
    pts=np.asarray(points_xy,dtype=np.float64)
    if pts.shape[0] < 2:
        return np.repeat(pts[:1], k, axis=0) if pts.size else np.zeros((k,2),dtype=np.float64)
    seg=pts[1:]-pts[:-1]; seg_len=np.linalg.norm(seg,axis=1)
    cum=np.concatenate([[0.0], np.cumsum(seg_len)]); L=cum[-1]
    if L < 1e-9: return np.tile(pts[:1], (k,1))
    s=np.linspace(0.0, L, k)
    out=[]; j=0
    for si in s:
        while j < len(seg_len)-1 and si > cum[j+1]: j+=1
        ds=si-cum[j]; r=0.0 if seg_len[j]<1e-9 else ds/seg_len[j]
        p=pts[j]+r*seg[j]; out.append(p)
    return np.asarray(out,dtype=np.float64)

# ==============================
# 角度辅助（相对起点切向）
# ==============================
def _wrap_pi(a):
    return ((a + np.pi) % (2*np.pi)) - np.pi

def estimate_start_tangent(xy, k=5):
    xy = np.asarray(xy, dtype=np.float64)
    if len(xy) < 2:
        return 0.0
    k = int(max(2, min(k, len(xy)-1)))
    v = np.diff(xy[:k+1], axis=0)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    u = v / n
    m = u.mean(axis=0)
    if np.linalg.norm(m) < 1e-12:
        m = xy[1] - xy[0]
    return float(np.arctan2(m[1], m[0]))

def angles_relative_to_start_tangent(points, k_hist, min_r=1e-3):
    P = np.asarray(points, dtype=np.float64)
    if len(P) == 0:
        return np.array([]), np.zeros(0, dtype=bool)
    o = P[0]
    phi0 = estimate_start_tangent(P, k=k_hist)
    v = P - o
    r = np.linalg.norm(v, axis=1)
    th = np.arctan2(v[:,1], v[:,0])
    th_rel = _wrap_pi(th - phi0)
    mask = (r > min_r)
    return th_rel, mask

def find_best_seed_by_angle_window_in_range(ref_traj_np, probe_pts, W, min_r=1e-3, stride=1,
                                            lo_idx=None, hi_idx=None,
                                            min_valid_frac=0.6, use_median=True):
    probe = np.asarray(probe_pts, dtype=np.float64)
    if probe.shape[0] < 2:
        return None
    W = int(max(2, min(W if W is not None else 10, probe.shape[0])))

    th_p, m_p = angles_relative_to_start_tangent(probe, k_hist=W, min_r=min_r)
    end = len(th_p) - 1
    start = max(0, end - (W - 1))
    th_p_win = th_p[start:end+1]
    m_p_win  = m_p[start:end+1]
    if m_p_win.sum() < max(2, int(np.ceil(min_valid_frac * len(m_p_win)))):
        return None

    ref = np.asarray(ref_traj_np, dtype=np.float64)
    N = len(ref)
    if N < W:
        return None

    lo = 0 if lo_idx is None else int(max(0, lo_idx))
    hi = (N-1) if hi_idx is None else int(min(N-1, hi_idx))
    i_min = max(W-1, lo)
    i_max = min(N-2, hi)

    best_i, best_cost = None, np.inf
    needed = max(2, int(np.ceil(min_valid_frac * len(th_p_win))))

    for i in range(i_min, i_max+1, stride):
        idx_win = np.arange(i-(W-1), i+1)
        seg = ref[idx_win]
        th_r_win, m_r_win = angles_relative_to_start_tangent(seg, k_hist=W, min_r=min_r)
        m = m_r_win & m_p_win
        if m.sum() < needed:
            continue
        diffs = _angle_wrap_diff(th_r_win[m], th_p_win[m])
        cost = np.median(diffs) if use_median else np.mean(diffs)
        if cost < best_cost:
            best_cost, best_i = cost, i
    return best_i

def build_relative_angles(xy, origin_idx=0, min_r=1e-6):
    P = np.asarray(xy, dtype=np.float64)
    N = len(P)
    if N == 0:
        return np.array([], dtype=np.float64)
    sub = P[origin_idx:]
    th_rel_sub, _ = angles_relative_to_start_tangent(sub, k_hist=K_HIST, min_r=min_r)
    out = np.full(N, np.nan, dtype=np.float64)
    out[origin_idx:origin_idx+len(th_rel_sub)] = th_rel_sub
    return out

def angle_diff(a, b):
    return _wrap_pi(a - b)

def _angle_wrap_diff(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.abs(((a - b + np.pi) % (2*np.pi)) - np.pi)

def crossed_multi_in_angle_rel(theta_from, theta_to, anchor_angles):
    """
    检测角度从 theta_from 到 theta_to 的变化路径上是否跨过 anchor_angles 中的角度。
    只支持 1 或 2 个 anchor 角。
    返回: (bool, count)
    """
    assert isinstance(anchor_angles, (list, tuple, np.ndarray))
    assert len(anchor_angles) in [1, 2], "只支持 1 或 2 个 anchor_angles"

    d_total = angle_diff(theta_to, theta_from)  # 差值（带符号，(-π,π]）

    crossed_count = 0
    for a in anchor_angles:
        d_anchor = angle_diff(a, theta_from)
        if d_total > 0 and 0 < d_anchor <= d_total:
            crossed_count += 1
        elif d_total < 0 and d_total <= d_anchor < 0:
            crossed_count += 1

    return crossed_count > 0, crossed_count

def estimate_similarity_by_anchor_vectors(ref_traj_np, probe_np, anchors, used_indices=None, agg='mean'):
    """
    通过锚点向量（相对起点）来估计旋转角度与缩放因子。
    返回 (dtheta, scale, used_count)
    """
    if ref_traj_np is None or probe_np is None or len(anchors) == 0:
        return None, None, 0

    ref_start = ref_traj_np[0]
    probe_start = probe_np[0]
    scales = []
    dthetas = []

    idx_list = range(len(anchors)) if used_indices is None else used_indices
    for k in idx_list:
        a = anchors[k]
        if 't_probe' not in a:
            continue
        i_ref = int(a['idx'])
        t_probe = a['t_probe']
        i_probe = int(round(t_probe * SAMPLE_HZ))

        if not (0 <= i_ref < len(ref_traj_np)) or not (0 <= i_probe < len(probe_np)):
            continue

        v_ref = ref_traj_np[i_ref] - ref_start
        v_probe = probe_np[i_probe] - probe_start

        norm_ref = np.linalg.norm(v_ref)
        norm_probe = np.linalg.norm(v_probe)
        if norm_ref < 1e-6 or norm_probe < 1e-6:
            continue

        scale = norm_probe / norm_ref
        theta_ref = np.arctan2(v_ref[1], v_ref[0])
        theta_probe = np.arctan2(v_probe[1], v_probe[0])
        dtheta = theta_probe - theta_ref

        scales.append(scale)
        dthetas.append(dtheta)

    if len(scales) == 0:
        return None, None, 0

    scale_agg = np.median(scales) if agg == 'median' else np.mean(scales)
    dtheta_u = np.unwrap(np.array(dthetas))
    dtheta_agg = np.median(dtheta_u) if agg == 'median' else np.mean(dtheta_u)

    return float(dtheta_agg), float(scale_agg), len(scales)


# ==============================
# 将参考系预测映射到新轨迹系
# ==============================
def align_and_scale_gp_prediction(
    ref_traj_np, seed_end, K_hist, preds_ref_np, probe_points,
    mode='angle',
    time_scale_override=None,
    time_scale_used_anchors=None,
    spatial_scale_override=None,        # ✅ 新增
    dtheta_override=None                # ✅ 新增
):
    assert seed_end >= K_hist - 1
    ref_seed = ref_traj_np[seed_end - (K_hist - 1): seed_end + 1]  # (K,2)
    probe = np.asarray(probe_points, dtype=np.float64)
    if probe.shape[0] >= K_HIST:
        probe_seed = probe[-K_HIST:, :]
    else:
        probe_seed = resample_to_k(probe, K_HIST)

    ref = np.asarray(ref_traj_np, dtype=np.float64)
    assert probe.shape[0] >= 2, "目标段太短"
    ref_start = ref[0]
    ref_anchor = ref[int(seed_end)]
    new_start = probe[0]
    new_anchor = probe[-1]

    # ======================== ANGLE 模式 ========================
    if mode == 'angle':
        v_ref = ref_anchor - ref_start
        v_new = new_anchor - new_start
        nr = np.linalg.norm(v_ref)
        nn = np.linalg.norm(v_new)
        if nr < 1e-9 or nn < 1e-9:
            raise ValueError("角度/尺度估计向量过短")
        ang_ref = np.arctan2(v_ref[1], v_ref[0])
        ang_new = np.arctan2(v_new[1], v_new[0])
        dtheta = ((ang_new - ang_ref + np.pi) % (2*np.pi)) - np.pi
        c, s_ = np.cos(dtheta), np.sin(dtheta)
        R = np.array([[c, -s_], [s_, c]], dtype=np.float64)

        spatial_scale = float(nn / nr)
        scale = spatial_scale
        if time_scale_override is not None:
            scale = float(time_scale_override)

        t = new_anchor - scale * (R @ ref_anchor)
        preds_new = (scale * (R @ preds_ref_np.T).T + t)
        params = dict(
            mode='angle',
            dtheta=float(dtheta), s=scale, t=t,
            ref_anchor=ref_anchor, new_anchor=new_anchor,
            ref_start=ref_start, new_start=new_start,
            spatial_scale=spatial_scale,
            time_scale=(None if time_scale_override is None else float(time_scale_override)),
            time_scale_used_anchors=(0 if time_scale_used_anchors is None else int(time_scale_used_anchors))
        )
        return preds_new, params

    # ======================== MANUAL 模式（手动旋转/缩放） ========================
    elif mode == 'manual':
        if dtheta_override is None or spatial_scale_override is None:
            raise ValueError("manual 模式需要提供 dtheta_override 和 spatial_scale_override")

        dtheta = float(dtheta_override)
        scale = float(spatial_scale_override)
        c, s_ = np.cos(dtheta), np.sin(dtheta)
        R = np.array([[c, -s_], [s_, c]], dtype=np.float64)

        t = new_anchor - scale * (R @ ref_anchor)
        preds_new = (scale * (R @ preds_ref_np.T).T + t)
        params = dict(
            mode='manual',
            dtheta=dtheta, s=scale, t=t,
            ref_anchor=ref_anchor, new_anchor=new_anchor,
            ref_start=ref_start, new_start=new_start,
            spatial_scale=scale,
            time_scale=(None if time_scale_override is None else float(time_scale_override)),
            time_scale_used_anchors=(0 if time_scale_used_anchors is None else int(time_scale_used_anchors))
        )
        return preds_new, params

    else:
        raise ValueError("mode must be 'angle' or 'manual'")

def last_window_rel_angles(points, W, min_r=1e-3):
    P = np.asarray(points, dtype=np.float64)
    if P.shape[0] < 2:
        return None, None
    W = int(max(2, min(W if W is not None else 10, P.shape[0])))
    th, m = angles_relative_to_start_tangent(P, k_hist=W, min_r=min_r)
    end = len(th) - 1
    start = max(0, end - (W - 1))
    return th[start:end+1], m[start:end+1]

def train_reference_from_array(ref_points):
    sampled = np.asarray(ref_points, dtype=np.float32)
    if sampled.shape[0] < K_HIST + 2:
        raise ValueError("轨迹太短")
    traj = torch.tensor(sampled, dtype=torch.float32)
    input_type, output_type = METHOD_CONFIGS[METHOD_ID - 1]
    X, Y = build_dataset(traj, K_HIST, input_type, output_type)
    (Xtr, Ytr), (_, _), _ = time_split(X, Y, 1.0)
    model_info = train_moe({'X_train': Xtr, 'Y_train': Ytr}, METHOD_ID)
    seed_end = max(K_HIST-1, min(traj.shape[0]-2, int(traj.shape[0]*0.33)))
    return {'model_info': model_info, 'sampled': traj, 'seed_end': seed_end}

def predict_trajectory_from_probe(model_bundle, probe_points):
    """
    使用 probe 点和已训练的 model_info 执行预测
    :param model_bundle: 来自 train_reference_from_array 的返回值
    :param probe_points: list of [x, y]
    :return: numpy array of predicted points in probe frame
    """
    if len(probe_points) < K_HIST + 1:
        raise ValueError("probe 点数不足")

    sampled = model_bundle['sampled']
    model_info = model_bundle['model_info']
    seed_end = model_bundle['seed_end']

    input_type, output_type = METHOD_CONFIGS[METHOD_ID - 1]

    # rollout 在参考轨迹坐标系中预测
    start_t = int(seed_end)
    h = sampled.shape[0] - (start_t + 1)
    preds_ref, _, _ = rollout_reference(model_info, sampled, start_t, h, K_HIST, input_type, output_type)
    preds_ref_np = preds_ref.numpy()

    ref_np = sampled.numpy()
    probe_np = np.asarray(probe_points, dtype=np.float64)

    # 用参考轨迹中的 anchor 向量来估计 dtheta, scale
    dtheta, scale, _ = estimate_similarity_by_anchor_vectors(
        ref_traj_np=ref_np,
        probe_np=probe_np,
        anchors=[{'idx': seed_end, 't_probe': len(probe_points) / SAMPLE_HZ}]
    )
    if dtheta is None or scale is None:
        dtheta = 0.0
        scale = 1.0

    preds_probe, _ = align_and_scale_gp_prediction(
        ref_traj_np=ref_np,
        seed_end=seed_end,
        K_hist=K_HIST,
        preds_ref_np=preds_ref_np,
        probe_points=probe_np,
        mode='manual',
        spatial_scale_override=scale,
        dtheta_override=dtheta
    )

    return preds_probe
# ==============================
# GUI
# ==============================
class DrawGPApp:
    def __init__(self):
        self.fig, self.ax = plt.subplots(1,1, figsize=(8,6))
        self.ax.set_xlim(DOMAIN['xmin'], DOMAIN['xmax'])
        self.ax.set_ylim(DOMAIN['ymin'], DOMAIN['ymax'])
        self.ax.set_aspect('equal'); self.ax.grid(True, alpha=0.3)
        self.ax.set_title("Trajectory Tracking and Visualization")
        rect = matplotlib.patches.Rectangle(
            (DOMAIN['xmin'], DOMAIN['ymin']),
            DOMAIN['xmax']-DOMAIN['xmin'], DOMAIN['ymax']-DOMAIN['ymin'],
            fill=False, linestyle='--', linewidth=1.0
        )
        self.ax.add_patch(rect)

        # 固定锚点（每 N 点）
        self.anchor_step = 40
        self.anchors = []            # [{'idx':..., 'angle':...}]
        self.anchor_markers = []     # 可视化句柄
        self.show_anchors = True
        self.anchor_count_total = 0
        self.ref_rel_angle = None
        self.last_end_idx = None

        self.last_probe_angle = 0.0
        self.probe_cross_count_session = 0
        self.current_anchor_ptr = 0

        self.probe_anchor_markers = []  # 显示 probe 上越过锚点的位置
        self.probe_predict_mode = 'ref-based'  # or 'probe-based'
        
        # 状态
        self.ref_pts=[]
        self.sampled=None
        self.model_info=None
        self.seed_end=None

        self.probe_pts=[]
        self.pred_scaled=None
        self.match_mode = MATCH_MODE

        self.anchor_window = K_HIST

        self.probe_crossed_set_session = set()
        self.probe_prev_contains = False
        
        self.lookahead_buffer = None  # None 表示未处于lookahead状态

        # 锚点向量可视化
        self.anchor_vecs_ref = []
        self.anchor_vecs_probe = []

        # 句柄
        self.line_ref_tmp, = self.ax.plot([], [], '-', color='gray', lw=1.0)
        self.line_ref = None
        self.line_samp, = self.ax.plot([], [], '.', color='#FF7F0E', markersize=2, label='Demonstrated Trajectory')
        self.line_probe, = self.ax.plot([], [], '-', color="#1f77b4", lw=LINE_WIDTHS['probe'], label='Target Segment')
        self.line_ps, = self.ax.plot([], [], '.', color='#2ca02c', markersize=2, label='Scaled Pred in Target')

        self.line_seed, = self.ax.plot([], [], '-', color='black', lw=LINE_WIDTHS['seed'], label='Seed Segment')
        self.line_pred, = self.ax.plot([], [], '-', color='green', lw=LINE_WIDTHS['pred'], label='Ref Prediction')
        self.line_gt,   = self.ax.plot([], [], '-', color='purple', lw=LINE_WIDTHS['gt'], label='Ref GT')

        # Angle 模式向量可视化
        self.line_vec_ref, = self.ax.plot([], [], '--', color='#4D4D4D', marker='o', lw=1.5, label='Reference Angle Vector')
        self.line_vec_new, = self.ax.plot([], [], '--', color="#ECD96DE6", marker='o', lw=1.5, label='New Angle Vector')

        self.ax.legend(fontsize=8, loc='upper right')

        # 事件
        self.drawing_left=False
        self.drawing_right=False
        self.cid_press   = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_move    = self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.cid_key     = self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        plt.tight_layout()
        plt.show(block=True)

    def predict_on_transformed_probe(self):
        """
        用锚点向量（起点→锚点）估计 dtheta + scale，变换 GP 输出
        """
        if self.model_info is None or self.sampled is None:
            print("❗请先训练参考轨迹")
            return
        if len(self.probe_pts) < K_HIST + 1:
            print("❗probe 太短")
            return
        if self.seed_end is None:
            self.seed_end = max(K_HIST-1, min(self.sampled.shape[0]-2, int(self.sampled.shape[0]*0.33)))

        # === 1. GP rollout in reference frame ===
        input_type, output_type = METHOD_CONFIGS[METHOD_ID - 1]
        try:
            start_t = int(self.seed_end)
            h = self.sampled.shape[0] - (start_t + 1)
            preds_ref, gt_ref, h_used = rollout_reference(
                self.model_info, self.sampled, start_t, h, K_HIST, input_type, output_type
            )
        except Exception as e:
            print(f"❗ 参考轨迹 GP rollout 失败: {e}")
            return

        preds_ref_np = preds_ref.numpy() if preds_ref is not None and preds_ref.numel() > 0 else np.zeros((0,2), dtype=np.float32)
        ref_np = self.sampled.numpy()
        probe_np = np.asarray(self.probe_pts, dtype=np.float64)

        # === 2. 用锚点向量估计 dtheta + spatial scale ===
        dtheta, spatial_scale, used = estimate_similarity_by_anchor_vectors(
            ref_traj_np=ref_np,
            probe_np=probe_np,
            anchors=self.anchors,
            used_indices=sorted(self.probe_crossed_set_session)
        )
        if dtheta is None or spatial_scale is None:
            print("⚠️ 无法估计 dtheta / scale，使用默认")
            dtheta = 0.0
            spatial_scale = 1.0
        else:
            print(f"📐 锚点向量估计: Δθ={np.degrees(dtheta):.2f}°, scale={spatial_scale:.3f} (from {used} anchors)")

        # === 3. 调用 align_and_scale_gp_prediction 做变换 ===
        try:
            preds_world, params = align_and_scale_gp_prediction(
                ref_traj_np=ref_np,
                seed_end=self.seed_end,
                K_hist=K_HIST,
                preds_ref_np=preds_ref_np,
                probe_points=self.probe_pts,
                mode='manual',  # ✅ 用手动给定参数
                spatial_scale_override=spatial_scale,
                dtheta_override=dtheta,
            )
        except Exception as e:
            print(f"❗ align_and_scale_gp_prediction 失败: {e}")
            return

        self.update_scaled_pred(preds_world)
        self.update_angle_vectors(params)
        print(f"✅ 预测完成 | 手动模式 manual | seed_end={self.seed_end}")
    
    # Supporting Lookahead
    def _register_anchor_cross(self, k):
        """注册越过第 k 个锚点"""
        if k in self.probe_crossed_set_session or k >= len(self.anchors):
            return
        self.probe_crossed_set_session.add(k)
        self.probe_cross_count_session += 1
        self.anchor_count_total += 1
        self.current_anchor_ptr = k + 1

        # 记录 probe 时间
        t_probe = len(self.probe_pts) / SAMPLE_HZ
        self.anchors[k]['t_probe'] = t_probe

        print(f"✅ 注册越过锚点 A{k} -> 当前累计 {self.probe_cross_count_session}")


    def _probe_check_cross_current_anchor(self):
        if len(self.probe_pts) < 2 or not self.anchors or self.current_anchor_ptr >= len(self.anchors):
            return 0

        th0 = self.last_probe_angle
        th1, mask = last_window_rel_angles(self.probe_pts, W=self.anchor_window, min_r=1e-3)
        if th1 is None or not mask[-1]:
            return 0

        # 当前、下一个、下下一个锚点索引
        idx0 = self.current_anchor_ptr
        idx1 = idx0 + 1
        idx2 = idx0 + 2

        crossed0 = crossed_multi_in_angle_rel(th0, th1[-1], [self.anchors[idx0]['angle']])[0] if idx0 < len(self.anchors) else False
        print(f"🔍 检测锚点 A{idx0} | th0={np.degrees(th0):.2f}°, th1={np.degrees(th1[-1]):.2f}° | crossed0={crossed0}")
        crossed1 = crossed_multi_in_angle_rel(th0, th1[-1], [self.anchors[idx1]['angle']])[0] if idx1 < len(self.anchors) else False
        crossed2 = crossed_multi_in_angle_rel(th0, th1[-1], [self.anchors[idx2]['angle']])[0] if idx2 < len(self.anchors) else False

        # === 正常越过当前锚点 ===
        if crossed0:
            self._register_anchor_cross(idx0)
            self.lookahead_buffer = None
            return 1

        # === 没越过当前，但越过了下一个 ===
        elif crossed1:
            self.lookahead_buffer = True
        
        if self.lookahead_buffer:
            if crossed2:
                # 连续越过两个锚点
                print("⏳ lookahead: 连续越过两个锚点，确认")
                self._register_anchor_cross(idx1)
                self._register_anchor_cross(idx2)
                self.lookahead_buffer = None
                return 2
            else:
                print("⏳ lookahead: 等待下一个锚点确认")
                return 0

        # === 清空 buffer ===
        # self.lookahead_buffer = None
        return 0

    # -------- 锚点可视化 --------
    def draw_anchors(self):
        for h in self.anchor_markers:
            try: h.remove()
            except Exception: pass
        self.anchor_markers.clear()

        if not self.show_anchors or self.sampled is None or not self.anchors:
            self.fig.canvas.draw_idle(); return

        ref_np = self.sampled.numpy()
        for k, a in enumerate(self.anchors):
            i = a['idx']
            if 0 <= i < len(ref_np):
                p = ref_np[i]
                m = self.ax.scatter(p[0], p[1], s=20, marker='o', color='black', zorder=4)
                txt = self.ax.text(
                    p[0], p[1],
                    f"A{k}", fontsize=7, color='black',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='black', alpha=0.6),
                    zorder=5
                )
                self.anchor_markers.extend([m, txt])

        self.fig.canvas.draw_idle()

    # -------- 交互事件 --------
    def on_press(self, event):
        if event.inaxes != self.ax: return
        if event.button == 1:   # 左键：参考
            self.drawing_left = True
            self.ref_pts.append([event.xdata, event.ydata])
            self.update_ref_line()
        elif event.button == 3: # 右键：目标段（开启新 prompt）
            self.drawing_right = True
            # 开启新一次绘制会话：清空 probe，重置“会话内”的计数与集合
            self.probe_pts = [[event.xdata, event.ydata]]
            self.update_probe_line()
            self.last_probe_angle = 0.0

            # ✅ 新会话：本次会话的越过计数清零；已越过集合清空
            self.probe_cross_count_session = 0
            self.probe_crossed_set_session = set()

            # ✅ 清除锚点中旧的 t_probe（以防污染）
            for a in self.anchors:
                if 't_probe' in a:
                    del a['t_probe']

            # ❌ 不要重置 self.current_anchor_ptr（从全局进度继续）
            self.current_anchor_ptr = 0  # 不要

    def on_release(self, event):
        if event.inaxes != self.ax: return
        if event.button == 1:
            self.drawing_left = False
        elif event.button == 3:
            self.drawing_right = False
            # 每次右键松开就触发一次预测（若点数不足则忽略）
            if self.model_info is None or self.sampled is None:
                print("❗先训练(T)")
                return
            if len(self.probe_pts) < 2:
                print("❗目标段点数太少，未预测")
                return

            print(f"📍(probe) 本次绘制越过锚点数量: {self.probe_cross_count_session}")

            # 只在“上一个锚点 → 当前锚点”区间内搜索 seed（若当前指针>=1）
            if self.current_anchor_ptr >= 1 and self.current_anchor_ptr < len(self.anchors):
                lo_ptr = self.current_anchor_ptr - 1
                hi_ptr = self.current_anchor_ptr
                lo_idx = int(self.anchors[lo_ptr]['idx'])
                hi_idx = int(self.anchors[hi_ptr]['idx'])

                best_seed = find_best_seed_by_angle_window_in_range(
                    ref_traj_np=self.sampled.numpy(),
                    probe_pts=self.probe_pts,
                    W=K_HIST, min_r=1e-3, stride=1,
                    lo_idx=lo_idx, hi_idx=hi_idx
                )
                if best_seed is None:
                    print(f"⚠️ (angle) 区间[A{lo_ptr}→A{hi_ptr}] 未匹配到合适 seed_end")
                    return
                self.seed_end = int(best_seed)
                print(f"📐(angle) 区间[A{lo_ptr}→A{hi_ptr}] 滑窗匹配 seed_end={self.seed_end}")
            else:
                # 若还在第一个锚点之前，可退化为在 [K_HIST-1, seed合法上界] 内做一次全局或宽区间匹配
                if self.seed_end is None:
                    self.seed_end = max(K_HIST-1, min(self.sampled.shape[0]-2, int(self.sampled.shape[0]*0.33)))

            # 预测+映射
            self.match_and_scale_predict()


    def on_move(self, event):
        if event.inaxes != self.ax: return
        if self.drawing_left:
            self.ref_pts.append([event.xdata, event.ydata])
            self.update_ref_line()

        if self.drawing_right:
            self.probe_pts.append([event.xdata, event.ydata])
            self.update_probe_line()
            self._probe_check_cross_current_anchor()
            # === 计算 probe 相对角度（相对于 probe 起点切向） ===
            probe_np = np.asarray(self.probe_pts, dtype=np.float64)
            if probe_np.shape[0] >= 2:
                probe_rel_angle, mask = angles_relative_to_start_tangent(
                    probe_np, k_hist=K_HIST, min_r=1e-6
                )
                if mask[-1]:
                    th_cur = float(probe_rel_angle[-1])
                    self.last_probe_angle = th_cur

    def on_key(self, event):
        key = event.key.lower()
        if key=='t': self.handle_train()
        elif key=='p': self.handle_predict_reference()
        elif key=='left': self.move_seed(-1)
        elif key=='right': self.move_seed(+1)
        elif key == 'v':
            if self.probe_predict_mode == 'ref-based':
                self.probe_predict_mode = 'probe-based'
            else:
                self.probe_predict_mode = 'ref-based'
            print(f"🔁 当前预测模式切换为: {self.probe_predict_mode}")
        elif key=='c': self.clear_all()
        elif key=='s': self.save_csv()
        elif key == 'g':  # 直接用 probe 坐标系 rollout 预测
            self.predict_on_transformed_probe()
        elif key=='a':
            self.show_anchors = not self.show_anchors
            self.draw_anchors()
            print(f"📍锚点显示: {'ON' if self.show_anchors else 'OFF'} | 已计数={self.anchor_count_total}")

    # -------- 可视化更新 --------
    def update_ref_line(self):
        if self.ref_pts:
            pts = np.asarray(self.ref_pts, dtype=np.float32)
            self.line_ref_tmp.set_data(pts[:, 0], pts[:, 1])
        else:
            self.line_ref_tmp.set_data([], [])
        self.fig.canvas.draw_idle()

    def update_probe_line(self):
        if self.probe_pts:
            pts=np.asarray(self.probe_pts,dtype=np.float32)
            self.line_probe.set_data(pts[:,0], pts[:,1])
        else:
            self.line_probe.set_data([],[])
        self.fig.canvas.draw_idle()

    def update_sample_line(self):
        if self.sampled is not None and len(self.sampled)>0:
            s=self.sampled
            self.line_samp.set_data(s[:,0], s[:,1])
        else:
            self.line_samp.set_data([],[])
        self.fig.canvas.draw_idle()

    def update_seed_line(self):
        if self.sampled is None or self.seed_end is None or self.seed_end < K_HIST-1:
            self.line_seed.set_data([],[])
        else:
            start_idx=self.seed_end-(K_HIST-1)
            seg=self.sampled[start_idx:self.seed_end+1]
            self.line_seed.set_data(seg[:,0], seg[:,1])
        self.fig.canvas.draw_idle()

    def update_ref_pred_gt(self, preds=None, gt=None):
        if preds is not None and len(preds)>0:
            self.line_pred.set_data(preds[:,0], preds[:,1])
        else:
            self.line_pred.set_data([],[])
        if gt is not None and len(gt)>0:
            self.line_gt.set_data(gt[:,0], gt[:,1])
        else:
            self.line_gt.set_data([],[])
        self.fig.canvas.draw_idle()

    def update_scaled_pred(self, preds_scaled=None):
        if preds_scaled is not None and len(preds_scaled)>0:
            self.line_ps.set_data(preds_scaled[:,0], preds_scaled[:,1])
        else:
            self.line_ps.set_data([],[])
        self.fig.canvas.draw_idle()

    def update_angle_vectors(self, params):
        if params is None or params.get('mode') not in ['angle', 'manual']:
            self.line_vec_ref.set_data([], [])
            self.line_vec_new.set_data([], [])
            # 清除锚点向量
            for h in self.anchor_vecs_ref + self.anchor_vecs_probe:
                try: h.remove()
                except: pass
            self.anchor_vecs_ref = []
            self.anchor_vecs_probe = []
            self.fig.canvas.draw_idle()
            return
        rs = params['ref_start']; ra = params['ref_anchor']
        ns = params['new_start']; na = params['new_anchor']
        self.line_vec_ref.set_data([rs[0], ra[0]], [rs[1], ra[1]])
        self.line_vec_new.set_data([ns[0], na[0]], [ns[1], na[1]])
        
        if params.get('mode') == 'manual':
            ref_start = params['ref_start']
            new_start = params['new_start']

            for h in self.anchor_vecs_ref + self.anchor_vecs_probe:
                try: h.remove()
                except: pass
            self.anchor_vecs_ref = []
            self.anchor_vecs_probe = []

            for k, a in enumerate(self.anchors):
                if 't_probe' not in a:
                    continue
                i_ref = int(a['idx'])
                t_probe = a['t_probe']
                i_probe = int(round(t_probe * SAMPLE_HZ))

                ref_traj = self.sampled.numpy()
                probe = np.asarray(self.probe_pts, dtype=np.float64)

                if not (0 <= i_ref < len(ref_traj)) or not (0 <= i_probe < len(probe)):
                    continue

                p_ref = ref_traj[i_ref]
                p_probe = probe[i_probe]

                # 向量：start -> anchor
                v_ref = np.stack([ref_start, p_ref])
                v_probe = np.stack([new_start, p_probe])

                # 绘图
                h1, = self.ax.plot(v_ref[:,0], v_ref[:,1], '--', color='gray', lw=1.0, zorder=3)
                h2, = self.ax.plot(v_probe[:,0], v_probe[:,1], '--', color='blue', lw=1.0, zorder=3)
                self.anchor_vecs_ref.append(h1)
                self.anchor_vecs_probe.append(h2)

            self.fig.canvas.draw_idle()

        self.fig.canvas.draw_idle()

    # -------- 训练/预测（参考系） --------
    def handle_train(self):
        if len(self.ref_pts) < 2:
            print("❗请先用左键画参考轨迹（至少2个点）"); return

        sampled = resample_polyline_equal_dt(self.ref_pts, SAMPLE_HZ, DEFAULT_SPEED)
        if sampled.shape[0] < K_HIST + 2:
            print(f"❗样本过少 {sampled.shape[0]} < {K_HIST+2}"); return

        self.sampled = torch.tensor(sampled, dtype=torch.float32)
        input_type, output_type = METHOD_CONFIGS[METHOD_ID-1]
        X, Y = build_dataset(self.sampled, K_HIST, input_type, output_type)
        (Xtr, Ytr), (Xte, Yte), ntr = time_split(X, Y, TRAIN_RATIO)
        ds = {'X_train': Xtr, 'Y_train': Ytr, 'X_test': Xte, 'Y_test': Yte, 'n_train': ntr}
        self.model_info = train_moe(ds, METHOD_ID)

        self.seed_end = max(K_HIST-1, min(self.sampled.shape[0]-2, int(self.sampled.shape[0]*0.33)))
        self.update_sample_line(); self.update_seed_line(); self.update_ref_pred_gt(None, None)

        # 隐藏画线时的临时轨迹
        self.line_ref_tmp.set_visible(False)

        # 删除旧的正式轨迹（如果有）
        if self.line_ref:
            self.line_ref.remove()

        # 显示新的正式轨迹（红色）
        pts = self.sampled.numpy()
        self.line_ref, = self.ax.plot(pts[:,0], pts[:,1], '-', color='red', lw=2.0, label='Final Reference Trajectory')
        self.ax.legend(fontsize=8, loc='upper right')
        self.fig.canvas.draw_idle()

        # —— 构建相对角度锚点（每 anchor_step 个点） ——
        ref_np = self.sampled.numpy()
        self.ref_rel_angle = build_relative_angles(ref_np, origin_idx=0, min_r=1e-6)

        self.anchors = []
        step = max(1, int(self.anchor_step))
        for i in range(0, len(self.ref_rel_angle), step):
            self.anchors.append({
                'idx': i,
                'angle': float(self.ref_rel_angle[i]),
                't_ref': i / SAMPLE_HZ   # ⚡ 参考时间戳（秒）
            })
        if (len(self.ref_rel_angle)-1) not in [a['idx'] for a in self.anchors]:
            j = len(self.ref_rel_angle)-1
            self.anchors.append({'idx': j, 'angle': float(self.ref_rel_angle[j])})
        # 去掉第一个点作为锚点
        if self.anchors and self.anchors[0]['idx'] == 0:
            self.anchors = self.anchors[1:]

        self.anchor_count_total = 0
        self.draw_anchors()
        self.last_end_idx = None
        self.current_anchor_ptr = 0
        self.probe_cross_count_session = 0
        self.probe_crossed_set_session = set()
        print(f"📍(relative) 固定锚点已生成 {len(self.anchors)} 个（步长={self.anchor_step}）")

    def handle_predict_reference(self):
        if self.model_info is None or self.sampled is None:
            print("❗先训练(T)"); return
        if self.seed_end is None:
            self.seed_end = K_HIST-1
        input_type, output_type = METHOD_CONFIGS[METHOD_ID-1]
        start_t=int(self.seed_end); h = self.sampled.shape[0] - (start_t+1)
        preds, gt, h_used = rollout_reference(self.model_info, self.sampled, start_t, h, K_HIST, input_type, output_type)
        preds_np = preds.numpy() if preds.numel()>0 else np.zeros((0,2),dtype=np.float32)
        gt_np    = gt.numpy()    if gt.numel()>0    else np.zeros((0,2),dtype=np.float32)
        self.update_ref_pred_gt(preds_np, gt_np)
        self.update_angle_vectors(None)
        mse = float(((preds-gt)**2).mean().item()) if gt.numel()>0 else float('nan')
        print(f"🔮 参考系预测: h={h_used} | MSE={mse:.6f}")

    def move_seed(self, delta):
        if self.sampled is None:
            print("❗先训练(T)"); return
        new_end = (self.seed_end if self.seed_end is not None else (K_HIST-1)) + int(delta)
        self.seed_end = max(K_HIST-1, min(self.sampled.shape[0]-2, new_end))
        self.update_seed_line()
        print(f"↔️ seed_end={self.seed_end}")

    # -------- 匹配 & 缩放预测（含相对角度锚点计数 + 局部 seed 搜索） --------
    def match_and_scale_predict(self):
        """
        两种预测模式：
        - ref-based：在参考轨迹上 rollout，再映射到 probe 系
        - probe-based：直接用 probe 的 seed rollout，不依赖参考轨迹
        """

        if self.model_info is None:
            print("❗请先训练")
            return

        input_type, output_type = METHOD_CONFIGS[METHOD_ID - 1]

        if self.probe_predict_mode == 'probe-based':
            self.predict_on_transformed_probe()
            return

        # ========== 模式 A：参考轨迹 rollout + 映射 ==========
        if self.sampled is None or self.seed_end is None:
            print("❗缺少参考轨迹或 seed_end")
            return

        if len(self.probe_pts) < 2:
            print("❗probe 太短")
            return

        try:
            start_t = int(self.seed_end)
            h = self.sampled.shape[0] - (start_t + 1)
            preds_ref, gt_ref, h_used = rollout_reference(
                self.model_info, self.sampled, start_t, h, K_HIST, input_type, output_type
            )
        except Exception as e:
            print(f"⚠️ 参考系 rollout 失败: {e}")
            return

        preds_ref_np = preds_ref.numpy() if preds_ref is not None and preds_ref.numel() > 0 else np.zeros((0,2), dtype=np.float32)
        ref_traj_np = self.sampled.numpy()

        try:
            preds_tar, params = align_and_scale_gp_prediction(
                ref_traj_np=ref_traj_np,
                seed_end=self.seed_end,
                K_hist=K_HIST,
                preds_ref_np=preds_ref_np,
                probe_points=self.probe_pts,
                mode=self.match_mode
            )
        except Exception as e:
            print(f"⚠️ 匹配失败: {e}")
            return

        self.update_scaled_pred(preds_tar)

        if params.get('mode') == 'angle':
            self.update_angle_vectors(params)
        else:
            self.update_angle_vectors(None)

        if gt_ref is not None and gt_ref.numel() > 0:
            mse_ref = float(((preds_ref - gt_ref)**2).mean().item())
            pretty = {k: (np.round(v, 4) if isinstance(v, np.ndarray) else v) for k, v in params.items()}
            print(f"🎯 ref-based 匹配完成 | 模式={self.match_mode} | seed_end={self.seed_end} | MSE={mse_ref:.6f} | 参数: {pretty}")
        else:
            print("🎯 ref-based 匹配完成")
            predicted_trajectory_index
    def draw_probe_anchors(self):
        # 移除旧的
        for h in self.probe_anchor_markers:
            try:
                h.remove()
            except Exception:
                pass
        self.probe_anchor_markers.clear()

        if len(self.probe_pts) < 1:
            self.fig.canvas.draw_idle()
            return

        pts = np.asarray(self.probe_pts, dtype=np.float64)
        for k, a in enumerate(self.anchors):
            if 't_probe' in a:
                t_probe = a['t_probe']
                idx = int(round(t_probe * SAMPLE_HZ))
                if 0 <= idx < len(pts):
                    p = pts[idx]
                    m = self.ax.scatter(p[0], p[1], s=20, marker='x', color='blue', zorder=5)
                    txt = self.ax.text(
                        p[0], p[1],
                        f"P{k}", fontsize=7, color='blue',
                        bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='blue', alpha=0.6),
                        zorder=6
                    )
                    self.probe_anchor_markers.extend([m, txt])

        self.fig.canvas.draw_idle()


    # -------- 杂项 --------
    def clear_all(self):
        self.ref_pts.clear(); self.probe_pts.clear()
        self.sampled=None; self.model_info=None; self.seed_end=None

        # —— 清理锚点数据和可视化 ——
        self.anchors = []
        self.ref_rel_angle = None
        self.anchor_count_total = 0
        for h in getattr(self, "anchor_markers", []):
            try:
                h.remove()
            except Exception:
                pass
        self.anchor_markers.clear()
        self.last_end_idx = None
        self.current_anchor_ptr = 0
        self.probe_cross_count_session = 0
        self.probe_crossed_set_session = set()
        self.probe_prev_contains = False

        # 下面保持不变
        self.line_ref_tmp.set_data([], []); self.line_ref_tmp.set_visible(True)
        if self.line_ref:
            self.line_ref.remove()
            self.line_ref = None
        self.update_probe_line()
        self.update_sample_line()
        self.update_scaled_pred(None)
        self.update_angle_vectors(None)
        self.update_ref_pred_gt(None, None)
        self.update_seed_line()
        print("🧹 已清空")

    def save_csv(self):
        if self.sampled is None:
            print("❗无参考等时数据"); return
        ts=datetime.now().strftime("%Y%m%d_%H%M%S")
        fname=f"ref_traj_{SAMPLE_HZ}hz_{ts}.csv"
        with open(fname,"w",newline="") as f:
            w=csv.writer(f); w.writerow(["x_actual","y_actual"])
            for p in self.sampled.numpy(): w.writerow([float(p[0]), float(p[1])])
        print(f"💾 已保存: {fname}")

# ==============================
# 入口
# ==============================
# if __name__ == "__main__":
#     DrawGPApp()
if __name__ == "__main__":
    # 输入轨迹
    ref = [[0, 0], [1, 0.5], [2, 1], [3, 1.5], [4, 2]]
    probe = [[5, 5], [6, 5.5], [7, 6], [8, 6.5], [9, 7]]

    # 训练
    model_bundle = train_reference_from_array(ref)

    # 预测
    predicted = predict_trajectory_from_probe(model_bundle, probe)

    print("✅ 预测轨迹：")
    print(predicted)