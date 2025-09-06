#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import traceback
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import csv
from icra_gp.skygp_online import SkyGP_MOE as SaoGP_MOE
import numpy as np
import csv

# ==============================
# 配置
# ==============================
SEED           = 0
SAMPLE_HZ      = 100            # 参考轨迹等时采样频率
K_HIST         = 10            # seed长度
TRAIN_RATIO    = 1.0           # 演示：全量训练
MAX_EXPERTS    = 40
NEAREST_K      = 1
MAX_DATA_PER_EXPERT = 1000
MIN_POINTS_OFFLINE  = 1
WINDOW_SIZE    = None
METHOD_ID      = 1             # 1=polar->delta; 5=polar+delta->delta
DOMAIN = dict(xmin=-1, xmax=1, ymin=-1, ymax=1)
DEFAULT_SPEED  = 0.01           # 把折线长度转时间，用于等时采样（不影响形状）
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
    ('polar+delta', 'absolute'),
    ('polar+delta', 'polar_next')  # 新增
]
METHOD_HPARAM = {
    1: {'adam_lr': 0.001, 'adam_steps': 200},
    3: {'adam_lr': 0.003, 'adam_steps': 250},
    5: {'adam_lr': 0.001, 'adam_steps': 200},
    7: {'adam_lr': 0.001, 'adam_steps': 200},
}

# ==============================
# GP 工具
# ==============================
def torch_to_np(x): return x.detach().cpu().numpy()

class Standardizer:
    def fit(self, X, Y):
        self.X_mean = X.mean(0)
        self.X_std = X.std(0).clamp_min(1e-8)
        self.Y_mean = Y.mean(0)
        self.Y_std = Y.std(0).clamp_min(1e-8)
        return self

    def x_transform(self, X): return (X - self.X_mean) / self.X_std

    def y_transform(self, Y): return (Y - self.Y_mean) / self.Y_std

    def y_inverse_transform(self, Yn):
        assert Yn.shape[-1] == self.Y_std.shape[0], f"维度不匹配: Yn.shape={Yn.shape}, std={self.Y_std.shape}"
        return Yn * self.Y_std + self.Y_mean

    # 兼容旧接口
    def y_inverse(self, Yn): return self.y_inverse_transform(Yn)

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

    ## 计算全局基准方向
    end_idx = min(10, traj.shape[0]-1)   # 防止轨迹不足10个点
    dirs = traj[1:end_idx+1] - traj[0]   # (end_idx, 2)
    global_base_dir = dirs.mean(dim=0)   # 平均方向
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
        elif output_type == 'polar_next':
            # 预测下一个点相对起点的极坐标
            next_pt = traj[t+1]
            origin = global_origin  # 起点作为极坐标原点
            v = next_pt - origin
            r = torch.norm(v)
            theta = torch.atan2(v[1], v[0])
            Ys.append(torch.tensor([r, torch.cos(theta), torch.sin(theta)], dtype=torch.float32))
        else:
            raise ValueError("Unsupported output_type")
    # print(f"构建数据集: 输入维度 {Xs[0].shape[0]}, 样本数 {len(Xs)}, Xs[100]: {Xs[100]}")
    return torch.stack(Xs), torch.stack(Ys)

def time_split(X, Y, train_ratio):
    N = X.shape[0]; ntr = int(N * train_ratio)
    return (X[:ntr], Y[:ntr]), (X[ntr:], Y[ntr:]), ntr

def train_moe(dataset, method_id=METHOD_ID):
    Xtr = dataset['X_train']; Ytr = dataset['Y_train']
    Din = Xtr.shape[1]
    Dout = Ytr.shape[1]
    scaler = Standardizer().fit(Xtr, Ytr)
    Xn = torch_to_np(scaler.x_transform(Xtr))
    Yn = torch_to_np(scaler.y_transform(Ytr))
    # Xn = torch_to_np(Xtr)
    # Yn = torch_to_np(Ytr)
    moe = SaoGP_MOE(
        x_dim=Din, y_dim=Dout, max_data_per_expert=MAX_DATA_PER_EXPERT,
        nearest_k=NEAREST_K, max_experts=MAX_EXPERTS,
        replacement=False, min_points=10**9, batch_step=10**9,
        window_size=256, light_maxiter=60
    )
    for i in range(Xn.shape[0]):
        moe.add_point(Xn[i], Yn[i])
    params = METHOD_HPARAM.get(method_id, {'adam_lr':0.001,'adam_steps':200})
    if hasattr(moe,"optimize_hyperparams") and params['adam_steps']>0:
        for e in range(len(moe.X_list)):
            if moe.localCount[e] >= MIN_POINTS_OFFLINE:
                for p in range(2):
                    moe.optimize_hyperparams(e, p, params['adam_steps'], WINDOW_SIZE, False, params['adam_lr'])
    return {'moe': moe, 'scaler': scaler, 'input_dim': Din}

def moe_predict(info, feat_1xD):
    moe, scaler = info['moe'], info['scaler']
    x = torch_to_np(feat_1xD.squeeze(0).float())  # shape: (D,)
    mu, var = moe.predict(torch_to_np(scaler.x_transform(torch.tensor(x))))
    mu = np.array(mu).reshape(1, -1)  # ✅ 保证是 shape (1, 2)
    y = torch_to_np(scaler.y_inverse(torch.tensor(mu)))  # shape (1, 2)
    return y, var  # 返回 shape (1, 2) 的 numpy

def rollout_reference(model_info, traj, start_t, h, k, input_type, output_type, scaler=None):
    assert start_t >= (k - 1), f"start_t={start_t} 太小，至少需要 {k - 1}"
    T = traj.shape[0]
    h = max(0, h)
    
    # ✅ 保持和训练时一致：使用 global origin 和 global base_dir
    global_origin = traj[0]
    if traj.shape[0] > 1:
        print("✅ 计算probe全局方向为前10段平均方向")
        end_idx = min(10, traj.shape[0]-1)
        dirs = traj[1:end_idx+1] - traj[0]
        global_base_dir = dirs.mean(dim=0)
    else:
        print("⚠️ 轨迹点不足2个，无法计算全局方向，使用默认方向")
        global_base_dir = torch.tensor([1.0, 0.0])

    # 初始化历史位置和 delta
    hist_pos = [traj[start_t - (k - 1) + i].clone() for i in range(k)]
    hist_del = []
    for i in range(k):
        idx = start_t - (k - 1) + i
        prev = traj[idx - 1] if idx - 1 >= 0 else traj[0]
        hist_del.append(traj[idx] - prev)

    cur_pos = hist_pos[-1].clone()
    preds_std = []  # 存储标准化预测
    preds_pos = []  # 存储实际位置（反标准化后）

    for _ in range(h):
        feats = []

        if 'polar' in input_type:
            # ✅ 使用 global_origin 保持训练一致性
            polar_feat = polar_feat_from_xy_torch(torch.stack(hist_pos[-k:]), global_origin)
            feats.append(polar_feat.reshape(1, -1))  # (1, 2K)

        if 'delta' in input_type:
            # ✅ 使用 global_base_dir 保持训练一致性
            delta_feat = rotate_to_fixed_frame(torch.stack(hist_del[-k:]), global_base_dir)
            feats.append(delta_feat.reshape(1, -1))  # (1, 2(K-1))

        x = torch.cat(feats, dim=1)  # shape (1, D)

        # GP预测
        y_pred, _ = moe_predict(model_info, x)  # shape (1, 2)
        y_pred = torch.tensor(y_pred, dtype=torch.float32)  # 确保 tensor 类型一致
        # print(f"Predicted (std space): {y_pred.numpy()}")
        preds_std.append(y_pred[0])

        # 反标准化输出仅在最后统一执行
        # 在 rollout 中仍然使用标准化空间的 step/delta 进行计算
        if output_type == 'delta':
            gb = global_base_dir / global_base_dir.norm()
            R = torch.stack([gb, torch.tensor([-gb[1], gb[0]])], dim=1)
            step_world = y_pred @ R.T  # shape (1, 2)
            next_pos = cur_pos + step_world[0]
            next_del = step_world[0]
        elif output_type == 'polar_next':
            r = y_pred[0, 0]
            cos_t = y_pred[0, 1]
            sin_t = y_pred[0, 2]
            next_pos = global_origin + r * torch.tensor([cos_t, sin_t], dtype=torch.float32)
            next_del = next_pos - cur_pos
        else:
            raise ValueError("Unsupported output_type")
        
        # 更新历史
        hist_pos.append(next_pos)
        hist_del.append(next_del)
        cur_pos = next_pos
        preds_pos.append(next_pos)
        
    preds = torch.stack(preds_pos, dim=0)

    # Ground truth (可选，仅调试用)
    gt = traj[start_t + 1: start_t + 1 + h]

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

def estimate_similarity_by_vectors_only(anchor_pairs):
    """
    给定若干个锚点配对（pt_ref, pt_probe），估计整体旋转角 dtheta 和缩放 scale。
    不需要时间信息 t_ref / t_probe，只用向量。
    """
    v_refs = []
    v_probes = []

    for pair in anchor_pairs:
        pt_ref = np.asarray(pair['pt_ref'], dtype=np.float64)
        pt_probe = np.asarray(pair['pt_probe'], dtype=np.float64)
        if 'ref_start' in pair:
            ref_start = np.asarray(pair['ref_start'], dtype=np.float64)
        else:
            ref_start = np.zeros(2)  # 默认起点为 (0,0)
        if 'probe_start' in pair:
            probe_start = np.asarray(pair['probe_start'], dtype=np.float64)
        else:
            probe_start = np.zeros(2)

        v_ref = pt_ref - ref_start
        v_probe = pt_probe - probe_start

        # 排除太短的向量，避免数值不稳定
        if np.linalg.norm(v_ref) < 1e-3 or np.linalg.norm(v_probe) < 1e-3:
            continue

        v_refs.append(v_ref)
        v_probes.append(v_probe)

    if len(v_refs) < 1:
        return None, None, 0  # 不足以估计

    v_refs = np.stack(v_refs, axis=0)
    v_probes = np.stack(v_probes, axis=0)

    # === 计算 Δθ（平均角度差）
    def angle_between(v1, v2):
        return np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])

    dthetas = [angle_between(vr, vp) for vr, vp in zip(v_refs, v_probes)]
    dtheta = np.mean(dthetas)

    # === 计算 scale（平均长度比）
    norms_ref = np.linalg.norm(v_refs, axis=1)
    norms_probe = np.linalg.norm(v_probes, axis=1)
    scales = norms_probe / norms_ref
    scale = np.mean(scales)

    return dtheta, scale, len(v_refs)

# ==============================
# 将参考系预测映射到新轨迹系
# ==============================
def align_and_scale_gp_prediction(
    ref_traj_np, seed_end, probe_end, K_hist, preds_ref_np, probe_points,
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
    new_anchor = probe[int(probe_end)]

    # ======================== ANGLE 模式 ========================
    if mode == 'angle':
        # --- ref 基准向量（前10段平均方向） ---
        k_hist_ref = min(10, ref.shape[0]-1)
        dirs_ref = ref[1:k_hist_ref+1] - ref[0]
        v_ref = dirs_ref.mean(axis=0)

        # --- probe 基准向量（前10段平均方向） ---
        k_hist_probe = min(10, probe.shape[0]-1)
        dirs_probe = probe[1:k_hist_probe+1] - probe[0]
        v_new = dirs_probe.mean(axis=0)

        ref_vector = ref_anchor - ref_start
        nr = np.linalg.norm(ref_vector)
        new_vector = new_anchor - new_start
        nn = np.linalg.norm(new_vector)
        if nr < 1e-9 or nn < 1e-9:
            raise ValueError("角度/尺度估计向量过短")
        ang_ref = np.arctan2(ref_vector[1], ref_vector[0])
        ang_new = np.arctan2(new_vector[1], new_vector[0])
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
        # if dtheta_override is None or spatial_scale_override is None:
        #     raise ValueError("manual 模式需要提供 dtheta_override 和 spatial_scale_override")

        # dtheta = float(dtheta_override)
        # scale = float(spatial_scale_override)
        # c, s_ = np.cos(dtheta), np.sin(dtheta)
        # R = np.array([[c, -s_], [s_, c]], dtype=np.float64)

        # t = new_anchor - scale * (R @ ref_anchor)
        # preds_new = (scale * (R @ preds_ref_np.T).T + t)
        # params = dict(
        #     mode='manual',
        #     dtheta=dtheta, s=scale, t=t,
        #     ref_anchor=ref_anchor, new_anchor=new_anchor,
        #     ref_start=ref_start, new_start=new_start,
        #     spatial_scale=scale,
        #     time_scale=(None if time_scale_override is None else float(time_scale_override)),
        #     time_scale_used_anchors=(0 if time_scale_used_anchors is None else int(time_scale_used_anchors))
        # )
        # return preds_new, params
        # --- ref 基准向量（前10段平均方向） ---
        k_hist_ref = min(10, ref.shape[0]-1)
        dirs_ref = ref[1:k_hist_ref+1] - ref[0]
        v_ref = dirs_ref.mean(axis=0)

        # --- probe 基准向量（前10段平均方向） ---
        k_hist_probe = min(10, probe.shape[0]-1)
        dirs_probe = probe[1:k_hist_probe+1] - probe[0]
        v_new = dirs_probe.mean(axis=0)

        ref_vector = ref_anchor - ref_start
        nr = np.linalg.norm(ref_vector)
        new_vector = new_anchor - new_start
        nn = np.linalg.norm(new_vector)
        if nr < 1e-9 or nn < 1e-9:
            raise ValueError("角度/尺度估计向量过短")
        ang_ref = np.arctan2(ref_vector[1], ref_vector[0])
        ang_new = np.arctan2(new_vector[1], new_vector[0])
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

def angle_diff_mod_pi(a, b):
    """计算两角之间的最小差值，范围 (-π, π]"""
    return ((a - b + np.pi) % (2 * np.pi)) - np.pi

import numpy as np
import matplotlib.pyplot as plt

# ==============================
# 角度变化绘图
# ==============================
def plot_angle_changes(ref_pts, probe_pts, k_hist=10, min_r=1e-3):
    """
    ref_pts: (N,2) numpy 数组，参考轨迹
    probe_pts: (M,2) numpy 数组，probe 轨迹
    k_hist: 用于估计切向的窗口长度（和你代码的 K_HIST 一致）
    """
    # --- 引用你已有的函数 ---
    def _wrap_pi(a): return ((a + np.pi) % (2*np.pi)) - np.pi

    def estimate_start_tangent(xy, k=5):
        xy = np.asarray(xy, dtype=np.float64)
        if len(xy) < 2: return 0.0
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

    # --- 计算角度序列 ---
    ref_angles, mask_ref = angles_relative_to_start_tangent(ref_pts, k_hist, min_r)
    probe_angles, mask_probe = angles_relative_to_start_tangent(probe_pts, k_hist, min_r)

    # --- 绘图 ---
    plt.figure(figsize=(10,4))
    plt.plot(ref_angles, label="Reference traj (relative angle)", color='red')
    plt.plot(probe_angles, label="Probe traj (relative angle)", color='blue')
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.xlabel("Point index")
    plt.ylabel("Relative angle (rad)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title("Relative angle changes of Reference vs Probe")
    plt.show()

    return ref_angles, probe_angles

# ==============================
# 在角度曲线和轨迹图上标出特定角度对应的点和向量
# ==============================
import numpy as np
import matplotlib.pyplot as plt

# === 直接复用你已有的角度工具 ===
def _wrap_pi(a): return ((a + np.pi) % (2*np.pi)) - np.pi

def estimate_start_tangent(xy, k=5):
    xy = np.asarray(xy, dtype=np.float64)
    if len(xy) < 2: return 0.0
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

def _angles_with_phi0(points, k_hist, min_r):
    """
    兼容包装器：
    - 如果你的 angles_relative_to_start_tangent 返回 (angles, mask, phi0)，就直接用；
    - 如果只返回 (angles, mask)，这里补算 phi0。
    """
    out = angles_relative_to_start_tangent(points, k_hist=k_hist, min_r=min_r)
    if isinstance(out, tuple) and len(out) == 3:
        return out  # (angles, mask, phi0)
    elif isinstance(out, tuple) and len(out) == 2:
        angles, mask = out
        phi0 = estimate_start_tangent(points, k=k_hist)
        return angles, mask, phi0
    else:
        raise RuntimeError("angles_relative_to_start_tangent 返回格式不符合预期")
    
# === 主函数：同图两子图，左ref右probe ===
import numpy as np
import matplotlib.pyplot as plt

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

def _angles_with_phi0(points, k_hist, min_r):
    """
    兼容包装器：
    - 如果你的 angles_relative_to_start_tangent 返回 (angles, mask, phi0)，就直接用；
    - 如果只返回 (angles, mask)，这里补算 phi0。
    """
    out = angles_relative_to_start_tangent(points, k_hist=k_hist, min_r=min_r)
    if isinstance(out, tuple) and len(out) == 3:
        return out  # (angles, mask, phi0)
    elif isinstance(out, tuple) and len(out) == 2:
        angles, mask = out
        phi0 = estimate_start_tangent(points, k=k_hist)
        return angles, mask, phi0
    else:
        raise RuntimeError("angles_relative_to_start_tangent 返回格式不符合预期")

def _compute_base_unit_vec(points, n_segments=10):
    pts = np.asarray(points, dtype=np.float64)
    m = min(n_segments, pts.shape[0]-1)
    if m < 1:
        return np.array([1.0, 0.0], dtype=np.float64)
    seg = np.diff(pts[:m+1], axis=0)                 # 前 m 段
    n = np.linalg.norm(seg, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    u = seg / n                                      # 单位切向
    v = u.mean(axis=0)                               # 平均方向
    if np.linalg.norm(v) < 1e-12:
        v = seg[0]
    return v / max(np.linalg.norm(v), 1e-12)

# ---------- main ----------
def plot_vectors_at_angle_ref_probe(
    ref_pts, probe_pts, angle_target, *,
    k_hist=10, min_r=1e-3, n_segments_base=10
):
    """
    在同一张图的两个子图中，画出：
      - 参考轨迹和 probe 的 target-angle 对应向量（起点->匹配点）
      - 参考轨迹和 probe 的“基准向量”（前 n_segments_base 段平均方向）

    ref_pts, probe_pts : (N,2)/(M,2)
    angle_target       : 目标相对角 (rad)
    k_hist             : 角度曲线里用于估计起点切向的窗口（与你项目 K_HIST 对齐）
    n_segments_base    : 基准向量使用的前段数（与你算法保持一致，默认 10）
    """
    ref_pts   = np.asarray(ref_pts,   dtype=np.float64)
    probe_pts = np.asarray(probe_pts, dtype=np.float64)
    assert ref_pts.shape[0] >= 2 and probe_pts.shape[0] >= 2, "ref/probe 点数至少为2"

    # 角度曲线 + 基准角
    ref_ang, ref_mask, _   = _angles_with_phi0(ref_pts,  k_hist=k_hist, min_r=min_r)
    pro_ang, pro_mask, _   = _angles_with_phi0(probe_pts, k_hist=k_hist, min_r=min_r)

    def _masked_nearest_idx(angles, mask, target):
        if not np.any(mask):
            return 0
        idxs = np.where(mask)[0]
        return int(idxs[np.argmin(np.abs(angles[idxs] - target))])

    i_ref = _masked_nearest_idx(ref_ang,  ref_mask,  angle_target)
    i_pro = _masked_nearest_idx(pro_ang,  pro_mask,  angle_target)

    # 起点与 target 向量
    o_ref, p_ref = ref_pts[0],   ref_pts[i_ref]
    v_ref        = p_ref - o_ref
    o_pro, p_pro = probe_pts[0], probe_pts[i_pro]
    v_pro        = p_pro - o_pro

    # 基准单位向量（严格按前 10 段的平均方向）
    u_ref = _compute_base_unit_vec(ref_pts,   n_segments=n_segments_base)
    u_pro = _compute_base_unit_vec(probe_pts, n_segments=n_segments_base)

    # 给基准向量一个合适的显示长度（只影响可视化，不改变方向）
    L_ref = max(np.linalg.norm(v_ref), 1e-6) * 0.6
    L_pro = max(np.linalg.norm(v_pro), 1e-6) * 0.6
    b_ref = u_ref * L_ref
    b_pro = u_pro * L_pro

    # ---- 绘图 ----
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5))

    # 左：参考
    axL.plot(ref_pts[:,0], ref_pts[:,1], '-', label='Reference traj')
    axL.scatter(o_ref[0], o_ref[1], c='k', s=30, label='Origin')
    axL.scatter(p_ref[0], p_ref[1], c='g', s=60, marker='x', label=f'target idx={i_ref}')
    # target 向量
    axL.plot([o_ref[0], o_ref[0] + v_ref[0]], [o_ref[1], o_ref[1] + v_ref[1]],
             linewidth=2, color='g', label='Target vector')
    # 基准向量（前10段平均方向）
    axL.plot([o_ref[0], o_ref[0] + b_ref[0]], [o_ref[1], o_ref[1] + b_ref[1]],
             linestyle='--', linewidth=2, color='r', label='Base tangent')
    axL.set_aspect('equal', adjustable='box')
    axL.grid(True, alpha=0.3)
    axL.set_title(f"Reference | target={angle_target:.2f} rad ({np.degrees(angle_target):.1f}°)")
    axL.legend(loc='best', fontsize=9)

    # 右：probe
    axR.plot(probe_pts[:,0], probe_pts[:,1], '-', label='Probe traj')
    axR.scatter(o_pro[0], o_pro[1], c='k', s=30, label='Origin')
    axR.scatter(p_pro[0], p_pro[1], c='g', s=60, marker='x', label=f'target idx={i_pro}')
    axR.plot([o_pro[0], o_pro[0] + v_pro[0]], [o_pro[1], o_pro[1] + v_pro[1]],
             linewidth=2, color='g', label='Target vector')
    
    axR.plot([o_pro[0], o_pro[0] + b_pro[0]], [o_pro[1], o_pro[1] + b_pro[1]],
             linestyle='--', linewidth=2, color='r', label='Base tangent')
    axR.set_aspect('equal', adjustable='box')
    axR.grid(True, alpha=0.3)
    axR.set_title("Probe (same target definition)")
    axR.legend(loc='best', fontsize=9)

    plt.suptitle("Target vector & Base tangent (Reference vs Probe)")
    plt.tight_layout()
    plt.show()

    return {
        "ref_index": i_ref,   "ref_vector": v_ref,   "ref_point": p_ref,   "ref_base_unit": u_ref,
        "probe_index": i_pro, "probe_vector": v_pro, "probe_point": p_pro, "probe_base_unit": u_pro
    }


# ==============================
# GUI
# ==============================
class GP_predictor:
    def __init__(self):
        # ---- 非阻塞绘图 ----
        plt.ion()

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
        self.anchor_step = 50
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
        self.probe_predict_mode = 'probe-based'  # or 'probe-based'
        
        # 状态
        self.ref_pts=[]
        self.sampled=None
        self.model_info=None
        self.seed_end=None

        # probe 结束点
        self.probe_end=None
        self.dtheta_manual = 0.0
        self.scale_manual = 1

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
        
        # 终点判断
        self.probe_goal = None          # 预测停止的目标点（probe坐标系）
        self.goal_stop_eps = 0.05       # 距离阈值（单位=坐标单位），可按需调
        
        # 多gp轨迹
        self.refs = []

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

        # 事件：仅保留你需要的
        self.drawing_left=False
        self.drawing_right=False
        # self.cid_press   = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        # self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_move    = self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.cid_key     = self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        plt.tight_layout()
        plt.show(block=False)
    
    
    def predict_on_transformed_probe(self):
        """
        使用锚点向量估计 Δθ 和 scale，将 probe 映射到参考轨迹帧，
        用其末尾 K_HIST 点作为 GP seed，在 ref 坐标系中 rollout，
        再将预测结果变换回 probe 坐标系。支持标准化。
        """
        if not hasattr(self, "best_ref") or self.best_ref is None:
            print("❗ 未找到最佳参考轨迹 (请先画 probe)")
            return

        if len(self.probe_pts) < K_HIST:
            print("❗ probe 太短")
            return

        # === Step 0: 准备数据 ===
        ref_np = self.best_ref['sampled'].numpy()
        model_info = self.best_ref['model_info']
        anchors = self.best_ref['anchors']
        crossed_set = self.best_ref.get('probe_crossed_set', set())
        probe_np = np.asarray(self.probe_pts, dtype=np.float64)

        # === Step 1: 估计 Δθ 和 scale ===
        anchor_pairs = []
        for idx in sorted(crossed_set):
            if idx >= len(anchors): continue
            anchor = anchors[idx]
            i_ref = anchor['idx']
            i_probe = anchor.get('probe_idx', None)
            if i_ref >= len(ref_np): continue
            pt_ref = ref_np[i_ref]
            pt_probe = probe_np[i_probe] if i_probe is not None and i_probe < len(probe_np) else probe_np[-1]
            anchor_pairs.append({
                'pt_ref': pt_ref,
                'pt_probe': pt_probe,
                'ref_start': ref_np[0],
                'probe_start': probe_np[0],
            })

        dtheta, spatial_scale, used = estimate_similarity_by_vectors_only(anchor_pairs)
        # 使用手动（由矢量可视化阶段设置）
        print("使用初始锚点向量估计......")
        # dtheta = self.dtheta_manual
        # spatial_scale = self.scale_manual
        dtheta = 0
        spatial_scale = 1
        print(f"📐 手动设定: Δθ={np.degrees(dtheta):.2f}°, scale={spatial_scale:.3f}")

        # === Step 2: 将 probe 映射到参考轨迹帧 ===
        c, s = np.cos(-dtheta), np.sin(-dtheta)
        R_inv = np.array([[c, -s], [s, c]])
        probe_origin = probe_np[0]
        probe_in_ref_frame = ((probe_np - probe_origin) @ R_inv.T) / spatial_scale

        # 目标终点（仅用于可视化/截断）
        c_f, s_f = np.cos(dtheta), np.sin(dtheta)
        R_fwd = np.array([[c_f, -s_f], [s_f, c_f]], dtype=np.float64)
        ref_vec_total = ref_np[-1] - ref_np[0]
        probe_goal = probe_origin + spatial_scale * (R_fwd @ ref_vec_total)
        self.probe_goal = probe_goal
        print(f"🎯 目标终点(Probe)：{probe_goal}")

        try:
            if getattr(self, "h_goal", None) is not None:
                try: self.h_goal.remove()
                except Exception: pass
            self.h_goal = self.ax.scatter(
                probe_goal[0], probe_goal[1],
                s=40, marker='*', color='magenta', zorder=6, label='Probe Goal'
            )
            self.ax.legend(fontsize=8, loc='upper right')
        except Exception:
            pass

        # 近终止判定
        if probe_np.shape[0] > 0:
            d_now = np.linalg.norm(probe_np[-1] - probe_goal)
            if d_now <= getattr(self, "goal_stop_eps", 0.05):
                print(f"🛑 已到终点阈值内 d={d_now:.3f} ≤ eps={self.goal_stop_eps:.3f}，不再预测。")
                self.update_scaled_pred([])   # 清空或保持现状
                return

        # === Step 3: GP seed ===
        if len(probe_in_ref_frame) < K_HIST:
            print(f"❗ probe_in_ref_frame 长度不足 {K_HIST}，无法作为 seed")
            return
        start_t = probe_in_ref_frame.shape[0] - 1

        # === Step 4: GP rollout（ref frame） ===
        h = 500
        input_type, output_type = METHOD_CONFIGS[METHOD_ID - 1]
        try:
            preds_ref, gt_ref, h_used = rollout_reference(
                model_info,
                torch.tensor(probe_in_ref_frame, dtype=torch.float32),
                start_t=start_t,
                h=h,
                k=K_HIST,
                input_type=input_type,
                output_type=output_type
            )
        except Exception as e:
            print(f"❗ GP rollout 失败: {e}")
            traceback.print_exc()
            return

        preds_ref_np = preds_ref.numpy() if preds_ref is not None and preds_ref.numel() > 0 else np.zeros((0, 2), dtype=np.float32)

        # === Step 5: 变回 probe 坐标系 ===
        c2, s2 = np.cos(dtheta), np.sin(dtheta)
        R = np.array([[c2, -s2], [s2, c2]])
        preds_world = (preds_ref_np * spatial_scale) @ R.T + probe_origin

        # 终点截断
        if self.probe_goal is not None and preds_world.shape[0] > 0:
            dists = np.linalg.norm(preds_world - self.probe_goal[None, :], axis=1)
            hit = np.where(dists <= self.goal_stop_eps)[0]
            if hit.size > 0:
                cut = int(hit[0]) + 1
                print(f"✂️ 预测在 idx={hit[0]} 进入阈值（d={dists[hit[0]]:.3f} ≤ {self.goal_stop_eps:.3f}），截断到 {cut} 点。")
                preds_world = preds_world[:cut]

        # === Step 6: 更新可视化 ===
        params = {
            'mode': 'probe→ref→probe (with standardization)',
            'dtheta': float(dtheta),
            's': float(spatial_scale),
            'probe_origin': probe_origin,
            'used_anchors': used
        }

        self.update_scaled_pred(preds_world)
        self.update_angle_vectors(params)

        print(f"✅ preds_world shape: {preds_world.shape}")
        if preds_world.shape[0] > 0:
            print("📌 First 3 points:\n", preds_world[:3])
            print("📌 Last 3 points:\n", preds_world[-3:])
        else:
            print("❗ preds_world is empty!")

        print(f"✅ 预测完成 | Δθ={np.degrees(dtheta):.1f}°, scale={spatial_scale:.3f}")

        from matplotlib.cm import get_cmap
        cmap = get_cmap('tab10')

        # 清除旧的向量句柄
        for h_ in self.anchor_vecs_ref + self.anchor_vecs_probe:
            try: h_.remove()
            except: pass
        self.anchor_vecs_ref = []
        self.anchor_vecs_probe = []

        # 绘制新的锚点向量对
        for i, pair in enumerate(anchor_pairs):
            pt_ref = pair['pt_ref']
            pt_probe = pair['pt_probe']
            ref_start = pair['ref_start']
            probe_start = pair['probe_start']

            color = cmap(i % 10)

            ref_vec = np.stack([ref_start, pt_ref], axis=0)
            h1, = self.ax.plot(ref_vec[:, 0], ref_vec[:, 1], '-', color=color, linewidth=1.5, label=f'ref_vec_{i}')

            probe_vec = np.stack([probe_start, pt_probe], axis=0)
            h2, = self.ax.plot(probe_vec[:, 0], probe_vec[:, 1], '--', color=color, linewidth=1.5, label=f'probe_vec_{i}')

            self.anchor_vecs_ref.append(h1)
            self.anchor_vecs_probe.append(h2)

        return preds_world

    def _probe_check_cross_current_anchor(self):
        if len(self.probe_pts) < 2 or not self.refs:
            return 0

        th0 = self.last_probe_angle
        th1, mask = last_window_rel_angles(self.probe_pts, W=self.anchor_window, min_r=1e-3)
        if th1 is None or not mask[-1]:
            return 0

        changed_refs = 0
        cur_probe_idx = len(self.probe_pts) - 1

        for ref in self.refs:
            anchors = ref['anchors']
            ptr = ref['current_anchor_ptr']
            buffer = ref.get('lookahead_buffer', None)

            idx0, idx1, idx2 = ptr, ptr + 1, ptr + 2

            def get_angle(i):
                return anchors[i]['angle'] if i < len(anchors) else None

            crossed0 = crossed_multi_in_angle_rel(th0, th1[-1], [get_angle(idx0)])[0] if idx0 < len(anchors) else False
            print(f"[ref] A{idx0} crossed (current) ⏳")
            crossed1 = crossed_multi_in_angle_rel(th0, th1[-1], [get_angle(idx1)])[0] if idx1 < len(anchors) else False
            crossed2 = crossed_multi_in_angle_rel(th0, th1[-1], [get_angle(idx2)])[0] if idx2 < len(anchors) else False

            if crossed0:
                ref['probe_crossed_set'].add(idx0)
                ref['current_anchor_ptr'] = idx0 + 1
                ref['lookahead_buffer'] = None
                anchors[idx0]['probe_idx'] = cur_probe_idx
                changed_refs += 1
                continue

            elif crossed1:
                print(f"[ref] A{idx1} crossed (lookahead) ⏳")
                ref['lookahead_buffer'] = {
                    'anchor_idx': idx1,
                    'probe_idx': cur_probe_idx
                }

            if buffer and crossed2:
                k1 = buffer['anchor_idx']
                k2 = idx2

                if 0 <= k1 < len(anchors) and k1 not in ref['probe_crossed_set']:
                    ref['probe_crossed_set'].add(k1)
                    anchors[k1]['probe_idx'] = buffer['probe_idx']
                if 0 <= k2 < len(anchors) and k2 not in ref['probe_crossed_set']:
                    ref['probe_crossed_set'].add(k2)
                    anchors[k2]['probe_idx'] = cur_probe_idx

                ref['current_anchor_ptr'] = k2 + 1
                ref['lookahead_buffer'] = None
                changed_refs += 1
                print(f"[ref] A{k1},{k2} crossed (lookahead) ✅✅")

            elif buffer:
                print("[ref] lookahead: waiting for next confirmation")

        return changed_refs

    # -------- 锚点可视化 --------
    def draw_anchors(self):
        for h in self.anchor_markers:
            try: h.remove()
            except Exception: pass
        self.anchor_markers.clear()

        if not self.show_anchors or self.sampled is None or not self.anchors:
            return

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

    # -------- 交互事件（保留 but 非必须） --------
    def train_gp(self, ref_traj):
        """
        ref_traj: (N,2) numpy array 或 list[list[float,float]]
        一次性设置参考轨迹并训练（保留所有plot）
        """
        ref_traj = np.asarray(ref_traj, dtype=np.float32)
        if ref_traj.ndim != 2 or ref_traj.shape[1] != 2:
            raise ValueError("ref_traj 需要是 (N,2) 形状的数组")
        # 覆盖到 ref_pts 并画线
        self.ref_pts = ref_traj.tolist()
        self.update_ref_line()
        # 训练（内部会等时重采样、建锚点、训练GP并绘制）
        self.handle_train()
        # 方便后续：默认把最后一条训练好的参考设为 best_ref
        if hasattr(self, "refs") and self.refs:
            self.best_ref = self.refs[-1]
        
        return self.model_info  # 按你先前接口约定返回model

    def predict_from_probe(self, probe_traj,model_info=None):
        """
        probe_traj: (M,2) numpy array 或 list[list[float,float]]
        一次性设置 probe，计算 Δθ/scale，rollout 并返回预测的 probe 坐标系轨迹 (K,2)
        """
        self.model_info = model_info
        if self.model_info is None:
            raise RuntimeError("请先调用 train_gp(ref_traj) 进行训练")
        if not hasattr(self, "refs") or not self.refs:
            raise RuntimeError("没有可用参考轨迹，请先训练")

        probe_traj = np.asarray(probe_traj, dtype=np.float64)
        if probe_traj.ndim != 2 or probe_traj.shape[1] != 2:
            raise ValueError("probe_traj 需要是 (M,2) 形状的数组")
        if probe_traj.shape[0] < 2:
            raise ValueError("probe_traj 至少需要两个点")
        
        # 统一替换内部 probe 点，并等时重采样（和 release 逻辑一致）
        self.probe_pts = probe_traj.tolist()
        probe_raw = np.asarray(self.probe_pts, dtype=np.float32)
        probe_eq = resample_polyline_equal_dt(probe_raw, SAMPLE_HZ, DEFAULT_SPEED)
        if probe_eq.shape[0] >= 2:
            self.probe_pts = probe_eq.tolist()
        self.update_probe_line()

        # 选择 best_ref（如果只有一条参考，直接用它）
        self.best_ref = self.refs[-1]

        # —— 角度对齐：用同一套可视化/取 seed 的逻辑 —— #
        if self.sampled is not None and len(self.probe_pts) > 1:
            ref_np = self.sampled.detach().cpu().numpy()
            probe_np = np.asarray(self.probe_pts, dtype=np.float64)

            # 可视化角度曲线（保留你的plot）
            plot_angle_changes(ref_np, probe_np, k_hist=K_HIST)

            # 选一个目标角（与你原代码一致 0.5 rad），得到 seed_end / probe_end、以及 manual Δθ/scale
            angle_target = 0.5
            out = plot_vectors_at_angle_ref_probe(
                ref_np, probe_np,
                angle_target=angle_target,
                k_hist=K_HIST,
                n_segments_base=10
            )
            self.seed_end = out['ref_index']
            self.probe_end = out['probe_index']
            v_ref = out['ref_vector']
            v_pro = out['probe_vector']
            self.dtheta_manual = float(np.arctan2(v_pro[1], v_pro[0]) - np.arctan2(v_ref[1], v_ref[0]))
            self.scale_manual = float(np.linalg.norm(v_pro) / max(np.linalg.norm(v_ref), 1e-6))
            print(out)
        else:
            print("❗参考或 probe 不足，无法绘制角度/向量对比")

        # —— 进行预测（内部会把 ref↔probe 做坐标映射 & rollout，并绘图）—— #
        preds_world = self.predict_on_transformed_probe()

        # 和你原来的逐点版本一致：清理每条参考的临时 probe 状态
        if hasattr(self, "refs"):
            for ref in self.refs:
                ref['current_anchor_ptr'] = 0
                ref['probe_crossed_set'] = set()
                ref['lookahead_buffer'] = None
                ref['reached_goal'] = False
                for a in ref.get('anchors', []):
                    a.pop('probe_idx', None)
        print("🧼 probe 状态已清空，准备下一次预测")

        # 返回 (K,2) 预测数组（若无法预测则返回 None）
        return preds_world

    def on_move(self, event):
        if event.inaxes != self.ax: return
        if self.drawing_left:
            self.ref_pts.append([event.xdata, event.ydata])
            self.update_ref_line()

        if self.drawing_right:
            self.probe_pts.append([event.xdata, event.ydata])
            self.update_probe_line()
            self._probe_check_cross_current_anchor()
            probe_np = np.asarray(self.probe_pts, dtype=np.float64)
            if probe_np.shape[0] >= 2:
                probe_rel_angle, mask = angles_relative_to_start_tangent(
                    probe_np, k_hist=K_HIST, min_r=1e-6
                )
                if mask[-1]:
                    th_cur = float(probe_rel_angle[-1])
                    self.last_probe_angle = th_cur

        if hasattr(self, 'refs') and self.refs:
            for ref in self.refs:
                if len(ref['probe_crossed_set']) == len(ref['anchors']):
                    final_angle = float(ref['anchors'][-1]['angle'])
                    if not ref.get('reached_goal', False) and len(self.probe_pts) >= 2:
                        th1, mask = angles_relative_to_start_tangent(self.probe_pts, k_hist=K_HIST, min_r=1e-6)
                        if mask[-1]:
                            th_cur = float(th1[-1])
                            crossed, _ = crossed_multi_in_angle_rel(self.last_probe_angle, th_cur, [final_angle])
                            if crossed:
                                ref['reached_goal'] = True
                                print("🎯 所有锚点已按顺序通过，且已跨过终点角度 🎉！任务完成")

    def on_key(self, event):
        key = event.key.lower()
        if key=='t': self.handle_train()
        elif key=='c': self.clear_all()
        elif key=='s': self.save_csv()
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

    def update_probe_line(self):
        if self.probe_pts:
            pts=np.asarray(self.probe_pts,dtype=np.float32)
            self.line_probe.set_data(pts[:,0], pts[:,1])
        else:
            self.line_probe.set_data([],[])

    def update_sample_line(self):
        if self.sampled is not None and len(self.sampled)>0:
            s=self.sampled
            self.line_samp.set_data(s[:,0], s[:,1])
        else:
            self.line_samp.set_data([],[])

    def update_seed_line(self):
        if self.sampled is None or self.seed_end is None or self.seed_end < K_HIST-1:
            self.line_seed.set_data([],[])
        else:
            start_idx=self.seed_end-(K_HIST-1)
            seg=self.sampled[start_idx:self.seed_end+1]
            self.line_seed.set_data(seg[:,0], seg[:,1])

    def update_ref_pred_gt(self, preds=None, gt=None):
        if preds is not None and len(preds)>0:
            self.line_pred.set_data(preds[:,0], preds[:,1])
        else:
            self.line_pred.set_data([],[])
        if gt is not None and len(gt)>0:
            self.line_gt.set_data(gt[:,0], gt[:,1])
        else:
            self.line_gt.set_data([],[])

    def update_scaled_pred(self, preds_scaled=None):
        if preds_scaled is not None and len(preds_scaled)>0:
            self.line_ps.set_data(preds_scaled[:,0], preds_scaled[:,1])
        else:
            self.line_ps.set_data([],[])


    def update_angle_vectors(self, params):
        if params is None or params.get('mode') not in ['angle', 'manual']:
            self.line_vec_ref.set_data([], [])
            self.line_vec_new.set_data([], [])
            for h in self.anchor_vecs_ref + self.anchor_vecs_probe:
                try: h.remove()
                except: pass
            self.anchor_vecs_ref = []
            self.anchor_vecs_probe = []
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
                if 'probe_idx' not in a:
                    continue
                i_ref = int(a['idx'])
                i_probe = int(a['probe_idx'])

                ref_traj = self.sampled.numpy()
                probe = np.asarray(self.probe_pts, dtype=np.float64)

                if not (0 <= i_ref < len(ref_traj)) or not (0 <= i_probe < len(probe)):
                    continue

                p_ref = ref_traj[i_ref]
                p_probe = probe[i_probe]

                v_ref = np.stack([ref_start, p_ref])
                v_probe = np.stack([new_start, p_probe])

                h1, = self.ax.plot(v_ref[:, 0], v_ref[:, 1], '--', color='gray', lw=1.0, zorder=3)
                h2, = self.ax.plot(v_probe[:, 0], v_probe[:, 1], '--', color='blue', lw=1.0, zorder=3)

                self.anchor_vecs_ref.append(h1)
                self.anchor_vecs_probe.append(h2)

    # -------- 训练/预测（参考系） --------
    def handle_train(self):
        if len(self.ref_pts) < 2:
            print("❗请先用左键画参考轨迹（至少2个点）"); return

        sampled = resample_polyline_equal_dt(self.ref_pts, SAMPLE_HZ, DEFAULT_SPEED)
        if sampled.shape[0] < K_HIST + 2:
            print(f"❗样本过少 {sampled.shape[0]} < {K_HIST+2}"); return

        NOISE_STD = 0.000
        sampled_noisy = sampled + np.random.normal(0, NOISE_STD, size=sampled.shape)

        self.sampled = torch.tensor(sampled_noisy, dtype=torch.float32)
        
        input_type, output_type = METHOD_CONFIGS[METHOD_ID-1]
        X, Y = build_dataset(self.sampled, K_HIST, input_type, output_type)
        (Xtr, Ytr), (Xte, Yte), ntr = time_split(X, Y, TRAIN_RATIO)
        print(Xtr.shape)
        ds = {'X_train': Xtr, 'Y_train': Ytr, 'X_test': Xte, 'Y_test': Yte, 'n_train': ntr}
        self.model_info = train_moe(ds, METHOD_ID)

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

        # —— 构建相对角度锚点（每 anchor_step 个点） ——
        ref_np = self.sampled.numpy()
        self.ref_rel_angle = build_relative_angles(ref_np, origin_idx=0, min_r=1e-6)

        self.anchors = []
        import math
        MIN_START_ANGLE_DIFF_DEG = 15
        MIN_START_ANGLE_DIFF = math.radians(MIN_START_ANGLE_DIFF_DEG)
        step = max(1, int(self.anchor_step))
        anchor_indices = []

        for i in range(step, len(self.ref_rel_angle), step):
            angle = self.ref_rel_angle[i]
            if np.isnan(angle):
                continue
            if len(anchor_indices) == 0:
                angle_diff = abs(angle_diff_mod_pi(angle, 0.0))
                if angle_diff >= MIN_START_ANGLE_DIFF:
                    anchor_indices.append(i)
            else:
                anchor_indices.append(i)

        if (len(self.ref_rel_angle) - 1) not in anchor_indices:
            anchor_indices.append(len(self.ref_rel_angle) - 1)

        self.anchors = []
        for i in anchor_indices:
            self.anchors.append({
                'idx': i,
                'angle': float(self.ref_rel_angle[i]),
                't_ref': i / SAMPLE_HZ
            })
        if (len(self.ref_rel_angle)-1) not in [a['idx'] for a in self.anchors]:
            j = len(self.ref_rel_angle)-1
            self.anchors.append({'idx': j, 'angle': float(self.ref_rel_angle[j])})
        if self.anchors and self.anchors[0]['idx'] == 0:
            self.anchors = self.anchors[1:]

        self.anchor_count_total = 0
        self.draw_anchors()
        self.last_end_idx = None
        self.current_anchor_ptr = 0
        self.probe_cross_count_session = 0
        self.probe_crossed_set_session = set()
        print(f"📍(relative) 固定锚点已生成 {len(self.anchors)} 个（步长={self.anchor_step}）")

        # 记录到 refs
        self.refs.append(dict(
            sampled=self.sampled,
            model_info=self.model_info,
            anchors=[dict(a) for a in self.anchors],
            current_anchor_ptr=0,
            probe_crossed_set=set(),
            lookahead_buffer=None,
            reached_goal=False
        ))
        print(f"🧠 已训练参考总数: {len(self.refs)}")

    # -------- 匹配 & 缩放预测 --------
    def match_and_scale_predict(self):
        if self.model_info is None:
            print("❗请先训练")
            return
        preds = self.predict_on_transformed_probe()
        return preds

    def draw_probe_anchors(self):
        for h in self.probe_anchor_markers:
            try:
                h.remove()
            except Exception:
                pass
        self.probe_anchor_markers.clear()

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

    # -------- 杂项 --------
    def clear_all(self):
        self.ref_pts.clear(); self.probe_pts.clear()
        self.sampled=None; self.model_info=None; self.seed_end=None; self.probe_end=None

        self.anchors = []
        self.ref_rel_angle = None
        self.anchor_count_total = 0
        for h in getattr(self, "anchor_markers", []):
            try: h.remove()
            except Exception:
                pass
        self.anchor_markers.clear()
        self.last_end_idx = None
        self.current_anchor_ptr = 0
        self.probe_cross_count_session = 0
        self.probe_crossed_set_session = set()
        self.probe_prev_contains = False

        if getattr(self, "h_goal", None) is not None:
            try: self.h_goal.remove()
            except Exception: pass
            self.h_goal = None
        self.probe_goal = None

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