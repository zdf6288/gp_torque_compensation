#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import os, sys, pickle, importlib.util
from custom_msgs.srv import AsyncGPpredict   # 刚才定义的 srv
import copy
import csv

class GPServer(Node):
    def __init__(self):
        super().__init__('gp_server')
        self.get_logger().info("[GPServer] starting...")

        self.declare_parameter('model_dir', './new_structure/gp/gp_models')
        self.model_dir = self.get_parameter('model_dir').value

        self.gp_models = {}
        self.gp_ready = False
        self.use_gp = True

        # 多点采样参数：在 state 附近采样 n 个点做预测并融合
        self.declare_parameter('gp_n_samples', 5)         # 每个关节输入附近的采样点数（不含中心）
        self.declare_parameter('gp_sample_radius', 0.3)   # 采样半径（在标准化后空间里）
        self.n_samples = int(self.get_parameter('gp_n_samples').value)
        self.sample_radius = float(self.get_parameter('gp_sample_radius').value)

        # 先确保 skygp 导入
        self._ensure_skygp_import()
        # 加载离线训练的模型
        self._load_gp_models(self.model_dir)

        # 创建 service
        self.srv = self.create_service(AsyncGPpredict, '/gp_predict', self.gp_predict_callback)
        self.get_logger().info("[GPServer] service /gp_predict ready")

        # === 日志缓存：方便之后存成 CSV 画图 ===
        self.enable_logging = True
        self.log_time = []           # 每次 service 调用的时间戳（秒）
        self.log_y_slow = []         # 每次调用: 7 维 slow 预测
        self.log_y_fast = []         # 每次调用: 7 维 fast 预测
        self.log_y_hat = []          # 每次调用: 7 维 融合后 y_hat
        self.log_tau_residual = []   # 每次调用: 7 维 真实残差


    def gp_predict_callback(self, request, response):
        """
        输入：
        - q
        - dq_des_joint / ddq_des_joint           （当前期望关节）
        - dq_des_joint_future / ddq_des_joint_future  （未来期望关节）
        - tau_residual
        输出：
        - y_hat[7]：slow + fast 融合后的补偿
        """
        try:
            q          = np.array(request.q, dtype=np.float32)
            dq_now     = np.array(request.dq_des_joint, dtype=np.float32)
            ddq_now    = np.array(request.ddq_des_joint, dtype=np.float32)
            dq_future  = np.array(request.dq_des_joint_future, dtype=np.float32)
            ddq_future = np.array(request.ddq_des_joint_future, dtype=np.float32)
            tau_residual = np.array(request.tau_residual, dtype=np.float32)

            # y_hat = self._gp_predict_and_update_twofeatures(
            #     q, dq_now, ddq_now, dq_future, ddq_future, tau_residual
            # )
            y_hat = self._gp_predict_and_update(q,dq_now,ddq_now,tau_residual)
            response.y_hat = y_hat.astype(np.float32).tolist()
        except Exception as e:
            self.get_logger().error(f"[GPServer] error in callback: {e}")
            response.y_hat = [0.0]*7

        return response

    # ==== 把你原来的这三个函数基本原样搬进来 ====
    def _ensure_skygp_import(self):
        """
        确保在当前进程里有名为 'skygp' 的模块。
        优先从工作空间的 src/new_structure/gp/skygp.py 加载。
        """
        import importlib.util
        import sys

        if "skygp" in sys.modules:
            return True

        candidates = []

        # 1) 以当前工作目录为基准：<ws>/src/new_structure/gp/skygp.py
        cwd = os.getcwd()
        candidates.append(os.path.join(cwd, "new_structure", "gp", "skygp.py"))

        # 2) 以安装后的 gp_server.py 为基准，再往上回到 src 目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates.append(os.path.abspath(os.path.join(
            script_dir, "..", "..", "..", "..", "src", "new_structure", "gp", "skygp.py"
        )))

        skygp_path = None
        for p in candidates:
            if os.path.isfile(p):
                skygp_path = p
                break

        if skygp_path is None:
            self.get_logger().error("[GP] skygp.py not found in candidates:")
            for p in candidates:
                self.get_logger().error(f"    {p}")
            return False

        try:
            self.get_logger().info(f"[GP] loading skygp from: {skygp_path}")
            spec = importlib.util.spec_from_file_location("skygp", skygp_path)
            skygp_mod = importlib.util.module_from_spec(spec)
            sys.modules["skygp"] = skygp_mod
            spec.loader.exec_module(skygp_mod)
            return True
        except Exception as e:
            self.get_logger().error(f"[GP] failed to import skygp from {skygp_path}: {e}")
            return False

    def _load_gp_models(self, dir_path=None):
        """
        对每个关节 j：
        - 只加载一个 joint{j}.pkl
        - 从里边的 model deepcopy 出两份：
            - slow：带延迟 GP（用未来关节特征）
            - fast：无延迟 GP（用当前关节特征）
        - stats (Xm, Xs, Ym, Ys) 两个模型共享
        """
        if dir_path is None:
            dir_path = self.model_dir

        abs_dir = os.path.abspath(dir_path)
        self.get_logger().info(f"[GP] loading models from: {abs_dir}")

        # 按关节 + 模型类型（slow/fast）定制运行参数
        # 你可以根据需要调这些
        per_joint_cfg_slow = {
            "default": dict(
                max_data_per_expert=500,
                nearest_k=2,
                max_experts=100,
                timescale=0.03,
            ),
            # 6: dict(
            #     max_data_per_expert=50,
            #     nearest_k=2,
            #     max_experts=50,
            #     timescale=0.05,
            # ),
        }

        per_joint_cfg_fast = {
            "default": dict(
                max_data_per_expert=50,   # 小模型记得少一点，更新快
                nearest_k=2,
                max_experts=50,
                timescale=0.02,           # timescale 小一点，forget 更快
            ),
            # 6: dict(
            #     max_data_per_expert=50,
            #     nearest_k=2,
            #     max_experts=30,
            #     timescale=0.03,
            # ),
        }

        loaded = 0
        self.gp_models = {}     # gp_models[j] = {"slow": {...}, "fast": {...}}

        for j in range(1, 7):
            p = os.path.join(dir_path, f"joint{j}.pkl")
            abs_p = os.path.abspath(p)
            if not os.path.isfile(p):
                self.get_logger().warn(f"[GP] model file not found: {abs_p}")
                continue

            try:
                with open(p, "rb") as f:
                    pack = pickle.load(f)
            except Exception as e:
                self.get_logger().error(f"[GP] fail loading {abs_p}: {e}")
                continue

            model_base = pack.get("model", None)
            stats = pack.get("stats", None)  # (Xm, Xs, Ym, Ys)
            if model_base is None or stats is None:
                self.get_logger().warn(f"[GP] bad model pack: {abs_p}")
                continue

            try:
                Xm, Xs, Ym, Ys = stats
                x_dim = int(np.asarray(Xm).shape[0])
            except Exception as e:
                self.get_logger().error(f"[GP] invalid stats in {abs_p}: {e}")
                continue

            self.gp_models[j] = {}

            # === 1) 生成 slow 实例 ===
            cfg_slow = per_joint_cfg_slow.get(j, per_joint_cfg_slow["default"])
            try:
                model_slow = copy.deepcopy(model_base)
                if hasattr(model_slow, "max_data_per_expert"):
                    model_slow.max_data_per_expert = int(cfg_slow["max_data_per_expert"])
                if hasattr(model_slow, "nearest_k"):
                    model_slow.nearest_k = int(cfg_slow["nearest_k"])
                if hasattr(model_slow, "max_experts"):
                    model_slow.max_experts = int(cfg_slow["max_experts"])
                if hasattr(model_slow, "timescale"):
                    model_slow.timescale = float(cfg_slow["timescale"])

                self.gp_models[j]["slow"] = {
                    "model": model_slow,
                    "stats": stats,
                    "x_dim": x_dim,
                }
                self.get_logger().info(
                    f"[GP] joint{j} slow loaded: x_dim={x_dim}, "
                    f"max_data_per_expert={getattr(model_slow, 'max_data_per_expert', 'NA')}, "
                    f"nearest_k={getattr(model_slow, 'nearest_k', 'NA')}, "
                    f"max_experts={getattr(model_slow, 'max_experts', 'NA')}, "
                    f"timescale={getattr(model_slow, 'timescale', 'NA')}"
                )
                loaded += 1
            except Exception as e:
                self.get_logger().error(f"[GP] joint{j} slow init failed: {e}")

            # === 2) 生成 fast 实例 ===
            cfg_fast = per_joint_cfg_fast.get(j, per_joint_cfg_fast["default"])
            try:
                model_fast = copy.deepcopy(model_base)
                if hasattr(model_fast, "max_data_per_expert"):
                    model_fast.max_data_per_expert = int(cfg_fast["max_data_per_expert"])
                if hasattr(model_fast, "nearest_k"):
                    model_fast.nearest_k = int(cfg_fast["nearest_k"])
                if hasattr(model_fast, "max_experts"):
                    model_fast.max_experts = int(cfg_fast["max_experts"])
                if hasattr(model_fast, "timescale"):
                    model_fast.timescale = float(cfg_fast["timescale"])

                self.gp_models[j]["fast"] = {
                    "model": model_fast,
                    "stats": stats,
                    "x_dim": x_dim,
                }
                self.get_logger().info(
                    f"[GP] joint{j} fast loaded: x_dim={x_dim}, "
                    f"max_data_per_expert={getattr(model_fast, 'max_data_per_expert', 'NA')}, "
                    f"nearest_k={getattr(model_fast, 'nearest_k', 'NA')}, "
                    f"max_experts={getattr(model_fast, 'max_experts', 'NA')}, "
                    f"timescale={getattr(model_fast, 'timescale', 'NA')}"
                )
                loaded += 1
            except Exception as e:
                self.get_logger().error(f"[GP] joint{j} fast init failed: {e}")

        self.gp_ready = (loaded > 0)
        self.get_logger().info(f"[GP] total GP instances (slow+fast): {loaded}, ready={self.gp_ready}")

    def _predict_one_model(self, pack, q_j, dq_j, ddq_j, y_real):
        """
        对单个 (某关节的 slow 或 fast) 模型：
          输入：q_j, dq_j, ddq_j, y_real
          输出：y_pred, var_est
        """
        model = pack["model"]
        Xm, Xs, Ym, Ys = pack["stats"]
        x_dim = pack["x_dim"]

        # 1) 构造输入 x
        if x_dim == 3:
            x = np.array([q_j, dq_j, ddq_j], dtype=np.float32)
        elif x_dim == 2:
            x = np.array([q_j, ddq_j], dtype=np.float32)
        else:
            x = np.array([q_j], dtype=np.float32)

        if not np.all(np.isfinite(x)):
            return 0.0, 1.0

        Xm = np.asarray(Xm, dtype=np.float32)
        Xs = np.asarray(Xs, dtype=np.float32)
        Ym = float(np.asarray(Ym)[0])
        Ys_val = float(np.asarray(Ys)[0])
        Ys = Ys_val if Ys_val != 0.0 else 1.0

        x_std = (x - Xm[:x_dim]) / Xs[:x_dim]

        try:
            mu_std_vec, var_std_vec = model.predict(x_std.astype(np.float32))
            mu_std = float(mu_std_vec[0])
            var_std = float(var_std_vec[0])
        except Exception as e:
            self.get_logger().error(f"[GP] predict failed: {e}")
            return 0.0, 1.0

        if not np.isfinite(mu_std) or not np.isfinite(var_std) or var_std <= 0.0:
            return 0.0, 1.0

        # 反标准化
        y_pred = mu_std * Ys + Ym
        var_est = var_std * (Ys ** 2)

        # 在线更新（只在中心点）
        if np.isfinite(y_real):
            y_std = (y_real - Ym) / Ys
            if np.isfinite(y_std):
                try:
                    model.add_point(x_std.astype(np.float32),
                                    np.array([y_std], dtype=np.float32))
                except Exception as e:
                    self.get_logger().error(f"[GP] add_point failed: {e}")

        return y_pred, var_est

    def _gp_predict_and_update(self, q, dq_des_joint, ddq_des_joint, tau_residual):
        """
        对每个关节：
          - 只使用 slow GP 模型 (未来特征版本结构相同，只是这里用的是当前 dq/ddq)
          - 构造输入 x (维度 x_dim)
          - 在中心点 x_std 上做一次预测
          - 用真实残差 tau_residual[j-1] 在中心点 x_std 做在线更新
        """
        if not self.gp_ready or not self.use_gp:
            return np.zeros(7, dtype=float)

        y_hat = np.zeros(7, dtype=float)

        for j in range(1, 7):
            # 只拿 slow 模型
            packs = self.gp_models.get(j, {})
            pack = packs.get("slow", None)
            if pack is None:
                self.get_logger().warn(f"[GP] joint {j}: no slow model, use 0")
                continue

            model = pack["model"]
            Xm, Xs, Ym, Ys = pack["stats"]
            x_dim = pack["x_dim"]

            # 1) 构造输入 x（当前关节状态）
            if x_dim == 3:
                x = np.array([q[j - 1], dq_des_joint[j - 1], ddq_des_joint[j - 1]], dtype=np.float32)
            elif x_dim == 2:
                x = np.array([q[j - 1], ddq_des_joint[j - 1]], dtype=np.float32)
            else:
                x = np.array([q[j - 1]], dtype=np.float32)

            if not np.all(np.isfinite(x)):
                self.get_logger().warn(f"[GP] joint {j}: invalid x={x}")
                continue

            Xm = np.asarray(Xm, dtype=np.float32)
            Xs = np.asarray(Xs, dtype=np.float32)
            Ym = float(np.asarray(Ym)[0])
            Ys_val = float(np.asarray(Ys)[0])
            Ys = Ys_val if Ys_val != 0.0 else 1.0

            # 2) 标准化
            x_std = (x - Xm[:x_dim]) / Xs[:x_dim]

            # 3) 预测
            try:
                mu_std_vec, var_std_vec = model.predict(x_std.astype(np.float32))
                mu_std = float(mu_std_vec[0])
                var_std = float(var_std_vec[0])
            except Exception as e:
                self.get_logger().error(f"[GP] joint {j}: predict failed: {e}")
                continue

            if (not np.isfinite(mu_std)) or (not np.isfinite(var_std)) or var_std <= 0.0:
                self.get_logger().warn(f"[GP] joint {j}: invalid predict (mu={mu_std}, var={var_std}), use 0")
                continue

            # 4) 反标准化
            y_pred = mu_std * Ys + Ym
            y_hat[j - 1] = y_pred

            # 5) 在线更新
            y_real = float(tau_residual[j - 1])
            if not np.isfinite(y_real):
                continue

            y_std = (y_real - Ym) / Ys
            if np.isfinite(y_std):
                try:
                    model.add_point(x_std.astype(np.float32),
                                    np.array([y_std], dtype=np.float32))
                except Exception as e:
                    self.get_logger().error(f"[GP] joint {j}: add_point failed: {e}")

        return y_hat


    def _gp_predict_and_update_twofeatures(
        self,
        q, dq_now, ddq_now,
        dq_future, ddq_future,
        tau_residual
    ):
        """
        对每个关节：
          - slow 模型用未来特征 (dq_future, ddq_future)
          - fast 模型用当前特征 (dq_now, ddq_now)
          - 对 slow / fast 的预测按方差加权融合
          - 在线更新分别在各自模型上做
          - 同时把 slow / fast / 融合 / 真实值 存一行日志
        """
        if not self.gp_ready or not self.use_gp:
            return np.zeros(7, dtype=float)

        y_hat = np.zeros(7, dtype=float)
        y_slow_all = np.zeros(7, dtype=float)
        y_fast_all = np.zeros(7, dtype=float)

        for j in range(1, 7):
            packs = self.gp_models.get(j, {})
            pack_slow = packs.get("slow", None)
            pack_fast = packs.get("fast", None)

            if pack_slow is None and pack_fast is None:
                continue

            q_j   = float(q[j-1])
            y_real = float(tau_residual[j-1])

            preds = []
            vars_ = []

            # ---- slow: 未来特征 ----
            if pack_slow is not None:
                y_slow, v_slow = self._predict_one_model(
                    pack_slow,
                    q_j,
                    float(dq_future[j-1]),
                    float(ddq_future[j-1]),
                    y_real
                )
                y_slow_all[j-1] = y_slow
                preds.append(y_slow)
                vars_.append(max(v_slow, 1e-6))

            # ---- fast: 当前特征 ----
            if pack_fast is not None:
                y_fast, v_fast = self._predict_one_model(
                    pack_fast,
                    q_j,
                    float(dq_now[j-1]),
                    float(ddq_now[j-1]),
                    y_real
                )
                y_fast_all[j-1] = y_fast
                preds.append(y_fast)
                vars_.append(max(v_fast, 1e-6))

            preds = np.array(preds, dtype=float)
            vars_ = np.array(vars_, dtype=float)

            if preds.size == 1:
                y_hat_j = preds[0]
            else:
                inv_vars = 1.0 / vars_
                w_sum = np.sum(inv_vars)
                if w_sum <= 0.0 or not np.isfinite(w_sum):
                    y_hat_j = np.mean(preds)
                else:
                    y_hat_j = float(np.sum(preds * inv_vars) / w_sum)

            y_hat[j-1] = y_hat_j

        # 数值保护 + 限幅
        y_hat = np.nan_to_num(y_hat, nan=0.0, posinf=0.0, neginf=0.0)
        y_hat = np.clip(y_hat, -10.0, 10.0)  # 每个关节 ±10Nm，可自己调

        # ====== 日志记录，一次 service 调用一行 ======
        if self.enable_logging:
            t_now = self.get_clock().now().nanoseconds / 1e9
            self.log_time.append(t_now)
            self.log_y_slow.append(y_slow_all.tolist())
            self.log_y_fast.append(y_fast_all.tolist())
            self.log_y_hat.append(y_hat.tolist())
            self.log_tau_residual.append(tau_residual.astype(float).tolist())

        return y_hat



    def _gp_response_callback(self, future):
        """GP service 异步响应回调：更新 self._latest_y_hat"""
        try:
            resp = future.result()
            if resp is None:
                return

            y_hat = np.array(resp.y_hat, dtype=float)
            if y_hat.shape != (7,):
                self.get_logger().warn(f"[Controller] GP returned wrong size y_hat: {y_hat.shape}")
                return

            with self._gp_lock:
                self._latest_y_hat = y_hat
            # 如果你想看一下效果，可以偶尔打印：
            # self.get_logger().debug(f"[Controller] updated y_hat = {y_hat}")
        except Exception as e:
            # 这里不需要太吓人，只是调试信息
            self.get_logger().warn(f"[Controller] GP response error: {e}")

    def save_logs_to_csv(self, filename="gp_debug_log.csv"):
        """
        把 slow / fast / y_hat / tau_residual 全部写到一个 CSV 里：
        每一行对应一次 service 调用：
          [time,
           slow_1..7,
           fast_1..7,
           y_hat_1..7,
           tau_residual_1..7]
        """
        if not self.log_time:
            self.get_logger().warn("[GP] no log data to save")
            return

        try:
            n = len(self.log_time)
            # 防御：取各 list 的最小长度
            n = min(
                n,
                len(self.log_y_slow),
                len(self.log_y_fast),
                len(self.log_y_hat),
                len(self.log_tau_residual),
            )

            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                header = ["time"]
                header += [f"y_slow_{i+1}" for i in range(7)]
                header += [f"y_fast_{i+1}" for i in range(7)]
                header += [f"y_hat_{i+1}" for i in range(7)]
                header += [f"tau_residual_{i+1}" for i in range(7)]
                writer.writerow(header)

                for k in range(n):
                    row = [self.log_time[k]]
                    row += self.log_y_slow[k]
                    row += self.log_y_fast[k]
                    row += self.log_y_hat[k]
                    row += self.log_tau_residual[k]
                    writer.writerow(row)

            self.get_logger().info(f"[GP] logs saved to {filename}, {n} rows")
        except Exception as e:
            self.get_logger().error(f"[GP] failed to save logs: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = GPServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("[GPServer] keyboard interrupt")
    finally:
        # 退出前保存日志
        try:
            node.save_logs_to_csv("gp_debug_log.csv")
        except Exception as e:
            # 这里 context 可能已经被 shutdown 了，所以不要再用 logger 打太多
            print(f"[GPServer] save_logs_to_csv error: {e}")

        # 安全销毁节点 + shutdown（避免重复 shutdown）
        try:
            node.destroy_node()
        except Exception as e:
            print(f"[GPServer] destroy_node error: {e}")

        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception as e:
            # 如果已经 shutdown 过了，就忽略这个错误
            print(f"[GPServer] rclpy.shutdown error: {e}")

if __name__ == '__main__':
    main()