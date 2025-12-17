#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import os, sys, pickle, importlib.util
from custom_msgs.srv import AsyncGPpredict   # 刚才定义的 srv
from collections import deque

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

        # ===== Memory buffer =====
        self.mem_X = []      # list of (x_dim,)
        self.mem_Y = []      # list of (7,)
        self.mem_max = 2000  # 最大容量（你可以调）

    # ============================================================
    # 主 Service 回调
    # ============================================================
    def gp_predict_callback(self, request, response):

        try:
            # -----------------------
            # 当前状态
            # -----------------------
            q   = np.array(request.q, dtype=np.float32)
            dq  = np.array(request.dq_des_joint, dtype=np.float32)
            ddq = np.array(request.ddq_des_joint, dtype=np.float32)
            tau = np.array(request.tau_residual, dtype=np.float32)

            # -----------------------
            # 未来主状态
            # -----------------------
            dq_f  = np.array(request.dq_des_joint_future, dtype=np.float32)
            ddq_f = np.array(request.ddq_des_joint_future, dtype=np.float32)

            # -----------------------
            # 多采样 future
            # -----------------------
            N = request.n_future_samples
            if N > 0:
                dq_samples  = np.array(request.dq_future_samples_flat, dtype=np.float32).reshape(N,7)
                ddq_samples = np.array(request.ddq_future_samples_flat, dtype=np.float32).reshape(N,7)
            else:
                dq_samples = np.zeros((0,7), dtype=np.float32)
                ddq_samples = np.zeros((0,7), dtype=np.float32)

            # =====================================================
            # 1) 当前 cloud GP + memory
            # =====================================================
            y_gp_cur, y_mem_cur, mem_dist_cur = \
                self._gp_predict_vector(q, dq, ddq, tau, update=True)

            # =====================================================
            # 2) 未来 cloud GP（主预测）
            # =====================================================
            y_gp_fut, _, _ = \
                self._gp_predict_vector(q, dq_f, ddq_f, None, update=False)

            # =====================================================
            # 3) 多采样 future cloud GP
            # =====================================================
            if N > 0:
                y_future_samples = []
                for k in range(N):
                    yk_gp, _, _ = self._gp_predict_vector(
                        q, dq_samples[k], ddq_samples[k],
                        None, update=False
                    )
                    y_future_samples.append(yk_gp)

                y_gp_future_agg = np.mean(np.array(y_future_samples), axis=0)
            else:
                y_gp_future_agg = np.zeros(7, dtype=np.float32)

            # =====================================================
            # 返回（不融合）
            # =====================================================
            response.y_local = y_gp_cur.tolist()
            response.y_cloud = y_gp_fut.tolist()
            response.y_mem   = y_mem_cur.tolist()
            response.mem_dist = float(mem_dist_cur)

        except Exception as e:
            self.get_logger().error(f"[GPServer] callback error: {e}")
            response.y_local = [0.0]*7
            response.y_cloud = [0.0]*7

        return response

    def _gp_predict_vector(self, q, dq, ddq, tau_residual=None, update=False):
        """
        返回：
            y_gp   : (7,) GP 预测
            y_mem  : (7,) memory 预测（若无则 0）
            mem_dist : float，与 memory 最近点的距离
        """

        if not self.gp_ready:
            return (
                np.zeros(7, dtype=float),
                np.zeros(7, dtype=float),
                np.inf
            )

        # ================================
        # 1) 构造统一输入 x_full
        # ================================
        x_full = np.concatenate([q, dq, ddq]).astype(np.float32)

        # ================================
        # 2) 用 joint1 的 stats 做标准化（统一）
        # ================================
        ref_pack = self.gp_models.get(1)
        Xm, Xs, _, _ = ref_pack["stats"]
        Xm = np.asarray(Xm, dtype=np.float32)
        Xs = np.asarray(Xs, dtype=np.float32)
        x_dim = ref_pack["x_dim"]

        x_std = (x_full[:x_dim] - Xm[:x_dim]) / Xs[:x_dim]

        # ================================
        # 3) memory 查询（⭐ 新增）
        # ================================
        y_mem, mem_dist = self._query_memory(x_std)
        if y_mem is None:
            y_mem = np.zeros(7, dtype=np.float32)
            mem_dist = np.inf

        # ================================
        # 4) GP 预测（原逻辑，几乎不动）
        # ================================
        y_gp = np.zeros(7, dtype=float)

        for j in range(1, 8):
            pack = self.gp_models.get(j)
            if pack is None:
                continue

            model = pack["model"]
            Xm, Xs, Ym, Ys = pack["stats"]
            x_dim = pack["x_dim"]

            Xm = np.asarray(Xm, dtype=np.float32)
            Xs = np.asarray(Xs, dtype=np.float32)
            Ym = float(np.asarray(Ym)[0])
            Ys = float(np.asarray(Ys)[0]) or 1.0

            x_std_j = (x_full[:x_dim] - Xm[:x_dim]) / Xs[:x_dim]

            mu_s, _ = model.predict(x_std_j.astype(np.float32))
            mu = float(mu_s[0])

            y_gp[j - 1] = mu * Ys + Ym

            # -------------------------------
            # online update（只对当前点）
            # -------------------------------
            if update and tau_residual is not None:
                y_real = float(tau_residual[j - 1])
                y_std = (y_real - Ym) / Ys

                if np.isfinite(y_std):
                    try:
                        model.add_point(
                            x_std_j.astype(np.float32),
                            np.array([y_std], dtype=np.float32)
                        )
                    except Exception as e:
                        self.get_logger().error(
                            f"[GPServer] joint{j} online update failed: {e}"
                        )

        # ================================
        # 5) 返回三个值
        # ================================
        self._store_memory(x_std,y_gp)
        return y_gp, y_mem, mem_dist


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
        从 dir_path 下加载 joint1.pkl ... joint6.pkl，
        并在加载后按关节覆盖 SkyGP_rBCM 的一些运行参数。
        """
        if dir_path is None:
            dir_path = self.model_dir

        abs_dir = os.path.abspath(dir_path)
        self.get_logger().info(f"[GP] loading models from: {abs_dir}")

        # ===== 按关节定制 GP 参数（你可以自己改这些值） =====
        # key = 关节号（1..6），"default" 为所有关节的默认配置
        per_joint_cfg = {
            "default": dict(
                max_data_per_expert=100,
                nearest_k=2,
                max_experts=8,
                timescale=0.03,
            ),
            # 举例：如果你想让 6 号关节忘得快一点、专家少一点，可以单独改：
            6: dict(
                max_data_per_expert=50,
                nearest_k=2,
                max_experts=8,
                timescale=0.05,
            ),
        }

        loaded = 0
        self.gp_models = {}

        for j in range(1, 7):
            p = os.path.join(dir_path, f"joint{j}_cloud.pkl")
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

            model = pack.get("model", None)
            stats = pack.get("stats", None)  # (Xm, Xs, Ym, Ys)
            if model is None or stats is None:
                self.get_logger().warn(f"[GP] bad model pack: {abs_p}")
                continue

            try:
                Xm, Xs, Ym, Ys = stats
                x_dim = int(np.asarray(Xm).shape[0])
            except Exception as e:
                self.get_logger().error(f"[GP] invalid stats in {abs_p}: {e}")
                continue

            # ===== 在这里覆盖 SkyGP_rBCM 的参数 =====
            cfg = per_joint_cfg.get(j, per_joint_cfg["default"])
            try:
                # 只有在模型里确实有这些属性时才改，避免旧版本崩溃
                if hasattr(model, "max_data_per_expert"):
                    model.max_data_per_expert = int(cfg["max_data_per_expert"])
                if hasattr(model, "nearest_k"):
                    model.nearest_k = int(cfg["nearest_k"])
                if hasattr(model, "max_experts"):
                    model.max_experts = int(cfg["max_experts"])
                if hasattr(model, "timescale"):
                    model.timescale = float(cfg["timescale"])

                self.get_logger().info(
                    f"[GP] joint{j} loaded: x_dim={x_dim}, "
                    f"max_data_per_expert={getattr(model, 'max_data_per_expert', 'NA')}, "
                    f"nearest_k={getattr(model, 'nearest_k', 'NA')}, "
                    f"max_experts={getattr(model, 'max_experts', 'NA')}, "
                    f"timescale={getattr(model, 'timescale', 'NA')}"
                )
            except Exception as e:
                self.get_logger().warn(
                    f"[GP] joint{j}: override model params failed: {e}"
                )

            # 存到字典里备用
            self.gp_models[j] = {
                "model": model,
                "stats": stats,
                "x_dim": x_dim,
            }
            loaded += 1

        self.gp_ready = (loaded > 0)
        self.get_logger().info(f"[GP] total loaded: {loaded}, ready={self.gp_ready}")

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

    # ============================================================
    # Memory: 查询最近邻
    # ============================================================
    def _query_memory(self, x_std):
        """
        返回：
            y_mem : (7,) or None
            dist  : float
        """
        if len(self.mem_X) == 0:
            return None, np.inf

        X = np.array(self.mem_X)   # (N, x_dim)
        Y = np.array(self.mem_Y)   # (N, 7)

        # 欧氏距离（标准化空间）
        dists = np.linalg.norm(X - x_std[None, :], axis=1)
        idx = np.argmin(dists)

        return Y[idx], float(dists[idx])



    # ============================================================
    # Memory: 写入
    # ============================================================
    def _store_memory(self, x_std, y_hat):
        """
        x_std : (x_dim,)
        y_hat : (7,)
        """
        self.mem_X.append(x_std.copy())
        self.mem_Y.append(y_hat.copy())

        # FIFO 限制容量
        if len(self.mem_X) > self.mem_max:
            self.mem_X.pop(0)
            self.mem_Y.pop(0)


def main(args=None):
    rclpy.init(args=args)
    node = GPServer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()