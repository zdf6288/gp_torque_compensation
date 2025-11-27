#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import os, sys, pickle, importlib.util
from custom_msgs.srv import AsyncGPpredict   # 刚才定义的 srv

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

    def gp_predict_callback(self, request, response):
        try:
            q = np.array(request.q, dtype=np.float32)
            dq_des_joint = np.array(request.dq_des_joint, dtype=np.float32)
            ddq_des_joint = np.array(request.ddq_des_joint, dtype=np.float32)
            tau_residual = np.array(request.tau_residual, dtype=np.float32)

            dq_des_joint_future = np.array(request.dq_des_joint_future, dtype=np.float32)
            ddq_des_joint_future = np.array(request.ddq_des_joint_future, dtype=np.float32)

            # 用「未来特征」做预测（当前特征你可以只用于分析或不用）
            y_hat = self._gp_predict_and_update(
                q,
                dq_des_joint_future,
                ddq_des_joint_future,
                tau_residual
            )
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
                max_data_per_expert=500,
                nearest_k=2,
                max_experts=100,
                timescale=0.03,
            ),
            # 举例：如果你想让 6 号关节忘得快一点、专家少一点，可以单独改：
            6: dict(
                max_data_per_expert=50,
                nearest_k=2,
                max_experts=50,
                timescale=0.05,
            ),
        }

        loaded = 0
        self.gp_models = {}

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

    def _gp_predict_and_update(self, q, dq_des_joint, ddq_des_joint, tau_residual):
        """
        对每个关节：
          - 构造输入 x (维度 x_dim)
          - 在 x_std 附近采样 n 个扰动点 x_std_k
          - 对每个 x_std_k 调用 model.predict
          - 用方差加权平均融合成一个预测 y_hat[j-1]
          - 用真实残差 tau_residual[j-1] 在中心点 x_std 做在线更新
        """
        if not self.gp_ready or not self.use_gp:
            return np.zeros(7, dtype=float)

        y_hat = np.zeros(7, dtype=float)

        # 多点采样配置
        n_samples = max(self.n_samples, 0)
        radius = float(self.sample_radius)

        for j in range(1, 7):
            pack = self.gp_models.get(j)
            if pack is None:
                continue
            model = pack["model"]
            Xm, Xs, Ym, Ys = pack["stats"]
            x_dim = pack["x_dim"]

            # ===== 1) 构造输入 x（当前 "state"） =====
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

            # 标准化中心点
            x_std_center = (x - Xm[:x_dim]) / Xs[:x_dim]

            # ===== 2) 构造采样点（标准化空间） =====
            # 第一个就是中心点本身
            samples = [x_std_center]

            # if n_samples > 0 and radius > 0.0:
            #     # 在单位高斯中采样，然后缩放到给定半径
            #     for _ in range(n_samples):
            #         # 标准正态扰动
            #         delta = np.random.normal(loc=0.0, scale=1.0, size=x_dim).astype(np.float32)
            #         # 归一化到单位球，再乘半径，避免太大的跳动
            #         norm = np.linalg.norm(delta)
            #         if norm < 1e-6:
            #             continue
            #         delta = delta / norm * radius
            #         samples.append(x_std_center + delta)

            mus_std = []
            vars_std = []

            # ===== 3) 对所有采样点调用 GP 预测 =====
            for x_std in samples:
                try:
                    mu_std_vec, var_std_vec = model.predict(x_std.astype(np.float32))
                    # 预测返回的是向量，这里只对单输出 y_dim=1 情况取 [0]
                    mu_std = float(mu_std_vec[0])
                    var_std = float(var_std_vec[0])
                except Exception as e:
                    self.get_logger().error(f"[GP] joint {j}: predict failed at sample: {e}")
                    continue

                if not np.isfinite(mu_std) or not np.isfinite(var_std) or var_std <= 0.0:
                    # 非法结果直接丢掉
                    continue

                mus_std.append(mu_std)
                vars_std.append(var_std)

            if len(mus_std) == 0:
                # 所有采样点都失败，就算了，给 0 输出
                self.get_logger().warn(f"[GP] joint {j}: all samples invalid, use 0")
                y_hat[j - 1] = 0.0
                continue

            mus_std = np.array(mus_std, dtype=float)
            vars_std = np.array(vars_std, dtype=float)

            # ===== 4) 方差加权平均（precision weighting）融合多个样本 =====
            inv_vars = 1.0 / vars_std
            weight_sum = np.sum(inv_vars)
            if weight_sum <= 0.0 or not np.isfinite(weight_sum):
                mu_std_comb = np.mean(mus_std)
            else:
                mu_std_comb = float(np.sum(mus_std * inv_vars) / weight_sum)

            # 如果你想要一个总方差也可以：
            # var_std_comb = 1.0 / weight_sum   （这里暂时没用到）

            # 把标准化预测还原到物理量空间
            y_pred = mu_std_comb * Ys + Ym

            # 这里如果你还想用 sigma 做置信度加权，可以再算一次：
            # sigma_std_comb = np.sqrt(var_std_comb)
            # sigma_comb = sigma_std_comb * Ys
            # alpha = 0.0  # 你之前的权重参数
            # w = 1.0 / (1.0 + alpha * sigma_comb**2)
            # y_pred_weighted = y_pred * w
            # 目前为了简单就直接用 y_pred：
            y_hat[j - 1] = y_pred

            # ===== 5) 在线更新：用真实残差在中心点更新 =====
            y_real = float(tau_residual[j - 1])
            if not np.isfinite(y_real):
                continue

            y_std = (y_real - Ym) / Ys
            if np.isfinite(y_std):
                try:
                    # 只在中心点 x_std_center 更新，保持“附近平均预测”的光滑性
                    model.add_point(x_std_center.astype(np.float32),
                                    np.array([y_std], dtype=np.float32))
                except Exception as e:
                    self.get_logger().error(f"[GP] joint {j}: add_point failed: {e}")

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


def main(args=None):
    rclpy.init(args=args)
    node = GPServer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()