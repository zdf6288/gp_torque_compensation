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

        # 先确保 skygp 导入
        self._ensure_skygp_import()
        # 加载离线训练的模型
        self._load_gp_models(self.model_dir)

        # 创建 service
        self.srv = self.create_service(AsyncGPpredict, '/gp_predict', self.gp_predict_callback)
        self.get_logger().info("[GPServer] service /gp_predict ready")

    def gp_predict_callback(self, request, response):
        """
        每次 controller 调用时：
        输入：q, dq_des_joint, ddq_des_joint, tau_residual
        输出：y_hat[7]
        """
        try:
            q = np.array(request.q, dtype=np.float32)
            dq_des_joint = np.array(request.dq_des_joint, dtype=np.float32)
            ddq_des_joint = np.array(request.ddq_des_joint, dtype=np.float32)
            tau_residual = np.array(request.tau_residual, dtype=np.float32)

            y_hat = self._gp_predict_and_update(q, dq_des_joint, ddq_des_joint, tau_residual)
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
        # 基本照搬你现有的版本，只是去掉 self.q 等共享变量，直接用函数参数
        if not self.gp_ready or not self.use_gp:
            return np.zeros(7, dtype=float)

        y_hat = np.zeros(7, dtype=float)
        for j in range(1, 7):
            pack = self.gp_models.get(j)
            if pack is None:
                continue
            model = pack["model"]
            Xm, Xs, Ym, Ys = pack["stats"]
            x_dim = pack["x_dim"]

            # 构造输入
            if x_dim == 3:
                x = np.array([q[j-1], dq_des_joint[j-1], ddq_des_joint[j-1]], dtype=np.float32)
            elif x_dim == 2:
                x = np.array([q[j-1], ddq_des_joint[j-1]], dtype=np.float32)
            else:
                x = np.array([q[j-1]], dtype=np.float32)

            if not np.all(np.isfinite(x)):
                self.get_logger().warn(f"[GP] joint {j}: invalid x={x}")
                continue

            Xm = np.asarray(Xm, dtype=np.float32)
            Xs = np.asarray(Xs, dtype=np.float32)
            Ym = float(np.asarray(Ym)[0])
            Ys = float(np.asarray(Ys)[0]) if float(np.asarray(Ys)[0]) != 0.0 else 1.0

            x_std = (x - Xm[:x_dim]) / Xs[:x_dim]

            try:
                mu_std, var_std = model.predict(x_std)
                mu_std = float(mu_std[0])
                sigma_std = float(np.sqrt(var_std[0])) if np.all(np.isfinite(var_std)) else 0.0
            except Exception as e:
                self.get_logger().error(f"[GP] joint {j}: predict failed: {e}")
                continue

            if not np.isfinite(mu_std):
                self.get_logger().warn(f"[GP] joint {j}: non-finite mu_std={mu_std}")
                mu_std = 0.0

            y_pred = mu_std * Ys + Ym
            sigma = sigma_std * Ys

            alpha = 0
            w = 1.0 / (1.0 + alpha * sigma**2)
            y_pred_weighted = y_pred * w
            y_hat[j-1] = y_pred_weighted

            # 在线更新
            y_real = float(tau_residual[j-1])
            if not np.isfinite(y_real):
                continue
            y_std = (y_real - Ym) / Ys
            if np.isfinite(y_std):
                try:
                    model.add_point(x_std, np.array([y_std], dtype=np.float32))
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
