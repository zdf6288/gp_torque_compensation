#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import os, sys, pickle, importlib.util, copy
from collections import deque
from custom_msgs.srv import AsyncGPpredict
import time

class GPServer(Node):
    """
    Two-GP comparison server:

      GP_now    : x(t)        -> tau(t)
      GP_delay  : x(t-delta)  -> tau(t)

    Both models are updated online.
    """

    def __init__(self):
        super().__init__('gp_server')
        self.get_logger().info('[GPServer] starting (dual-GP mode)')

        # ================= parameters =================
        self.declare_parameter('model_dir', './new_structure/gp/gp_models')
        self.declare_parameter('delay_steps', 10)

        self.model_dir   = self.get_parameter('model_dir').value
        self.delay_steps = int(self.get_parameter('delay_steps').value)

        # ================= import skygp =================
        self._ensure_skygp_import()

        # ================= load base GP =================
        base_models = self._load_gp_models(self.model_dir)

        # ================= duplicate GP =================
        self.gp_models_now   = copy.deepcopy(base_models)
        self.gp_models_delay = copy.deepcopy(base_models)

        self.gp_ready = len(self.gp_models_now) > 0

        # ================= delay buffer =================
        self.x_buffer = deque(maxlen=self.delay_steps + 1)

        # ================= ROS service =================
        self.srv = self.create_service(
            AsyncGPpredict,
            '/gp_predict',
            self.gp_predict_callback
        )

        self.get_logger().info(
            f'[GPServer] ready | delay_steps={self.delay_steps}'
        )

    # ============================================================
    # Service callback
    # ============================================================
    def gp_predict_callback(self, request, response):
        try:
            q   = np.array(request.q, dtype=np.float32)
            dq  = np.array(request.dq_des_joint, dtype=np.float32)
            ddq = np.array(request.ddq_des_joint, dtype=np.float32)
            tau = np.array(request.tau_residual, dtype=np.float32)

            x_now = np.concatenate([q, dq, ddq]).astype(np.float32)
            self.x_buffer.append(x_now)

            if len(self.x_buffer) < self.delay_steps:
                response.y_local = [0.0]*7
                response.y_cloud = [0.0]*7
                return response

            x_delay = self.x_buffer[0]

            # ===== 1) PREDICT =====
            y_delay = self._gp_predict(x_now, self.gp_models_delay)
            y_now   = self._gp_predict(x_now, self.gp_models_now)

            # ===== 2) UPDATE =====
            self._safe_gp_update(x_delay, tau, self.gp_models_delay)
            self._gp_update(x_now, tau, self.gp_models_now)

            # ===== 3) return =====
            response.y_local = y_delay.tolist()   # baseline
            response.y_cloud = y_now.tolist()     # delay-compensated

        except Exception as e:
            self.get_logger().error(f'[GPServer] callback error: {e}')
            response.y_local = [0.0]*7
            response.y_cloud = [0.0]*7

        return response


    def _safe_gp_update(self, x_full, tau, gp_models):
        # 1) 数值合法性
        if not np.all(np.isfinite(x_full)):
            return

        # 2) 用 joint1 的 stats 判断是否 OOD
        Xm, Xs, _, _ = gp_models[1]['stats']
        x_dim = len(Xm)

        x_std = (x_full[:x_dim] - Xm[:x_dim]) / Xs[:x_dim]

        # 3) 标准化空间阈值（非常关键）
        if np.linalg.norm(x_std) > 6.0:
            print("tiaoguo")
            return   # ❌ 不更新，直接跳过

        # 4) 真正 update
        self._gp_update(x_full, tau, gp_models)


    # ============================================================
    # Core GP logic
    # ============================================================
    def _gp_predict(self, x_full, gp_models):
        y_hat = np.zeros(7, dtype=np.float32)

        for j in range(1, 8):
            pack = gp_models.get(j)
            if pack is None:
                continue

            model = pack['model']
            Xm, Xs, Ym, Ys = pack['stats']
            x_dim = len(Xm)

            Xm = np.asarray(Xm, dtype=np.float32)
            Xs = np.asarray(Xs, dtype=np.float32)
            Ym = float(Ym[0])
            Ys = float(Ys[0]) if float(Ys[0]) != 0.0 else 1.0

            x_std = (x_full[:x_dim] - Xm[:x_dim]) / Xs[:x_dim]

            mu_std, _ = model.predict(x_std.astype(np.float32))
            mu_std = float(mu_std[0])

            y_hat[j - 1] = mu_std * Ys + Ym

        return y_hat

    def _gp_update(self, x_full, tau_residual, gp_models):
        for j in range(1, 8):
            pack = gp_models.get(j)
            if pack is None:
                continue

            model = pack['model']
            Xm, Xs, Ym, Ys = pack['stats']
            x_dim = len(Xm)

            Xm = np.asarray(Xm, dtype=np.float32)
            Xs = np.asarray(Xs, dtype=np.float32)
            Ym = float(Ym[0])
            Ys = float(Ys[0]) if float(Ys[0]) != 0.0 else 1.0

            x_std = (x_full[:x_dim] - Xm[:x_dim]) / Xs[:x_dim]

            y_real = float(tau_residual[j - 1])
            y_std  = (y_real - Ym) / Ys

            if np.isfinite(y_std):
                model.add_point(
                    x_std.astype(np.float32),
                    np.array([y_std], dtype=np.float32)
                )

    # ============================================================
    # Load GP models
    # ============================================================
    def _load_gp_models(self, dir_path):
        self.get_logger().info(f'[GP] loading models from {dir_path}')
        gp_models = {}

        for j in range(1, 8):
            pkl = os.path.join(dir_path, f'joint{j}_cloud.pkl')
            if not os.path.isfile(pkl):
                continue

            with open(pkl, 'rb') as f:
                pack = pickle.load(f)

            gp_models[j] = {
                'model': pack['model'],
                'stats': pack['stats']
            }

            self.get_logger().info(f'[GP] joint{j} loaded')

        return gp_models

    # ============================================================
    # Ensure skygp import
    # ============================================================
    def _ensure_skygp_import(self):
        if 'skygp' in sys.modules:
            return

        path = os.path.join(
            os.getcwd(), 'new_structure', 'gp', 'skygp.py'
        )
        spec = importlib.util.spec_from_file_location('skygp', path)
        mod  = importlib.util.module_from_spec(spec)
        sys.modules['skygp'] = mod
        spec.loader.exec_module(mod)


def main(args=None):
    rclpy.init(args=args)
    node = GPServer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
