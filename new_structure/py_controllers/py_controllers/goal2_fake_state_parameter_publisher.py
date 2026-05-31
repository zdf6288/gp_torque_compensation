#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node

from custom_msgs.msg import StateParameter


class Goal2FakeStateParameterPublisher(Node):
    """Publish synthetic StateParameter messages for GOAL2 fake/sim smoke only."""

    def __init__(self):
        super().__init__('goal2_fake_state_parameter_publisher')

        self.declare_parameter('use_fake_hardware', False)
        self.declare_parameter('publish_rate_hz', 50.0)
        self.declare_parameter('arm_id', 'panda')

        self.use_fake_hardware = bool(self.get_parameter('use_fake_hardware').value)
        self.arm_id = str(self.get_parameter('arm_id').value)
        self.publish_rate_hz = self._sanitize_publish_rate(
            self.get_parameter('publish_rate_hz').value
        )

        self.get_logger().warn(
            'GOAL2 fake/sim only /state_parameter publisher starting. '
            'This node must never be used for real robot operation.'
        )
        self.get_logger().info(
            f'use_fake_hardware={self.use_fake_hardware}, '
            f'publish_rate_hz={self.publish_rate_hz:.3f}, arm_id={self.arm_id}'
        )

        # GOAL2 fake/sim smoke 的 synthetic input：只用于触发 callback timing / CSV save path。
        # 真实机器人路径仍必须由 cpp_relayer 发布真实 Franka state_parameter。
        # 这些 dummy values 不代表真实 Franka dynamics。
        self._template_msg = self._build_template_message()

        if not self.use_fake_hardware:
            # 双重防误用边界：launch 要显式开启，本 node 内部也要求 use_fake_hardware=true。
            self.get_logger().error(
                'use_fake_hardware is false; fake /state_parameter publishing is disabled.'
            )
            return

        self.publisher = self.create_publisher(StateParameter, '/state_parameter', 10)
        self.timer = self.create_timer(1.0 / self.publish_rate_hz, self._publish_once)

    def _sanitize_publish_rate(self, raw_value):
        try:
            publish_rate_hz = float(raw_value)
        except (TypeError, ValueError):
            self.get_logger().warn(
                f'Invalid publish_rate_hz={raw_value!r}; falling back to 50.0 Hz.'
            )
            return 50.0

        if not 1.0 <= publish_rate_hz <= 1000.0:
            self.get_logger().warn(
                f'publish_rate_hz={publish_rate_hz:.3f} outside [1.0, 1000.0]; '
                'falling back to 50.0 Hz.'
            )
            return 50.0

        return publish_rate_hz

    def _build_template_message(self):
        msg = StateParameter()
        msg.header.frame_id = 'goal2_fake'
        msg.position = [0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.0]
        msg.velocity = [0.0] * 7
        msg.effort_measured = [0.0] * 7
        msg.gravity = [0.0] * 7
        msg.coriolis = [0.0] * 7

        o_t_f = np.eye(4, dtype=float)
        o_t_f[0, 3] = 0.35
        o_t_f[1, 3] = 0.0
        o_t_f[2, 3] = 0.65
        msg.o_t_f = o_t_f.flatten(order='F').tolist()

        mass = np.eye(7, dtype=float)
        msg.mass = mass.flatten(order='F').tolist()

        zero_jacobian = np.zeros((6, 7), dtype=float)
        zero_jacobian[0, 0] = 1.0
        zero_jacobian[1, 1] = 1.0
        zero_jacobian[2, 2] = 1.0
        zero_jacobian[3, 3] = 0.1
        zero_jacobian[4, 4] = 0.1
        zero_jacobian[5, 5] = 0.1
        msg.zero_jacobian_flange = zero_jacobian.flatten(order='F').tolist()

        return msg

    def _publish_once(self):
        msg = StateParameter()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._template_msg.header.frame_id
        msg.position = list(self._template_msg.position)
        msg.velocity = list(self._template_msg.velocity)
        msg.effort_measured = list(self._template_msg.effort_measured)
        msg.gravity = list(self._template_msg.gravity)
        msg.o_t_f = list(self._template_msg.o_t_f)
        msg.mass = list(self._template_msg.mass)
        msg.coriolis = list(self._template_msg.coriolis)
        msg.zero_jacobian_flange = list(self._template_msg.zero_jacobian_flange)
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = Goal2FakeStateParameterPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
