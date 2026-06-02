#!/usr/bin/env python3

import math

import numpy as np
import rclpy
from custom_msgs.msg import StateParameter
from rclpy.node import Node
from std_msgs.msg import Header


class Goal1MockStateParameterPublisher(Node):
    """Offline/no-motion StateParameter publisher for GOAL1 O sequencing checks."""

    def __init__(self):
        super().__init__('goal1_mock_state_parameter_publisher')

        self.declare_parameter('publish_rate_hz', 50.0)
        self.declare_parameter(
            'q',
            [0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.0],
        )
        self.declare_parameter('dq', [0.0] * 7)
        self.declare_parameter('mock_ee_x', 0.35)
        self.declare_parameter('mock_ee_y', 0.0)
        self.declare_parameter('mock_ee_z', 0.65)

        self.publish_rate_hz = self._positive_float_parameter('publish_rate_hz', 50.0)
        self.q = self._seven_float_parameter('q')
        self.dq = self._seven_float_parameter('dq')
        self.mock_ee_xyz = np.array([
            float(self.get_parameter('mock_ee_x').value),
            float(self.get_parameter('mock_ee_y').value),
            float(self.get_parameter('mock_ee_z').value),
        ])

        self.publisher = self.create_publisher(StateParameter, '/state_parameter', 10)
        self.timer = self.create_timer(
            1.0 / self.publish_rate_hz,
            self.publish_state_parameter,
        )

        self.get_logger().warn(
            'GOAL1 O FAKE/NO-MOTION node: publishing mock /state_parameter only. '
            'This node does not publish /effort_command, does not start controllers, '
            'and must not be used as a real robot state source.'
        )
        self.get_logger().info(
            f'Mock q={self.q.tolist()}, dq={self.dq.tolist()}, '
            f'ee_xyz={self.mock_ee_xyz.tolist()}, rate={self.publish_rate_hz} Hz'
        )

    def _positive_float_parameter(self, name, default_value):
        try:
            value = float(self.get_parameter(name).value)
        except (TypeError, ValueError):
            value = float(default_value)

        if not math.isfinite(value) or value <= 0.0:
            self.get_logger().warn(
                f"Parameter '{name}' must be finite and > 0.0; using {default_value}"
            )
            return float(default_value)
        return value

    def _seven_float_parameter(self, name):
        raw_value = self.get_parameter(name).value
        values = np.array(raw_value, dtype=float)
        if values.shape != (7,) or not np.all(np.isfinite(values)):
            raise ValueError(f"Parameter '{name}' must contain 7 finite values")
        return values

    def publish_state_parameter(self):
        msg = StateParameter()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'goal1_fake_no_motion'

        msg.position = self.q.tolist()
        msg.velocity = self.dq.tolist()
        msg.effort_measured = [0.0] * 7
        msg.gravity = [0.0] * 7

        o_t_f = np.eye(4, dtype=float)
        o_t_f[:3, 3] = self.mock_ee_xyz
        msg.o_t_f = o_t_f.reshape(16, order='F').tolist()

        mass = np.eye(7, dtype=float)
        msg.mass = mass.reshape(49, order='F').tolist()
        msg.coriolis = [0.0] * 7

        zero_jacobian = np.zeros((6, 7), dtype=float)
        zero_jacobian[:6, :6] = np.eye(6, dtype=float)
        msg.zero_jacobian_flange = zero_jacobian.reshape(42, order='F').tolist()

        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = Goal1MockStateParameterPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
