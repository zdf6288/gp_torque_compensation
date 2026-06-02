#!/usr/bin/env python3

import csv
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node

from custom_msgs.msg import StateParameter, EffortCommand


class EffortPDController(Node):
    """
    GOAL1 q7-specific effort probe.

    Safety semantics:
    - default off
    - q7-only sinusoidal target around measured initial q7
    - q1-q6 only do small posture hold around measured initial pose
    - q7 torque has its own low clip
    - short duration
    - saves CSV for q range / tau range check
    """

    def __init__(self):
        super().__init__('effort_pd')

        self.param_subscription = self.create_subscription(
            StateParameter, '/state_parameter', self.stateParameterCallback, 10)

        self.effort_publisher = self.create_publisher(
            EffortCommand, '/effort_command', 10)

        self.declare_parameter('goal1_q7_probe_enabled', False)
        self.declare_parameter('goal1_q7_amplitude_rad', 0.02)
        self.declare_parameter('goal1_q7_frequency_hz', 0.10)
        self.declare_parameter('goal1_hold_sec', 1.0)
        self.declare_parameter('goal1_motion_duration_sec', 10.0)

        # q1-q6 posture hold should stay modest. q7 has separate lower clip.
        self.declare_parameter('k_gains', [8.0, 8.0, 8.0, 8.0, 4.0, 3.0, 0.8])
        self.declare_parameter('d_gains', [1.5, 1.5, 1.5, 1.0, 0.8, 0.8, 0.15])
        self.declare_parameter('tau_clip_nm', 3.0)
        self.declare_parameter('q7_tau_clip_nm', 0.2)
        self.declare_parameter('output_csv', 'goal1_effort_pd_q7_probe_data.csv')

        self.goal1_q7_probe_enabled = bool(
            self.get_parameter('goal1_q7_probe_enabled').value)
        self.goal1_q7_amplitude_rad = float(
            self.get_parameter('goal1_q7_amplitude_rad').value)
        self.goal1_q7_frequency_hz = float(
            self.get_parameter('goal1_q7_frequency_hz').value)
        self.goal1_hold_sec = float(
            self.get_parameter('goal1_hold_sec').value)
        self.goal1_motion_duration_sec = float(
            self.get_parameter('goal1_motion_duration_sec').value)

        self.k_gains = np.array(self.get_parameter('k_gains').value, dtype=float)
        self.d_gains = np.array(self.get_parameter('d_gains').value, dtype=float)
        self.tau_clip_nm = float(self.get_parameter('tau_clip_nm').value)
        self.q7_tau_clip_nm = float(self.get_parameter('q7_tau_clip_nm').value)
        self.output_csv = str(self.get_parameter('output_csv').value)

        self.q_initial = None
        self.t_initial = None
        self.done = False
        self.csv_saved = False

        self.rows = []
        self.effort_msg = EffortCommand()

        self.get_logger().info('Effort PD q7 probe node started')
        self.get_logger().info(
            '[GOAL1] q7_probe_enabled='
            f'{self.goal1_q7_probe_enabled}, '
            f'amp_rad={self.goal1_q7_amplitude_rad}, '
            f'freq_hz={self.goal1_q7_frequency_hz}, '
            f'hold_sec={self.goal1_hold_sec}, '
            f'motion_duration_sec={self.goal1_motion_duration_sec}, '
            f'tau_clip_nm={self.tau_clip_nm}, '
            f'q7_tau_clip_nm={self.q7_tau_clip_nm}'
        )

    def _publish_zero(self):
        self.effort_msg.efforts = [0.0] * 7
        self.effort_publisher.publish(self.effort_msg)

    def _save_csv_once(self):
        if self.csv_saved:
            return
        self.csv_saved = True

        path = Path(self.output_csv)
        header = ['time_s']
        header += [f'q_{i+1}' for i in range(7)]
        header += [f'dq_{i+1}' for i in range(7)]
        header += [f'q_des_{i+1}' for i in range(7)]
        header += [f'dq_des_{i+1}' for i in range(7)]
        header += [f'tau_{i+1}' for i in range(7)]

        with path.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(self.rows)

        self.get_logger().info(f'[GOAL1] saved {len(self.rows)} rows to {path.resolve()}')

    def stateParameterCallback(self, msg):
        try:
            t_now = self.get_clock().now()

            q = np.array(msg.position, dtype=float)
            dq = np.array(msg.velocity, dtype=float)

            if self.t_initial is None:
                self.t_initial = t_now
                self.q_initial = q.copy()
                self.get_logger().info(
                    '[GOAL1] latched q_initial='
                    + np.array2string(self.q_initial, precision=5)
                )

            t_elapsed = (t_now - self.t_initial).nanoseconds / 1e9

            if not self.goal1_q7_probe_enabled:
                self._publish_zero()
                return

            total_duration = self.goal1_hold_sec + self.goal1_motion_duration_sec
            if t_elapsed > total_duration:
                if not self.done:
                    self.done = True
                    self.get_logger().info('[GOAL1] q7 probe duration complete; publishing zero torque and saving CSV')
                    self._save_csv_once()
                self._publish_zero()
                return

            q_des = self.q_initial.copy()
            dq_des = np.zeros(7, dtype=float)

            if t_elapsed >= self.goal1_hold_sec:
                t_probe = t_elapsed - self.goal1_hold_sec
                omega = 2.0 * np.pi * self.goal1_q7_frequency_hz
                q7_offset = self.goal1_q7_amplitude_rad * np.sin(omega * t_probe)
                dq7_des = self.goal1_q7_amplitude_rad * omega * np.cos(omega * t_probe)

                q_des[6] = self.q_initial[6] + q7_offset
                dq_des[6] = dq7_des

            tau = self.k_gains * (q_des - q) + self.d_gains * (dq_des - dq)

            # General clip for posture hold, then stricter q7 clip for the probe channel.
            tau = np.clip(tau, -self.tau_clip_nm, self.tau_clip_nm)
            tau[6] = float(np.clip(tau[6], -self.q7_tau_clip_nm, self.q7_tau_clip_nm))

            self.rows.append(
                [t_elapsed]
                + q.tolist()
                + dq.tolist()
                + q_des.tolist()
                + dq_des.tolist()
                + tau.tolist()
            )

            self.effort_msg.efforts = tau.tolist()
            self.effort_publisher.publish(self.effort_msg)

        except Exception as e:
            self.get_logger().error(f'Parameter error: {str(e)}')
            self._publish_zero()

    def destroy_node(self):
        try:
            self._publish_zero()
            self._save_csv_once()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = EffortPDController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt received; saving CSV and publishing zero torque')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
