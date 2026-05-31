#!/usr/bin/env python3

import argparse
import csv
from dataclasses import dataclass, field
import math
from pathlib import Path
import sys
from typing import List, Optional, Sequence


DEFAULT_CSV_PATH = 'outputs/goal1_joint_trajectory/goal1_allq_spatial_rich_60s_50hz.csv'
DEFAULT_DRY_RUN = True
DEFAULT_START_REPLAY = False
DEFAULT_PUBLISH_EFFORT = False
DEFAULT_STATE_ONLY = False
DEFAULT_MAX_DURATION = 3.0
DEFAULT_START_TIME = 0.0
DEFAULT_STATE_TOPIC = '/state_parameter'
DEFAULT_EFFORT_TOPIC = '/effort_command'
DEFAULT_START_POSITION_TOLERANCE_RAD = 0.05
DEFAULT_STATE_TIMEOUT_SEC = 0.5
DEFAULT_COMMAND_TIMEOUT_SEC = 0.1
DEFAULT_KP = [2.0] * 7
DEFAULT_KD = [0.2] * 7
DEFAULT_TORQUE_CLIP_NM = 0.5
DEFAULT_TORQUE_RATE_LIMIT_NM_PER_S = 5.0
DEFAULT_HOLD_FINAL = False
DEFAULT_HOLD_DURATION = 0.0
DEFAULT_ROBOT_IP = ''


@dataclass
class ReplayConfig:
    csv_path: str = DEFAULT_CSV_PATH
    dry_run: bool = DEFAULT_DRY_RUN
    start_replay: bool = DEFAULT_START_REPLAY
    publish_effort: bool = DEFAULT_PUBLISH_EFFORT
    state_only: bool = DEFAULT_STATE_ONLY
    max_duration: float = DEFAULT_MAX_DURATION
    start_time: float = DEFAULT_START_TIME
    state_topic: str = DEFAULT_STATE_TOPIC
    effort_topic: str = DEFAULT_EFFORT_TOPIC
    start_position_tolerance_rad: float = DEFAULT_START_POSITION_TOLERANCE_RAD
    state_timeout_sec: float = DEFAULT_STATE_TIMEOUT_SEC
    command_timeout_sec: float = DEFAULT_COMMAND_TIMEOUT_SEC
    kp: List[float] = field(default_factory=lambda: list(DEFAULT_KP))
    kd: List[float] = field(default_factory=lambda: list(DEFAULT_KD))
    torque_clip_nm: float = DEFAULT_TORQUE_CLIP_NM
    torque_rate_limit_nm_per_s: float = DEFAULT_TORQUE_RATE_LIMIT_NM_PER_S
    hold_final: bool = DEFAULT_HOLD_FINAL
    hold_duration: float = DEFAULT_HOLD_DURATION
    robot_ip: str = DEFAULT_ROBOT_IP


@dataclass
class TrajectoryPoint:
    csv_time: float
    time_from_start: float
    q: List[float]
    dq: List[float]
    ddq: List[float]
    jerk: Optional[List[float]]


@dataclass
class LoadedTrajectory:
    header: List[str]
    points: List[TrajectoryPoint]
    source_start_time: float
    source_end_time: float
    has_jerk: bool


def parse_bool(value, name: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in ('1', 'true', 'yes', 'on'):
        return True
    if normalized in ('0', 'false', 'no', 'off'):
        return False
    raise ValueError(f'{name} must be a boolean value, got {value!r}')


def parse_gain_list(value, name: str) -> List[float]:
    if isinstance(value, str):
        items = [item.strip() for item in value.split(',') if item.strip()]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = [value]

    gains = [float(item) for item in items]
    if len(gains) == 1:
        gains = gains * 7
    if len(gains) != 7:
        raise ValueError(f'{name} must be a scalar or 7 values, got {len(gains)} values')
    if not all(math.isfinite(gain) for gain in gains):
        raise ValueError(f'{name} must contain finite values')
    if any(gain < 0.0 for gain in gains):
        raise ValueError(f'{name} must be nonnegative')
    return gains


def positive_float(value, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f'{name} must be a finite value > 0.0, got {value}')
    return value


def nonnegative_float(value, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f'{name} must be a finite value >= 0.0, got {value}')
    return value


def normalize_config(config: ReplayConfig) -> ReplayConfig:
    config.dry_run = parse_bool(config.dry_run, 'dry_run')
    config.start_replay = parse_bool(config.start_replay, 'start_replay')
    config.publish_effort = parse_bool(config.publish_effort, 'publish_effort')
    config.state_only = parse_bool(config.state_only, 'state_only')
    config.max_duration = positive_float(config.max_duration, 'max_duration')
    config.start_time = nonnegative_float(config.start_time, 'start_time')
    config.state_topic = str(config.state_topic).strip()
    config.effort_topic = str(config.effort_topic).strip()
    if not config.state_topic:
        raise ValueError('state_topic must not be empty')
    if not config.effort_topic and not config.state_only:
        raise ValueError('effort_topic must not be empty')
    config.start_position_tolerance_rad = nonnegative_float(
        config.start_position_tolerance_rad, 'start_position_tolerance_rad'
    )
    config.state_timeout_sec = positive_float(config.state_timeout_sec, 'state_timeout_sec')
    config.command_timeout_sec = positive_float(config.command_timeout_sec, 'command_timeout_sec')
    config.kp = parse_gain_list(config.kp, 'kp')
    config.kd = parse_gain_list(config.kd, 'kd')
    config.torque_clip_nm = positive_float(config.torque_clip_nm, 'torque_clip_nm')
    config.torque_rate_limit_nm_per_s = positive_float(
        config.torque_rate_limit_nm_per_s, 'torque_rate_limit_nm_per_s'
    )
    config.hold_final = parse_bool(config.hold_final, 'hold_final')
    config.hold_duration = nonnegative_float(config.hold_duration, 'hold_duration')
    config.robot_ip = str(config.robot_ip).strip()
    return config


def required_columns() -> List[str]:
    return (
        ['time']
        + [f'joint_pos_{idx}' for idx in range(1, 8)]
        + [f'joint_vel_{idx}' for idx in range(1, 8)]
        + [f'joint_acc_{idx}' for idx in range(1, 8)]
    )


def parse_finite_float(row, column: str, row_number: int) -> float:
    try:
        value = float(row[column])
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Invalid numeric value at row {row_number}, column {column}') from exc
    if not math.isfinite(value):
        raise ValueError(f'Non-finite value at row {row_number}, column {column}: {value}')
    return value


def load_goal1_csv(config: ReplayConfig) -> LoadedTrajectory:
    config = normalize_config(config)
    csv_path = Path(config.csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f'CSV file not found: {csv_path}')

    all_points: List[TrajectoryPoint] = []
    previous_time: Optional[float] = None
    jerk_columns = [f'joint_jerk_{idx}' for idx in range(1, 8)]

    with csv_path.open(newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        header = reader.fieldnames or []
        missing = [column for column in required_columns() if column not in header]
        if missing:
            raise ValueError(f'Missing required CSV columns: {missing}')
        present_jerk_columns = [column for column in jerk_columns if column in header]
        if present_jerk_columns and len(present_jerk_columns) != 7:
            missing_jerk = [column for column in jerk_columns if column not in header]
            raise ValueError(f'Incomplete optional jerk columns, missing: {missing_jerk}')
        has_jerk = len(present_jerk_columns) == 7

        for row_number, row in enumerate(reader, start=2):
            csv_time = parse_finite_float(row, 'time', row_number)
            if previous_time is not None and csv_time <= previous_time:
                raise ValueError(
                    f'CSV time column must be strictly increasing: '
                    f'row {row_number} has {csv_time}, previous {previous_time}'
                )
            previous_time = csv_time

            q = [parse_finite_float(row, f'joint_pos_{idx}', row_number) for idx in range(1, 8)]
            dq = [parse_finite_float(row, f'joint_vel_{idx}', row_number) for idx in range(1, 8)]
            ddq = [parse_finite_float(row, f'joint_acc_{idx}', row_number) for idx in range(1, 8)]
            jerk = (
                [parse_finite_float(row, f'joint_jerk_{idx}', row_number) for idx in range(1, 8)]
                if has_jerk else None
            )
            all_points.append(
                TrajectoryPoint(
                    csv_time=csv_time,
                    time_from_start=0.0,
                    q=q,
                    dq=dq,
                    ddq=ddq,
                    jerk=jerk,
                )
            )

    if not all_points:
        raise ValueError(f'CSV contains no trajectory rows: {csv_path}')

    source_end_limit = config.start_time + config.max_duration
    selected = [
        point for point in all_points
        if config.start_time <= point.csv_time <= source_end_limit
    ]
    if not selected:
        raise ValueError(
            f'No trajectory points selected from {csv_path} with '
            f'start_time={config.start_time} and max_duration={config.max_duration}'
        )

    first_selected_time = selected[0].csv_time
    points = [
        TrajectoryPoint(
            csv_time=point.csv_time,
            time_from_start=point.csv_time - first_selected_time,
            q=point.q,
            dq=point.dq,
            ddq=point.ddq,
            jerk=point.jerk,
        )
        for point in selected
    ]

    return LoadedTrajectory(
        header=header,
        points=points,
        source_start_time=points[0].csv_time,
        source_end_time=points[-1].csv_time,
        has_jerk=has_jerk,
    )


def max_abs_per_joint(points: Sequence[TrajectoryPoint], field_name: str) -> Optional[List[float]]:
    max_values = [0.0] * 7
    for point in points:
        values = getattr(point, field_name)
        if values is None:
            return None
        for idx, value in enumerate(values):
            max_values[idx] = max(max_values[idx], abs(value))
    return max_values


def rounded(values: Sequence[float], digits: int = 9) -> List[float]:
    return [round(value, digits) for value in values]


def print_validation_summary(config: ReplayConfig, trajectory: LoadedTrajectory) -> None:
    first = trajectory.points[0]
    last = trajectory.points[-1]
    max_abs_dq = max_abs_per_joint(trajectory.points, 'dq')
    max_abs_ddq = max_abs_per_joint(trajectory.points, 'ddq')
    max_abs_jerk = max_abs_per_joint(trajectory.points, 'jerk')

    print('GOAL1 joint-space torque replay validation')
    print('real_robot_safe: false')
    print('gp: false')
    print('compensation: false')
    print(f'csv_path: {config.csv_path}')
    print(f'selected_point_count: {len(trajectory.points)}')
    print(
        'source_time_range: '
        f'{trajectory.source_start_time:.9f} to {trajectory.source_end_time:.9f}'
    )
    print(f'first_q: {rounded(first.q)}')
    print(f'last_q: {rounded(last.q)}')
    print(f'max_abs_dq_per_joint: {rounded(max_abs_dq)}')
    print(f'max_abs_ddq_per_joint: {rounded(max_abs_ddq)}')
    if max_abs_jerk is None:
        print('max_abs_jerk_per_joint: not_available')
    else:
        print(f'max_abs_jerk_per_joint: {rounded(max_abs_jerk)}')
    print(f'dry_run: {config.dry_run}')
    print(f'start_replay: {config.start_replay}')
    print(f'publish_effort: {config.publish_effort}')
    print(f'state_only: {config.state_only}')
    print('no_torque_published: true')


def print_refusal(reason: str) -> None:
    print(f'refusal_reason: {reason}')
    print('no_torque_published: true')


def estimate_command_period(trajectory: LoadedTrajectory, command_timeout_sec: float) -> float:
    if len(trajectory.points) < 2:
        return command_timeout_sec
    deltas = [
        trajectory.points[idx + 1].time_from_start - trajectory.points[idx].time_from_start
        for idx in range(len(trajectory.points) - 1)
    ]
    positive_deltas = [delta for delta in deltas if delta > 0.0]
    if not positive_deltas:
        return command_timeout_sec
    return min(min(positive_deltas), command_timeout_sec)


def subtract(lhs: Sequence[float], rhs: Sequence[float]) -> List[float]:
    return [left - right for left, right in zip(lhs, rhs)]


def max_abs(values: Sequence[float]) -> float:
    return max(abs(value) for value in values)


def run_state_only_path(
    config: ReplayConfig, trajectory: LoadedTrajectory, argv: Sequence[str]
) -> None:
    import rclpy
    from rclpy.node import Node
    from custom_msgs.msg import StateParameter

    class Goal1StateOnlyValidationNode(Node):
        def __init__(self):
            super().__init__('goal1_joint_space_replay_state_only')
            self.config = config
            self.trajectory = trajectory
            self.latest_state = None
            self.latest_state_time = None
            self.refusal_reason = ''
            self.state_subscription = self.create_subscription(
                StateParameter, config.state_topic, self.state_callback, 10
            )
            self.get_logger().warn(
                'GOAL1 state-only no-motion check: subscribing to state only; '
                'no /effort_command publisher is created.'
            )

        def state_callback(self, msg):
            if len(msg.position) != 7 or len(msg.velocity) != 7:
                self.refusal_reason = (
                    'state_parameter position/velocity must each contain 7 values'
                )
                return
            q = [float(value) for value in msg.position]
            dq = [float(value) for value in msg.velocity]
            if not all(math.isfinite(value) for value in q + dq):
                self.refusal_reason = 'state_parameter contains non-finite q or dq'
                return
            self.latest_state = (q, dq)
            self.latest_state_time = self.get_clock().now()

        def state_age_sec(self) -> Optional[float]:
            if self.latest_state is None or self.latest_state_time is None:
                return None
            return (self.get_clock().now() - self.latest_state_time).nanoseconds / 1e9

        def has_fresh_state(self) -> bool:
            age = self.state_age_sec()
            return age is not None and age <= self.config.state_timeout_sec

        def wait_for_initial_state(self) -> bool:
            deadline = self.get_clock().now().nanoseconds + int(
                self.config.state_timeout_sec * 1e9
            )
            while rclpy.ok() and self.get_clock().now().nanoseconds <= deadline:
                rclpy.spin_once(self, timeout_sec=0.02)
                if self.refusal_reason:
                    return False
                if self.has_fresh_state():
                    return True
            self.refusal_reason = (
                f'no fresh state on {self.config.state_topic} within '
                f'{self.config.state_timeout_sec:.3f}s'
            )
            return False

        def print_result(self) -> bool:
            first_q = self.trajectory.points[0].q
            state_age = self.state_age_sec()
            state_fresh = self.has_fresh_state()

            print('GOAL1 state-only no-motion validation')
            print('real_robot_safe: false')
            print('gp: false')
            print('compensation: false')
            print('state_only: true')
            print(f'dry_run: {self.config.dry_run}')
            print(f'start_replay: {self.config.start_replay}')
            print(f'publish_effort_requested: {self.config.publish_effort}')
            print(f'state_topic: {self.config.state_topic}')
            print(f'csv_path: {self.config.csv_path}')
            print(f'selected_point_count: {len(self.trajectory.points)}')
            print(
                'source_time_range: '
                f'{self.trajectory.source_start_time:.9f} to '
                f'{self.trajectory.source_end_time:.9f}'
            )
            print('no_effort_publisher_created: true')
            print('no_torque_published: true')
            print(f'state_received: {self.latest_state is not None}')
            if state_age is None:
                print('state_age_sec: not_available')
            else:
                print(f'state_age_sec: {state_age:.9f}')
            print(f'state_fresh: {state_fresh}')
            print(f'csv_first_q: {rounded(first_q)}')
            print(
                'start_position_tolerance_rad: '
                f'{self.config.start_position_tolerance_rad:.9f}'
            )

            if self.latest_state is None:
                print('current_q: not_available')
                print('current_dq: not_available')
                print('q_mismatch_per_joint: not_available')
                print('max_q_mismatch: not_available')
                print('state_only_pass: false')
                print_refusal(self.refusal_reason)
                return False

            q, dq = self.latest_state
            q_mismatch = subtract(first_q, q)
            abs_q_mismatch = [abs(value) for value in q_mismatch]
            max_q_mismatch = max_abs(q_mismatch)
            passed = (
                state_fresh
                and max_q_mismatch <= self.config.start_position_tolerance_rad
                and not self.refusal_reason
            )

            print(f'current_q: {rounded(q)}')
            print(f'current_dq: {rounded(dq)}')
            print(f'q_mismatch_per_joint: {rounded(q_mismatch)}')
            print(f'q_mismatch_csv_minus_current_per_joint: {rounded(q_mismatch)}')
            print(f'q_mismatch_abs_per_joint: {rounded(abs_q_mismatch)}')
            print(f'max_q_mismatch: {max_q_mismatch:.9f}')
            print(f'state_only_pass: {str(passed).lower()}')
            if not passed:
                if not self.refusal_reason:
                    if not state_fresh:
                        self.refusal_reason = 'state is stale before state-only validation report'
                    else:
                        self.refusal_reason = (
                            'start pose mismatch: '
                            f'max_abs_error={max_q_mismatch:.6f} rad, '
                            f'tolerance={self.config.start_position_tolerance_rad:.6f} rad'
                        )
                print_refusal(self.refusal_reason)
            return passed

    rclpy.init(args=[sys.argv[0]] + list(argv))
    node = Goal1StateOnlyValidationNode()
    try:
        node.wait_for_initial_state()
        passed = node.print_result()
    finally:
        node.destroy_node()
        rclpy.shutdown()
    if not passed:
        raise SystemExit(1)


def run_publish_path(config: ReplayConfig, trajectory: LoadedTrajectory, argv: Sequence[str]) -> None:
    import rclpy
    from rclpy.node import Node
    from custom_msgs.msg import EffortCommand, StateParameter

    class Goal1JointSpaceReplayNode(Node):
        def __init__(self):
            super().__init__('goal1_joint_space_replay')
            self.config = config
            self.trajectory = trajectory
            self.latest_state = None
            self.latest_state_time = None
            self.replay_start_time = None
            self.last_timer_time = None
            self.last_command_time = None
            self.last_tau = [0.0] * 7
            self.current_index = 0
            self.done = False
            self.stop_reason = ''
            self.command_period_sec = estimate_command_period(
                trajectory, config.command_timeout_sec
            )

            self.state_subscription = self.create_subscription(
                StateParameter, config.state_topic, self.state_callback, 10
            )
            self.effort_publisher = self.create_publisher(
                EffortCommand, config.effort_topic, 10
            )
            self.timer = None
            self.get_logger().warn(
                'GOAL1 joint-space torque replay skeleton is not real-robot safe by itself. '
                'No GP, no compensation, no inverse dynamics.'
            )

        def state_callback(self, msg):
            if len(msg.position) != 7 or len(msg.velocity) != 7:
                self.refuse('state_parameter position/velocity must each contain 7 values')
                return
            q = [float(value) for value in msg.position]
            dq = [float(value) for value in msg.velocity]
            if not all(math.isfinite(value) for value in q + dq):
                self.refuse('state_parameter contains non-finite q or dq')
                return
            self.latest_state = (q, dq)
            self.latest_state_time = self.get_clock().now()

        def has_fresh_state(self) -> bool:
            if self.latest_state is None or self.latest_state_time is None:
                return False
            age = (self.get_clock().now() - self.latest_state_time).nanoseconds / 1e9
            return age <= self.config.state_timeout_sec

        def wait_for_initial_state(self) -> bool:
            deadline = self.get_clock().now().nanoseconds + int(
                self.config.state_timeout_sec * 1e9
            )
            while rclpy.ok() and self.get_clock().now().nanoseconds <= deadline:
                rclpy.spin_once(self, timeout_sec=0.02)
                if self.has_fresh_state():
                    return True
            self.refuse(
                f'no fresh state on {self.config.state_topic} within '
                f'{self.config.state_timeout_sec:.3f}s'
            )
            return False

        def validate_start_pose(self) -> bool:
            if not self.has_fresh_state():
                self.refuse('state is stale before replay start')
                return False
            q, _ = self.latest_state
            first_q = self.trajectory.points[0].q
            error = subtract(first_q, q)
            max_error = max_abs(error)
            if max_error > self.config.start_position_tolerance_rad:
                self.refuse(
                    'start pose mismatch: '
                    f'max_abs_error={max_error:.6f} rad, '
                    f'tolerance={self.config.start_position_tolerance_rad:.6f} rad'
                )
                return False
            self.get_logger().info(
                f'start pose check passed: max_abs_error={max_error:.6f} rad'
            )
            return True

        def start(self):
            self.replay_start_time = self.get_clock().now()
            self.last_timer_time = self.replay_start_time
            self.last_command_time = self.replay_start_time
            self.timer = self.create_timer(self.command_period_sec, self.timer_callback)
            self.get_logger().warn(
                'Publishing guarded joint PD efforts. This is a short skeleton path only.'
            )

        def timer_callback(self):
            now = self.get_clock().now()
            if self.last_timer_time is not None:
                timer_dt = (now - self.last_timer_time).nanoseconds / 1e9
                if timer_dt > self.config.command_timeout_sec:
                    self.refuse(
                        f'command timer overrun: dt={timer_dt:.6f}s, '
                        f'timeout={self.config.command_timeout_sec:.6f}s'
                    )
                    return
            self.last_timer_time = now

            if not self.has_fresh_state():
                self.refuse('state became stale during replay')
                return

            elapsed = (now - self.replay_start_time).nanoseconds / 1e9
            replay_end = self.trajectory.points[-1].time_from_start
            hold_end = replay_end + (
                self.config.hold_duration if self.config.hold_final else 0.0
            )
            if elapsed > hold_end:
                self.done = True
                self.stop_reason = (
                    'selected segment complete; no shutdown torque command is sent'
                )
                self.get_logger().warn(
                    self.stop_reason
                    + '; cpp_relayer last-command behavior needs lab-side review.'
                )
                if self.timer is not None:
                    self.timer.cancel()
                return

            if elapsed <= replay_end:
                while (
                    self.current_index + 1 < len(self.trajectory.points)
                    and self.trajectory.points[self.current_index + 1].time_from_start <= elapsed
                ):
                    self.current_index += 1
                point = self.trajectory.points[self.current_index]
            else:
                point = self.trajectory.points[-1]

            q, dq = self.latest_state
            tau = [
                self.config.kp[idx] * (point.q[idx] - q[idx])
                + self.config.kd[idx] * (point.dq[idx] - dq[idx])
                for idx in range(7)
            ]
            tau = [
                max(-self.config.torque_clip_nm, min(self.config.torque_clip_nm, value))
                for value in tau
            ]
            command_dt = (now - self.last_command_time).nanoseconds / 1e9
            command_dt = max(command_dt, 1e-6)
            max_delta = self.config.torque_rate_limit_nm_per_s * command_dt
            tau = [
                self.last_tau[idx]
                + max(-max_delta, min(max_delta, tau[idx] - self.last_tau[idx]))
                for idx in range(7)
            ]

            msg = EffortCommand()
            msg.header.stamp = now.to_msg()
            msg.efforts = list(tau)
            self.effort_publisher.publish(msg)
            self.last_tau = list(tau)
            self.last_command_time = now

        def refuse(self, reason: str):
            if self.done:
                return
            self.done = True
            self.stop_reason = reason
            self.get_logger().error(f'Refusing GOAL1 replay: {reason}. No new torque command sent.')
            if self.timer is not None:
                self.timer.cancel()

    rclpy.init(args=[sys.argv[0]] + list(argv))
    node = Goal1JointSpaceReplayNode()
    try:
        if not node.wait_for_initial_state():
            return
        if not node.validate_start_pose():
            return
        node.start()
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Default-disabled GOAL1 joint-space torque replay skeleton.'
    )
    parser.add_argument('--csv-path', default=DEFAULT_CSV_PATH)
    parser.add_argument('--dry-run', action=argparse.BooleanOptionalAction, default=DEFAULT_DRY_RUN)
    parser.add_argument(
        '--start-replay', action=argparse.BooleanOptionalAction, default=DEFAULT_START_REPLAY
    )
    parser.add_argument(
        '--publish-effort', action=argparse.BooleanOptionalAction, default=DEFAULT_PUBLISH_EFFORT
    )
    parser.add_argument(
        '--state-only', action=argparse.BooleanOptionalAction, default=DEFAULT_STATE_ONLY
    )
    parser.add_argument('--max-duration', type=float, default=DEFAULT_MAX_DURATION)
    parser.add_argument('--start-time', type=float, default=DEFAULT_START_TIME)
    parser.add_argument('--state-topic', default=DEFAULT_STATE_TOPIC)
    parser.add_argument('--effort-topic', default=DEFAULT_EFFORT_TOPIC)
    parser.add_argument(
        '--start-position-tolerance-rad',
        type=float,
        default=DEFAULT_START_POSITION_TOLERANCE_RAD,
    )
    parser.add_argument('--state-timeout-sec', type=float, default=DEFAULT_STATE_TIMEOUT_SEC)
    parser.add_argument('--command-timeout-sec', type=float, default=DEFAULT_COMMAND_TIMEOUT_SEC)
    parser.add_argument('--kp', default=','.join(str(value) for value in DEFAULT_KP))
    parser.add_argument('--kd', default=','.join(str(value) for value in DEFAULT_KD))
    parser.add_argument('--torque-clip-nm', type=float, default=DEFAULT_TORQUE_CLIP_NM)
    parser.add_argument(
        '--torque-rate-limit-nm-per-s',
        type=float,
        default=DEFAULT_TORQUE_RATE_LIMIT_NM_PER_S,
    )
    parser.add_argument(
        '--hold-final', action=argparse.BooleanOptionalAction, default=DEFAULT_HOLD_FINAL
    )
    parser.add_argument('--hold-duration', type=float, default=DEFAULT_HOLD_DURATION)
    parser.add_argument('--robot-ip', default=DEFAULT_ROBOT_IP)
    return parser


def config_from_cli(argv: Sequence[str]) -> ReplayConfig:
    args = cli_parser().parse_args(argv)
    return ReplayConfig(
        csv_path=args.csv_path,
        dry_run=args.dry_run,
        start_replay=args.start_replay,
        publish_effort=args.publish_effort,
        state_only=args.state_only,
        max_duration=args.max_duration,
        start_time=args.start_time,
        state_topic=args.state_topic,
        effort_topic=args.effort_topic,
        start_position_tolerance_rad=args.start_position_tolerance_rad,
        state_timeout_sec=args.state_timeout_sec,
        command_timeout_sec=args.command_timeout_sec,
        kp=parse_gain_list(args.kp, 'kp'),
        kd=parse_gain_list(args.kd, 'kd'),
        torque_clip_nm=args.torque_clip_nm,
        torque_rate_limit_nm_per_s=args.torque_rate_limit_nm_per_s,
        hold_final=args.hold_final,
        hold_duration=args.hold_duration,
        robot_ip=args.robot_ip,
    )


def read_ros_params(argv: Sequence[str]) -> ReplayConfig:
    import rclpy
    from rcl_interfaces.msg import ParameterDescriptor

    rclpy.init(args=[sys.argv[0]] + list(argv))
    node = rclpy.create_node('goal1_joint_space_replay')
    try:
        gain_parameter_descriptor = ParameterDescriptor(dynamic_typing=True)
        node.declare_parameter('csv_path', DEFAULT_CSV_PATH)
        node.declare_parameter('dry_run', DEFAULT_DRY_RUN)
        node.declare_parameter('start_replay', DEFAULT_START_REPLAY)
        node.declare_parameter('publish_effort', DEFAULT_PUBLISH_EFFORT)
        node.declare_parameter('state_only', DEFAULT_STATE_ONLY)
        node.declare_parameter('max_duration', DEFAULT_MAX_DURATION)
        node.declare_parameter('start_time', DEFAULT_START_TIME)
        node.declare_parameter('state_topic', DEFAULT_STATE_TOPIC)
        node.declare_parameter('effort_topic', DEFAULT_EFFORT_TOPIC)
        node.declare_parameter(
            'start_position_tolerance_rad', DEFAULT_START_POSITION_TOLERANCE_RAD
        )
        node.declare_parameter('state_timeout_sec', DEFAULT_STATE_TIMEOUT_SEC)
        node.declare_parameter('command_timeout_sec', DEFAULT_COMMAND_TIMEOUT_SEC)
        node.declare_parameter(
            'kp',
            ','.join(str(value) for value in DEFAULT_KP),
            gain_parameter_descriptor,
        )
        node.declare_parameter(
            'kd',
            ','.join(str(value) for value in DEFAULT_KD),
            gain_parameter_descriptor,
        )
        node.declare_parameter('torque_clip_nm', DEFAULT_TORQUE_CLIP_NM)
        node.declare_parameter(
            'torque_rate_limit_nm_per_s', DEFAULT_TORQUE_RATE_LIMIT_NM_PER_S
        )
        node.declare_parameter('hold_final', DEFAULT_HOLD_FINAL)
        node.declare_parameter('hold_duration', DEFAULT_HOLD_DURATION)
        node.declare_parameter('robot_ip', DEFAULT_ROBOT_IP)
        config = ReplayConfig(
            csv_path=node.get_parameter('csv_path').value,
            dry_run=node.get_parameter('dry_run').value,
            start_replay=node.get_parameter('start_replay').value,
            publish_effort=node.get_parameter('publish_effort').value,
            state_only=node.get_parameter('state_only').value,
            max_duration=node.get_parameter('max_duration').value,
            start_time=node.get_parameter('start_time').value,
            state_topic=node.get_parameter('state_topic').value,
            effort_topic=node.get_parameter('effort_topic').value,
            start_position_tolerance_rad=node.get_parameter(
                'start_position_tolerance_rad'
            ).value,
            state_timeout_sec=node.get_parameter('state_timeout_sec').value,
            command_timeout_sec=node.get_parameter('command_timeout_sec').value,
            kp=node.get_parameter('kp').value,
            kd=node.get_parameter('kd').value,
            torque_clip_nm=node.get_parameter('torque_clip_nm').value,
            torque_rate_limit_nm_per_s=node.get_parameter(
                'torque_rate_limit_nm_per_s'
            ).value,
            hold_final=node.get_parameter('hold_final').value,
            hold_duration=node.get_parameter('hold_duration').value,
            robot_ip=node.get_parameter('robot_ip').value,
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return config


def run(config: ReplayConfig, argv: Sequence[str]) -> None:
    config = normalize_config(config)
    trajectory = load_goal1_csv(config)

    if config.state_only:
        run_state_only_path(config, trajectory, argv)
        return

    if config.dry_run or not config.publish_effort:
        print_validation_summary(config, trajectory)
        return
    if not config.start_replay:
        print_validation_summary(config, trajectory)
        print_refusal('start_replay is false')
        return

    run_publish_path(config, trajectory, argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if '--ros-args' in argv:
        config = read_ros_params(argv)
    else:
        config = config_from_cli(argv)
    run(config, argv)


if __name__ == '__main__':
    main()
