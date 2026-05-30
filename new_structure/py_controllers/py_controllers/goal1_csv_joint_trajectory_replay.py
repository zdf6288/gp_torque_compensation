#!/usr/bin/env python3

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import List, Optional, Sequence


DEFAULT_CSV_PATH = 'outputs/goal1_joint_trajectory/goal1_allq_spatial_rich_60s_50hz.csv'
DEFAULT_JOINT_NAMES = [
    'panda_joint1',
    'panda_joint2',
    'panda_joint3',
    'panda_joint4',
    'panda_joint5',
    'panda_joint6',
    'panda_joint7',
]
DEFAULT_CONTROLLER_NAME = 'goal1_joint_trajectory_controller'
DEFAULT_TOPIC_OR_ACTION = 'topic'
DEFAULT_MAX_DURATION = 5.0
DEFAULT_START_TIME = 0.0
DEFAULT_TIME_SCALE = 1.0


@dataclass
class ReplayConfig:
    csv_path: str = DEFAULT_CSV_PATH
    joint_names: Optional[List[str]] = None
    controller_name: str = DEFAULT_CONTROLLER_NAME
    topic_or_action: str = DEFAULT_TOPIC_OR_ACTION
    max_duration: float = DEFAULT_MAX_DURATION
    start_time: float = DEFAULT_START_TIME
    time_scale: float = DEFAULT_TIME_SCALE
    dry_run: bool = True
    hold_final: bool = False
    prepend_current_state: bool = False
    ramp_duration: float = 0.0

    def __post_init__(self):
        if self.joint_names is None:
            self.joint_names = list(DEFAULT_JOINT_NAMES)


@dataclass
class TrajectoryPointData:
    csv_time: float
    time_from_start: float
    positions: List[float]


@dataclass
class LoadedTrajectory:
    header: List[str]
    points: List[TrajectoryPointData]
    source_start_time: float
    source_end_time: float


def parse_joint_names(value) -> List[str]:
    if isinstance(value, (list, tuple)):
        names = [str(item).strip() for item in value]
    else:
        names = [item.strip() for item in str(value).split(',')]
    names = [name for name in names if name]
    if len(names) != 7:
        raise ValueError(f'Expected 7 joint names, got {len(names)}: {names}')
    return names


def positive_float(value: float, name: str) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError(f'{name} must be > 0.0, got {value}')
    return value


def nonnegative_float(value: float, name: str) -> float:
    value = float(value)
    if value < 0.0:
        raise ValueError(f'{name} must be >= 0.0, got {value}')
    return value


def normalize_config(config: ReplayConfig) -> ReplayConfig:
    config.joint_names = parse_joint_names(config.joint_names)
    config.topic_or_action = str(config.topic_or_action).strip().lower()
    if config.topic_or_action not in ('topic', 'action'):
        raise ValueError("topic_or_action must be 'topic' or 'action'")
    config.max_duration = positive_float(config.max_duration, 'max_duration')
    config.start_time = nonnegative_float(config.start_time, 'start_time')
    config.time_scale = positive_float(config.time_scale, 'time_scale')
    config.ramp_duration = nonnegative_float(config.ramp_duration, 'ramp_duration')
    return config


def load_goal1_csv(config: ReplayConfig) -> LoadedTrajectory:
    config = normalize_config(config)
    csv_path = Path(config.csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f'CSV file not found: {csv_path}')

    required_columns = ['time'] + [f'joint_pos_{idx}' for idx in range(1, 8)]
    points = []
    previous_time: Optional[float] = None
    first_selected_time: Optional[float] = None
    source_end_limit = config.start_time + config.max_duration

    with csv_path.open(newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        header = reader.fieldnames or []
        missing = [column for column in required_columns if column not in header]
        if missing:
            raise ValueError(f'Missing required CSV columns: {missing}')

        for row in reader:
            csv_time = float(row['time'])
            if previous_time is not None and csv_time < previous_time:
                raise ValueError('CSV time column must be nondecreasing')
            previous_time = csv_time

            if csv_time < config.start_time:
                continue
            if csv_time > source_end_limit:
                break

            if first_selected_time is None:
                first_selected_time = csv_time
            positions = [float(row[f'joint_pos_{idx}']) for idx in range(1, 8)]
            time_from_start = (csv_time - first_selected_time) / config.time_scale
            points.append(TrajectoryPointData(csv_time, time_from_start, positions))

    if not points:
        raise ValueError(
            f'No trajectory points selected from {csv_path} with '
            f'start_time={config.start_time} and max_duration={config.max_duration}'
        )

    return LoadedTrajectory(
        header=header,
        points=points,
        source_start_time=points[0].csv_time,
        source_end_time=points[-1].csv_time,
    )


def print_dry_run_summary(config: ReplayConfig, trajectory: LoadedTrajectory) -> None:
    first = trajectory.points[0]
    last = trajectory.points[-1]
    replay_duration = last.time_from_start
    print('GOAL1 CSV joint trajectory replay dry-run')
    print('fake_only: true')
    print('real_robot: false')
    print('gp: false')
    print('torque_or_effort_command: false')
    print(f'csv_path: {config.csv_path}')
    print(f'columns: {trajectory.header}')
    print(f'joint_names: {config.joint_names}')
    print(f'controller_name: {config.controller_name}')
    print(f'topic_or_action: {config.topic_or_action}')
    print(f'point_count: {len(trajectory.points)}')
    print(f'source_time_start: {trajectory.source_start_time:.9f}')
    print(f'source_time_end: {trajectory.source_end_time:.9f}')
    print(f'replay_duration: {replay_duration:.9f}')
    print(f'first_q: {[round(value, 9) for value in first.positions]}')
    print(f'last_q: {[round(value, 9) for value in last.positions]}')
    if config.prepend_current_state or config.ramp_duration > 0.0:
        print(
            'start_pose_caveat: prepend_current_state/ramp_duration are placeholders '
            'in this fake-only v1 and do not make this real-robot safe.'
        )


def build_joint_trajectory_msg(config: ReplayConfig, trajectory: LoadedTrajectory):
    from builtin_interfaces.msg import Duration
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    message = JointTrajectory()
    message.joint_names = list(config.joint_names)

    for point in trajectory.points:
        trajectory_point = JointTrajectoryPoint()
        trajectory_point.positions = list(point.positions)
        seconds = int(point.time_from_start)
        nanoseconds = int(round((point.time_from_start - seconds) * 1_000_000_000))
        if nanoseconds >= 1_000_000_000:
            seconds += 1
            nanoseconds -= 1_000_000_000
        trajectory_point.time_from_start = Duration(sec=seconds, nanosec=nanoseconds)
        message.points.append(trajectory_point)

    if config.hold_final and message.points:
        final_point = JointTrajectoryPoint()
        final_point.positions = list(message.points[-1].positions)
        final_time = trajectory.points[-1].time_from_start + 1.0
        final_seconds = int(final_time)
        final_nanoseconds = int(round((final_time - final_seconds) * 1_000_000_000))
        if final_nanoseconds >= 1_000_000_000:
            final_seconds += 1
            final_nanoseconds -= 1_000_000_000
        final_point.time_from_start = Duration(sec=final_seconds, nanosec=final_nanoseconds)
        message.points.append(final_point)

    return message


def publish_trajectory(config: ReplayConfig, trajectory: LoadedTrajectory) -> None:
    if config.topic_or_action == 'action':
        raise NotImplementedError('topic_or_action=action is reserved for a later fake-only extension')

    import rclpy
    from trajectory_msgs.msg import JointTrajectory

    rclpy.init(args=None)
    node = rclpy.create_node('goal1_csv_joint_trajectory_replay')
    try:
        topic = f'/{config.controller_name}/joint_trajectory'
        publisher = node.create_publisher(JointTrajectory, topic, 10)
        message = build_joint_trajectory_msg(config, trajectory)
        node.get_logger().info(
            'Publishing GOAL1 fake-only JointTrajectory to %s with %d points. '
            'No real robot, no GP, no torque, no effort command.'
            % (topic, len(message.points))
        )
        deadline = node.get_clock().now().nanoseconds + 1_000_000_000
        while rclpy.ok() and node.get_clock().now().nanoseconds < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        publisher.publish(message)
        rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Fake-only GOAL1 CSV to trajectory_msgs/JointTrajectory replay helper.'
    )
    parser.add_argument('--csv-path', default=DEFAULT_CSV_PATH)
    parser.add_argument('--joint-names', default=','.join(DEFAULT_JOINT_NAMES))
    parser.add_argument('--controller-name', default=DEFAULT_CONTROLLER_NAME)
    parser.add_argument('--topic-or-action', default=DEFAULT_TOPIC_OR_ACTION)
    parser.add_argument('--max-duration', type=float, default=DEFAULT_MAX_DURATION)
    parser.add_argument('--start-time', type=float, default=DEFAULT_START_TIME)
    parser.add_argument('--time-scale', type=float, default=DEFAULT_TIME_SCALE)
    parser.add_argument('--dry-run', action='store_true', default=True)
    parser.add_argument('--publish', action='store_false', dest='dry_run')
    parser.add_argument('--hold-final', action='store_true')
    parser.add_argument('--prepend-current-state', action='store_true')
    parser.add_argument('--ramp-duration', type=float, default=0.0)
    return parser


def config_from_cli(argv: Sequence[str]) -> ReplayConfig:
    parser = cli_parser()
    args = parser.parse_args(argv)
    return ReplayConfig(
        csv_path=args.csv_path,
        joint_names=parse_joint_names(args.joint_names),
        controller_name=args.controller_name,
        topic_or_action=args.topic_or_action,
        max_duration=args.max_duration,
        start_time=args.start_time,
        time_scale=args.time_scale,
        dry_run=args.dry_run,
        hold_final=args.hold_final,
        prepend_current_state=args.prepend_current_state,
        ramp_duration=args.ramp_duration,
    )


def config_from_ros_params():
    import rclpy

    rclpy.init(args=sys.argv)
    node = rclpy.create_node('goal1_csv_joint_trajectory_replay')
    try:
        node.declare_parameter('csv_path', DEFAULT_CSV_PATH)
        node.declare_parameter('joint_names', ','.join(DEFAULT_JOINT_NAMES))
        node.declare_parameter('controller_name', DEFAULT_CONTROLLER_NAME)
        node.declare_parameter('topic_or_action', DEFAULT_TOPIC_OR_ACTION)
        node.declare_parameter('max_duration', DEFAULT_MAX_DURATION)
        node.declare_parameter('start_time', DEFAULT_START_TIME)
        node.declare_parameter('time_scale', DEFAULT_TIME_SCALE)
        node.declare_parameter('dry_run', True)
        node.declare_parameter('hold_final', False)
        node.declare_parameter('prepend_current_state', False)
        node.declare_parameter('ramp_duration', 0.0)

        config = ReplayConfig(
            csv_path=node.get_parameter('csv_path').value,
            joint_names=parse_joint_names(node.get_parameter('joint_names').value),
            controller_name=node.get_parameter('controller_name').value,
            topic_or_action=node.get_parameter('topic_or_action').value,
            max_duration=node.get_parameter('max_duration').value,
            start_time=node.get_parameter('start_time').value,
            time_scale=node.get_parameter('time_scale').value,
            dry_run=node.get_parameter('dry_run').value,
            hold_final=node.get_parameter('hold_final').value,
            prepend_current_state=node.get_parameter('prepend_current_state').value,
            ramp_duration=node.get_parameter('ramp_duration').value,
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()

    return config


def should_use_cli(argv: Sequence[str]) -> bool:
    if '--ros-args' in argv:
        return False
    return any(argument.startswith('--') and argument != '--ros-args' for argument in argv)


def run(config: ReplayConfig) -> None:
    config = normalize_config(config)
    trajectory = load_goal1_csv(config)
    if config.dry_run:
        print_dry_run_summary(config, trajectory)
        return
    publish_trajectory(config, trajectory)


def main(argv: Optional[Sequence[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if should_use_cli(argv):
        config = config_from_cli(argv)
    else:
        config = config_from_ros_params()
    run(config)


if __name__ == '__main__':
    main()
