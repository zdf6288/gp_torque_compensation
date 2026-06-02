#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    robot_ip_parameter_name = 'robot_ip'
    csv_path_parameter_name = 'csv_path'
    start_time_parameter_name = 'start_time'
    max_duration_parameter_name = 'max_duration'
    dry_run_parameter_name = 'dry_run'
    start_replay_parameter_name = 'start_replay'
    publish_effort_parameter_name = 'publish_effort'
    publish_reference_parameter_name = 'publish_reference'
    state_only_parameter_name = 'state_only'
    state_source_parameter_name = 'state_source'
    state_topic_parameter_name = 'state_topic'
    joint_state_topic_parameter_name = 'joint_state_topic'
    reference_topic_parameter_name = 'reference_topic'
    kp_parameter_name = 'kp'
    kd_parameter_name = 'kd'
    torque_clip_nm_parameter_name = 'torque_clip_nm'
    torque_rate_limit_nm_per_s_parameter_name = 'torque_rate_limit_nm_per_s'
    start_position_tolerance_rad_parameter_name = 'start_position_tolerance_rad'

    robot_ip = LaunchConfiguration(robot_ip_parameter_name)
    csv_path = LaunchConfiguration(csv_path_parameter_name)
    start_time = LaunchConfiguration(start_time_parameter_name)
    max_duration = LaunchConfiguration(max_duration_parameter_name)
    dry_run = LaunchConfiguration(dry_run_parameter_name)
    start_replay = LaunchConfiguration(start_replay_parameter_name)
    publish_effort = LaunchConfiguration(publish_effort_parameter_name)
    publish_reference = LaunchConfiguration(publish_reference_parameter_name)
    state_only = LaunchConfiguration(state_only_parameter_name)
    state_source = LaunchConfiguration(state_source_parameter_name)
    state_topic = LaunchConfiguration(state_topic_parameter_name)
    joint_state_topic = LaunchConfiguration(joint_state_topic_parameter_name)
    reference_topic = LaunchConfiguration(reference_topic_parameter_name)
    kp = LaunchConfiguration(kp_parameter_name)
    kd = LaunchConfiguration(kd_parameter_name)
    torque_clip_nm = LaunchConfiguration(torque_clip_nm_parameter_name)
    torque_rate_limit_nm_per_s = LaunchConfiguration(
        torque_rate_limit_nm_per_s_parameter_name
    )
    start_position_tolerance_rad = LaunchConfiguration(
        start_position_tolerance_rad_parameter_name
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            robot_ip_parameter_name,
            default_value='',
            description=(
                'Robot IP is exposed for review symmetry only. '
                'This launch does not include franka.launch.py.'
            ),
        ),
        DeclareLaunchArgument(
            csv_path_parameter_name,
            default_value=(
                'outputs/goal1_joint_trajectory/'
                'goal1_allq_spatial_rich_60s_50hz.csv'
            ),
            description='GOAL1 joint-space CSV path.',
        ),
        DeclareLaunchArgument(
            start_time_parameter_name,
            default_value='0.0',
            description='Source CSV start time in seconds.',
        ),
        DeclareLaunchArgument(
            max_duration_parameter_name,
            default_value='3.0',
            description='Maximum selected replay duration in seconds.',
        ),
        DeclareLaunchArgument(
            dry_run_parameter_name,
            default_value='true',
            description='Validate only; no effort publishing.',
        ),
        DeclareLaunchArgument(
            start_replay_parameter_name,
            default_value='false',
            description='Additional guard required before any replay attempt.',
        ),
        DeclareLaunchArgument(
            publish_effort_parameter_name,
            default_value='false',
            description='Additional guard required before publishing /effort_command.',
        ),
        DeclareLaunchArgument(
            publish_reference_parameter_name,
            default_value='false',
            description=(
                'Publish JointSpaceCommand references instead of /effort_command. '
                'Still requires dry_run:=false and start_replay:=true.'
            ),
        ),
        DeclareLaunchArgument(
            state_only_parameter_name,
            default_value='false',
            description=(
                'State-only preflight checks the start pose only; '
                'no /effort_command publisher is created.'
            ),
        ),
        DeclareLaunchArgument(
            state_source_parameter_name,
            default_value='state_parameter',
            description=(
                'State source for state_only preflight or final effort replay: '
                'state_parameter uses /state_parameter; joint_states uses '
                '/franka/joint_states and avoids /state_parameter as the replay '
                'state input. publish_reference still uses state_parameter.'
            ),
        ),
        DeclareLaunchArgument(
            state_topic_parameter_name,
            default_value='/state_parameter',
            description='StateParameter topic used when state_source:=state_parameter.',
        ),
        DeclareLaunchArgument(
            joint_state_topic_parameter_name,
            default_value='/franka/joint_states',
            description=(
                'JointState topic used when state_source:=joint_states. '
                'In state_only mode, no /effort_command publisher is created.'
            ),
        ),
        DeclareLaunchArgument(
            reference_topic_parameter_name,
            default_value='/joint_space_command',
            description='JointSpaceCommand topic used when publish_reference:=true.',
        ),
        DeclareLaunchArgument(
            kp_parameter_name,
            default_value='2.0',
            description='Low joint PD proportional gain scalar or comma-separated 7 values.',
        ),
        DeclareLaunchArgument(
            kd_parameter_name,
            default_value='0.2',
            description='Low joint PD derivative gain scalar or comma-separated 7 values.',
        ),
        DeclareLaunchArgument(
            torque_clip_nm_parameter_name,
            default_value='0.5',
            description='Per-joint torque clip in Nm.',
        ),
        DeclareLaunchArgument(
            torque_rate_limit_nm_per_s_parameter_name,
            default_value='5.0',
            description='Per-joint torque rate limit in Nm/s.',
        ),
        DeclareLaunchArgument(
            start_position_tolerance_rad_parameter_name,
            default_value='0.05',
            description='Required current q vs first CSV q tolerance in rad.',
        ),
        Node(
            package='py_controllers',
            executable='goal1_joint_space_replay',
            name='goal1_joint_space_replay',
            output='screen',
            parameters=[{
                robot_ip_parameter_name: robot_ip,
                csv_path_parameter_name: csv_path,
                start_time_parameter_name: ParameterValue(start_time, value_type=float),
                max_duration_parameter_name: ParameterValue(max_duration, value_type=float),
                dry_run_parameter_name: ParameterValue(dry_run, value_type=bool),
                start_replay_parameter_name: ParameterValue(start_replay, value_type=bool),
                publish_effort_parameter_name: ParameterValue(publish_effort, value_type=bool),
                publish_reference_parameter_name: ParameterValue(
                    publish_reference, value_type=bool
                ),
                state_only_parameter_name: ParameterValue(state_only, value_type=bool),
                state_source_parameter_name: state_source,
                state_topic_parameter_name: state_topic,
                joint_state_topic_parameter_name: joint_state_topic,
                reference_topic_parameter_name: reference_topic,
                kp_parameter_name: kp,
                kd_parameter_name: kd,
                torque_clip_nm_parameter_name: ParameterValue(
                    torque_clip_nm, value_type=float
                ),
                torque_rate_limit_nm_per_s_parameter_name: ParameterValue(
                    torque_rate_limit_nm_per_s, value_type=float
                ),
                start_position_tolerance_rad_parameter_name: ParameterValue(
                    start_position_tolerance_rad, value_type=float
                ),
            }]
        ),
    ])
