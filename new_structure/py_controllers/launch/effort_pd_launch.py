#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    robot_ip_parameter_name = 'robot_ip'
    load_gripper_parameter_name = 'load_gripper'
    use_fake_hardware_parameter_name = 'use_fake_hardware'
    fake_sensor_commands_parameter_name = 'fake_sensor_commands'
    use_rviz_parameter_name = 'use_rviz'

    goal1_q7_probe_enabled_parameter_name = 'goal1_q7_probe_enabled'
    goal1_q7_amplitude_rad_parameter_name = 'goal1_q7_amplitude_rad'
    goal1_q7_frequency_hz_parameter_name = 'goal1_q7_frequency_hz'
    goal1_hold_sec_parameter_name = 'goal1_hold_sec'
    goal1_motion_duration_sec_parameter_name = 'goal1_motion_duration_sec'
    tau_clip_nm_parameter_name = 'tau_clip_nm'
    q7_tau_clip_nm_parameter_name = 'q7_tau_clip_nm'
    output_csv_parameter_name = 'output_csv'

    robot_ip = LaunchConfiguration(robot_ip_parameter_name)
    load_gripper = LaunchConfiguration(load_gripper_parameter_name)
    use_fake_hardware = LaunchConfiguration(use_fake_hardware_parameter_name)
    fake_sensor_commands = LaunchConfiguration(fake_sensor_commands_parameter_name)
    use_rviz = LaunchConfiguration(use_rviz_parameter_name)

    goal1_q7_probe_enabled = LaunchConfiguration(goal1_q7_probe_enabled_parameter_name)
    goal1_q7_amplitude_rad = LaunchConfiguration(goal1_q7_amplitude_rad_parameter_name)
    goal1_q7_frequency_hz = LaunchConfiguration(goal1_q7_frequency_hz_parameter_name)
    goal1_hold_sec = LaunchConfiguration(goal1_hold_sec_parameter_name)
    goal1_motion_duration_sec = LaunchConfiguration(goal1_motion_duration_sec_parameter_name)
    tau_clip_nm = LaunchConfiguration(tau_clip_nm_parameter_name)
    q7_tau_clip_nm = LaunchConfiguration(q7_tau_clip_nm_parameter_name)
    output_csv = LaunchConfiguration(output_csv_parameter_name)

    return LaunchDescription([
        DeclareLaunchArgument(
            robot_ip_parameter_name,
            description='Hostname or IP address of the robot.'),
        DeclareLaunchArgument(
            use_rviz_parameter_name,
            default_value='false',
            description='Visualize the robot in Rviz'),
        DeclareLaunchArgument(
            use_fake_hardware_parameter_name,
            default_value='false',
            description='Use fake hardware'),
        DeclareLaunchArgument(
            fake_sensor_commands_parameter_name,
            default_value='false',
            description="Fake sensor commands. Only valid when use_fake_hardware is true"),
        DeclareLaunchArgument(
            load_gripper_parameter_name,
            default_value='false',
            description='Load Franka gripper. Default false for GOAL1 q7 probe.'),

        DeclareLaunchArgument(
            goal1_q7_probe_enabled_parameter_name,
            default_value='false',
            description='Enable GOAL1 q7-only effort PD probe.'),
        DeclareLaunchArgument(
            goal1_q7_amplitude_rad_parameter_name,
            default_value='0.02',
            description='GOAL1 q7 sine amplitude in radians.'),
        DeclareLaunchArgument(
            goal1_q7_frequency_hz_parameter_name,
            default_value='0.10',
            description='GOAL1 q7 sine frequency in Hz.'),
        DeclareLaunchArgument(
            goal1_hold_sec_parameter_name,
            default_value='1.0',
            description='Initial hold duration before q7 sine starts.'),
        DeclareLaunchArgument(
            goal1_motion_duration_sec_parameter_name,
            default_value='10.0',
            description='q7 probe motion duration.'),
        DeclareLaunchArgument(
            tau_clip_nm_parameter_name,
            default_value='3.0',
            description='General per-joint torque clip for posture hold.'),
        DeclareLaunchArgument(
            q7_tau_clip_nm_parameter_name,
            default_value='0.2',
            description='Strict q7 torque clip for the probe channel.'),
        DeclareLaunchArgument(
            output_csv_parameter_name,
            default_value='goal1_effort_pd_q7_probe_data.csv',
            description='Output CSV path for q7 probe data.'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([PathJoinSubstitution(
                [FindPackageShare('new_bringup'), 'launch', 'franka.launch.py'])]),
            launch_arguments={
                robot_ip_parameter_name: robot_ip,
                load_gripper_parameter_name: load_gripper,
                use_fake_hardware_parameter_name: use_fake_hardware,
                fake_sensor_commands_parameter_name: fake_sensor_commands,
                use_rviz_parameter_name: use_rviz,
            }.items(),
        ),

        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['cpp_relayer'],
            output='screen',
        ),
        Node(
            package='py_controllers',
            executable='effort_pd',
            name='effort_pd',
            output='screen',
            parameters=[{
                goal1_q7_probe_enabled_parameter_name: ParameterValue(
                    goal1_q7_probe_enabled, value_type=bool),
                goal1_q7_amplitude_rad_parameter_name: ParameterValue(
                    goal1_q7_amplitude_rad, value_type=float),
                goal1_q7_frequency_hz_parameter_name: ParameterValue(
                    goal1_q7_frequency_hz, value_type=float),
                goal1_hold_sec_parameter_name: ParameterValue(
                    goal1_hold_sec, value_type=float),
                goal1_motion_duration_sec_parameter_name: ParameterValue(
                    goal1_motion_duration_sec, value_type=float),
                tau_clip_nm_parameter_name: ParameterValue(
                    tau_clip_nm, value_type=float),
                q7_tau_clip_nm_parameter_name: ParameterValue(
                    q7_tau_clip_nm, value_type=float),
                output_csv_parameter_name: output_csv,
            }],
        ),
    ])
