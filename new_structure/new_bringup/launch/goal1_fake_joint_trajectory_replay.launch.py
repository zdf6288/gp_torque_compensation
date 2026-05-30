#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.conditions import IfCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    robot_ip_parameter_name = 'robot_ip'
    load_gripper_parameter_name = 'load_gripper'
    use_rviz_parameter_name = 'use_rviz'
    controller_config_parameter_name = 'controller_config'
    csv_path_parameter_name = 'csv_path'
    start_replay_parameter_name = 'start_replay'
    dry_run_parameter_name = 'dry_run'
    max_duration_parameter_name = 'max_duration'
    joint_names_parameter_name = 'joint_names'
    controller_name_parameter_name = 'controller_name'

    robot_ip = LaunchConfiguration(robot_ip_parameter_name)
    load_gripper = LaunchConfiguration(load_gripper_parameter_name)
    use_rviz = LaunchConfiguration(use_rviz_parameter_name)
    controller_config = LaunchConfiguration(controller_config_parameter_name)
    csv_path = LaunchConfiguration(csv_path_parameter_name)
    start_replay = LaunchConfiguration(start_replay_parameter_name)
    dry_run = LaunchConfiguration(dry_run_parameter_name)
    max_duration = LaunchConfiguration(max_duration_parameter_name)
    joint_names = LaunchConfiguration(joint_names_parameter_name)
    controller_name = LaunchConfiguration(controller_name_parameter_name)

    franka_xacro_file = PathJoinSubstitution(
        [FindPackageShare('franka_description'), 'robots', 'panda_arm.urdf.xacro']
    )
    robot_description = ParameterValue(
        Command([
            FindExecutable(name='xacro'),
            ' ',
            franka_xacro_file,
            ' hand:=',
            load_gripper,
            ' robot_ip:=',
            robot_ip,
            ' use_fake_hardware:=true',
            ' fake_sensor_commands:=false',
        ]),
        value_type=str,
    )
    rviz_file = PathJoinSubstitution(
        [FindPackageShare('franka_description'), 'rviz', 'visualize_franka.rviz']
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            robot_ip_parameter_name,
            default_value='0.0.0.0',
            description='Dummy robot IP for local fake hardware only.'),
        DeclareLaunchArgument(
            load_gripper_parameter_name,
            default_value='true',
            description='Load the Panda hand in the fake hardware robot description.'),
        DeclareLaunchArgument(
            use_rviz_parameter_name,
            default_value='true',
            description='Visualize the fake hardware robot in RViz.'),
        DeclareLaunchArgument(
            controller_config_parameter_name,
            default_value=PathJoinSubstitution([
                FindPackageShare('new_bringup'),
                'config',
                'goal1_fake_joint_trajectory_controllers.yaml',
            ]),
            description='Fake-only controller config for GOAL1 joint trajectory replay.'),
        DeclareLaunchArgument(
            csv_path_parameter_name,
            default_value='outputs/goal1_joint_trajectory/goal1_allq_spatial_rich_60s_50hz.csv',
            description='GOAL1 joint trajectory CSV path.'),
        DeclareLaunchArgument(
            start_replay_parameter_name,
            default_value='false',
            description='Start the CSV replay node. Defaults false for fake-only safety.'),
        DeclareLaunchArgument(
            dry_run_parameter_name,
            default_value='true',
            description='Run the replay node without publishing a trajectory.'),
        DeclareLaunchArgument(
            max_duration_parameter_name,
            default_value='5.0',
            description='Maximum replay segment duration in seconds.'),
        DeclareLaunchArgument(
            joint_names_parameter_name,
            default_value='panda_joint1,panda_joint2,panda_joint3,panda_joint4,panda_joint5,panda_joint6,panda_joint7',
            description='Comma-separated Panda fake hardware joint names.'),
        DeclareLaunchArgument(
            controller_name_parameter_name,
            default_value='goal1_joint_trajectory_controller',
            description='Joint trajectory controller name.'),

        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description}],
        ),
        Node(
            package='controller_manager',
            executable='ros2_control_node',
            parameters=[{'robot_description': robot_description}, controller_config],
            output={'stdout': 'screen', 'stderr': 'screen'},
            on_exit=Shutdown(),
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['joint_state_broadcaster'],
            output='screen',
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['goal1_joint_trajectory_controller'],
            output='screen',
        ),
        Node(
            package='py_controllers',
            executable='goal1_csv_joint_trajectory_replay',
            name='goal1_csv_joint_trajectory_replay',
            output='screen',
            condition=IfCondition(start_replay),
            parameters=[{
                csv_path_parameter_name: ParameterValue(csv_path, value_type=str),
                joint_names_parameter_name: ParameterValue(joint_names, value_type=str),
                controller_name_parameter_name: ParameterValue(controller_name, value_type=str),
                dry_run_parameter_name: ParameterValue(dry_run, value_type=bool),
                max_duration_parameter_name: ParameterValue(max_duration, value_type=float),
            }],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['--display-config', rviz_file],
            condition=IfCondition(use_rviz),
        ),
    ])
