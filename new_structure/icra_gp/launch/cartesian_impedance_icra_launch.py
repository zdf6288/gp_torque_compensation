#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    robot_ip_parameter_name = 'robot_ip'
    load_gripper_parameter_name = 'load_gripper'
    use_fake_hardware_parameter_name = 'use_fake_hardware'
    fake_sensor_commands_parameter_name = 'fake_sensor_commands'
    use_rviz_parameter_name = 'use_rviz'
    mode_parameter_name = 'mode'

    robot_ip = LaunchConfiguration(robot_ip_parameter_name)
    load_gripper = LaunchConfiguration(load_gripper_parameter_name)
    use_fake_hardware = LaunchConfiguration(use_fake_hardware_parameter_name)
    fake_sensor_commands = LaunchConfiguration(fake_sensor_commands_parameter_name)
    use_rviz = LaunchConfiguration(use_rviz_parameter_name)
    mode = LaunchConfiguration(mode_parameter_name)

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
            description="Fake sensor commands. Only valid when '{}' is true".format(
                use_fake_hardware_parameter_name)),
        DeclareLaunchArgument(
            load_gripper_parameter_name,
            default_value='true',
            description='Use Franka Gripper as an end-effector, otherwise, the robot is loaded '
                        'without an end-effector.'),
        DeclareLaunchArgument(
            mode_parameter_name,
            default_value='data',
            description='Mode selection: data or validation'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([PathJoinSubstitution(
                [FindPackageShare('new_bringup'), 'launch', 'franka.launch.py'])]),
            launch_arguments={robot_ip_parameter_name: robot_ip,
                              load_gripper_parameter_name: load_gripper,
                              use_fake_hardware_parameter_name: use_fake_hardware,
                              fake_sensor_commands_parameter_name: fake_sensor_commands,
                              use_rviz_parameter_name: use_rviz
                              }.items(),
        ),

        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['cpp_relayer'],
            output='screen',
        ),
        Node(
            package='icra_gp',
            executable='cartesian_impedance_icra_data',
            name='cartesian_impedance_icra_data',
            output='screen',
                condition=IfCondition(
                    PythonExpression(["'", mode, "' == 'data'"]),
                ),
        ),
        Node(
            package='icra_gp',
            executable='cartesian_impedance_icra_validation',
            name='cartesian_impedance_icra_validation',
            output='screen',
            condition=IfCondition(
                PythonExpression(["'", mode, "' == 'validation'"]),
            ),
        ),
        Node(
            package='icra_gp',
            executable='trajectory_publisher_icra_data',
            name='trajectory_publisher_icra_data',
            output='screen',
            condition=IfCondition(
                PythonExpression(["'", mode, "' == 'data'"]),
            ),
        ),
        Node(
            package='icra_gp',
            executable='trajectory_publisher_icra_validation',
            name='trajectory_publisher_icra_validation',
            output='screen',
            condition=IfCondition(
                PythonExpression(["'", mode, "' == 'validation'"]),
            ),
        ),
        Node(
            package='icra_gp',
            executable='gp_trajectory',
            name='gp_trajectory',
            output='screen',
        ),
    ])
