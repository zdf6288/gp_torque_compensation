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
    # Stage 1: frozen GP / compensation experiment 参数，默认值保持安全。
    gp_prediction_enabled_parameter_name = 'gp_prediction_enabled'
    gp_online_update_enabled_parameter_name = 'gp_online_update_enabled'
    gp_model_dir_parameter_name = 'gp_model_dir'
    gp_compensation_enabled_parameter_name = 'gp_compensation_enabled'
    gp_compensation_source_parameter_name = 'gp_compensation_source'
    gp_compensation_scale_parameter_name = 'gp_compensation_scale'
    gp_compensation_clip_nm_parameter_name = 'gp_compensation_clip_nm'
    goal1_orientation_command_enabled_parameter_name = 'goal1_orientation_command_enabled'
    goal1_orientation_max_abs_rad_parameter_name = 'goal1_orientation_max_abs_rad'
    # Stage 3A trajectory 参数默认保持 planar_circle，只有显式传参才启用 z modulation。
    trajectory_mode_parameter_name = 'trajectory_mode'
    z_amplitude_parameter_name = 'z_amplitude'
    z_frequency_multiplier_parameter_name = 'z_frequency_multiplier'
    circle_frequency_parameter_name = 'circle_frequency'
    transition_duration_parameter_name = 'transition_duration'
    rounds_per_mode_parameter_name = 'rounds_per_mode'
    goal1_x_amplitude_parameter_name = 'goal1_x_amplitude'
    goal1_y_amplitude_parameter_name = 'goal1_y_amplitude'
    goal1_z_amplitude_parameter_name = 'goal1_z_amplitude'
    goal1_x_frequency_multiplier_parameter_name = 'goal1_x_frequency_multiplier'
    goal1_y_frequency_multiplier_parameter_name = 'goal1_y_frequency_multiplier'
    goal1_z_frequency_multiplier_parameter_name = 'goal1_z_frequency_multiplier'
    goal1_roll_amplitude_parameter_name = 'goal1_roll_amplitude'
    goal1_pitch_amplitude_parameter_name = 'goal1_pitch_amplitude'
    goal1_yaw_amplitude_parameter_name = 'goal1_yaw_amplitude'
    goal1_roll_frequency_multiplier_parameter_name = 'goal1_roll_frequency_multiplier'
    goal1_pitch_frequency_multiplier_parameter_name = 'goal1_pitch_frequency_multiplier'
    goal1_yaw_frequency_multiplier_parameter_name = 'goal1_yaw_frequency_multiplier'

    robot_ip = LaunchConfiguration(robot_ip_parameter_name)
    load_gripper = LaunchConfiguration(load_gripper_parameter_name)
    use_fake_hardware = LaunchConfiguration(use_fake_hardware_parameter_name)
    fake_sensor_commands = LaunchConfiguration(fake_sensor_commands_parameter_name)
    use_rviz = LaunchConfiguration(use_rviz_parameter_name)
    gp_prediction_enabled = LaunchConfiguration(gp_prediction_enabled_parameter_name)
    gp_online_update_enabled = LaunchConfiguration(gp_online_update_enabled_parameter_name)
    gp_model_dir = LaunchConfiguration(gp_model_dir_parameter_name)
    gp_compensation_enabled = LaunchConfiguration(gp_compensation_enabled_parameter_name)
    gp_compensation_source = LaunchConfiguration(gp_compensation_source_parameter_name)
    gp_compensation_scale = LaunchConfiguration(gp_compensation_scale_parameter_name)
    gp_compensation_clip_nm = LaunchConfiguration(gp_compensation_clip_nm_parameter_name)
    goal1_orientation_command_enabled = LaunchConfiguration(goal1_orientation_command_enabled_parameter_name)
    goal1_orientation_max_abs_rad = LaunchConfiguration(goal1_orientation_max_abs_rad_parameter_name)
    trajectory_mode = LaunchConfiguration(trajectory_mode_parameter_name)
    z_amplitude = LaunchConfiguration(z_amplitude_parameter_name)
    z_frequency_multiplier = LaunchConfiguration(z_frequency_multiplier_parameter_name)
    circle_frequency = LaunchConfiguration(circle_frequency_parameter_name)
    transition_duration = LaunchConfiguration(transition_duration_parameter_name)
    rounds_per_mode = LaunchConfiguration(rounds_per_mode_parameter_name)
    goal1_x_amplitude = LaunchConfiguration(goal1_x_amplitude_parameter_name)
    goal1_y_amplitude = LaunchConfiguration(goal1_y_amplitude_parameter_name)
    goal1_z_amplitude = LaunchConfiguration(goal1_z_amplitude_parameter_name)
    goal1_x_frequency_multiplier = LaunchConfiguration(goal1_x_frequency_multiplier_parameter_name)
    goal1_y_frequency_multiplier = LaunchConfiguration(goal1_y_frequency_multiplier_parameter_name)
    goal1_z_frequency_multiplier = LaunchConfiguration(goal1_z_frequency_multiplier_parameter_name)
    goal1_roll_amplitude = LaunchConfiguration(goal1_roll_amplitude_parameter_name)
    goal1_pitch_amplitude = LaunchConfiguration(goal1_pitch_amplitude_parameter_name)
    goal1_yaw_amplitude = LaunchConfiguration(goal1_yaw_amplitude_parameter_name)
    goal1_roll_frequency_multiplier = LaunchConfiguration(goal1_roll_frequency_multiplier_parameter_name)
    goal1_pitch_frequency_multiplier = LaunchConfiguration(goal1_pitch_frequency_multiplier_parameter_name)
    goal1_yaw_frequency_multiplier = LaunchConfiguration(goal1_yaw_frequency_multiplier_parameter_name)

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
            gp_prediction_enabled_parameter_name,
            default_value='true',
            description='Enable GP prediction path in the controller.'),
        DeclareLaunchArgument(
            gp_online_update_enabled_parameter_name,
            default_value='true',
            description='Enable online GP model updates in the controller.'),
        DeclareLaunchArgument(
            gp_model_dir_parameter_name,
            default_value='./new_structure/gp/gp_models',
            description='Directory containing offline GP model pickle files.'),
        DeclareLaunchArgument(
            gp_compensation_enabled_parameter_name,
            default_value='false',
            description='Enable GP torque compensation.'),
        DeclareLaunchArgument(
            gp_compensation_source_parameter_name,
            default_value='local',
            description='GP compensation source: local, cloud, or combined.'),
        DeclareLaunchArgument(
            gp_compensation_scale_parameter_name,
            default_value='0.1',
            description='Scale applied to GP torque compensation before clipping.'),
        DeclareLaunchArgument(
            gp_compensation_clip_nm_parameter_name,
            default_value='0.5',
            description='Per-joint GP compensation clip in Nm.'),
        DeclareLaunchArgument(
            goal1_orientation_command_enabled_parameter_name,
            default_value='false',
            description='Enable GOAL1 small orientation command parsing in cartesian_impedance.'),
        DeclareLaunchArgument(
            goal1_orientation_max_abs_rad_parameter_name,
            default_value='0.035',
            description='Safety clip for GOAL1 roll/pitch/yaw command in radians.'),
        # Stage 3A launch defaults 保持 Stage 1 / Stage 2A 的平面圆轨迹行为。
        DeclareLaunchArgument(
            trajectory_mode_parameter_name,
            default_value='planar_circle',
            description='Trajectory mode: planar_circle or z_modulated_circle.'),
        DeclareLaunchArgument(
            z_amplitude_parameter_name,
            default_value='0.0',
            description='Z modulation amplitude in meters for z_modulated_circle.'),
        DeclareLaunchArgument(
            z_frequency_multiplier_parameter_name,
            default_value='0.5',
            description='Z modulation frequency multiplier relative to circle omega.'),
        DeclareLaunchArgument(
            circle_frequency_parameter_name,
            default_value='0.1',
            description='Circle trajectory frequency in Hz.'),
        DeclareLaunchArgument(
            transition_duration_parameter_name,
            default_value='3.0',
            description='Smooth transition duration before trajectory recording starts.'),
        DeclareLaunchArgument(
            rounds_per_mode_parameter_name,
            default_value='6',
            description='Number of trajectory rounds before auto stop / GP mode switch.'),
        DeclareLaunchArgument(
            goal1_x_amplitude_parameter_name,
            default_value='0.025',
            description='GOAL1 spatial-rich x amplitude in meters.'),
        DeclareLaunchArgument(
            goal1_y_amplitude_parameter_name,
            default_value='0.025',
            description='GOAL1 spatial-rich y amplitude in meters.'),
        DeclareLaunchArgument(
            goal1_z_amplitude_parameter_name,
            default_value='0.015',
            description='GOAL1 spatial-rich z amplitude in meters.'),
        DeclareLaunchArgument(
            goal1_x_frequency_multiplier_parameter_name,
            default_value='1.0',
            description='GOAL1 spatial-rich x frequency multiplier relative to circle omega.'),
        DeclareLaunchArgument(
            goal1_y_frequency_multiplier_parameter_name,
            default_value='1.5',
            description='GOAL1 spatial-rich y frequency multiplier relative to circle omega.'),
        DeclareLaunchArgument(
            goal1_z_frequency_multiplier_parameter_name,
            default_value='0.5',
            description='GOAL1 spatial-rich z frequency multiplier relative to circle omega.'),
        DeclareLaunchArgument(
            goal1_roll_amplitude_parameter_name,
            default_value='0.02',
            description='GOAL1 orientation-rich roll amplitude in radians.'),
        DeclareLaunchArgument(
            goal1_pitch_amplitude_parameter_name,
            default_value='0.02',
            description='GOAL1 orientation-rich pitch amplitude in radians.'),
        DeclareLaunchArgument(
            goal1_yaw_amplitude_parameter_name,
            default_value='0.02',
            description='GOAL1 orientation-rich yaw amplitude in radians.'),
        DeclareLaunchArgument(
            goal1_roll_frequency_multiplier_parameter_name,
            default_value='0.5',
            description='GOAL1 orientation-rich roll frequency multiplier relative to circle omega.'),
        DeclareLaunchArgument(
            goal1_pitch_frequency_multiplier_parameter_name,
            default_value='0.75',
            description='GOAL1 orientation-rich pitch frequency multiplier relative to circle omega.'),
        DeclareLaunchArgument(
            goal1_yaw_frequency_multiplier_parameter_name,
            default_value='1.0',
            description='GOAL1 orientation-rich yaw frequency multiplier relative to circle omega.'),

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
            package='py_controllers',
            executable='gp_server',          # 和 setup.py 里 entry_points 名字一致
            name='gp_server',
            output='screen',
        ),

        Node(
            package='py_controllers',
            executable='cartesian_impedance',
            name='cartesian_impedance',
            output='screen',
            parameters=[{
                gp_prediction_enabled_parameter_name: gp_prediction_enabled,
                gp_online_update_enabled_parameter_name: gp_online_update_enabled,
                gp_model_dir_parameter_name: gp_model_dir,
                gp_compensation_enabled_parameter_name: gp_compensation_enabled,
                gp_compensation_source_parameter_name: gp_compensation_source,
                gp_compensation_scale_parameter_name: gp_compensation_scale,
                gp_compensation_clip_nm_parameter_name: gp_compensation_clip_nm,
                goal1_orientation_command_enabled_parameter_name: ParameterValue(
                    goal1_orientation_command_enabled, value_type=bool),
                goal1_orientation_max_abs_rad_parameter_name: ParameterValue(
                    goal1_orientation_max_abs_rad, value_type=float),
            }]
        ),
        Node(
            package='py_controllers',
            executable='trajectory_publisher',
            name='trajectory_publisher',
            output='screen',
            parameters=[{
                trajectory_mode_parameter_name: ParameterValue(trajectory_mode, value_type=str),
                z_amplitude_parameter_name: ParameterValue(z_amplitude, value_type=float),
                z_frequency_multiplier_parameter_name: ParameterValue(
                    z_frequency_multiplier, value_type=float),
                circle_frequency_parameter_name: ParameterValue(circle_frequency, value_type=float),
                transition_duration_parameter_name: ParameterValue(
                    transition_duration, value_type=float),
                rounds_per_mode_parameter_name: ParameterValue(rounds_per_mode, value_type=int),
                goal1_x_amplitude_parameter_name: ParameterValue(
                    goal1_x_amplitude, value_type=float),
                goal1_y_amplitude_parameter_name: ParameterValue(
                    goal1_y_amplitude, value_type=float),
                goal1_z_amplitude_parameter_name: ParameterValue(
                    goal1_z_amplitude, value_type=float),
                goal1_x_frequency_multiplier_parameter_name: ParameterValue(
                    goal1_x_frequency_multiplier, value_type=float),
                goal1_y_frequency_multiplier_parameter_name: ParameterValue(
                    goal1_y_frequency_multiplier, value_type=float),
                goal1_z_frequency_multiplier_parameter_name: ParameterValue(
                    goal1_z_frequency_multiplier, value_type=float),
                goal1_roll_amplitude_parameter_name: ParameterValue(
                    goal1_roll_amplitude, value_type=float),
                goal1_pitch_amplitude_parameter_name: ParameterValue(
                    goal1_pitch_amplitude, value_type=float),
                goal1_yaw_amplitude_parameter_name: ParameterValue(
                    goal1_yaw_amplitude, value_type=float),
                goal1_roll_frequency_multiplier_parameter_name: ParameterValue(
                    goal1_roll_frequency_multiplier, value_type=float),
                goal1_pitch_frequency_multiplier_parameter_name: ParameterValue(
                    goal1_pitch_frequency_multiplier, value_type=float),
                goal1_yaw_frequency_multiplier_parameter_name: ParameterValue(
                    goal1_yaw_frequency_multiplier, value_type=float),
            }]
        ),
        # Node(
        #     package='py_controllers',
        #     executable='trajectory_eclipse_publisher',
        #     name='trajectory_eclipse_publisher',
        #     output='screen',
        #     # parameters=[{
        #     #     'circle_radius': 0.2,
        #     #     'circle_frequency': 0.5,
        #     #     'circle_center_x': 0.5,
        #     #     'circle_center_y': 0.0,
        #     #     'circle_center_z': 0.3,
        #     #     'bank_angle': 10.0,
        #     # }]
        # )
    ])
