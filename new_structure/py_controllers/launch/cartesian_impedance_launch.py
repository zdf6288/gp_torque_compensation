#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
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
    save_csv_on_shutdown_parameter_name = 'save_csv_on_shutdown'
    enable_runtime_plotting_parameter_name = 'enable_runtime_plotting'
    run_ablation_on_shutdown_parameter_name = 'run_ablation_on_shutdown'
    shutdown_hold_duration_parameter_name = 'shutdown_hold_duration'

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
    save_csv_on_shutdown = LaunchConfiguration(save_csv_on_shutdown_parameter_name)
    enable_runtime_plotting = LaunchConfiguration(enable_runtime_plotting_parameter_name)
    run_ablation_on_shutdown = LaunchConfiguration(run_ablation_on_shutdown_parameter_name)
    shutdown_hold_duration = LaunchConfiguration(shutdown_hold_duration_parameter_name)

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
            description='Enable GP prediction in the controller.'),
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
            save_csv_on_shutdown_parameter_name,
            default_value='true',
            description='Save controller CSV data on shutdown.'),
        DeclareLaunchArgument(
            enable_runtime_plotting_parameter_name,
            default_value='false',
            description='Enable plotting or other heavy shutdown-time analysis in the controller.'),
        DeclareLaunchArgument(
            run_ablation_on_shutdown_parameter_name,
            default_value='false',
            description='Run ablation.py from the controller shutdown path.'),
        DeclareLaunchArgument(
            shutdown_hold_duration_parameter_name,
            default_value='1.0',
            description='Duration in seconds to hold the final trajectory pose before shutdown.'),

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
                save_csv_on_shutdown_parameter_name: save_csv_on_shutdown,
                enable_runtime_plotting_parameter_name: enable_runtime_plotting,
                run_ablation_on_shutdown_parameter_name: run_ablation_on_shutdown,
            }]
        ),
        Node(
            package='py_controllers',
            executable='trajectory_publisher',
            name='trajectory_publisher',
            output='screen',
            parameters=[{
                shutdown_hold_duration_parameter_name: shutdown_hold_duration,
            }],
            # parameters=[{
            #     'circle_radius': 0.2,
            #     'circle_frequency': 0.5,
            #     'circle_center_x': 0.5,
            #     'circle_center_y': 0.0,
            #     'circle_center_z': 0.3,
            # }]
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
