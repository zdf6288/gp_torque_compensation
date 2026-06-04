#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
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
    spawn_cpp_relayer_parameter_name = 'spawn_cpp_relayer'
    spawn_gp_server_parameter_name = 'spawn_gp_server'
    spawn_fake_state_parameter_publisher_parameter_name = 'spawn_fake_state_parameter_publisher'
    control_frequency_parameter_name = 'control_frequency'
    run_name_parameter_name = 'run_name'
    data_output_dir_parameter_name = 'data_output_dir'
    # Stage 1: frozen GP / compensation experiment 参数，默认值保持安全。
    gp_prediction_enabled_parameter_name = 'gp_prediction_enabled'
    gp_online_update_enabled_parameter_name = 'gp_online_update_enabled'
    gp_model_dir_parameter_name = 'gp_model_dir'
    gp_compensation_enabled_parameter_name = 'gp_compensation_enabled'
    gp_compensation_source_parameter_name = 'gp_compensation_source'
    gp_compensation_scale_parameter_name = 'gp_compensation_scale'
    gp_compensation_clip_nm_parameter_name = 'gp_compensation_clip_nm'
    # GOAL2 D2: cloud-like control-step / takt delay；默认 2 保持旧 controller 行为。
    delay_steps_parameter_name = 'delay_steps'
    # GOAL2 D timing instrumentation 参数默认关闭，不改变 controller 行为。
    timing_logging_enabled_parameter_name = 'timing_logging_enabled'
    timing_log_stride_parameter_name = 'timing_log_stride'
    timing_output_dir_parameter_name = 'timing_output_dir'
    deadline_ratio_warn_threshold_parameter_name = 'deadline_ratio_warn_threshold'
    # Stage 3A trajectory 参数默认保持 planar_circle，只有显式传参才启用 z modulation。
    trajectory_mode_parameter_name = 'trajectory_mode'
    z_amplitude_parameter_name = 'z_amplitude'
    z_frequency_multiplier_parameter_name = 'z_frequency_multiplier'
    circle_frequency_parameter_name = 'circle_frequency'
    transition_duration_parameter_name = 'transition_duration'
    goal1_multisine_x_primary_amplitude_parameter_name = 'goal1_multisine_x_primary_amplitude'
    goal1_multisine_x_secondary_amplitude_parameter_name = 'goal1_multisine_x_secondary_amplitude'
    goal1_multisine_y_primary_amplitude_parameter_name = 'goal1_multisine_y_primary_amplitude'
    goal1_multisine_y_secondary_amplitude_parameter_name = 'goal1_multisine_y_secondary_amplitude'
    goal1_multisine_z_primary_amplitude_parameter_name = 'goal1_multisine_z_primary_amplitude'
    goal1_multisine_z_secondary_amplitude_parameter_name = 'goal1_multisine_z_secondary_amplitude'
    goal1_multisine_x_primary_frequency_multiplier_parameter_name = 'goal1_multisine_x_primary_frequency_multiplier'
    goal1_multisine_x_secondary_frequency_multiplier_parameter_name = 'goal1_multisine_x_secondary_frequency_multiplier'
    goal1_multisine_y_primary_frequency_multiplier_parameter_name = 'goal1_multisine_y_primary_frequency_multiplier'
    goal1_multisine_y_secondary_frequency_multiplier_parameter_name = 'goal1_multisine_y_secondary_frequency_multiplier'
    goal1_multisine_z_primary_frequency_multiplier_parameter_name = 'goal1_multisine_z_primary_frequency_multiplier'
    goal1_multisine_z_secondary_frequency_multiplier_parameter_name = 'goal1_multisine_z_secondary_frequency_multiplier'
    goal1_multisine_phi_x2_parameter_name = 'goal1_multisine_phi_x2'
    goal1_multisine_phi_y1_parameter_name = 'goal1_multisine_phi_y1'
    goal1_multisine_phi_y2_parameter_name = 'goal1_multisine_phi_y2'
    goal1_multisine_phi_z1_parameter_name = 'goal1_multisine_phi_z1'
    goal1_multisine_phi_z2_parameter_name = 'goal1_multisine_phi_z2'

    robot_ip = LaunchConfiguration(robot_ip_parameter_name)
    load_gripper = LaunchConfiguration(load_gripper_parameter_name)
    use_fake_hardware = LaunchConfiguration(use_fake_hardware_parameter_name)
    fake_sensor_commands = LaunchConfiguration(fake_sensor_commands_parameter_name)
    use_rviz = LaunchConfiguration(use_rviz_parameter_name)
    spawn_cpp_relayer = LaunchConfiguration(spawn_cpp_relayer_parameter_name)
    spawn_gp_server = LaunchConfiguration(spawn_gp_server_parameter_name)
    spawn_fake_state_parameter_publisher = LaunchConfiguration(
        spawn_fake_state_parameter_publisher_parameter_name)
    control_frequency = LaunchConfiguration(control_frequency_parameter_name)
    run_name = LaunchConfiguration(run_name_parameter_name)
    data_output_dir = LaunchConfiguration(data_output_dir_parameter_name)
    gp_prediction_enabled = LaunchConfiguration(gp_prediction_enabled_parameter_name)
    gp_online_update_enabled = LaunchConfiguration(gp_online_update_enabled_parameter_name)
    gp_model_dir = LaunchConfiguration(gp_model_dir_parameter_name)
    gp_compensation_enabled = LaunchConfiguration(gp_compensation_enabled_parameter_name)
    gp_compensation_source = LaunchConfiguration(gp_compensation_source_parameter_name)
    gp_compensation_scale = LaunchConfiguration(gp_compensation_scale_parameter_name)
    gp_compensation_clip_nm = LaunchConfiguration(gp_compensation_clip_nm_parameter_name)
    delay_steps = LaunchConfiguration(delay_steps_parameter_name)
    timing_logging_enabled = LaunchConfiguration(timing_logging_enabled_parameter_name)
    timing_log_stride = LaunchConfiguration(timing_log_stride_parameter_name)
    timing_output_dir = LaunchConfiguration(timing_output_dir_parameter_name)
    deadline_ratio_warn_threshold = LaunchConfiguration(
        deadline_ratio_warn_threshold_parameter_name)
    trajectory_mode = LaunchConfiguration(trajectory_mode_parameter_name)
    z_amplitude = LaunchConfiguration(z_amplitude_parameter_name)
    z_frequency_multiplier = LaunchConfiguration(z_frequency_multiplier_parameter_name)
    circle_frequency = LaunchConfiguration(circle_frequency_parameter_name)
    transition_duration = LaunchConfiguration(transition_duration_parameter_name)
    goal1_multisine_x_primary_amplitude = LaunchConfiguration(
        goal1_multisine_x_primary_amplitude_parameter_name)
    goal1_multisine_x_secondary_amplitude = LaunchConfiguration(
        goal1_multisine_x_secondary_amplitude_parameter_name)
    goal1_multisine_y_primary_amplitude = LaunchConfiguration(
        goal1_multisine_y_primary_amplitude_parameter_name)
    goal1_multisine_y_secondary_amplitude = LaunchConfiguration(
        goal1_multisine_y_secondary_amplitude_parameter_name)
    goal1_multisine_z_primary_amplitude = LaunchConfiguration(
        goal1_multisine_z_primary_amplitude_parameter_name)
    goal1_multisine_z_secondary_amplitude = LaunchConfiguration(
        goal1_multisine_z_secondary_amplitude_parameter_name)
    goal1_multisine_x_primary_frequency_multiplier = LaunchConfiguration(
        goal1_multisine_x_primary_frequency_multiplier_parameter_name)
    goal1_multisine_x_secondary_frequency_multiplier = LaunchConfiguration(
        goal1_multisine_x_secondary_frequency_multiplier_parameter_name)
    goal1_multisine_y_primary_frequency_multiplier = LaunchConfiguration(
        goal1_multisine_y_primary_frequency_multiplier_parameter_name)
    goal1_multisine_y_secondary_frequency_multiplier = LaunchConfiguration(
        goal1_multisine_y_secondary_frequency_multiplier_parameter_name)
    goal1_multisine_z_primary_frequency_multiplier = LaunchConfiguration(
        goal1_multisine_z_primary_frequency_multiplier_parameter_name)
    goal1_multisine_z_secondary_frequency_multiplier = LaunchConfiguration(
        goal1_multisine_z_secondary_frequency_multiplier_parameter_name)
    goal1_multisine_phi_x2 = LaunchConfiguration(goal1_multisine_phi_x2_parameter_name)
    goal1_multisine_phi_y1 = LaunchConfiguration(goal1_multisine_phi_y1_parameter_name)
    goal1_multisine_phi_y2 = LaunchConfiguration(goal1_multisine_phi_y2_parameter_name)
    goal1_multisine_phi_z1 = LaunchConfiguration(goal1_multisine_phi_z1_parameter_name)
    goal1_multisine_phi_z2 = LaunchConfiguration(goal1_multisine_phi_z2_parameter_name)

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
            spawn_cpp_relayer_parameter_name,
            default_value='true',
            description='Allow disabling cpp_relayer only for GOAL2 fake/sim smoke where '
                        'fake hardware lacks Franka semantic model interfaces.'),
        DeclareLaunchArgument(
            spawn_gp_server_parameter_name,
            default_value='true',
            description='Allow disabling gp_server for no-GP fake/sim smoke.'),
        DeclareLaunchArgument(
            spawn_fake_state_parameter_publisher_parameter_name,
            default_value='false',
            description='Only for GOAL2 fake/sim smoke; publishes synthetic /state_parameter '
                        'when use_fake_hardware:=true.'),
        DeclareLaunchArgument(
            control_frequency_parameter_name,
            default_value='50',
            choices=['25', '50'],
            description='GOAL2-B true controller, trajectory, and fake-state frequency in Hz.'),
        DeclareLaunchArgument(
            run_name_parameter_name,
            default_value='',
            description='Optional unique GOAL2-B run name written to CSV metadata and filename.'),
        DeclareLaunchArgument(
            data_output_dir_parameter_name,
            default_value='.',
            description='Directory for the controller data CSV.'),
        DeclareLaunchArgument(
            gp_prediction_enabled_parameter_name,
            default_value='true',
            description='Enable GP prediction path in the controller.'),
        DeclareLaunchArgument(
            gp_online_update_enabled_parameter_name,
            default_value='false',
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
            delay_steps_parameter_name,
            default_value='2',
            description='Cloud-like control-step delay; not real network cloud latency.'),
        DeclareLaunchArgument(
            timing_logging_enabled_parameter_name,
            default_value='false',
            description='Enable GOAL2 D controller timing CSV logging.'),
        DeclareLaunchArgument(
            timing_log_stride_parameter_name,
            default_value='1',
            description='Record one controller timing row every N callbacks.'),
        DeclareLaunchArgument(
            timing_output_dir_parameter_name,
            default_value='outputs/goal2d_controller_timing',
            description='Directory for GOAL2 D controller timing CSV output.'),
        DeclareLaunchArgument(
            deadline_ratio_warn_threshold_parameter_name,
            default_value='0.8',
            description='Warn in timing summary when max callback deadline ratio reaches this threshold.'),
        # Stage 3A launch defaults 保持 Stage 1 / Stage 2A 的平面圆轨迹行为。
        DeclareLaunchArgument(
            trajectory_mode_parameter_name,
            default_value='planar_circle',
            description='Trajectory mode: planar_circle, z_modulated_circle, or goal1_spatial_multisine.'),
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
        # GOAL2-B 只迁移 spatial multisine position 参数；不迁移 GOAL1 historical/orientation/q7。
        DeclareLaunchArgument(
            goal1_multisine_x_primary_amplitude_parameter_name,
            default_value='0.040',
            description='GOAL2-B spatial multisine x primary amplitude in meters.'),
        DeclareLaunchArgument(
            goal1_multisine_x_secondary_amplitude_parameter_name,
            default_value='0.012',
            description='GOAL2-B spatial multisine x secondary amplitude in meters.'),
        DeclareLaunchArgument(
            goal1_multisine_y_primary_amplitude_parameter_name,
            default_value='0.035',
            description='GOAL2-B spatial multisine y primary amplitude in meters.'),
        DeclareLaunchArgument(
            goal1_multisine_y_secondary_amplitude_parameter_name,
            default_value='0.012',
            description='GOAL2-B spatial multisine y secondary amplitude in meters.'),
        DeclareLaunchArgument(
            goal1_multisine_z_primary_amplitude_parameter_name,
            default_value='0.030',
            description='GOAL2-B spatial multisine z primary amplitude in meters.'),
        DeclareLaunchArgument(
            goal1_multisine_z_secondary_amplitude_parameter_name,
            default_value='0.010',
            description='GOAL2-B spatial multisine z secondary amplitude in meters.'),
        DeclareLaunchArgument(
            goal1_multisine_x_primary_frequency_multiplier_parameter_name,
            default_value='1.0',
            description='GOAL2-B spatial multisine x primary frequency multiplier.'),
        DeclareLaunchArgument(
            goal1_multisine_x_secondary_frequency_multiplier_parameter_name,
            default_value='2.1',
            description='GOAL2-B spatial multisine x secondary frequency multiplier.'),
        DeclareLaunchArgument(
            goal1_multisine_y_primary_frequency_multiplier_parameter_name,
            default_value='1.3',
            description='GOAL2-B spatial multisine y primary frequency multiplier.'),
        DeclareLaunchArgument(
            goal1_multisine_y_secondary_frequency_multiplier_parameter_name,
            default_value='2.4',
            description='GOAL2-B spatial multisine y secondary frequency multiplier.'),
        DeclareLaunchArgument(
            goal1_multisine_z_primary_frequency_multiplier_parameter_name,
            default_value='0.7',
            description='GOAL2-B spatial multisine z primary frequency multiplier.'),
        DeclareLaunchArgument(
            goal1_multisine_z_secondary_frequency_multiplier_parameter_name,
            default_value='1.9',
            description='GOAL2-B spatial multisine z secondary frequency multiplier.'),
        DeclareLaunchArgument(
            goal1_multisine_phi_x2_parameter_name,
            default_value='0.7',
            description='GOAL2-B spatial multisine x secondary phase in radians.'),
        DeclareLaunchArgument(
            goal1_multisine_phi_y1_parameter_name,
            default_value='0.4',
            description='GOAL2-B spatial multisine y primary phase in radians.'),
        DeclareLaunchArgument(
            goal1_multisine_phi_y2_parameter_name,
            default_value='1.3',
            description='GOAL2-B spatial multisine y secondary phase in radians.'),
        DeclareLaunchArgument(
            goal1_multisine_phi_z1_parameter_name,
            default_value='0.2',
            description='GOAL2-B spatial multisine z primary phase in radians.'),
        DeclareLaunchArgument(
            goal1_multisine_phi_z2_parameter_name,
            default_value='1.1',
            description='GOAL2-B spatial multisine z secondary phase in radians.'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([PathJoinSubstitution(
                [FindPackageShare('new_bringup'), 'launch', 'franka.launch.py'])]),
            launch_arguments={robot_ip_parameter_name: robot_ip,
                              load_gripper_parameter_name: load_gripper,
                              use_fake_hardware_parameter_name: use_fake_hardware,
                              fake_sensor_commands_parameter_name: fake_sensor_commands,
                              use_rviz_parameter_name: use_rviz,
                              control_frequency_parameter_name: control_frequency
                              }.items(),
        ),

        # GOAL2 fake/sim smoke gate: fake hardware 缺少 Franka semantic interfaces；默认 true 保持真机行为不变。
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['cpp_relayer'],
            output='screen',
            condition=IfCondition(spawn_cpp_relayer),
        ),
        Node(
            package='py_controllers',
            executable='gp_server',          # 和 setup.py 里 entry_points 名字一致
            name='gp_server',
            output='screen',
            condition=IfCondition(spawn_gp_server),
        ),
        Node(
            package='py_controllers',
            executable='goal2_fake_state_parameter_publisher',
            name='goal2_fake_state_parameter_publisher',
            output='screen',
            condition=IfCondition(spawn_fake_state_parameter_publisher),
            parameters=[{
                use_fake_hardware_parameter_name: ParameterValue(
                    use_fake_hardware, value_type=bool),
                'publish_rate_hz': ParameterValue(
                    control_frequency, value_type=float),
                'arm_id': 'panda',
            }],
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
                delay_steps_parameter_name: ParameterValue(
                    delay_steps, value_type=int),
                control_frequency_parameter_name: ParameterValue(
                    control_frequency, value_type=float),
                run_name_parameter_name: ParameterValue(run_name, value_type=str),
                data_output_dir_parameter_name: ParameterValue(data_output_dir, value_type=str),
                trajectory_mode_parameter_name: ParameterValue(trajectory_mode, value_type=str),
                timing_logging_enabled_parameter_name: ParameterValue(
                    timing_logging_enabled, value_type=bool),
                timing_log_stride_parameter_name: ParameterValue(
                    timing_log_stride, value_type=int),
                timing_output_dir_parameter_name: ParameterValue(
                    timing_output_dir, value_type=str),
                deadline_ratio_warn_threshold_parameter_name: ParameterValue(
                    deadline_ratio_warn_threshold, value_type=float),
            }]
        ),
        Node(
            package='py_controllers',
            executable='trajectory_publisher',
            name='trajectory_publisher',
            output='screen',
            parameters=[{
                control_frequency_parameter_name: ParameterValue(
                    control_frequency, value_type=float),
                trajectory_mode_parameter_name: ParameterValue(trajectory_mode, value_type=str),
                z_amplitude_parameter_name: ParameterValue(z_amplitude, value_type=float),
                z_frequency_multiplier_parameter_name: ParameterValue(
                    z_frequency_multiplier, value_type=float),
                circle_frequency_parameter_name: ParameterValue(circle_frequency, value_type=float),
                transition_duration_parameter_name: ParameterValue(
                    transition_duration, value_type=float),
                goal1_multisine_x_primary_amplitude_parameter_name: ParameterValue(
                    goal1_multisine_x_primary_amplitude, value_type=float),
                goal1_multisine_x_secondary_amplitude_parameter_name: ParameterValue(
                    goal1_multisine_x_secondary_amplitude, value_type=float),
                goal1_multisine_y_primary_amplitude_parameter_name: ParameterValue(
                    goal1_multisine_y_primary_amplitude, value_type=float),
                goal1_multisine_y_secondary_amplitude_parameter_name: ParameterValue(
                    goal1_multisine_y_secondary_amplitude, value_type=float),
                goal1_multisine_z_primary_amplitude_parameter_name: ParameterValue(
                    goal1_multisine_z_primary_amplitude, value_type=float),
                goal1_multisine_z_secondary_amplitude_parameter_name: ParameterValue(
                    goal1_multisine_z_secondary_amplitude, value_type=float),
                goal1_multisine_x_primary_frequency_multiplier_parameter_name: ParameterValue(
                    goal1_multisine_x_primary_frequency_multiplier, value_type=float),
                goal1_multisine_x_secondary_frequency_multiplier_parameter_name: ParameterValue(
                    goal1_multisine_x_secondary_frequency_multiplier, value_type=float),
                goal1_multisine_y_primary_frequency_multiplier_parameter_name: ParameterValue(
                    goal1_multisine_y_primary_frequency_multiplier, value_type=float),
                goal1_multisine_y_secondary_frequency_multiplier_parameter_name: ParameterValue(
                    goal1_multisine_y_secondary_frequency_multiplier, value_type=float),
                goal1_multisine_z_primary_frequency_multiplier_parameter_name: ParameterValue(
                    goal1_multisine_z_primary_frequency_multiplier, value_type=float),
                goal1_multisine_z_secondary_frequency_multiplier_parameter_name: ParameterValue(
                    goal1_multisine_z_secondary_frequency_multiplier, value_type=float),
                goal1_multisine_phi_x2_parameter_name: ParameterValue(
                    goal1_multisine_phi_x2, value_type=float),
                goal1_multisine_phi_y1_parameter_name: ParameterValue(
                    goal1_multisine_phi_y1, value_type=float),
                goal1_multisine_phi_y2_parameter_name: ParameterValue(
                    goal1_multisine_phi_y2, value_type=float),
                goal1_multisine_phi_z1_parameter_name: ParameterValue(
                    goal1_multisine_phi_z1, value_type=float),
                goal1_multisine_phi_z2_parameter_name: ParameterValue(
                    goal1_multisine_phi_z2, value_type=float),
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
