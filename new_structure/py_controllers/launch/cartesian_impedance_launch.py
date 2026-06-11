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
    spawn_gp_server_parameter_name = 'spawn_gp_server'
    spawn_fake_state_parameter_publisher_parameter_name = 'spawn_fake_state_parameter_publisher'
    control_frequency_parameter_name = 'control_frequency'
    run_name_parameter_name = 'run_name'
    data_output_dir_parameter_name = 'data_output_dir'
    reference_mode_parameter_name = 'reference_mode'
    joint_space_command_topic_parameter_name = 'joint_space_command_topic'
    # Stage 1: frozen GP / compensation experiment 参数，默认值保持安全。
    gp_prediction_enabled_parameter_name = 'gp_prediction_enabled'
    gp_online_update_enabled_parameter_name = 'gp_online_update_enabled'
    gp_prediction_stride_parameter_name = 'gp_prediction_stride'
    gp_output_timeout_sec_parameter_name = 'gp_output_timeout_sec'
    gp_model_dir_parameter_name = 'gp_model_dir'
    gp_compensation_enabled_parameter_name = 'gp_compensation_enabled'
    gp_compensation_source_parameter_name = 'gp_compensation_source'
    gp_compensation_scale_parameter_name = 'gp_compensation_scale'
    gp_compensation_clip_nm_parameter_name = 'gp_compensation_clip_nm'
    gp_compensation_disable_joint7_parameter_name = 'gp_compensation_disable_joint7'
    delay_steps_parameter_name = 'delay_steps'
    timing_logging_enabled_parameter_name = 'timing_logging_enabled'
    timing_log_stride_parameter_name = 'timing_log_stride'
    timing_output_dir_parameter_name = 'timing_output_dir'
    deadline_ratio_warn_threshold_parameter_name = 'deadline_ratio_warn_threshold'
    gp_historical_db_enabled_parameter_name = 'gp_historical_db_enabled'
    gp_historical_db_path_parameter_name = 'gp_historical_db_path'
    gp_historical_db_k_parameter_name = 'gp_historical_db_k'
    gp_historical_db_q_scale_parameter_name = 'gp_historical_db_q_scale'
    gp_historical_db_dq_scale_parameter_name = 'gp_historical_db_dq_scale'
    gp_historical_db_max_distance_parameter_name = 'gp_historical_db_max_distance'
    gp_historical_db_query_stride_parameter_name = 'gp_historical_db_query_stride'
    gp_historical_db_disable_online_parameter_name = (
        'gp_historical_db_disable_when_online_update'
    )
    gp_historical_db_fallback_source_parameter_name = 'gp_historical_db_fallback_source'
    gp_historical_soft_shadow_enabled_parameter_name = (
        'gp_historical_soft_shadow_enabled'
    )
    gp_historical_soft_alpha_parameter_name = 'gp_historical_soft_alpha'
    gp_historical_soft_distance_threshold_parameter_name = (
        'gp_historical_soft_distance_threshold'
    )
    gp_historical_soft_online_scale_parameter_name = (
        'gp_historical_soft_online_scale'
    )
    gp_historical_soft_non_online_scale_parameter_name = (
        'gp_historical_soft_non_online_scale'
    )
    future_trajectory_request_stride_parameter_name = 'future_trajectory_request_stride'
    # Stage 3A trajectory 参数默认保持 planar_circle，只有显式传参才启用 z modulation。
    trajectory_mode_parameter_name = 'trajectory_mode'
    z_amplitude_parameter_name = 'z_amplitude'
    z_frequency_multiplier_parameter_name = 'z_frequency_multiplier'
    circle_frequency_parameter_name = 'circle_frequency'
    transition_duration_parameter_name = 'transition_duration'

    robot_ip = LaunchConfiguration(robot_ip_parameter_name)
    load_gripper = LaunchConfiguration(load_gripper_parameter_name)
    use_fake_hardware = LaunchConfiguration(use_fake_hardware_parameter_name)
    fake_sensor_commands = LaunchConfiguration(fake_sensor_commands_parameter_name)
    use_rviz = LaunchConfiguration(use_rviz_parameter_name)
    spawn_gp_server = LaunchConfiguration(spawn_gp_server_parameter_name)
    control_frequency = LaunchConfiguration(control_frequency_parameter_name)
    run_name = LaunchConfiguration(run_name_parameter_name)
    data_output_dir = LaunchConfiguration(data_output_dir_parameter_name)
    reference_mode = LaunchConfiguration(reference_mode_parameter_name)
    joint_space_command_topic = LaunchConfiguration(joint_space_command_topic_parameter_name)
    gp_prediction_enabled = LaunchConfiguration(gp_prediction_enabled_parameter_name)
    gp_online_update_enabled = LaunchConfiguration(gp_online_update_enabled_parameter_name)
    gp_prediction_stride = LaunchConfiguration(gp_prediction_stride_parameter_name)
    gp_output_timeout_sec = LaunchConfiguration(gp_output_timeout_sec_parameter_name)
    gp_model_dir = LaunchConfiguration(gp_model_dir_parameter_name)
    gp_compensation_enabled = LaunchConfiguration(gp_compensation_enabled_parameter_name)
    gp_compensation_source = LaunchConfiguration(gp_compensation_source_parameter_name)
    gp_compensation_scale = LaunchConfiguration(gp_compensation_scale_parameter_name)
    gp_compensation_clip_nm = LaunchConfiguration(gp_compensation_clip_nm_parameter_name)
    gp_compensation_disable_joint7 = LaunchConfiguration(
        gp_compensation_disable_joint7_parameter_name
    )
    delay_steps = LaunchConfiguration(delay_steps_parameter_name)
    timing_logging_enabled = LaunchConfiguration(timing_logging_enabled_parameter_name)
    timing_log_stride = LaunchConfiguration(timing_log_stride_parameter_name)
    timing_output_dir = LaunchConfiguration(timing_output_dir_parameter_name)
    deadline_ratio_warn_threshold = LaunchConfiguration(
        deadline_ratio_warn_threshold_parameter_name
    )
    gp_historical_db_enabled = LaunchConfiguration(gp_historical_db_enabled_parameter_name)
    gp_historical_db_path = LaunchConfiguration(gp_historical_db_path_parameter_name)
    gp_historical_db_k = LaunchConfiguration(gp_historical_db_k_parameter_name)
    gp_historical_db_q_scale = LaunchConfiguration(gp_historical_db_q_scale_parameter_name)
    gp_historical_db_dq_scale = LaunchConfiguration(gp_historical_db_dq_scale_parameter_name)
    gp_historical_db_max_distance = LaunchConfiguration(
        gp_historical_db_max_distance_parameter_name
    )
    gp_historical_db_query_stride = LaunchConfiguration(
        gp_historical_db_query_stride_parameter_name
    )
    gp_historical_db_disable_online = LaunchConfiguration(
        gp_historical_db_disable_online_parameter_name
    )
    gp_historical_db_fallback_source = LaunchConfiguration(
        gp_historical_db_fallback_source_parameter_name
    )
    gp_historical_soft_shadow_enabled = LaunchConfiguration(
        gp_historical_soft_shadow_enabled_parameter_name
    )
    gp_historical_soft_alpha = LaunchConfiguration(
        gp_historical_soft_alpha_parameter_name
    )
    gp_historical_soft_distance_threshold = LaunchConfiguration(
        gp_historical_soft_distance_threshold_parameter_name
    )
    gp_historical_soft_online_scale = LaunchConfiguration(
        gp_historical_soft_online_scale_parameter_name
    )
    gp_historical_soft_non_online_scale = LaunchConfiguration(
        gp_historical_soft_non_online_scale_parameter_name
    )
    future_trajectory_request_stride = LaunchConfiguration(
        future_trajectory_request_stride_parameter_name
    )
    trajectory_mode = LaunchConfiguration(trajectory_mode_parameter_name)
    z_amplitude = LaunchConfiguration(z_amplitude_parameter_name)
    z_frequency_multiplier = LaunchConfiguration(z_frequency_multiplier_parameter_name)
    circle_frequency = LaunchConfiguration(circle_frequency_parameter_name)
    transition_duration = LaunchConfiguration(transition_duration_parameter_name)

    return LaunchDescription([
        DeclareLaunchArgument(
            robot_ip_parameter_name,
            description='Hostname or IP address of the robot.'),
        DeclareLaunchArgument(
            use_rviz_parameter_name,
            default_value='false',
            description='Visualize the robot in Rviz'),
        DeclareLaunchArgument(
            spawn_gp_server_parameter_name,
            default_value='false',
            description=(
                'Start standalone gp_server only when explicitly requested; '
                'default false for GOAL1 real shadow validation.'
            )),
        DeclareLaunchArgument(
            spawn_fake_state_parameter_publisher_parameter_name,
            default_value='false',
            description=(
                'Reserved for GOAL2 fake/sim smoke; current branch has no '
                'goal2 fake state publisher executable, so this launch does '
                'not start an extra node.'
            )),
        DeclareLaunchArgument(
            control_frequency_parameter_name,
            default_value='50',
            description='Controller manager, trajectory, and controller frequency in Hz.'),
        DeclareLaunchArgument(
            run_name_parameter_name,
            default_value='',
            description='Optional run name written to controller CSV metadata and filename.'),
        DeclareLaunchArgument(
            data_output_dir_parameter_name,
            default_value='.',
            description='Directory for controller data CSV output.'),
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
            reference_mode_parameter_name,
            default_value='cartesian',
            description='Controller reference mode: cartesian or joint.'),
        DeclareLaunchArgument(
            joint_space_command_topic_parameter_name,
            default_value='/joint_space_command',
            description='JointSpaceCommand topic used only when reference_mode:=joint.'),
        DeclareLaunchArgument(
            gp_prediction_enabled_parameter_name,
            default_value='true',
            description='Enable GP prediction path in the controller.'),
        DeclareLaunchArgument(
            gp_online_update_enabled_parameter_name,
            default_value='true',
            description='Enable online GP model updates in the controller.'),
        DeclareLaunchArgument(
            gp_prediction_stride_parameter_name,
            default_value='1',
            description=(
                'Run heavy GP predict/update once every N controller callbacks. '
                '1 preserves per-callback GP updates.'
            )),
        DeclareLaunchArgument(
            gp_output_timeout_sec_parameter_name,
            default_value='0.5',
            description='Maximum age in seconds for held GP output to enter active compensation.'),
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
            description='GP compensation source: local, cloud, combined, or hist_db.'),
        DeclareLaunchArgument(
            gp_compensation_scale_parameter_name,
            default_value='0.1',
            description='Scale applied to GP torque compensation before clipping.'),
        DeclareLaunchArgument(
            gp_compensation_clip_nm_parameter_name,
            default_value='0.5',
            description='Per-joint GP compensation clip in Nm.'),
        DeclareLaunchArgument(
            gp_compensation_disable_joint7_parameter_name,
            default_value='false',
            description='Disable active GP applied torque on joint7 only when explicitly true.'),
        DeclareLaunchArgument(
            delay_steps_parameter_name,
            default_value='0',
            description='Cloud-like control-step delay; not real network cloud latency.'),
        DeclareLaunchArgument(
            timing_logging_enabled_parameter_name,
            default_value='false',
            description='Enable controller timing CSV logging.'),
        DeclareLaunchArgument(
            timing_log_stride_parameter_name,
            default_value='1',
            description='Record one controller timing row every N callbacks.'),
        DeclareLaunchArgument(
            timing_output_dir_parameter_name,
            default_value='outputs/goal12_controller_timing',
            description='Directory for controller timing CSV output.'),
        DeclareLaunchArgument(
            deadline_ratio_warn_threshold_parameter_name,
            default_value='0.8',
            description='Warn in timing summary when max callback deadline ratio reaches this threshold.'),
        DeclareLaunchArgument(
            gp_historical_db_enabled_parameter_name,
            default_value='false',
            description=(
                'Enable persistent historical DB CSV query only when explicitly '
                'requested; active torque uses it only with gp_compensation_source:=hist_db.'
            )),
        DeclareLaunchArgument(
            gp_historical_db_path_parameter_name,
            default_value='',
            description=(
                'Persistent historical DB .npz path for shadow logging or explicit hist_db source; '
                'empty keeps the DB unavailable.'
            )),
        DeclareLaunchArgument(
            gp_historical_db_k_parameter_name,
            default_value='25',
            description='Persistent historical DB top-k size for validation.'),
        DeclareLaunchArgument(
            gp_historical_db_q_scale_parameter_name,
            default_value='0.1',
            description='Persistent historical DB joint-position distance scale.'),
        DeclareLaunchArgument(
            gp_historical_db_dq_scale_parameter_name,
            default_value='0.1',
            description='Persistent historical DB joint-velocity distance scale.'),
        DeclareLaunchArgument(
            gp_historical_db_max_distance_parameter_name,
            default_value='1.0',
            description='Persistent historical DB nearest-distance hard gate.'),
        DeclareLaunchArgument(
            gp_historical_db_query_stride_parameter_name,
            default_value='1',
            description=(
                'Query stride for persistent historical DB KNN lookup. '
                '1 preserves per-callback queries; larger values reuse the previous query result '
                'between lookup callbacks to reduce load.'
            ),
        ),
        DeclareLaunchArgument(
            gp_historical_db_disable_online_parameter_name,
            default_value='true',
            description=(
                'Keep persistent historical DB unavailable while GP online update is '
                'enabled; hist_db active source therefore uses zero fallback in that mode.'
            )),
        DeclareLaunchArgument(
            gp_historical_db_fallback_source_parameter_name,
            default_value='cloud',
            description='Shadow-only fallback source: none, local, cloud, or combined.'),
        DeclareLaunchArgument(
            gp_historical_soft_shadow_enabled_parameter_name,
            default_value='false',
            description=(
                'Enable GOAL1 historical soft-weight CSV shadow logging only when '
                'explicitly requested for validation; no active torque compensation.'
            )),
        DeclareLaunchArgument(
            gp_historical_soft_alpha_parameter_name,
            default_value='1.0',
            description=(
                'GOAL1 historical soft-shadow alpha for validation logging only; '
                'no active torque compensation.'
            )),
        DeclareLaunchArgument(
            gp_historical_soft_distance_threshold_parameter_name,
            default_value='0.2',
            description=(
                'GOAL1 historical soft-shadow nearest-distance threshold for '
                'validation logging only; no active torque compensation.'
            )),
        DeclareLaunchArgument(
            gp_historical_soft_online_scale_parameter_name,
            default_value='0.02',
            description=(
                'GOAL1 historical soft-shadow online-mode historical scale for '
                'validation logging only; no active torque compensation.'
            )),
        DeclareLaunchArgument(
            gp_historical_soft_non_online_scale_parameter_name,
            default_value='1.0',
            description=(
                'GOAL1 historical soft-shadow non-online historical scale for '
                'validation logging only; no active torque compensation.'
            )),
        DeclareLaunchArgument(
            future_trajectory_request_stride_parameter_name,
            default_value='1',
            description=(
                'Request future task-space trajectory once every N callbacks. '
                '1 preserves per-callback request attempts; pending requests are not stacked.'
            )),
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
            condition=IfCondition(spawn_gp_server),
        ),

        Node(
            package='py_controllers',
            executable='cartesian_impedance',
            name='cartesian_impedance',
            output='screen',
            parameters=[{
                reference_mode_parameter_name: ParameterValue(reference_mode, value_type=str),
                joint_space_command_topic_parameter_name: joint_space_command_topic,
                gp_prediction_enabled_parameter_name: gp_prediction_enabled,
                gp_online_update_enabled_parameter_name: gp_online_update_enabled,
                gp_prediction_stride_parameter_name: ParameterValue(
                    gp_prediction_stride, value_type=int),
                gp_output_timeout_sec_parameter_name: ParameterValue(
                    gp_output_timeout_sec, value_type=float),
                gp_model_dir_parameter_name: gp_model_dir,
                gp_compensation_enabled_parameter_name: gp_compensation_enabled,
                gp_compensation_source_parameter_name: gp_compensation_source,
                gp_compensation_scale_parameter_name: gp_compensation_scale,
                gp_compensation_clip_nm_parameter_name: gp_compensation_clip_nm,
                gp_compensation_disable_joint7_parameter_name: gp_compensation_disable_joint7,
                delay_steps_parameter_name: ParameterValue(
                    delay_steps, value_type=int),
                control_frequency_parameter_name: ParameterValue(
                    control_frequency, value_type=float),
                run_name_parameter_name: ParameterValue(run_name, value_type=str),
                data_output_dir_parameter_name: ParameterValue(data_output_dir, value_type=str),
                timing_logging_enabled_parameter_name: ParameterValue(
                    timing_logging_enabled, value_type=bool),
                timing_log_stride_parameter_name: ParameterValue(
                    timing_log_stride, value_type=int),
                timing_output_dir_parameter_name: ParameterValue(
                    timing_output_dir, value_type=str),
                deadline_ratio_warn_threshold_parameter_name: ParameterValue(
                    deadline_ratio_warn_threshold, value_type=float),
                gp_historical_db_enabled_parameter_name: ParameterValue(
                    gp_historical_db_enabled,
                    value_type=bool),
                gp_historical_db_path_parameter_name: ParameterValue(
                    gp_historical_db_path,
                    value_type=str),
                gp_historical_db_k_parameter_name: ParameterValue(
                    gp_historical_db_k,
                    value_type=int),
                gp_historical_db_q_scale_parameter_name: ParameterValue(
                    gp_historical_db_q_scale,
                    value_type=float),
                gp_historical_db_dq_scale_parameter_name: ParameterValue(
                    gp_historical_db_dq_scale,
                    value_type=float),
                gp_historical_db_max_distance_parameter_name: ParameterValue(
                    gp_historical_db_max_distance,
                    value_type=float),
                gp_historical_db_query_stride_parameter_name: ParameterValue(
                    gp_historical_db_query_stride,
                    value_type=int),
                gp_historical_db_disable_online_parameter_name: ParameterValue(
                    gp_historical_db_disable_online,
                    value_type=bool),
                gp_historical_db_fallback_source_parameter_name: ParameterValue(
                    gp_historical_db_fallback_source,
                    value_type=str),
                gp_historical_soft_shadow_enabled_parameter_name: ParameterValue(
                    gp_historical_soft_shadow_enabled,
                    value_type=bool),
                gp_historical_soft_alpha_parameter_name: ParameterValue(
                    gp_historical_soft_alpha,
                    value_type=float),
                gp_historical_soft_distance_threshold_parameter_name: ParameterValue(
                    gp_historical_soft_distance_threshold,
                    value_type=float),
                gp_historical_soft_online_scale_parameter_name: ParameterValue(
                    gp_historical_soft_online_scale,
                    value_type=float),
                gp_historical_soft_non_online_scale_parameter_name: ParameterValue(
                    gp_historical_soft_non_online_scale,
                    value_type=float),
                future_trajectory_request_stride_parameter_name: ParameterValue(
                    future_trajectory_request_stride,
                    value_type=int),
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
