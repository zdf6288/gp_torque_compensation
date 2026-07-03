#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    run_name = LaunchConfiguration('run_name')
    data_output_dir = LaunchConfiguration('data_output_dir')
    csv_output_profile = LaunchConfiguration('csv_output_profile')
    trajectory_mode = LaunchConfiguration('trajectory_mode')
    circle_frequency = LaunchConfiguration('circle_frequency')
    control_frequency = LaunchConfiguration('control_frequency')
    trajectory_publish_rate = LaunchConfiguration('trajectory_publish_rate')
    state_parameter_publish_rate = LaunchConfiguration(
        'state_parameter_publish_rate'
    )
    transition_duration = LaunchConfiguration('transition_duration')
    torque_rate_limit_nm_per_s = LaunchConfiguration(
        'torque_rate_limit_nm_per_s'
    )
    gp_model_dir = LaunchConfiguration('gp_model_dir')
    gp_compensation_source = LaunchConfiguration('gp_compensation_source')
    gp_compensation_scale = LaunchConfiguration('gp_compensation_scale')
    gp_compensation_clip_nm = LaunchConfiguration('gp_compensation_clip_nm')
    gp_compensation_disable_joint7 = LaunchConfiguration(
        'gp_compensation_disable_joint7'
    )
    delay_steps = LaunchConfiguration('delay_steps')
    gp_prediction_stride = LaunchConfiguration('gp_prediction_stride')
    future_trajectory_request_stride = LaunchConfiguration(
        'future_trajectory_request_stride'
    )
    gp_output_timeout_sec = LaunchConfiguration('gp_output_timeout_sec')
    gp_historical_db_enabled = LaunchConfiguration('gp_historical_db_enabled')
    gp_historical_db_path = LaunchConfiguration('gp_historical_db_path')
    gp_historical_db_k = LaunchConfiguration('gp_historical_db_k')
    gp_historical_db_q_scale = LaunchConfiguration('gp_historical_db_q_scale')
    gp_historical_db_dq_scale = LaunchConfiguration('gp_historical_db_dq_scale')
    gp_historical_db_max_distance = LaunchConfiguration(
        'gp_historical_db_max_distance'
    )
    gp_historical_db_query_stride = LaunchConfiguration(
        'gp_historical_db_query_stride'
    )
    gp_historical_db_disable_when_online_update = LaunchConfiguration(
        'gp_historical_db_disable_when_online_update'
    )
    gp_historical_db_fallback_source = LaunchConfiguration(
        'gp_historical_db_fallback_source'
    )
    gp_historical_db_preflight_enabled = LaunchConfiguration(
        'gp_historical_db_preflight_enabled'
    )
    gp_historical_db_preflight_required = LaunchConfiguration(
        'gp_historical_db_preflight_required'
    )
    gp_historical_db_preflight_mode = LaunchConfiguration(
        'gp_historical_db_preflight_mode'
    )
    gp_historical_db_preflight_duration_sec = LaunchConfiguration(
        'gp_historical_db_preflight_duration_sec'
    )
    gp_historical_db_preflight_min_samples = LaunchConfiguration(
        'gp_historical_db_preflight_min_samples'
    )
    gp_historical_db_preflight_min_pass_ratio = LaunchConfiguration(
        'gp_historical_db_preflight_min_pass_ratio'
    )
    gp_historical_db_preflight_p95_max_distance = LaunchConfiguration(
        'gp_historical_db_preflight_p95_max_distance'
    )
    gp_historical_db_preflight_max_distance = LaunchConfiguration(
        'gp_historical_db_preflight_max_distance'
    )
    gp_disable_silent_hist_fallback = LaunchConfiguration(
        'gp_disable_silent_hist_fallback'
    )
    gp_triple_weight_mode = LaunchConfiguration('gp_triple_weight_mode')
    gp_triple_weight_local = LaunchConfiguration('gp_triple_weight_local')
    gp_triple_weight_cloud = LaunchConfiguration('gp_triple_weight_cloud')
    gp_triple_weight_hist = LaunchConfiguration('gp_triple_weight_hist')
    gp_triple_weight_normalize = LaunchConfiguration(
        'gp_triple_weight_normalize'
    )
    gp_triple_rmse_local = LaunchConfiguration('gp_triple_rmse_local')
    gp_triple_rmse_cloud = LaunchConfiguration('gp_triple_rmse_cloud')
    gp_triple_rmse_hist = LaunchConfiguration('gp_triple_rmse_hist')
    gp_triple_hist_distance_scale = LaunchConfiguration(
        'gp_triple_hist_distance_scale'
    )
    gp_triple_hist_distance_power = LaunchConfiguration(
        'gp_triple_hist_distance_power'
    )
    gp_triple_hist_weight_cap = LaunchConfiguration(
        'gp_triple_hist_weight_cap'
    )
    gp_triple_hist_min_weight = LaunchConfiguration(
        'gp_triple_hist_min_weight'
    )
    gp_triple_min_weight_local = LaunchConfiguration(
        'gp_triple_min_weight_local'
    )
    gp_triple_min_weight_cloud = LaunchConfiguration(
        'gp_triple_min_weight_cloud'
    )
    gp_triple_require_hist_available = LaunchConfiguration(
        'gp_triple_require_hist_available'
    )
    gp_triple_fallback_source = LaunchConfiguration(
        'gp_triple_fallback_source'
    )
    gp_triple_debug_safety_log_enabled = LaunchConfiguration(
        'gp_triple_debug_safety_log_enabled'
    )
    gp_triple_debug_safety_log_first_n = LaunchConfiguration(
        'gp_triple_debug_safety_log_first_n'
    )
    gp_triple_combined_base_shadow_enabled = LaunchConfiguration(
        'gp_triple_combined_base_shadow_enabled'
    )
    gp_triple_combined_base_hist_weight_cap = LaunchConfiguration(
        'gp_triple_combined_base_hist_weight_cap'
    )
    gp_triple_combined_base_hist_weight_ramp_sec = LaunchConfiguration(
        'gp_triple_combined_base_hist_weight_ramp_sec'
    )
    gp_historical_shadow_enabled = LaunchConfiguration(
        'gp_historical_shadow_enabled'
    )
    gp_historical_source_mode = LaunchConfiguration(
        'gp_historical_source_mode'
    )
    gp_historical_soft_shadow_enabled = LaunchConfiguration(
        'gp_historical_soft_shadow_enabled'
    )
    gp_historical_soft_alpha = LaunchConfiguration(
        'gp_historical_soft_alpha'
    )
    gp_historical_soft_distance_threshold = LaunchConfiguration(
        'gp_historical_soft_distance_threshold'
    )
    gp_historical_soft_online_scale = LaunchConfiguration(
        'gp_historical_soft_online_scale'
    )
    gp_historical_soft_non_online_scale = LaunchConfiguration(
        'gp_historical_soft_non_online_scale'
    )

    controller_node = Node(
        package='py_controllers',
        executable='cartesian_impedance',
        name='cartesian_impedance',
        output='screen',
        on_exit=Shutdown(
            reason=(
                'cartesian_impedance exited; stopping Python-only GP '
                'compensation launch'
            )
        ),
        parameters=[{
            'effort_output_mode': 'active',
            'reference_mode': 'cartesian',
            'gp_prediction_enabled': True,
            'gp_online_update_enabled': True,
            'gp_compensation_enabled': True,
            'gp_compensation_source': ParameterValue(
                gp_compensation_source, value_type=str
            ),
            'gp_compensation_scale': ParameterValue(
                gp_compensation_scale, value_type=float
            ),
            'gp_compensation_clip_nm': ParameterValue(
                gp_compensation_clip_nm, value_type=float
            ),
            'gp_compensation_disable_joint7': ParameterValue(
                gp_compensation_disable_joint7, value_type=bool
            ),
            'gp_model_dir': ParameterValue(gp_model_dir, value_type=str),
            'delay_steps': ParameterValue(delay_steps, value_type=int),
            'gp_prediction_stride': ParameterValue(
                gp_prediction_stride, value_type=int
            ),
            'future_trajectory_request_stride': ParameterValue(
                future_trajectory_request_stride, value_type=int
            ),
            'gp_output_timeout_sec': ParameterValue(
                gp_output_timeout_sec, value_type=float
            ),
            'gp_historical_shadow_enabled': ParameterValue(
                gp_historical_shadow_enabled, value_type=bool
            ),
            'gp_historical_source_mode': ParameterValue(
                gp_historical_source_mode, value_type=str
            ),
            'gp_historical_db_enabled': ParameterValue(
                gp_historical_db_enabled, value_type=bool
            ),
            'gp_historical_db_path': ParameterValue(
                gp_historical_db_path, value_type=str
            ),
            'gp_historical_db_k': ParameterValue(
                gp_historical_db_k, value_type=int
            ),
            'gp_historical_db_q_scale': ParameterValue(
                gp_historical_db_q_scale, value_type=float
            ),
            'gp_historical_db_dq_scale': ParameterValue(
                gp_historical_db_dq_scale, value_type=float
            ),
            'gp_historical_db_max_distance': ParameterValue(
                gp_historical_db_max_distance, value_type=float
            ),
            'gp_historical_db_query_stride': ParameterValue(
                gp_historical_db_query_stride, value_type=int
            ),
            'gp_historical_db_disable_when_online_update': ParameterValue(
                gp_historical_db_disable_when_online_update, value_type=bool
            ),
            'gp_historical_db_fallback_source': ParameterValue(
                gp_historical_db_fallback_source, value_type=str
            ),
            'gp_historical_db_preflight_enabled': ParameterValue(
                gp_historical_db_preflight_enabled, value_type=bool
            ),
            'gp_historical_db_preflight_required': ParameterValue(
                gp_historical_db_preflight_required, value_type=bool
            ),
            'gp_historical_db_preflight_mode': ParameterValue(
                gp_historical_db_preflight_mode, value_type=str
            ),
            'gp_historical_db_preflight_duration_sec': ParameterValue(
                gp_historical_db_preflight_duration_sec, value_type=float
            ),
            'gp_historical_db_preflight_min_samples': ParameterValue(
                gp_historical_db_preflight_min_samples, value_type=int
            ),
            'gp_historical_db_preflight_min_pass_ratio': ParameterValue(
                gp_historical_db_preflight_min_pass_ratio, value_type=float
            ),
            'gp_historical_db_preflight_p95_max_distance': ParameterValue(
                gp_historical_db_preflight_p95_max_distance, value_type=float
            ),
            'gp_historical_db_preflight_max_distance': ParameterValue(
                gp_historical_db_preflight_max_distance, value_type=float
            ),
            'gp_disable_silent_hist_fallback': ParameterValue(
                gp_disable_silent_hist_fallback, value_type=bool
            ),
            'gp_triple_weight_mode': ParameterValue(
                gp_triple_weight_mode, value_type=str
            ),
            'gp_triple_weight_local': ParameterValue(
                gp_triple_weight_local, value_type=float
            ),
            'gp_triple_weight_cloud': ParameterValue(
                gp_triple_weight_cloud, value_type=float
            ),
            'gp_triple_weight_hist': ParameterValue(
                gp_triple_weight_hist, value_type=float
            ),
            'gp_triple_weight_normalize': ParameterValue(
                gp_triple_weight_normalize, value_type=bool
            ),
            'gp_triple_rmse_local': ParameterValue(
                gp_triple_rmse_local, value_type=float
            ),
            'gp_triple_rmse_cloud': ParameterValue(
                gp_triple_rmse_cloud, value_type=float
            ),
            'gp_triple_rmse_hist': ParameterValue(
                gp_triple_rmse_hist, value_type=float
            ),
            'gp_triple_hist_distance_scale': ParameterValue(
                gp_triple_hist_distance_scale, value_type=float
            ),
            'gp_triple_hist_distance_power': ParameterValue(
                gp_triple_hist_distance_power, value_type=float
            ),
            'gp_triple_hist_weight_cap': ParameterValue(
                gp_triple_hist_weight_cap, value_type=float
            ),
            'gp_triple_hist_min_weight': ParameterValue(
                gp_triple_hist_min_weight, value_type=float
            ),
            'gp_triple_min_weight_local': ParameterValue(
                gp_triple_min_weight_local, value_type=float
            ),
            'gp_triple_min_weight_cloud': ParameterValue(
                gp_triple_min_weight_cloud, value_type=float
            ),
            'gp_triple_require_hist_available': ParameterValue(
                gp_triple_require_hist_available, value_type=bool
            ),
            'gp_triple_fallback_source': ParameterValue(
                gp_triple_fallback_source, value_type=str
            ),
            'gp_triple_debug_safety_log_enabled': ParameterValue(
                gp_triple_debug_safety_log_enabled, value_type=bool
            ),
            'gp_triple_debug_safety_log_first_n': ParameterValue(
                gp_triple_debug_safety_log_first_n, value_type=int
            ),
            'gp_triple_combined_base_shadow_enabled': ParameterValue(
                gp_triple_combined_base_shadow_enabled, value_type=bool
            ),
            'gp_triple_combined_base_hist_weight_cap': ParameterValue(
                gp_triple_combined_base_hist_weight_cap, value_type=float
            ),
            'gp_triple_combined_base_hist_weight_ramp_sec': ParameterValue(
                gp_triple_combined_base_hist_weight_ramp_sec, value_type=float
            ),
            'gp_historical_soft_shadow_enabled': ParameterValue(
                gp_historical_soft_shadow_enabled, value_type=bool
            ),
            'gp_historical_soft_alpha': ParameterValue(
                gp_historical_soft_alpha, value_type=float
            ),
            'gp_historical_soft_distance_threshold': ParameterValue(
                gp_historical_soft_distance_threshold, value_type=float
            ),
            'gp_historical_soft_online_scale': ParameterValue(
                gp_historical_soft_online_scale, value_type=float
            ),
            'gp_historical_soft_non_online_scale': ParameterValue(
                gp_historical_soft_non_online_scale, value_type=float
            ),
            'timing_logging_enabled': False,
            'control_frequency': ParameterValue(
                control_frequency, value_type=float
            ),
            'trajectory_publish_rate': ParameterValue(
                trajectory_publish_rate, value_type=float
            ),
            'state_parameter_publish_rate': ParameterValue(
                state_parameter_publish_rate, value_type=float
            ),
            'trajectory_mode': ParameterValue(
                trajectory_mode, value_type=str
            ),
            'circle_frequency': ParameterValue(
                circle_frequency, value_type=float
            ),
            'transition_duration': ParameterValue(
                transition_duration, value_type=float
            ),
            'run_name': ParameterValue(run_name, value_type=str),
            'data_output_dir': ParameterValue(
                data_output_dir, value_type=str
            ),
            'csv_output_profile': ParameterValue(
                csv_output_profile, value_type=str
            ),
            'startup_distance_refuse_enabled': False,
            'startup_torque_clip_nm': 10.0,
            'startup_torque_rate_limit_from_zero': True,
            'torque_rate_limit_enabled': True,
            'torque_rate_limit_nm_per_s': ParameterValue(
                torque_rate_limit_nm_per_s, value_type=float
            ),
        }],
    )

    trajectory_node = Node(
        package='py_controllers',
        executable='trajectory_publisher',
        name='trajectory_publisher',
        output='screen',
        on_exit=Shutdown(
            reason=(
                'trajectory_publisher exited; stopping Python-only GP '
                'compensation launch'
            )
        ),
        parameters=[{
            'control_frequency': ParameterValue(
                control_frequency, value_type=float
            ),
            'trajectory_publish_rate': ParameterValue(
                trajectory_publish_rate, value_type=float
            ),
            'trajectory_mode': ParameterValue(
                trajectory_mode, value_type=str
            ),
            'circle_frequency': ParameterValue(
                circle_frequency, value_type=float
            ),
            'transition_duration': ParameterValue(
                transition_duration, value_type=float
            ),
            # Preserve the formal fixed-start comparison. Enabling this for
            # goal1_spatial_multisine would shift the center to the measured
            # pose after /joint_position_adjust is called.
            'anchor_trajectory_start_to_current_pose': False,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'run_name',
            default_value='MANUAL_F50_GP_COMPENSATION_STARTUP_TRAJ_TEST',
            description='Controller CSV run name.',
        ),
        DeclareLaunchArgument(
            'data_output_dir',
            default_value='.',
            description='Directory for controller CSV output.',
        ),
        DeclareLaunchArgument(
            'csv_output_profile',
            default_value='full',
            description='Controller CSV output profile: full or final.',
        ),
        DeclareLaunchArgument(
            'trajectory_mode',
            default_value='goal1_spatial_multisine',
            description='Trajectory mode passed to both Python nodes.',
        ),
        DeclareLaunchArgument(
            'circle_frequency',
            default_value='0.05',
            description='Base trajectory frequency in Hz.',
        ),
        DeclareLaunchArgument(
            'control_frequency',
            default_value='50',
            description='Python controller frequency metadata in Hz.',
        ),
        DeclareLaunchArgument(
            'trajectory_publish_rate',
            default_value='50',
            description='trajectory_publisher timer rate in Hz.',
        ),
        DeclareLaunchArgument(
            'state_parameter_publish_rate',
            default_value='50',
            description='Expected external cpp_relayer state publish rate in Hz.',
        ),
        DeclareLaunchArgument(
            'transition_duration',
            default_value='10.0',
            description='Fixed-start to trajectory-start transition duration in seconds.',
        ),
        DeclareLaunchArgument(
            'torque_rate_limit_nm_per_s',
            default_value='20.0',
            description='Per-joint torque slew-rate limit in Nm/s.',
        ),
        DeclareLaunchArgument(
            'gp_model_dir',
            default_value='./new_structure/gp/gp_models',
            description='Directory containing offline GP model pickle files.',
        ),
        DeclareLaunchArgument(
            'gp_compensation_source',
            default_value='local',
            description='Active GP compensation source.',
        ),
        DeclareLaunchArgument(
            'gp_compensation_scale',
            default_value='0.25',
            description='Scale applied to selected GP compensation torque.',
        ),
        DeclareLaunchArgument(
            'gp_compensation_clip_nm',
            default_value='0.5',
            description='Per-joint absolute clip for GP compensation torque in Nm.',
        ),
        DeclareLaunchArgument(
            'gp_compensation_disable_joint7',
            default_value='true',
            description='Disable active GP compensation on joint 7.',
        ),
        DeclareLaunchArgument(
            'delay_steps',
            default_value='0',
            description='Cloud-like GP delay in controller callbacks.',
        ),
        DeclareLaunchArgument(
            'gp_prediction_stride',
            default_value='5',
            description='Run GP predict/update once every N controller callbacks.',
        ),
        DeclareLaunchArgument(
            'future_trajectory_request_stride',
            default_value='5',
            description='Request /future_task_space once every N controller callbacks.',
        ),
        DeclareLaunchArgument(
            'gp_output_timeout_sec',
            default_value='0.5',
            description='Maximum local/cloud GP output age before active compensation fails closed.',
        ),
        DeclareLaunchArgument(
            'gp_historical_db_enabled',
            default_value='false',
            description='Load and query persistent historical residual DB.',
        ),
        DeclareLaunchArgument(
            'gp_historical_db_path',
            default_value='',
            description='Path to persistent historical residual DB .npz.',
        ),
        DeclareLaunchArgument(
            'gp_historical_db_k',
            default_value='25',
            description='KNN neighbor count for historical residual DB.',
        ),
        DeclareLaunchArgument(
            'gp_historical_db_q_scale',
            default_value='0.1',
            description='Joint position scale for historical DB KNN.',
        ),
        DeclareLaunchArgument(
            'gp_historical_db_dq_scale',
            default_value='0.1',
            description='Joint velocity scale for historical DB KNN.',
        ),
        DeclareLaunchArgument(
            'gp_historical_db_max_distance',
            default_value='1.0',
            description='Maximum scaled nearest-neighbor distance for hist DB availability.',
        ),
        DeclareLaunchArgument(
            'gp_historical_db_query_stride',
            default_value='1',
            description='Query historical DB once every N controller callbacks.',
        ),
        DeclareLaunchArgument(
            'gp_historical_db_disable_when_online_update',
            default_value='true',
            description='Disable hist DB availability while online GP update is enabled.',
        ),
        DeclareLaunchArgument(
            'gp_historical_db_fallback_source',
            default_value='cloud',
            description='hist_db fallback source: none, local, cloud, or combined.',
        ),
        DeclareLaunchArgument(
            'gp_historical_db_preflight_enabled',
            default_value='false',
            description='Enable historical DB distance preflight before active hist compensation.',
        ),
        DeclareLaunchArgument(
            'gp_historical_db_preflight_required',
            default_value='false',
            description='Require hist DB preflight pass before active hist compensation.',
        ),
        DeclareLaunchArgument(
            'gp_historical_db_preflight_mode',
            default_value='segment',
            description='Hist DB preflight mode: single, segment, or single_and_segment.',
        ),
        DeclareLaunchArgument(
            'gp_historical_db_preflight_duration_sec',
            default_value='5.0',
            description='Hist DB segment preflight duration.',
        ),
        DeclareLaunchArgument(
            'gp_historical_db_preflight_min_samples',
            default_value='50',
            description='Minimum hist DB preflight samples.',
        ),
        DeclareLaunchArgument(
            'gp_historical_db_preflight_min_pass_ratio',
            default_value='0.95',
            description='Minimum hist DB preflight distance-pass ratio.',
        ),
        DeclareLaunchArgument(
            'gp_historical_db_preflight_p95_max_distance',
            default_value='1.5',
            description='Maximum p95 nearest distance for hist DB preflight.',
        ),
        DeclareLaunchArgument(
            'gp_historical_db_preflight_max_distance',
            default_value='2.0',
            description='Maximum nearest distance for hist DB preflight.',
        ),
        DeclareLaunchArgument(
            'gp_disable_silent_hist_fallback',
            default_value='false',
            description='Fail closed instead of silently using hist DB fallback.',
        ),
        DeclareLaunchArgument(
            'gp_triple_weight_mode',
            default_value='inverse_rmse',
            description='Triple fusion weight mode: fixed or inverse_rmse.',
        ),
        DeclareLaunchArgument(
            'gp_triple_weight_local',
            default_value='0.10',
            description='Fixed local weight for triple fusion.',
        ),
        DeclareLaunchArgument(
            'gp_triple_weight_cloud',
            default_value='0.20',
            description='Fixed cloud weight for triple fusion.',
        ),
        DeclareLaunchArgument(
            'gp_triple_weight_hist',
            default_value='0.70',
            description='Fixed historical DB weight for triple fusion.',
        ),
        DeclareLaunchArgument(
            'gp_triple_weight_normalize',
            default_value='true',
            description='Normalize fixed triple fusion weights.',
        ),
        DeclareLaunchArgument(
            'gp_triple_rmse_local',
            default_value='0.330269',
            description='Local RMSE prior for inverse-RMSE triple weights.',
        ),
        DeclareLaunchArgument(
            'gp_triple_rmse_cloud',
            default_value='0.330278',
            description='Cloud RMSE prior for inverse-RMSE triple weights.',
        ),
        DeclareLaunchArgument(
            'gp_triple_rmse_hist',
            default_value='0.093071',
            description='Historical DB RMSE prior for inverse-RMSE triple weights.',
        ),
        DeclareLaunchArgument(
            'gp_triple_hist_distance_scale',
            default_value='2.0',
            description='Distance scale for dynamic triple hist penalty.',
        ),
        DeclareLaunchArgument(
            'gp_triple_hist_distance_power',
            default_value='2.0',
            description='Distance power for dynamic triple hist penalty.',
        ),
        DeclareLaunchArgument(
            'gp_triple_hist_weight_cap',
            default_value='0.70',
            description='Maximum hist weight in triple fusion.',
        ),
        DeclareLaunchArgument(
            'gp_triple_hist_min_weight',
            default_value='0.0',
            description='Minimum hist weight in triple fusion.',
        ),
        DeclareLaunchArgument(
            'gp_triple_min_weight_local',
            default_value='0.05',
            description='Minimum local weight in triple dynamic fusion.',
        ),
        DeclareLaunchArgument(
            'gp_triple_min_weight_cloud',
            default_value='0.05',
            description='Minimum cloud weight in triple dynamic fusion.',
        ),
        DeclareLaunchArgument(
            'gp_triple_require_hist_available',
            default_value='true',
            description='Require hist DB availability before normal triple fusion.',
        ),
        DeclareLaunchArgument(
            'gp_triple_fallback_source',
            default_value='combined',
            description='Triple fallback source: none, local, cloud, combined, or hist_db.',
        ),
        DeclareLaunchArgument(
            'gp_triple_debug_safety_log_enabled',
            default_value='true',
            description='Log first triple active-safety diagnostics.',
        ),
        DeclareLaunchArgument(
            'gp_triple_debug_safety_log_first_n',
            default_value='5',
            description='Number of triple active-safety diagnostics to log.',
        ),
        DeclareLaunchArgument(
            'gp_triple_combined_base_shadow_enabled',
            default_value='false',
            description='Enable combined-base plus hist shadow logging only.',
        ),
        DeclareLaunchArgument(
            'gp_triple_combined_base_hist_weight_cap',
            default_value='0.50',
            description='Hist weight cap for combined-base shadow.',
        ),
        DeclareLaunchArgument(
            'gp_triple_combined_base_hist_weight_ramp_sec',
            default_value='0.0',
            description='Ramp time for combined-base shadow hist weight.',
        ),
        DeclareLaunchArgument(
            'gp_historical_shadow_enabled',
            default_value='false',
            description='Enable runtime historical prediction-pool shadow logging.',
        ),
        DeclareLaunchArgument(
            'gp_historical_source_mode',
            default_value='none',
            description='Historical shadow source mode: none or local_prediction_pool.',
        ),
        DeclareLaunchArgument(
            'gp_historical_soft_shadow_enabled',
            default_value='false',
            description='Enable historical soft-fusion shadow logging only.',
        ),
        DeclareLaunchArgument(
            'gp_historical_soft_alpha',
            default_value='1.0',
            description='Historical soft-fusion distance decay alpha.',
        ),
        DeclareLaunchArgument(
            'gp_historical_soft_distance_threshold',
            default_value='0.2',
            description='Historical soft-fusion distance threshold.',
        ),
        DeclareLaunchArgument(
            'gp_historical_soft_online_scale',
            default_value='0.02',
            description='Historical soft-fusion hist scale during online update.',
        ),
        DeclareLaunchArgument(
            'gp_historical_soft_non_online_scale',
            default_value='1.0',
            description='Historical soft-fusion hist scale without online update.',
        ),
        LogInfo(
            msg=(
                'PYTHON-ONLY GP COMPENSATION TRAJECTORY: expects external '
                'IMPL bringup and external cpp_relayer to be active already. '
                'Starts only cartesian_impedance and trajectory_publisher. '
                'GP prediction, online update, and torque compensation are '
                'enabled with conservative local defaults; no bringup, '
                'controller manager, or relayer activation is started here.'
            )
        ),
        controller_node,
        trajectory_node,
    ])
