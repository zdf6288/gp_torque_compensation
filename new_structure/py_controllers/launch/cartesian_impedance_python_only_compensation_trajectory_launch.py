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
            'gp_historical_shadow_enabled': False,
            'gp_historical_db_enabled': False,
            'gp_historical_db_preflight_enabled': False,
            'gp_historical_db_preflight_required': False,
            'gp_triple_combined_base_shadow_enabled': False,
            'gp_historical_soft_shadow_enabled': False,
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
