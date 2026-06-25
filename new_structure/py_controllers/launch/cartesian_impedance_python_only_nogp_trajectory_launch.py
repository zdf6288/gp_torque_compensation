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

    controller_node = Node(
        package='py_controllers',
        executable='cartesian_impedance',
        name='cartesian_impedance',
        output='screen',
        on_exit=Shutdown(
            reason='cartesian_impedance exited; stopping Python-only no-GP launch'
        ),
        parameters=[{
            'effort_output_mode': 'active',
            'reference_mode': 'cartesian',
            'gp_prediction_enabled': False,
            'gp_online_update_enabled': False,
            'gp_compensation_enabled': False,
            'gp_compensation_source': 'local',
            'gp_compensation_scale': 0.0,
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
            reason='trajectory_publisher exited; stopping Python-only no-GP launch'
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
            default_value='MANUAL_F50_NOGP_STARTUP_TRAJ_TEST',
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
        LogInfo(
            msg=(
                'PYTHON-ONLY NO-GP TRAJECTORY: expects IMPL bringup and '
                'cpp_relayer to be active already. Starts only '
                'cartesian_impedance and trajectory_publisher. GP prediction, '
                'online update, and compensation are disabled; fixed-start '
                'anchoring remains disabled.'
            )
        ),
        controller_node,
        trajectory_node,
    ])
