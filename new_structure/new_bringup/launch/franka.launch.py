#  Copyright (c) 2021 Franka Emika GmbH
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


import math
import os
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction, Shutdown
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _positive_int_or_fallback(value_text, fallback_text, default_value=50):
    try:
        value = float(value_text)
        if math.isfinite(value) and value > 0.0:
            return int(round(value))
    except (TypeError, ValueError):
        pass

    try:
        fallback = float(fallback_text)
        if math.isfinite(fallback) and fallback > 0.0:
            return int(round(fallback))
    except (TypeError, ValueError):
        pass

    return int(default_value)


def _positive_float_or_raise(value_text, parameter_name):
    try:
        value = float(value_text)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f'{parameter_name} must be positive and finite; got {value_text!r}.'
        ) from exc

    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError(
            f'{parameter_name} must be positive and finite; got {value_text!r}.'
        )
    return value


def _bool_or_raise(value_text, parameter_name):
    normalized = str(value_text).strip().lower()
    if normalized in ('true', '1', 'yes', 'on'):
        return True
    if normalized in ('false', '0', 'no', 'off'):
        return False
    raise RuntimeError(
        f'{parameter_name} must be a boolean value; got {value_text!r}.'
    )


def _make_ros2_control_node(
        context,
        robot_description,
        franka_controllers,
        ros2_control_update_rate,
        control_frequency,
        allow_high_ros2_control_rate):
    control_rate = _positive_float_or_raise(
        control_frequency.perform(context), 'control_frequency')
    ros2_control_rate = _positive_float_or_raise(
        ros2_control_update_rate.perform(context), 'ros2_control_update_rate')
    allow_high_rate = _bool_or_raise(
        allow_high_ros2_control_rate.perform(context), 'allow_high_ros2_control_rate')

    if ros2_control_rate > control_rate and not allow_high_rate:
        raise RuntimeError(
            'High-rate communication mode blocked: '
            'ros2_control_update_rate > control_frequency requires '
            'allow_high_ros2_control_rate:=true '
            f'(control_frequency={control_rate:.3f}, '
            f'ros2_control_update_rate={ros2_control_rate:.3f}).'
        )

    update_rate = _positive_int_or_fallback(
        ros2_control_rate,
        control_rate,
    )
    with tempfile.NamedTemporaryFile(
        mode='w',
        prefix='controller_manager_update_rate_',
        suffix='.yaml',
        delete=False,
    ) as param_file:
        param_file.write('/controller_manager:\n')
        param_file.write('  ros__parameters:\n')
        param_file.write(f'    update_rate: {update_rate}\n')
        update_rate_param_file = param_file.name

    return [
        LogInfo(
            msg=(
                'WARNING: high-rate ros2_control communication is experimental.'
                if ros2_control_rate > control_rate
                else 'High-rate communication mode disabled or inactive in franka.launch.py.'
            )
        ),
        LogInfo(
            msg=(
                'controller_manager update_rate override param file after '
                f'controllers.yaml: {update_rate} Hz'
            )
        ),
        Node(
            package='controller_manager',
            executable='ros2_control_node',
            name='controller_manager',
            namespace='',
            parameters=[
                {'robot_description': robot_description},
                franka_controllers,
                update_rate_param_file,
            ],
            remappings=[('joint_states', 'franka/joint_states')],
            output={
                'stdout': 'screen',
                'stderr': 'screen',
            },
            on_exit=Shutdown(),
        ),
    ]


def generate_launch_description():
    robot_ip_parameter_name = 'robot_ip'
    load_gripper_parameter_name = 'load_gripper'
    use_fake_hardware_parameter_name = 'use_fake_hardware'
    fake_sensor_commands_parameter_name = 'fake_sensor_commands'
    use_rviz_parameter_name = 'use_rviz'
    control_frequency_parameter_name = 'control_frequency'
    allow_high_ros2_control_rate_parameter_name = 'allow_high_ros2_control_rate'
    ros2_control_update_rate_parameter_name = 'ros2_control_update_rate'

    robot_ip = LaunchConfiguration(robot_ip_parameter_name)
    load_gripper = LaunchConfiguration(load_gripper_parameter_name)
    use_fake_hardware = LaunchConfiguration(use_fake_hardware_parameter_name)
    fake_sensor_commands = LaunchConfiguration(fake_sensor_commands_parameter_name)
    use_rviz = LaunchConfiguration(use_rviz_parameter_name)
    control_frequency = LaunchConfiguration(control_frequency_parameter_name)
    allow_high_ros2_control_rate = LaunchConfiguration(
        allow_high_ros2_control_rate_parameter_name)
    ros2_control_update_rate = LaunchConfiguration(ros2_control_update_rate_parameter_name)

    franka_xacro_file = os.path.join(get_package_share_directory('franka_description'), 'robots',
                                     'panda_arm.urdf.xacro')
    robot_description = Command(
        [FindExecutable(name='xacro'), ' ', franka_xacro_file, ' hand:=', load_gripper,
         ' robot_ip:=', robot_ip, ' use_fake_hardware:=', use_fake_hardware,
         ' fake_sensor_commands:=', fake_sensor_commands])

    rviz_file = os.path.join(get_package_share_directory('franka_description'), 'rviz',
                             'visualize_franka.rviz')

    franka_controllers = PathJoinSubstitution(
        [
            FindPackageShare('new_bringup'),
            'config',
            'controllers.yaml',
        ]
    )

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
            control_frequency_parameter_name,
            default_value='50',
            description='Legacy umbrella frequency; used as fallback for ros2_control_update_rate.'),
        DeclareLaunchArgument(
            allow_high_ros2_control_rate_parameter_name,
            default_value='false',
            description='Hard opt-in for ros2_control update rates above control_frequency.'),
        DeclareLaunchArgument(
            ros2_control_update_rate_parameter_name,
            default_value=control_frequency,
            description='Controller manager update_rate override in Hz.'),
        LogInfo(
            msg=[
                'Frequency config: control_frequency=',
                control_frequency,
                ', allow_high_ros2_control_rate=',
                allow_high_ros2_control_rate,
                ', ros2_control_update_rate=',
                ros2_control_update_rate,
                ' Hz',
            ]
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description}],
        ),
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            parameters=[
                {'source_list': ['franka/joint_states', 'panda_gripper/joint_states'],
                 'rate': 30}],
        ),
        OpaqueFunction(
            function=_make_ros2_control_node,
            args=[
                robot_description,
                franka_controllers,
                ros2_control_update_rate,
                control_frequency,
                allow_high_ros2_control_rate,
            ],
        ),
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=['joint_state_broadcaster'],
            output='screen',
        ),
        
        # The franka_robot_state_broadcaster is closed to save resource for controller_manager.
        # If it is opened, there will be an error in the log like 
        # [ros2_control_node-3] [ERROR] [XXX.XXX] [controller_manager]: The update call of the following controller 
        # returned an error: 'franka_robot_state_broadcaster'
        # As far as we know this error won't affect robot operation.

        # Node(
        #     package='controller_manager',
        #     executable='spawner',
        #     arguments=['franka_robot_state_broadcaster'],
        #     output='screen',
        #     condition=UnlessCondition(use_fake_hardware),
        # ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([PathJoinSubstitution(
                [FindPackageShare('franka_gripper'), 'launch', 'gripper.launch.py'])]),
            launch_arguments={robot_ip_parameter_name: robot_ip,
                              use_fake_hardware_parameter_name: use_fake_hardware}.items(),
            condition=IfCondition(load_gripper)
        ),

        Node(package='rviz2',
             executable='rviz2',
             name='rviz2',
             arguments=['--display-config', rviz_file],
             condition=IfCondition(use_rviz)
             )

    ])
