import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import SetEnvironmentVariable

def generate_launch_description():
    # package_dir = get_package_share_directory('endowrist_control')
    # config_file_path = os.path.join(package_dir, 'example_config', 'endoWrist.yaml')

    return LaunchDescription([
        SetEnvironmentVariable('ROS_DOMAIN_ID', '9'),
        Node(
            package='endowrist_control',
            executable='endowrist_control_node',
            output='screen',
            # parameters=[config_file_path]
            prefix=["sudo -E env \"PYTHONPATH=$PYTHONPATH\" \"LD_LIBRARY_PATH=$LD_LIBRARY_PATH\" \"PATH=$PATH\" \"USER=$USER\"  \"ROS_DOMAIN_ID=9\" \"RMW_IMPLEMENTATION=rmw_cyclonedds_cpp\" bash -c "],
            shell=True,
        )
    ])