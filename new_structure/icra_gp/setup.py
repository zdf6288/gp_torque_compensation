from setuptools import find_packages, setup

package_name = 'icra_gp'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name + '/launch', ['launch/cartesian_impedance_icra_launch.py']),
        ('share/' + package_name + '/launch', ['launch/cartesian_impedance_multi_launch.py']),
        ('share/' + package_name + '/launch', ['launch/cartesian_impedance_validation_launch.py']),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ziyang',
    maintainer_email='mengziyang168@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cartesian_impedance_icra_data = icra_gp.cartesian_impedance_icra_data:main',
            'cartesian_impedance_icra_validation = icra_gp.cartesian_impedance_icra_validation:main',
            'trajectory_publisher_icra_data = icra_gp.trajectory_publisher_icra_data:main',
            'trajectory_publisher_icra_validation = icra_gp.trajectory_publisher_icra_validation:main',
            'gp_trajectory = icra_gp.gp_trajectory:main',
            'gp_train = icra_gp.gp_train:main',
            'cartesian_impedance_multi_data = icra_gp.cartesian_impedance_multi_data:main',
            'cartesian_impedance_multi_validation = icra_gp.cartesian_impedance_multi_validation:main',
            'trajectory_publisher_multi_data = icra_gp.trajectory_publisher_multi_data:main',
            'trajectory_publisher_multi_validation = icra_gp.trajectory_publisher_multi_validation:main',
            'gp_trajectory_multi = icra_gp.gp_trajectory_multi:main',
            'gp_train_multi = icra_gp.gp_train_multi:main',
            'gp_trajectory_validation = icra_gp.gp_trajectory_validation:main',
        ],
    },
)
