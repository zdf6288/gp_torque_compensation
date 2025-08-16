from setuptools import find_packages, setup

package_name = 'py_controllers'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/effort_pd_launch.py']),
        ('share/' + package_name + '/launch', ['launch/cartesian_impedance_launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ziyang',
    maintainer_email='mengziyang168@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'effort_pd = py_controllers.effort_pd:main',
            'cartesian_impedance = py_controllers.cartesian_impedance:main',
        ],
    },
)
