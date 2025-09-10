from setuptools import find_packages, setup

package_name = 'icra_plot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nuc_6g_life_3',
    maintainer_email='mengziyang168@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'plot_xy_icra_data = icra_plot.plot_xy_icra_data:main',
            'plot_xy_icra_validation = icra_plot.plot_xy_icra_validation:main',
            'plot_xy_multi_data = icra_plot.plot_xy_multi_data:main',
            'plot_xy_multi_validation = icra_plot.plot_xy_multi_validation:main',
        ],
    },
)
