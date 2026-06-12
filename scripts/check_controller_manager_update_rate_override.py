#!/usr/bin/env python3
"""Static checks for the controller_manager update_rate override scaffold."""

from pathlib import Path
import importlib.util


REPO_ROOT = Path(__file__).resolve().parents[1]
FRANKA_LAUNCH = REPO_ROOT / "new_structure/new_bringup/launch/franka.launch.py"
CONTROLLERS_YAML = REPO_ROOT / "new_structure/new_bringup/config/controllers.yaml"
CARTESIAN_LAUNCH = (
    REPO_ROOT / "new_structure/py_controllers/launch/cartesian_impedance_launch.py"
)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def load_franka_launch_module():
    spec = importlib.util.spec_from_file_location("franka_launch", FRANKA_LAUNCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    franka_launch = load_franka_launch_module()

    high_rate_yaml = franka_launch._controller_manager_update_rate_yaml_text(1000)
    legacy_yaml = franka_launch._controller_manager_update_rate_yaml_text(50)

    require(
        high_rate_yaml.splitlines()[0] == "controller_manager:",
        "high-rate override YAML must use controller_manager key",
    )
    require(
        "  ros__parameters:\n    update_rate: 1000\n" in high_rate_yaml,
        "high-rate override YAML must set update_rate: 1000",
    )
    require(
        "/" + "controller_manager" not in high_rate_yaml,
        "high-rate override YAML must not use an absolute /controller_manager key",
    )
    require(
        "  ros__parameters:\n    update_rate: 50\n" in legacy_yaml,
        "legacy override YAML must set update_rate: 50",
    )

    controllers_text = CONTROLLERS_YAML.read_text()
    require(
        "controller_manager:\n  ros__parameters:\n    update_rate: 50" in controllers_text,
        "controllers.yaml must keep its legacy update_rate: 50 baseline",
    )

    franka_launch_text = FRANKA_LAUNCH.read_text()
    require(
        "parameters=[\n                {'robot_description': robot_description},\n"
        "                franka_controllers,\n                update_rate_param_file,\n"
        "            ]" in franka_launch_text,
        "ros2_control_node parameters must append the temp override after controllers.yaml",
    )
    require(
        "param_file.write('/controller_manager" not in franka_launch_text,
        "override writer must not write /controller_manager",
    )

    cartesian_launch_text = CARTESIAN_LAUNCH.read_text()
    require(
        "arguments=['cpp_relayer']" in cartesian_launch_text,
        "cpp_relayer spawner must remain arguments=['cpp_relayer']",
    )
    require(
        "'--param-file'" not in cartesian_launch_text,
        "cpp_relayer spawner must not use a runtime --param-file override",
    )

    print("controller_manager update_rate override static checks passed")


if __name__ == "__main__":
    main()
