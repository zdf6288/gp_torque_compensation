#!/usr/bin/env python3

"""Static checks for the fake-only update_rate_diagnostic controller."""

from pathlib import Path
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path):
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def require_contains(relative_path, needle):
    content = read_text(relative_path)
    require(needle in content, f"{relative_path} missing {needle!r}")
    return content


def check_controller_plugin():
    header = require_contains(
        "new_structure/cpp_relayer/include/cpp_relayer/"
        "update_rate_diagnostic_controller.hpp",
        "class UpdateRateDiagnosticController",
    )
    source = require_contains(
        "new_structure/cpp_relayer/src/update_rate_diagnostic_controller.cpp",
        "PLUGINLIB_EXPORT_CLASS",
    )

    combined = header + "\n" + source
    banned_tokens = [
        "franka_semantic_components",
        "FrankaRobotModel",
        "franka_robot_model",
        "LoanedCommandInterface",
        "command_interfaces_",
        "state_interfaces_",
        "EffortCommand",
        "set_value(",
    ]
    for token in banned_tokens:
        require(token not in combined, f"diagnostic controller must not use {token}")

    require(
        source.count("interface_configuration_type::NONE") >= 2,
        "diagnostic controller must request NONE command and state interfaces",
    )
    require(
        "expected_update_rate_hz" in source
        and "expected_publish_rate_hz" in source
        and "diagnostics_log_period_sec" in source,
        "diagnostic controller parameters missing",
    )

    cmake = require_contains(
        "new_structure/cpp_relayer/CMakeLists.txt",
        "src/update_rate_diagnostic_controller.cpp",
    )
    require("${PROJECT_NAME}" in cmake, "cpp_relayer library target missing")

    plugin_xml = require_contains(
        "new_structure/cpp_relayer/cpp_relayer.xml",
        "cpp_relayer/UpdateRateDiagnosticController",
    )
    require(
        "cpp_relayer::UpdateRateDiagnosticController" in plugin_xml,
        "plugin XML type missing",
    )


def check_config_and_launch():
    controllers = require_contains(
        "new_structure/new_bringup/config/controllers.yaml",
        "type: cpp_relayer/UpdateRateDiagnosticController",
    )
    require(
        "expected_update_rate_hz: 1000.0" in controllers,
        "default expected_update_rate_hz must remain 1000.0",
    )
    require(
        "expected_publish_rate_hz: 50.0" in controllers,
        "default expected_publish_rate_hz must remain 50.0",
    )

    launch = read_text("new_structure/py_controllers/launch/cartesian_impedance_launch.py")
    require(
        "spawn_update_rate_diagnostic_parameter_name = 'spawn_update_rate_diagnostic'"
        in launch,
        "spawn_update_rate_diagnostic launch arg missing",
    )
    require(
        "spawn_cpp_relayer_parameter_name = 'spawn_cpp_relayer'" in launch,
        "spawn_cpp_relayer launch arg missing",
    )
    require(
        "default_value='false'" in launch
        and "Spawn fake-only update_rate_diagnostic controller" in launch,
        "spawn_update_rate_diagnostic must default false",
    )
    require(
        "default_value='true'" in launch
        and "Spawn cpp_relayer controller unless explicitly disabled." in launch,
        "spawn_cpp_relayer must default true",
    )
    require(
        "update_rate_diagnostic is fake-only and requires " in launch
        and "use_fake_hardware:=true" in launch,
        "fake-only guard missing",
    )
    require(
        "spawn_cpp_relayer:=false so cpp_relayer is not activated" in launch,
        "diagnostic-only cpp_relayer guard missing",
    )
    guard_index = launch.index("function=_guard_frequency_config")
    for parameter_name in (
        "use_fake_hardware_parameter_name",
        "spawn_cpp_relayer_parameter_name",
        "spawn_update_rate_diagnostic_parameter_name",
    ):
        declaration_index = launch.index(f"DeclareLaunchArgument(\n            {parameter_name}")
        require(
            declaration_index < guard_index,
            f"{parameter_name} must be declared before _guard_frequency_config",
        )
    require(
        "condition=IfCondition(spawn_cpp_relayer)" in launch,
        "cpp_relayer spawner condition missing",
    )
    require(
        "condition=IfCondition(spawn_update_rate_diagnostic)" in launch,
        "update_rate_diagnostic spawner condition missing",
    )
    compact_launch = re.sub(r"\s+", "", launch)
    require(
        "arguments=['cpp_relayer']" in compact_launch,
        "cpp_relayer spawner arguments must remain ['cpp_relayer']",
    )

    cpp_relayer_spawner = re.search(
        r"Node\(\s*package='controller_manager',\s*"
        r"executable='spawner',\s*arguments=\['cpp_relayer'\].*?\)",
        launch,
        re.DOTALL,
    )
    require(cpp_relayer_spawner is not None, "cpp_relayer spawner block missing")
    require(
        "--param-file" not in cpp_relayer_spawner.group(0),
        "cpp_relayer spawner must not use runtime --param-file",
    )


def main():
    try:
        check_controller_plugin()
        check_config_and_launch()
    except AssertionError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print("OK: update_rate_diagnostic static checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
