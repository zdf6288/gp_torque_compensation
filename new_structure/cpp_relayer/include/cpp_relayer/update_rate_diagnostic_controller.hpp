#pragma once

#include <cstdint>

#include <controller_interface/controller_interface.hpp>
#include <rclcpp_lifecycle/state.hpp>
#include <rclcpp/time.hpp>

namespace cpp_relayer
{

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

class UpdateRateDiagnosticController : public controller_interface::ControllerInterface
{
public:
  CallbackReturn on_init() override;

  controller_interface::InterfaceConfiguration command_interface_configuration() const override;
  controller_interface::InterfaceConfiguration state_interface_configuration() const override;

  CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override;
  CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;
  CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;

  controller_interface::return_type update(
    const rclcpp::Time & time,
    const rclcpp::Duration & period) override;

private:
  void log_summary(const rclcpp::Time & time, const char * phase);
  double sanitize_positive_double(
    double value,
    double fallback,
    const char * parameter_name) const;

  bool diagnostics_enabled_{true};
  double diagnostics_log_period_sec_{5.0};
  double expected_update_rate_hz_{1000.0};
  double expected_publish_rate_hz_{50.0};
  double warn_ratio_low_{0.8};
  double warn_ratio_high_{1.2};

  bool timing_started_{false};
  std::uint64_t update_count_{0};
  std::uint64_t last_report_update_count_{0};
  rclcpp::Time start_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_report_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_update_time_{0, 0, RCL_ROS_TIME};
};

}  // namespace cpp_relayer
