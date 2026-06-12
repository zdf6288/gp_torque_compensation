#include "cpp_relayer/update_rate_diagnostic_controller.hpp"

#include <algorithm>
#include <cmath>

#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/logging.hpp>

namespace cpp_relayer
{

CallbackReturn UpdateRateDiagnosticController::on_init()
{
  auto_declare<bool>("diagnostics_enabled", true);
  auto_declare<double>("diagnostics_log_period_sec", 5.0);
  auto_declare<double>("expected_update_rate_hz", 1000.0);
  auto_declare<double>("expected_publish_rate_hz", 50.0);
  auto_declare<double>("warn_ratio_low", 0.8);
  auto_declare<double>("warn_ratio_high", 1.2);
  return CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration
UpdateRateDiagnosticController::command_interface_configuration() const
{
  return {
    controller_interface::interface_configuration_type::NONE,
    {}
  };
}

controller_interface::InterfaceConfiguration
UpdateRateDiagnosticController::state_interface_configuration() const
{
  return {
    controller_interface::interface_configuration_type::NONE,
    {}
  };
}

CallbackReturn UpdateRateDiagnosticController::on_configure(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  const auto node = get_node();
  diagnostics_enabled_ = node->get_parameter("diagnostics_enabled").as_bool();
  diagnostics_log_period_sec_ = sanitize_positive_double(
    node->get_parameter("diagnostics_log_period_sec").as_double(),
    5.0,
    "diagnostics_log_period_sec");
  expected_update_rate_hz_ = sanitize_positive_double(
    node->get_parameter("expected_update_rate_hz").as_double(),
    1000.0,
    "expected_update_rate_hz");
  expected_publish_rate_hz_ = sanitize_positive_double(
    node->get_parameter("expected_publish_rate_hz").as_double(),
    50.0,
    "expected_publish_rate_hz");
  warn_ratio_low_ = sanitize_positive_double(
    node->get_parameter("warn_ratio_low").as_double(),
    0.8,
    "warn_ratio_low");
  warn_ratio_high_ = sanitize_positive_double(
    node->get_parameter("warn_ratio_high").as_double(),
    1.2,
    "warn_ratio_high");

  if (warn_ratio_low_ >= warn_ratio_high_) {
    RCLCPP_WARN(
      node->get_logger(),
      "Invalid warn_ratio range low=%.3f high=%.3f; falling back to 0.8/1.2.",
      warn_ratio_low_,
      warn_ratio_high_);
    warn_ratio_low_ = 0.8;
    warn_ratio_high_ = 1.2;
  }

  RCLCPP_INFO(
    node->get_logger(),
    "UpdateRateDiagnosticController configured: expected_update_rate_hz=%.3f, "
    "expected_publish_rate_hz=%.3f, diagnostics_log_period_sec=%.3f, "
    "diagnostics_enabled=%s. This controller declares no command or state interfaces.",
    expected_update_rate_hz_,
    expected_publish_rate_hz_,
    diagnostics_log_period_sec_,
    diagnostics_enabled_ ? "true" : "false");
  return CallbackReturn::SUCCESS;
}

CallbackReturn UpdateRateDiagnosticController::on_activate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  timing_started_ = false;
  update_count_ = 0;
  last_report_update_count_ = 0;
  start_time_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
  last_report_time_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
  last_update_time_ = rclcpp::Time(0, 0, RCL_ROS_TIME);

  RCLCPP_INFO(
    get_node()->get_logger(),
    "UpdateRateDiagnosticController activated in log-only mode.");
  return CallbackReturn::SUCCESS;
}

CallbackReturn UpdateRateDiagnosticController::on_deactivate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  if (diagnostics_enabled_ && timing_started_) {
    log_summary(last_update_time_, "deactivate");
  }
  RCLCPP_INFO(
    get_node()->get_logger(),
    "UpdateRateDiagnosticController deactivated after %lu updates.",
    update_count_);
  return CallbackReturn::SUCCESS;
}

controller_interface::return_type UpdateRateDiagnosticController::update(
  const rclcpp::Time & time,
  const rclcpp::Duration & /*period*/)
{
  ++update_count_;
  last_update_time_ = time;

  if (!diagnostics_enabled_) {
    return controller_interface::return_type::OK;
  }

  if (!timing_started_) {
    timing_started_ = true;
    start_time_ = time;
    last_report_time_ = time;
    last_report_update_count_ = update_count_;
    return controller_interface::return_type::OK;
  }

  const double elapsed_since_report = (time - last_report_time_).seconds();
  if (elapsed_since_report >= diagnostics_log_period_sec_) {
    log_summary(time, "periodic");
    last_report_time_ = time;
    last_report_update_count_ = update_count_;
  }

  return controller_interface::return_type::OK;
}

void UpdateRateDiagnosticController::log_summary(
  const rclcpp::Time & time,
  const char * phase)
{
  const double elapsed_since_start = std::max((time - start_time_).seconds(), 0.0);
  const double elapsed_since_report = std::max((time - last_report_time_).seconds(), 0.0);
  const std::uint64_t updates_since_report = update_count_ - last_report_update_count_;
  const double estimated_rate = elapsed_since_report > 0.0 ?
    static_cast<double>(updates_since_report) / elapsed_since_report : 0.0;
  const double ratio = expected_update_rate_hz_ > 0.0 ?
    estimated_rate / expected_update_rate_hz_ : 0.0;
  const bool within_threshold = ratio >= warn_ratio_low_ && ratio <= warn_ratio_high_;

  const auto logger = get_node()->get_logger();
  const char * status = within_threshold ? "OK" : "WARN";
  if (within_threshold) {
    RCLCPP_INFO(
      logger,
      "[%s] update_rate_diagnostic status=%s total_updates=%lu "
      "updates_since_report=%lu elapsed_total_sec=%.3f elapsed_report_sec=%.3f "
      "estimated_update_rate_hz=%.3f expected_update_rate_hz=%.3f ratio=%.3f "
      "expected_publish_rate_hz=%.3f diagnostics_log_period_sec=%.3f",
      phase,
      status,
      update_count_,
      updates_since_report,
      elapsed_since_start,
      elapsed_since_report,
      estimated_rate,
      expected_update_rate_hz_,
      ratio,
      expected_publish_rate_hz_,
      diagnostics_log_period_sec_);
  } else {
    RCLCPP_WARN(
      logger,
      "[%s] update_rate_diagnostic status=%s total_updates=%lu "
      "updates_since_report=%lu elapsed_total_sec=%.3f elapsed_report_sec=%.3f "
      "estimated_update_rate_hz=%.3f expected_update_rate_hz=%.3f ratio=%.3f "
      "expected range=[%.3f, %.3f] expected_publish_rate_hz=%.3f "
      "diagnostics_log_period_sec=%.3f",
      phase,
      status,
      update_count_,
      updates_since_report,
      elapsed_since_start,
      elapsed_since_report,
      estimated_rate,
      expected_update_rate_hz_,
      ratio,
      warn_ratio_low_,
      warn_ratio_high_,
      expected_publish_rate_hz_,
      diagnostics_log_period_sec_);
  }
}

double UpdateRateDiagnosticController::sanitize_positive_double(
  double value,
  double fallback,
  const char * parameter_name) const
{
  if (std::isfinite(value) && value > 0.0) {
    return value;
  }

  RCLCPP_WARN(
    get_node()->get_logger(),
    "Invalid %s=%.6f; falling back to %.6f.",
    parameter_name,
    value,
    fallback);
  return fallback;
}

}  // namespace cpp_relayer

PLUGINLIB_EXPORT_CLASS(
  cpp_relayer::UpdateRateDiagnosticController,
  controller_interface::ControllerInterface)
