#include <cpp_relayer/cpp_relayer.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <exception>
#include <functional>
#include <string>

#include <Eigen/Eigen>

#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"

namespace cpp_relayer {

namespace {
constexpr double kDefaultCommandTimeoutSec = 0.2;
constexpr double kDefaultNegativeCommandAgeToleranceSec = 0.002;
constexpr double kDefaultStateParameterPublishRate = 50.0;
constexpr int kWarningThrottleMs = 2000;
}  // namespace

controller_interface::InterfaceConfiguration
CPPRelayer::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/effort");
  }
  return config;
}

controller_interface::InterfaceConfiguration
CPPRelayer::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/position");
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/velocity");
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/effort");
  }
  for (const auto& franka_robot_model_name : franka_robot_model_->get_state_interface_names()) {
    config.names.push_back(franka_robot_model_name);
  }
  return config;
}

controller_interface::return_type CPPRelayer::update(
    const rclcpp::Time& /*time*/,
    const rclcpp::Duration& /*period*/) {

  ++update_count_;
  rclcpp::Time now;
  Vector7d command_to_write = Vector7d::Zero();
  bool has_command = false;
  bool command_is_fresh = false;
  bool small_negative_command_age = false;
  bool large_negative_command_age = false;
  double raw_command_age_sec = 0.0;
  double command_age_sec = 0.0;

  {
    std::lock_guard<std::mutex> lock(command_mutex_);
    // A 50 Hz Python /effort_command stream is expected to feed a 1 kHz
    // ros2_control update loop. Snapshot the command timestamp and now while
    // holding the same mutex, then intentionally hold the last valid command
    // across multiple hardware updates until it is genuinely stale. Tiny
    // timestamp-ordering anomalies must not create a single-sample zero torque.
    now = get_node()->get_clock()->now();
    has_command = received_effort_command_;
    if (has_command) {
      const auto command_age = now - last_command_time_;
      raw_command_age_sec = command_age.seconds();
      command_age_sec = raw_command_age_sec;
      if (raw_command_age_sec < min_command_age_sec_) {
        min_command_age_sec_ = raw_command_age_sec;
      }
      if (raw_command_age_sec < 0.0) {
        ++negative_command_age_anomaly_count_;
        last_negative_command_age_sec_ = raw_command_age_sec;
        if (std::abs(raw_command_age_sec) <= negative_command_age_tolerance_sec_) {
          small_negative_command_age = true;
          command_age_sec = 0.0;
        } else {
          large_negative_command_age = true;
        }
      }
      command_is_fresh = isCommandFresh(command_age_sec);
      last_command_age_sec_ = command_age_sec;
      if (command_age_sec > max_command_age_sec_) {
        max_command_age_sec_ = command_age_sec;
      }
      if (command_is_fresh) {
        command_to_write = tau_d_received_;
      }
    }
  }

  if (command_is_fresh) {
    if (small_negative_command_age && diagnostics_enabled_ &&
        negative_command_age_event_log_count_ <
            static_cast<std::uint64_t>(log_first_n_stale_events_)) {
      ++negative_command_age_event_log_count_;
      RCLCPP_WARN(
          get_node()->get_logger(),
          "[CPPRelayerDiag] small negative command_age %.9f s observed "
          "(tolerance %.9f s); treating as fresh and holding last valid command. "
          "update_count=%llu, received_command_count=%llu.",
          raw_command_age_sec, negative_command_age_tolerance_sec_,
          static_cast<unsigned long long>(update_count_),
          static_cast<unsigned long long>(received_command_count_));
    }
    for (int i = 0; i < num_joints; ++i) {
      command_interfaces_[i].set_value(command_to_write(i));
    }
  } else {
    ++zero_fallback_count_;
    setZeroCommandInterfaces();
    if (has_command) {
      ++stale_command_count_;
      // Diagnostics only: confirm whether Python /effort_command publish jitter triggers fallback.
      if (diagnostics_enabled_ &&
          stale_event_log_count_ < static_cast<std::uint64_t>(log_first_n_stale_events_)) {
        ++stale_event_log_count_;
        RCLCPP_WARN(
            get_node()->get_logger(),
            "[CPPRelayerDiag] stale fallback event #%llu: command_age=%.6f s, "
            "timeout=%.6f s, update_count=%llu, received_command_count=%llu. Writing zero.",
            static_cast<unsigned long long>(stale_event_log_count_),
            command_age_sec, command_timeout_sec_,
            static_cast<unsigned long long>(update_count_),
            static_cast<unsigned long long>(received_command_count_));
      }
      if (large_negative_command_age) {
        RCLCPP_WARN_THROTTLE(
            get_node()->get_logger(), *get_node()->get_clock(), kWarningThrottleMs,
            "Refusing EffortCommand with large negative age %.6f s beyond tolerance %.6f s. "
            "Writing zero.",
            raw_command_age_sec, negative_command_age_tolerance_sec_);
      } else {
        RCLCPP_WARN_THROTTLE(
            get_node()->get_logger(), *get_node()->get_clock(), kWarningThrottleMs,
            "Refusing stale EffortCommand: age %.3f s, timeout %.3f s. Writing zero.",
            command_age_sec, command_timeout_sec_);
      }
    }
  }

  if (shouldPublishStateParameter(now)) {
    publishStateParameter(now);
  }
  maybeLogDiagnostics(now);

  return controller_interface::return_type::OK;
}

CallbackReturn CPPRelayer::on_init() {
  try {
    auto_declare<std::string>("arm_id", "panda");
    auto_declare<double>("command_timeout_sec", kDefaultCommandTimeoutSec);
    auto_declare<double>(
        "negative_command_age_tolerance_sec", kDefaultNegativeCommandAgeToleranceSec);
    auto_declare<bool>("require_fresh_command_on_activate", false);
    auto_declare<bool>("diagnostics_enabled", true);
    auto_declare<double>("diagnostics_log_period_sec", 5.0);
    auto_declare<int>("log_first_n_stale_events", 5);
    auto_declare<double>("state_parameter_publish_rate", kDefaultStateParameterPublishRate);
  } 
  catch (const std::exception& e) {
    fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

CallbackReturn CPPRelayer::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  try {
    arm_id_ = get_node()->get_parameter("arm_id").as_string();
    command_timeout_sec_ = get_node()->get_parameter("command_timeout_sec").as_double();
    negative_command_age_tolerance_sec_ =
        get_node()->get_parameter("negative_command_age_tolerance_sec").as_double();
    require_fresh_command_on_activate_ =
        get_node()->get_parameter("require_fresh_command_on_activate").as_bool();
    diagnostics_enabled_ = get_node()->get_parameter("diagnostics_enabled").as_bool();
    diagnostics_log_period_sec_ =
        get_node()->get_parameter("diagnostics_log_period_sec").as_double();
    log_first_n_stale_events_ =
        get_node()->get_parameter("log_first_n_stale_events").as_int();
    state_parameter_publish_rate_ =
        get_node()->get_parameter("state_parameter_publish_rate").as_double();
    if (!std::isfinite(command_timeout_sec_) || command_timeout_sec_ <= 0.0) {
      RCLCPP_WARN(
          get_node()->get_logger(),
          "Invalid command_timeout_sec %.3f. Falling back to safe default %.3f s.",
          command_timeout_sec_, kDefaultCommandTimeoutSec);
      command_timeout_sec_ = kDefaultCommandTimeoutSec;
    }
    if (!std::isfinite(negative_command_age_tolerance_sec_) ||
        negative_command_age_tolerance_sec_ < 0.0) {
      RCLCPP_WARN(
          get_node()->get_logger(),
          "Invalid negative_command_age_tolerance_sec %.6f. Falling back to %.6f s.",
          negative_command_age_tolerance_sec_, kDefaultNegativeCommandAgeToleranceSec);
      negative_command_age_tolerance_sec_ = kDefaultNegativeCommandAgeToleranceSec;
    }
    if (!std::isfinite(diagnostics_log_period_sec_) || diagnostics_log_period_sec_ <= 0.0) {
      RCLCPP_WARN(
          get_node()->get_logger(),
          "Invalid diagnostics_log_period_sec %.3f. Falling back to 5.000 s.",
          diagnostics_log_period_sec_);
      diagnostics_log_period_sec_ = 5.0;
    }
    if (log_first_n_stale_events_ < 0) {
      RCLCPP_WARN(
          get_node()->get_logger(),
          "Invalid log_first_n_stale_events %d. Falling back to 5.",
          log_first_n_stale_events_);
      log_first_n_stale_events_ = 5;
    }
    if (!std::isfinite(state_parameter_publish_rate_) ||
        state_parameter_publish_rate_ <= 0.0) {
      RCLCPP_WARN(
          get_node()->get_logger(),
          "Invalid state_parameter_publish_rate %.3f. Publishing /state_parameter "
          "on every controller update as a compatibility fallback.",
          state_parameter_publish_rate_);
      state_parameter_publish_rate_ = 0.0;
      state_parameter_publish_period_sec_ = 0.0;
    } else {
      state_parameter_publish_period_sec_ = 1.0 / state_parameter_publish_rate_;
    }
  } 
  catch (const std::exception& e) {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to get cpp_relayer parameters: %s", e.what());
    return CallbackReturn::ERROR;
  }

  // Subscribe to /effort_command
  effort_command_sub_ = get_node()->create_subscription<custom_msgs::msg::EffortCommand>(
      "effort_command", 10, 
      std::bind(&CPPRelayer::effortCommandCallback, this, std::placeholders::_1));

  // Publish on /state_parameter
  state_param_pub_ = get_node()->create_publisher<custom_msgs::msg::StateParameter>(
      "state_parameter", 10);

  // franka_semantic_components::FrankaRobotModel
  try {
    franka_robot_model_ = std::make_unique<franka_semantic_components::FrankaRobotModel>(
      franka_semantic_components::FrankaRobotModel(arm_id_ + "/" + k_robot_model_interface_name,
                                                   arm_id_ + "/" + k_robot_state_interface_name));

    RCLCPP_DEBUG(get_node()->get_logger(), "configured successfully");
    RCLCPP_INFO(
        get_node()->get_logger(),
        "cpp_relayer configured with final parameter values: command_timeout_sec=%.3f s, "
        "negative_command_age_tolerance_sec=%.6f s, "
        "require_fresh_command_on_activate=%s; stale commands will be zeroed. "
        "diagnostics_enabled=%s, diagnostics_log_period_sec=%.3f, "
        "log_first_n_stale_events=%d, state_parameter_publish_rate=%.3f Hz.",
        command_timeout_sec_,
        negative_command_age_tolerance_sec_,
        require_fresh_command_on_activate_ ? "true" : "false",
        diagnostics_enabled_ ? "true" : "false",
        diagnostics_log_period_sec_,
        log_first_n_stale_events_,
        state_parameter_publish_rate_);
    return CallbackReturn::SUCCESS;
  } 
  catch (const std::exception& e) {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to configure controller: %s", e.what());
    return CallbackReturn::ERROR;
  }
}

CallbackReturn CPPRelayer::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  q_ = Vector7d::Zero();
  dq_ = Vector7d::Zero();
  tau_measured_ = Vector7d::Zero();
  o_t_f_.fill(0.0);
  mass_.fill(0.0);
  coriolis_.fill(0.0);
  zero_jacobian_flange_.fill(0.0);
  gravity_.fill(0.0);

  auto now = get_node()->get_clock()->now();
  last_diagnostics_log_time_ = now;
  last_state_parameter_publish_time_ = rclcpp::Time();
  state_parameter_publish_count_ = 0;
  last_diagnostics_state_parameter_publish_count_ = 0;
  last_state_parameter_publish_age_sec_ = 0.0;
  Vector7d command_to_write = Vector7d::Zero();
  bool has_command = false;
  bool command_is_fresh = false;
  bool command_values_are_finite = false;
  double command_age_sec = 0.0;

  if (require_fresh_command_on_activate_) {
    {
      std::lock_guard<std::mutex> lock(command_mutex_);
      now = get_node()->get_clock()->now();
      has_command = received_effort_command_;
      if (has_command) {
        const auto command_age = now - last_command_time_;
        command_age_sec = command_age.seconds();
        if (command_age_sec < 0.0 &&
            std::abs(command_age_sec) <= negative_command_age_tolerance_sec_) {
          command_age_sec = 0.0;
        }
        command_is_fresh = isCommandFresh(command_age_sec);
        command_values_are_finite = tau_d_received_.allFinite();
        if (command_is_fresh && command_values_are_finite) {
          command_to_write = tau_d_received_;
        }
      }
    }

    if (!has_command) {
      setZeroCommandInterfaces();
      RCLCPP_ERROR(
          get_node()->get_logger(),
          "Refusing cpp_relayer activation: no cached EffortCommand is available "
          "(has_command=false, command_age_sec=not_available, timeout %.3f s, "
          "require_fresh_command_on_activate=true).",
          command_timeout_sec_);
      return CallbackReturn::FAILURE;
    }

    if (!command_is_fresh || !command_values_are_finite) {
      setZeroCommandInterfaces();
      RCLCPP_ERROR(
          get_node()->get_logger(),
          "Refusing cpp_relayer activation: cached EffortCommand is not usable "
          "(has_command=true, command_age_sec=%.3f s, timeout %.3f s, fresh=%s, "
          "finite=%s, require_fresh_command_on_activate=true).",
          command_age_sec, command_timeout_sec_,
          command_is_fresh ? "true" : "false",
          command_values_are_finite ? "true" : "false");
      return CallbackReturn::FAILURE;
    }

    for (int i = 0; i < num_joints; ++i) {
      command_interfaces_[i].set_value(command_to_write(i));
    }
  } else {
    {
      std::lock_guard<std::mutex> lock(command_mutex_);
      tau_d_received_ = Vector7d::Zero();
      received_effort_command_ = false;
      last_command_time_ = now;
    }
    setZeroCommandInterfaces();
    RCLCPP_WARN(
        get_node()->get_logger(),
        "cpp_relayer legacy zero activation is enabled "
        "(require_fresh_command_on_activate=false).");
  }
  
  franka_robot_model_->assign_loaned_state_interfaces(state_interfaces_);
  updateStateParam();

  if (require_fresh_command_on_activate_) {
    RCLCPP_INFO(
        get_node()->get_logger(),
        "cpp_relayer activated with fresh cached EffortCommand "
        "(command_age_sec=%.3f s, timeout %.3f s).",
        command_age_sec, command_timeout_sec_);
  } else {
    RCLCPP_INFO(
        get_node()->get_logger(),
        "cpp_relayer activated: zero fallback enabled until a valid fresh EffortCommand arrives "
        "(timeout %.3f s).",
        command_timeout_sec_);
  }

  return CallbackReturn::SUCCESS;
}

CallbackReturn CPPRelayer::on_deactivate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  setZeroCommandInterfaces();
  {
    std::lock_guard<std::mutex> lock(command_mutex_);
    tau_d_received_ = Vector7d::Zero();
    received_effort_command_ = false;
    last_command_time_ = get_node()->get_clock()->now();
  }
  RCLCPP_INFO(get_node()->get_logger(), "cpp_relayer deactivated: command interfaces zeroed.");
  return CallbackReturn::SUCCESS;
}

void CPPRelayer::effortCommandCallback(const custom_msgs::msg::EffortCommand::SharedPtr msg) {
  if (msg->efforts.size() != static_cast<std::size_t>(num_joints)) {
    {
      std::lock_guard<std::mutex> lock(command_mutex_);
      received_effort_command_ = false;
    }
    RCLCPP_WARN_THROTTLE(
        get_node()->get_logger(), *get_node()->get_clock(), kWarningThrottleMs,
        "Refusing invalid EffortCommand: expected %d efforts, got %zu. Writing zero.",
        num_joints, msg->efforts.size());
    return;
  }

  Vector7d command = Vector7d::Zero();
  for (std::size_t i = 0; i < msg->efforts.size(); ++i) {
    if (!std::isfinite(msg->efforts[i])) {
      {
        std::lock_guard<std::mutex> lock(command_mutex_);
        received_effort_command_ = false;
      }
      RCLCPP_WARN_THROTTLE(
          get_node()->get_logger(), *get_node()->get_clock(), kWarningThrottleMs,
          "Refusing invalid EffortCommand: effort[%zu] is not finite. Writing zero.",
          i);
      return;
    }
    command(static_cast<int>(i)) = msg->efforts[i];
  }

  {
    std::lock_guard<std::mutex> lock(command_mutex_);
    tau_d_received_ = command;
    last_command_time_ = get_node()->get_clock()->now();
    received_effort_command_ = true;
    ++received_command_count_;
  }
}

void CPPRelayer::setZeroCommandInterfaces() {
  for (auto& command_interface : command_interfaces_) {
    command_interface.set_value(0.0);
  }
}

bool CPPRelayer::isCommandFresh(double effective_command_age_sec) const {
  return effective_command_age_sec >= 0.0 &&
         effective_command_age_sec <= command_timeout_sec_;
}

bool CPPRelayer::shouldPublishStateParameter(const rclcpp::Time& now) {
  if (state_parameter_publish_period_sec_ <= 0.0) {
    last_state_parameter_publish_age_sec_ = 0.0;
    return true;
  }

  if (last_state_parameter_publish_time_.nanoseconds() == 0) {
    last_state_parameter_publish_age_sec_ = 0.0;
    return true;
  }

  const auto publish_age = now - last_state_parameter_publish_time_;
  if (publish_age.nanoseconds() < 0) {
    last_state_parameter_publish_age_sec_ = 0.0;
    return true;
  }

  last_state_parameter_publish_age_sec_ = publish_age.seconds();
  return last_state_parameter_publish_age_sec_ >= state_parameter_publish_period_sec_;
}

void CPPRelayer::publishStateParameter(const rclcpp::Time& now) {
  // 解耦 controller_manager update rate 和 Python state callback rate：
  // 1000 Hz update 仍写 torque interface，/state_parameter 只按配置频率发布。
  updateStateParam();
  custom_msgs::msg::StateParameter state_param;
  state_param.header.stamp = now;
  Eigen::Map<Eigen::VectorXd>(state_param.position.data(), num_joints) = q_;
  Eigen::Map<Eigen::VectorXd>(state_param.velocity.data(), num_joints) = dq_;
  Eigen::Map<Eigen::VectorXd>(state_param.effort_measured.data(), num_joints) = tau_measured_;
  std::copy(o_t_f_.begin(), o_t_f_.end(), state_param.o_t_f.begin());
  std::copy(mass_.begin(), mass_.end(), state_param.mass.begin());
  std::copy(coriolis_.begin(), coriolis_.end(), state_param.coriolis.begin());
  std::copy(zero_jacobian_flange_.begin(), zero_jacobian_flange_.end(), state_param.zero_jacobian_flange.begin());
  std::copy(gravity_.begin(), gravity_.end(), state_param.gravity.begin());
  state_param_pub_->publish(state_param);
  last_state_parameter_publish_time_ = now;
  last_state_parameter_publish_age_sec_ = 0.0;
  ++state_parameter_publish_count_;
}

void CPPRelayer::maybeLogDiagnostics(const rclcpp::Time& now) {
  if (!diagnostics_enabled_) {
    return;
  }

  double observed_state_parameter_publish_rate = 0.0;
  if (last_diagnostics_log_time_.nanoseconds() != 0) {
    const auto elapsed = now - last_diagnostics_log_time_;
    if (elapsed.nanoseconds() >= 0 && elapsed.seconds() < diagnostics_log_period_sec_) {
      return;
    }
    if (elapsed.nanoseconds() > 0) {
      const auto published_since_last_log =
          state_parameter_publish_count_ - last_diagnostics_state_parameter_publish_count_;
      observed_state_parameter_publish_rate =
          static_cast<double>(published_since_last_log) / elapsed.seconds();
    }
  }
  last_diagnostics_log_time_ = now;
  last_diagnostics_state_parameter_publish_count_ = state_parameter_publish_count_;

  RCLCPP_INFO(
      get_node()->get_logger(),
      "[CPPRelayerDiag] updates=%llu, received_effort_commands=%llu, "
      "stale_command_count=%llu, zero_fallback_count=%llu, "
      "last_command_age=%.6f s, max_command_age=%.6f s, timeout=%.6f s, "
      "negative_command_age_anomaly_count=%llu, last_negative_command_age=%.9f s, "
      "min_command_age=%.9f s, negative_command_age_tolerance=%.9f s, "
      "state_parameter_publish_count=%llu, state_parameter_publish_rate=%.3f Hz, "
      "observed_state_parameter_publish_rate=%.3f Hz, "
      "last_state_parameter_publish_age=%.6f s.",
      static_cast<unsigned long long>(update_count_),
      static_cast<unsigned long long>(received_command_count_),
      static_cast<unsigned long long>(stale_command_count_),
      static_cast<unsigned long long>(zero_fallback_count_),
      last_command_age_sec_, max_command_age_sec_, command_timeout_sec_,
      static_cast<unsigned long long>(negative_command_age_anomaly_count_),
      last_negative_command_age_sec_, min_command_age_sec_,
      negative_command_age_tolerance_sec_,
      static_cast<unsigned long long>(state_parameter_publish_count_),
      state_parameter_publish_rate_,
      observed_state_parameter_publish_rate,
      last_state_parameter_publish_age_sec_);
}

void CPPRelayer::updateStateParam() {
  // joint position, velocity, and effort
  for (auto i = 0; i < num_joints; ++i) {
    const auto& position_interface = state_interfaces_.at(3 * i);
    const auto& velocity_interface = state_interfaces_.at(3 * i + 1);
    const auto& effort_interface = state_interfaces_.at(3 * i + 2);
    assert(position_interface.get_interface_name() == "position");
    assert(velocity_interface.get_interface_name() == "velocity");
    assert(effort_interface.get_interface_name() == "effort");
    q_(i) = position_interface.get_value();
    dq_(i) = velocity_interface.get_value();
    tau_measured_(i) = effort_interface.get_value();
  }

  // get kinematics and dynamics parameters from franka_robot_model_(franka_semantic_components)
  if (franka_robot_model_) {
    try {
      o_t_f_ = franka_robot_model_->getPose(franka::Frame::kFlange);
      mass_ = franka_robot_model_->getMass();
      coriolis_ = franka_robot_model_->getCoriolis();
      zero_jacobian_flange_ = franka_robot_model_->getZeroJacobian(franka::Frame::kFlange);
      gravity_ = franka_robot_model_->getGravity();
    } 
    catch (const std::exception& e) {
      RCLCPP_WARN(get_node()->get_logger(), "Failed to compute dynamics: %s", e.what());
    }
  }
}

}  // namespace cpp_relayer
#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(cpp_relayer::CPPRelayer,
                       controller_interface::ControllerInterface)
