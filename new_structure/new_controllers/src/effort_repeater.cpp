#include <new_controllers/effort_repeater.hpp>

#include <cassert>
#include <cmath>
#include <exception>
#include <string>

#include <Eigen/Eigen>

namespace new_controllers {

controller_interface::InterfaceConfiguration
EffortRepeater::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/effort");
  }
  return config;
}

controller_interface::InterfaceConfiguration
EffortRepeater::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  return config;
}

controller_interface::return_type EffortRepeater::update(
    const rclcpp::Time& /*time*/,
    const rclcpp::Duration& period) {

  // EffortCommand message received
  for (int i = 0; i < num_joints; ++i) {
    command_interfaces_[i].set_value(tau_d_received_(i));
  }
  // if not received, automatically send latest tau_d_received_

  return controller_interface::return_type::OK;
}

CallbackReturn EffortRepeater::on_init() {
  try {
    auto_declare<std::string>("arm_id", "panda");
  } catch (const std::exception& e) {
    fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

CallbackReturn EffortRepeater::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  arm_id_ = get_node()->get_parameter("arm_id").as_string();

  // Subscription
  effort_command_sub_ = get_node()->create_subscription<custom_msgs::msg::EffortCommand>(
      "effort_command", 10, 
      std::bind(&EffortRepeater::effortCommandCallback, this, std::placeholders::_1));

  return CallbackReturn::SUCCESS;
}

CallbackReturn EffortRepeater::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  tau_d_received_ = Vector7d::Zero();
  received_effort_command_ = false;

  return CallbackReturn::SUCCESS;
}

void EffortRepeater::effortCommandCallback(const custom_msgs::msg::EffortCommand::SharedPtr msg) {
  tau_d_received_ = Vector7d(msg->efforts.data());
}

}  // namespace new_controllers
#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(new_controllers::EffortRepeater,
                       controller_interface::ControllerInterface)
