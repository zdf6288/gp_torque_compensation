#include <cpp_relayer/cpp_relayer.hpp>

#include <cassert>
#include <cmath>
#include <exception>
#include <string>

#include <Eigen/Eigen>

namespace cpp_relayer {

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
  }
  return config;
}

controller_interface::return_type CPPRelayer::update(
    const rclcpp::Time& /*time*/,
    const rclcpp::Duration& /*period*/) {

  // EffortCommand message received
  for (int i = 0; i < num_joints; ++i) {
    command_interfaces_[i].set_value(tau_d_received_(i));
  }
  // if not received, automatically send latest tau_d_received_

  // publish state parameter
  updateStateParam();
  custom_msgs::msg::StateParameter state_param;
  state_param.header.stamp = get_node()->now();
  Eigen::Map<Eigen::VectorXd>(state_param.position.data(), num_joints) = q_;
  Eigen::Map<Eigen::VectorXd>(state_param.velocity.data(), num_joints) = dq_;
  
  // 直接复制数组数据，因为消息字段是std::array类型
  std::copy(mass_.begin(), mass_.end(), state_param.mass.begin());
  std::copy(coriolis_.begin(), coriolis_.end(), state_param.coriolis.begin());
  std::copy(body_jacobian_flange_.begin(), body_jacobian_flange_.end(), state_param.body_jacobian_flange.begin());
  
  state_param_pub_->publish(state_param);

  return controller_interface::return_type::OK;
}

CallbackReturn CPPRelayer::on_init() {
  try {
    auto_declare<std::string>("arm_id", "panda");
  } catch (const std::exception& e) {
    fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

CallbackReturn CPPRelayer::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  arm_id_ = get_node()->get_parameter("arm_id").as_string();

  // 使用franka_semantic_components::FrankaRobotModel
  try {
    franka_robot_model_ = std::make_unique<franka_semantic_components::FrankaRobotModel>(
        arm_id_ + "/robot_model", arm_id_ + "/robot_state");
  } catch (const std::exception& e) {
    fprintf(stderr, "Failed to initialize Franka robot model: %s\n", e.what());
    return CallbackReturn::ERROR;
  }

  // Subscribe to /effort_command
  effort_command_sub_ = get_node()->create_subscription<custom_msgs::msg::EffortCommand>(
      "effort_command", 10, 
      std::bind(&CPPRelayer::effortCommandCallback, this, std::placeholders::_1));

  // Publish on /state_parameter
  state_param_pub_ = get_node()->create_publisher<custom_msgs::msg::StateParameter>(
      "state_parameter", 10);

  return CallbackReturn::SUCCESS;
}

CallbackReturn CPPRelayer::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  q_ = Vector7d::Zero();
  dq_ = Vector7d::Zero();
  tau_d_received_ = Vector7d::Zero();
  
  // 初始化数组为零
  mass_.fill(0.0);
  coriolis_.fill(0.0);
  body_jacobian_flange_.fill(0.0);
  
  updateStateParam();

  return CallbackReturn::SUCCESS;
}

void CPPRelayer::effortCommandCallback(const custom_msgs::msg::EffortCommand::SharedPtr msg) {
  tau_d_received_ = Vector7d(msg->efforts.data());
}

void CPPRelayer::updateStateParam() {
  // joint position and velocity
  for (auto i = 0; i < num_joints; ++i) {
    const auto& position_interface = state_interfaces_.at(2 * i);
    const auto& velocity_interface = state_interfaces_.at(2 * i + 1);

    assert(position_interface.get_interface_name() == "position");
    assert(velocity_interface.get_interface_name() == "velocity");

    q_(i) = position_interface.get_value();
    dq_(i) = velocity_interface.get_value();
  }

  // 使用FrankaRobotModel计算动力学参数
  if (franka_robot_model_) {
    try {
      // FrankaRobotModel会自动处理状态接口，不需要手动创建RobotState
      mass_ = franka_robot_model_->getMassMatrix();
      coriolis_ = franka_robot_model_->getCoriolisForceVector();
      body_jacobian_flange_ = franka_robot_model_->getBodyJacobian(franka::Frame::kFlange);
    } catch (const std::exception& e) {
      // 如果计算失败，保持默认值
      RCLCPP_WARN(get_node()->get_logger(), "Failed to compute dynamics: %s", e.what());
    }
  }
}

}  // namespace cpp_relayer
#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(cpp_relayer::CPPRelayer,
                       controller_interface::ControllerInterface)
