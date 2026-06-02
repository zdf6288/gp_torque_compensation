#pragma once

#include <array>
#include <mutex>
#include <string>

#include <Eigen/Eigen>
#include <controller_interface/controller_interface.hpp>
#include <rclcpp/rclcpp.hpp>
#include <custom_msgs/msg/effort_command.hpp>
#include <custom_msgs/msg/state_parameter.hpp>  
#include "franka_semantic_components/franka_robot_model.hpp"

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace cpp_relayer {


class CPPRelayer : public controller_interface::ControllerInterface {
 public:
  using Vector7d = Eigen::Matrix<double, 7, 1>;
  [[nodiscard]] controller_interface::InterfaceConfiguration command_interface_configuration()
      const override;
  [[nodiscard]] controller_interface::InterfaceConfiguration state_interface_configuration()
      const override;
  controller_interface::return_type update(const rclcpp::Time& time,
                                           const rclcpp::Duration& period) override;
  CallbackReturn on_init() override;
  CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;

 private:
  std::string arm_id_;
  const int num_joints = 7;
  double command_timeout_sec_{0.2};
  bool require_fresh_command_on_activate_{true};
  Vector7d q_;               // from state interface
  Vector7d dq_;              // from state interface
  Vector7d tau_measured_;    // from state interface
  Vector7d tau_d_received_;  // from effort command, to be sent via command interface
  std::array<double, 16> o_t_f_;
  std::array<double, 49> mass_;
  std::array<double, 7> coriolis_;
  std::array<double, 42> zero_jacobian_flange_;
  std::array<double, 7> gravity_;

  // Franka robot model for kinematics and dynamics
  std::unique_ptr<franka_semantic_components::FrankaRobotModel> franka_robot_model_;
  const std::string k_robot_state_interface_name{"robot_state"};
  const std::string k_robot_model_interface_name{"robot_model"};

  // Subuscriber
  rclcpp::Subscription<custom_msgs::msg::EffortCommand>::SharedPtr effort_command_sub_ ;
  bool received_effort_command_{false} ;
  rclcpp::Time last_command_time_;
  std::mutex command_mutex_;
  void effortCommandCallback(const custom_msgs::msg::EffortCommand::SharedPtr msg) ;
  void setZeroCommandInterfaces();
  bool isCommandFresh(const rclcpp::Time& now, const rclcpp::Time& last_command_time) const;


  // Publisher
  rclcpp::Publisher<custom_msgs::msg::StateParameter>::SharedPtr state_param_pub_;
  void updateStateParam();


};

}  // namespace cpp_relayer
