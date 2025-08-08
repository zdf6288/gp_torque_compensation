#pragma once

#include <string>

#include <Eigen/Eigen>
#include <controller_interface/controller_interface.hpp>
#include <rclcpp/rclcpp.hpp>
#include <custom_msgs/msg/effort_command.hpp>
#include <custom_msgs/msg/state_parameter.hpp>  

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

 private:
  std::string arm_id_;
  const int num_joints = 7;
  Vector7d q_;               // from state interface
  Vector7d dq_;              // from state interface
  Vector7d tau_d_received_;  // from effort command, to be sent via command interface

  // Subuscriber
  rclcpp::Subscription<custom_msgs::msg::EffortCommand>::SharedPtr effort_command_sub_ ;
  bool received_effort_command_{false} ;
  void effortCommandCallback(const custom_msgs::msg::EffortCommand::SharedPtr msg) ;


  // Publisher
  rclcpp::Publisher<custom_msgs::msg::StateParameter>::SharedPtr state_param_pub_;
  void updateStateParam();

};

}  // namespace cpp_relayer