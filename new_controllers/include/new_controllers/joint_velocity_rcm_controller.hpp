#pragma once

#include <Eigen/Eigen>
#include <controller_interface/controller_interface.hpp>
#include <rclcpp/rclcpp.hpp>
#include <string>
#include <array>

#include "std_msgs/msg/string.hpp"
#include "custom_msgs/msg/lambda_command.hpp"
#include "custom_msgs/msg/motor_measurement.hpp"
#include "custom_msgs/msg/motor_command.hpp"

#ifndef RCM_KINEMATICS_HPP
#define RCM_KINEMATICS_HPP
#include "rcm_kinematics.hpp"
#endif

#define ROBOT_TO_CONSOLE_RATIO 2.0
#define MAX_RAD_PER_SEC 80.0 / 60.0 * 2.0 * M_PI
#define MIN_RAD_PER_SEC -80.0 / 60.0 * 2.0 * M_PI
#define MOTOR_ENDOWRIST_PULLEY_RADIUS 0.1  // 0.008 // m
#define HAPTIC_CONSOLE_MESSAGE_LENGTH 23
#define FRAME_MESSAGE_LENGTH 16
#define ENDOWRIST_MESSAGE_LENGTH 4
#define MAX_POSITION_LEFT_JAW 85.0 / 180.0 * M_PI    // rad
#define MIN_POSITION_LEFT_JAW -85.0 / 180.0 * M_PI   // rad
#define MAX_POSITION_RIGHT_JAW 85.0 / 180.0 * M_PI   // rad
#define MIN_POSITION_RIGHT_JAW -85.0 / 180.0 * M_PI  // rad
#define MAX_POSITION_WRIST 70.0 / 180.0 * M_PI       // rad
#define MIN_POSITION_WRIST -70.0 / 180.0 * M_PI      // rad
#define MAX_POSITION_SHAFT 325.0 / 180.0 * M_PI      // rad
#define MIN_POSITION_SHAFT -325.0 / 180.0 * M_PI     // rad

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace new_controllers {


class JointVelocityRCMController : public controller_interface::ControllerInterface {
   public:
    [[nodiscard]] controller_interface::InterfaceConfiguration command_interface_configuration() const override;
    [[nodiscard]] controller_interface::InterfaceConfiguration state_interface_configuration() const override;
    controller_interface::return_type update(const rclcpp::Time& time, const rclcpp::Duration& period) override;
    CallbackReturn on_init() override;
    CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;
    CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;

   private:
    std::string arm_id_;
    const int num_joints = 7;

    /* Declare Control Variables */
    std::array<double, 7> q_meas_;
    Eigen::Matrix<double, 8, 1> q_, dq_, dq_old_;
    Eigen::Matrix<double, 8, 1> qmax, qmin;
    Eigen::Matrix<double, 4, 4> T_rcm, T_endo;
    Eigen::Matrix<double, 3, 1> p_rcm, p_endo, p_rcm_init_, p_endo_ref, v_endo_ref, error_endo, error_rcm;
    Eigen::Matrix<double, 3, 8> J_rcm, J_endo, J_endo_proj;
    Eigen::Matrix<double, 8, 3> J_rcm_plus, J_endo_proj_plus;
    Eigen::Matrix<double, 8, 1> w_func, w_func_old_, dwdq;
    Eigen::Matrix<double, 8, 8> Null_Proj_1, Null_Proj_2, eye8;
    Eigen::Matrix<double, 3, 3> eye3;
    double k_rcm_, k_endo_;
    double eta_;

    /* caculate motor velocities */
    double v_gripper = 0.;
    double v_gripper_scale_ = 1.; // TODO: read from configuration file
    std::array<double, 4> motor_position_ = {0., 0., 0., 0.};
    std::array<double, 4> motor_actual_torque_ = {0., 0., 0., 0.};
    std::array<double, 4> motor_actual_velocity_ = {0., 0., 0., 0.};
    std::array<double, 4> motor_actual_position_ = {0., 0., 0., 0.};
    std::array<double, 4> motor_integral_position_ = {0., 0., 0., 0.};

    double v_x = 0.;
    double v_y = 0.;
    double v_z = 0.;

    const double gripper_gain = 5.0;
    Eigen::Matrix4d T_lambda_motor; // TODO: initialize as const

    /* Subscriber */
    custom_msgs::msg::LambdaCommand lambda_command_;
    rclcpp::Subscription<custom_msgs::msg::LambdaCommand>::SharedPtr lambda_subscription_;

    /* Update Joint States */
    void updateJointStates();

    /* Subscriber Callbacks */
    void lambdaCommandCallback(const custom_msgs::msg::LambdaCommand::SharedPtr msg) { 
        lambda_command_ = *msg;

        v_x = std ::min(2.0, std ::max(-2.0, -lambda_command_.linear.y)) / ROBOT_TO_CONSOLE_RATIO;  
        v_y = std ::min(2.0, std ::max(-2.0, lambda_command_.linear.x)) / ROBOT_TO_CONSOLE_RATIO;  
        v_z = std ::min(2.0, std ::max(-2.0, lambda_command_.linear.z)) / ROBOT_TO_CONSOLE_RATIO;
    }

};


}  // namespace new_controllers