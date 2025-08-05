// Copyright (c) 2021 Franka Emika GmbH
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <Eigen/Eigen>
#include <controller_interface/controller_interface.hpp>
#include <rclcpp/rclcpp.hpp>
#include <string>
#include <array>
#include "franka_semantic_components/franka_robot_model.hpp"
#include "hardware_interface/types/hardware_interface_return_values.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"

#include "std_msgs/msg/string.hpp"
#include "custom_msgs/msg/lambda_command.hpp"
#include "custom_msgs/msg/motor_measurement.hpp"
#include "custom_msgs/msg/motor_command.hpp"
#include "custom_msgs/msg/custom_var.hpp"
#include <std_msgs/msg/float64_multi_array.hpp>

#ifndef RCM_KINEMATICS_HPP
#define RCM_KINEMATICS_HPP
#include "rcm_kinematics.hpp"
#endif

#define ROBOT_TO_CONSOLE_RATIO 4.0
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
    const std::string k_robot_model_interface_name{"robot_model"};
    const std::string k_robot_state_interface_name{"robot_state"};
    std::unique_ptr<franka_semantic_components::FrankaRobotModel> franka_robot_model_;

    /* Declare Control Variables */
    std::array<double, 7> q_meas_, dq_meas_, dq_meas_prev_, q_errI_;
    std::array<double, 7> tau_d_prev_, tau_meas_, tau_err_integral_;
    Eigen::Matrix<double, 8, 1> q_, dq_, dq_old_, ddq_, ddq_old_, q_init_;
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
    double time_;

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
    rclcpp::Publisher<custom_msgs::msg::CustomVar>::SharedPtr intermediate_var_publisher_;
    rclcpp::TimerBase::SharedPtr publish_timer_;
    std::vector<std::reference_wrapper<const hardware_interface::LoanedStateInterface>> effort_interfaces_;

    void publishIntermediateVariables() {
        auto message = custom_msgs::msg::CustomVar();
        message.q_des.data.resize(7);  
        for (size_t i = 0; i < 7; ++i) {
            message.q_des.data[i] = q_(i);
        }
        message.q_meas.data.resize(7);  
        for (size_t i = 0; i < 7; ++i) {
            message.q_meas.data[i] = q_meas_[i];
        }
        message.dq_des.data.resize(7); 
        for (size_t i = 0; i < 7; ++i) {
            message.dq_des.data[i] = dq_(i);
        }
        message.dq_meas.data.resize(7); 
        for (size_t i = 0; i < 7; ++i) {
            message.dq_meas.data[i] = dq_meas_[i];
        }
        message.ddq_computed.data.resize(7); 
        for (size_t i = 0; i < 7; ++i) {
            message.ddq_computed.data[i] = ddq_(i);
        }
        message.error_endo.data.resize(3);
        for (size_t i = 0; i < 3; ++i) {
            message.error_endo.data[i] = error_endo(i);
        }
        message.error_rcm.data.resize(3);
        for (size_t i = 0; i < 3; ++i) {
            message.error_rcm.data[i] = error_rcm(i);
        }
        intermediate_var_publisher_->publish(message);
    };



    /* Update Joint States */
    void updateJointStates();

    /* Mass eigenvalue decomposition */
    std::array<double, 7> computeEigenvalues(const std::array<double, 49>& mass_matrix);

    int counter = 1;
    /* Subscriber Callbacks */
    void lambdaCommandCallback(const custom_msgs::msg::LambdaCommand::SharedPtr msg) { 
        lambda_command_ = *msg;

        v_x = std ::min(2.0, std ::max(-2.0, lambda_command_.linear.x)) / ROBOT_TO_CONSOLE_RATIO;  
        v_y = std ::min(2.0, std ::max(-2.0, lambda_command_.linear.y)) / ROBOT_TO_CONSOLE_RATIO;  
        v_z = std ::min(2.0, std ::max(-2.0, lambda_command_.linear.z)) / ROBOT_TO_CONSOLE_RATIO;

    }

};


}  // namespace franka_example_controllers