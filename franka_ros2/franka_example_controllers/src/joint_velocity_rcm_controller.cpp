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

#include <Eigen/Eigen>
#include <cassert>
#include <cmath>
#include <exception>
#include <franka_example_controllers/joint_velocity_rcm_controller.hpp>
#include <franka_example_controllers/rcm_kinematics.hpp>
#include <franka_msgs/srv/set_full_collision_behavior.hpp>
#include <string>
#include <chrono>

using namespace std::chrono_literals;

namespace franka_example_controllers {



controller_interface::InterfaceConfiguration JointVelocityRCMController::command_interface_configuration() const {
    controller_interface::InterfaceConfiguration config;
    config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

    for (int i = 1; i <= num_joints; ++i) {
        config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/velocity");
    }
    return config;
}



controller_interface::InterfaceConfiguration JointVelocityRCMController::state_interface_configuration() const {
    controller_interface::InterfaceConfiguration config;
    config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
    for (int i = 1; i <= num_joints; ++i) {
        config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/position");
        config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/velocity");
    }
    return config;
}

controller_interface::return_type JointVelocityRCMController::update(const rclcpp::Time& /*time*/, const rclcpp::Duration& period) {

    /* Retrieve sample step */
    double dt = period.seconds();

    /* Obtain positions and Jacobians from Endowrist and RCM (from measurement + forward kinematics) */
    updateJointStates();
    RCMForwardKinematics(q_meas_, eta_, T_endo, T_rcm, J_endo, J_rcm);
    p_rcm  = T_rcm.block<3, 1>(0, 3);
    p_endo = T_endo.block<3, 1>(0, 3);

    /* Obtain reference trajectory */
    v_endo_ref << v_x, v_y, v_z;                   // TODO: mutex protection
    p_endo_ref << p_endo_ref + dt * v_endo_ref;

    /* Calculate RCM and tracking error */
    error_endo = p_endo_ref - p_endo;
    error_rcm = p_rcm_init_ - p_rcm;

    /* Nullspace objective (maximize distance to joint limits) */
    for (int i = 0; i < 8; i++) {
        w_func[i] = (-1 / 16) * pow(((q_[i] - (qmax[i] - qmin[i]) / 2) / (qmax[i] - qmin[i])), 2);
    }
    dwdq = (w_func - w_func_old_) / dt;
    w_func_old_ = w_func;

    /* Nullspace projections */
    J_rcm_plus = J_rcm.transpose() * (J_rcm * J_rcm.transpose()).inverse(); //inverse 1
    Null_Proj_1 = eye8 - J_rcm_plus * J_rcm;
    J_endo_proj = J_endo * Null_Proj_1;
    J_endo_proj_plus = J_endo_proj.transpose() * (J_endo_proj * J_endo_proj.transpose()).inverse(); //inverse 2
    Null_Proj_2 = Null_Proj_1 * (eye8 - J_endo_proj_plus * J_endo_proj);


    /* RCM multipriority control law */
    dq_ = J_rcm_plus * (k_rcm_ * eye3 * (error_rcm)) + 
                J_endo_proj_plus * ((v_endo_ref + k_endo_ * eye3 * (error_endo)) - J_endo * J_rcm_plus * (k_rcm_ * eye3 * (error_rcm))) 
                        + Null_Proj_2 * dwdq;

    /* Integrate to get joint angles*/
    q_ = q_ + dq_ * dt;
    eta_ = q_[7];

    /* Saturate commandes velocities */
    for (auto idx = 0; idx < 7; idx++) {
        // Filter joint velocities to avoid discontinuities 
        dq_[idx] = 0.8*dq_old_[idx] + 0.2*dq_[idx];
        dq_[idx] = std ::min(10.0 * dt + dq_old_[idx], std ::max(-10.0 * dt + dq_old_[idx], dq_[idx])); 
        dq_old_[idx] = dq_[idx];
    }

    /* Send command to Joint Velocity Interface */
    for (int i = 0; i < num_joints; i++) {
        command_interfaces_[i].set_value(dq_[i]);
    }

    return controller_interface::return_type::OK;
}



CallbackReturn JointVelocityRCMController::on_init() {

    /* Get node */
    auto node = get_node();

    /* Create subscriber */
    lambda_subscription_ = node->create_subscription<custom_msgs::msg::LambdaCommand>("/TwistRight", 10, std::bind(&JointVelocityRCMController::lambdaCommandCallback, this, std::placeholders::_1));

    /* Init Franka Arm */
    try {
        auto_declare<std::string>("arm_id", "panda");

    } catch (const std::exception& e) {
        fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
        return CallbackReturn::ERROR;
    }

    /* Init Control Variables */
    q_.setZero();
    dq_old_.setZero();
    qmax << 2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5165, 3.0159, 0.59;
    qmin << -2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159, 0;
    p_rcm.setZero();
    p_rcm_init_.setZero();
    p_endo_ref.setZero();
    v_endo_ref.setZero();
    w_func_old_.setZero();
    eye3.setIdentity();
    eye8.setIdentity();
    eta_ = 0.435;

    k_rcm_ = 5.0;  //TODO: Read from configuration / ROS param
    k_endo_ = 5.0; //TODO: Read from configuration / ROS param

    return CallbackReturn::SUCCESS;
}



CallbackReturn JointVelocityRCMController::on_configure(const rclcpp_lifecycle::State& /*previous_state*/) {

    /* Initialize robot state / model interface */
    arm_id_ = get_node()->get_parameter("arm_id").as_string();

    auto client = get_node()->create_client<franka_msgs::srv::SetFullCollisionBehavior>("service_server/set_full_collision_behavior");
    auto request = std::make_shared<franka_msgs::srv::SetFullCollisionBehavior::Request>();

    request->lower_torque_thresholds_acceleration = {20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0};
    request->upper_torque_thresholds_acceleration = {20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0};
    request->lower_torque_thresholds_nominal = {20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0};
    request->upper_torque_thresholds_nominal = {20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0};
    request->lower_force_thresholds_acceleration = {20.0, 20.0, 20.0, 25.0, 25.0, 25.0};
    request->upper_force_thresholds_acceleration = {20.0, 20.0, 20.0, 25.0, 25.0, 25.0};
    request->lower_force_thresholds_nominal = {20.0, 20.0, 20.0, 25.0, 25.0, 25.0};
    request->upper_force_thresholds_nominal = {20.0, 20.0, 20.0, 25.0, 25.0, 25.0};

    if (!client->wait_for_service(20s)) {
        RCLCPP_FATAL(get_node()->get_logger(), "service server can't be found.");
        return CallbackReturn::FAILURE;
    }

    client->async_send_request(request);

    return CallbackReturn::SUCCESS;
}



CallbackReturn JointVelocityRCMController::on_activate(const rclcpp_lifecycle::State& /*previous_state*/) {
    updateJointStates();
    
    RCMForwardKinematics(q_meas_, eta_, T_endo, T_rcm, J_endo, J_rcm);
    p_rcm  = T_rcm.block<3, 1>(0, 3);
    p_endo = T_endo.block<3, 1>(0, 3);

    p_rcm_init_ = p_rcm;
    p_endo_ref = p_endo;

    Eigen::Map<const Eigen::Matrix<double, 7, 1>> q0(q_meas_.data());
    q_ << q0, eta_;
    return CallbackReturn::SUCCESS;
}



void JointVelocityRCMController::updateJointStates() {
    for (auto i = 0; i < num_joints; ++i) {
        const auto& position_interface = state_interfaces_.at(2 * i);

        assert(position_interface.get_interface_name() == "position");

        q_meas_[i] = position_interface.get_value();
    }
}

}  // namespace franka_example_controllers
#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(franka_example_controllers::JointVelocityRCMController, controller_interface::ControllerInterface)