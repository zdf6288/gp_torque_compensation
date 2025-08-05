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
#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <exception>
#include <new_controllers/joint_velocity_rcm_controller.hpp>
#include <new_controllers/rcm_kinematics.hpp>
#include <franka_msgs/srv/set_full_collision_behavior.hpp>
#include <string>
#include <chrono>
#include <franka/rate_limiting.h>
#include <franka/lowpass_filter.h>
#include <franka/robot_state.h>


using namespace std::chrono_literals;

namespace new_controllers {



controller_interface::InterfaceConfiguration JointVelocityRCMController::command_interface_configuration() const {
    controller_interface::InterfaceConfiguration config;
    config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

    for (int i = 1; i <= num_joints; ++i) {
        config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/velocity");
        // config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/effort");

    }
    return config;
}



controller_interface::InterfaceConfiguration JointVelocityRCMController::state_interface_configuration() const {
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

controller_interface::return_type JointVelocityRCMController::update(const rclcpp::Time& /*time*/, const rclcpp::Duration& period) {

    /* Retrieve sample step */
    double dt = period.seconds();
    time_ += dt;
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

    Eigen::Matrix<double, 8, 1> Null2_error;

    for (size_t i = 0; i < 8; ++i) {
        Null2_error[i] = q_init_[i] - q_meas_[i];
        if (i == 7) { // eta is the last element in q_
            Null2_error[7] = q_init_[i] - eta_; // eta is not controlled, so we set its error to 0
        }
    }

    /* RCM multipriority control law */
    dq_ = J_rcm_plus * (k_rcm_ * eye3 * (error_rcm)) + 
                J_endo_proj_plus * ((v_endo_ref + k_endo_ * eye3 * (error_endo)) - J_endo * J_rcm_plus * (k_rcm_ * eye3 * (error_rcm))) 
                        + Null_Proj_2 * (10 * Null2_error) ;

    /* Saturate commandes velocities */
    for (auto idx = 0; idx < 7; idx++) {
        double alpha = exp(-2 * M_PI * 5 * dt); 
        // if ( idx == 4 || idx == 5 || idx == 6) {
        //     alpha = exp(-2 * M_PI * 1 * dt); 
        // } else {
        //     alpha = exp(-2 * M_PI * 2.5 * dt); 
        // }
        alpha = 0.939;
        dq_[idx] = alpha * dq_old_[idx] + (1 - alpha) * dq_[idx];
        dq_[idx] = std ::min(10.0 * dt + dq_old_[idx], std ::max(-10.0 * dt + dq_old_[idx], dq_[idx])); 
        // Do velocity limitation based on rate_limiting.h

        ddq_[idx] = (dq_[idx] - dq_old_[idx]) / dt;
        // ddq_[idx] = franka::lowpassFilter(dt, ddq_[idx], ddq_old_[idx], 1);

    }

    // std::array<double, 7> upper_velocity_limits = franka::computeUpperLimitsJointVelocity(
    //     std::array<double, 7>{q_[0], q_[1], q_[2], q_[3], q_[4], q_[5], q_[6]});
    // std::array<double, 7> lower_velocity_limits = franka::computeLowerLimitsJointVelocity(
    //     std::array<double, 7>{q_[0], q_[1], q_[2], q_[3], q_[4], q_[5], q_[6]});

    // auto limited_dq = franka::limitRate(
    //     upper_velocity_limits,
    //     lower_velocity_limits,
    //     franka::kMaxJointAcceleration,
    //     franka::kMaxJointJerk,
    //     std::array<double, 7>{dq_[0], dq_[1], dq_[2], dq_[3], dq_[4], dq_[5], dq_[6]},                // commanded velocities
    //     std::array<double, 7>{ddq_[0], ddq_[1], ddq_[2], ddq_[3], ddq_[4], ddq_[5], ddq_[6]},                // last commanded velocities
    //     std::array<double, 7>{ddq_old_[0], ddq_old_[1], ddq_old_[2], ddq_old_[3], ddq_old_[4], ddq_old_[5], ddq_old_[6]}                // last commanded velocities
    // );
    // double dq_8 = dq_[7];
    // dq_.head<7>() = Eigen::Map<const Eigen::Matrix<double, 7, 1>>(limited_dq.data());
    // dq_[7] = dq_8;  
    // std::cout<<"limited dq: "<<limited_dq[0]<<" "<<limited_dq[1]<<" "<<limited_dq[2]<<" "<<limited_dq[3]<<" "<<limited_dq[4]<<" "<<limited_dq[5]<<" "<<limited_dq[6]<<std::endl;

    q_ = q_ + dq_ * dt;
    eta_ = q_[7];


    /* Send command to Joint Velocity Interface */
    for (int i = 0; i < num_joints; i++) {
        command_interfaces_[i].set_value(dq_[i]);
    }

    // save previous velocity and acc
    for (size_t i = 0; i < 7; ++i) {
        dq_old_[i] = dq_[i];
        ddq_old_[i] = ddq_[i];
    }


    // // std::array<double, 7> K_q = {12, 12, 12, 3, 1, 0.5, 0.2};

    // std::array<double, 7> K_q = {12, 12, 12, 3, 2, 1, 0.85};
    // std::array<double, 7> K_i = {1, 1, 1, 1, 0.2, 0.2, 0.1}; //

    // std::array<double, 7> tau_d_desired;
    // std::array<double, 49> mass_matrix = franka_robot_model_->getMassMatrix();
    // // std::array<double, 7> coriolis_vector = franka_robot_model_->getCoriolisForceVector();
    // std::array<double, 7> gravity = franka_robot_model_->getGravityForceVector();
    // // std::array<double, 7> eigenvalues = computeEigenvalues(mass_matrix);

    // std::array<double, 7> lambda = {5, 5, 5, 5, 5, 5, 5};
    // double ratio_of_lambda = 0.5;
    // std::array<double, 7> dq_r;
    // std::array<double, 7> ddq_r;
    // std::array<double, 7> s;
    // for (size_t i = 0; i < 7; ++i) {
    //     double vel_err = dq_[i] - dq_meas_[i];
    //     double pos_err = q_[i] - q_meas_[i];
    //     // K_q[i] = 10 * K_q[i];
    //     // tau_d_desired[i] = 2 * M_PI * sqrt(K_q[i]) * vel_err +  1 * K_q[i] * pos_err;

    //     // passivity based controller part --> these params temporally working
    //     // dq_r[i] = dq_[i] + ratio_of_lambda * lambda[i] * pos_err;
    //     // s[i] = vel_err+ ratio_of_lambda * lambda[i] * pos_err;
    //     // ddq_r[i] = ddq_[i] + ratio_of_lambda * lambda[i] * vel_err;
    //     // for (size_t j = 0; j < 7; ++j) {
    //     //     tau_d_desired[i] += mass_matrix[i * 7 + j] * ddq_r[j];
    //     // }
    //     // K_q[i] = 10 * K_q[i];
    //     // tau_d_desired[i] += 0.25 * sqrt(K_q[i]) * s[i];

    //     // PD control w.r.t. velocity reference
    //     for (size_t j = 0; j < 7; ++j) {
    //         tau_d_desired[i] += mass_matrix[i * 7 + j] * ddq_[j];
    //     }
    //     K_q[i] = 10 * K_q[i];
    //     tau_d_desired[i] = 2 * 3 * sqrt(K_q[i]) * vel_err + 1 * K_q[i] * pos_err;

    //     if ( i == 4 || i == 5 || i == 6) {
    //         tau_d_desired[i] = franka::lowpassFilter(dt, tau_d_desired[i], tau_d_prev_[i], 2);        
    //     } else {
    //         tau_d_desired[i] = franka::lowpassFilter(dt, tau_d_desired[i], tau_d_prev_[i], 5);
    //     }
        
    // }

    // std::array<double, 7> max_delta_tau;
    // for (size_t i = 0; i < 7; ++i)
    // max_delta_tau[i] = franka::kMaxTorqueRate[i] * 0.25;
    // // for (size_t i = 0; i < 7; ++i){
    // //     tau_err_integral_[i] += (tau_d_desired[i] - tau_meas_[i] + gravity[i]) * dt;
    // //     tau_d_desired[i] += 0.1 * tau_err_integral_[i];
    // // }
    // std::array<double, 7> tau_d_limited = franka::limitRate(max_delta_tau, tau_d_desired, tau_d_prev_);
    // for (size_t i = 0; i < 7; ++i)
    // command_interfaces_[i].set_value(tau_d_limited[i]);
    // // std::cout<<"tau_d_limited: "<<tau_d_limited[0]<<" "<<tau_d_limited[1]<<" "<<tau_d_limited[2]<<" "<<tau_d_limited[3]<<" "<<tau_d_limited[4]<<" "<<tau_d_limited[5]<<" "<<tau_d_limited[6]<<std::endl;

    // for (size_t i = 0; i < 7; ++i) {
    //     tau_d_prev_[i] = tau_d_limited[i];
    //     dq_meas_prev_[i] = dq_meas_[i];
    // }


    return controller_interface::return_type::OK;
}



CallbackReturn JointVelocityRCMController::on_init() {

    /* Get node */
    auto node = get_node();

    /* Create subscriber */
    lambda_subscription_ = node->create_subscription<custom_msgs::msg::LambdaCommand>("/TwistRight", 10, std::bind(&JointVelocityRCMController::lambdaCommandCallback, this, std::placeholders::_1));

    auto node_pub = get_node(); 

    // Create publisher for intermediate variables
    intermediate_var_publisher_ = node_pub->create_publisher<custom_msgs::msg::CustomVar>(
        "/left_arm/intermediate_variables", 10);

    // Optional: Create a timer for periodic publishing (if needed)
    publish_timer_ = node->create_wall_timer(
        std::chrono::milliseconds(1),  // Publish every 100ms
        std::bind(&JointVelocityRCMController::publishIntermediateVariables, this));



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
    ddq_old_.setZero();
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

    k_rcm_ = 0.2;  //TODO: Read from configuration / ROS param
    k_endo_ = 0.05; //TODO: Read from configuration / ROS param

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

    // configure the model interface
    franka_robot_model_ = std::make_unique<franka_semantic_components::FrankaRobotModel>(
    franka_semantic_components::FrankaRobotModel(arm_id_ + "/" + k_robot_model_interface_name, arm_id_ + "/" + k_robot_state_interface_name));
    RCLCPP_DEBUG(get_node()->get_logger(), "configured successfully");

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
    q_init_ = q_;
    franka_robot_model_->assign_loaned_state_interfaces(state_interfaces_);

    return CallbackReturn::SUCCESS;
}



void JointVelocityRCMController::updateJointStates() {
    for (auto i = 0; i < num_joints; ++i) {

        const auto& position_interface = state_interfaces_.at(3 * i);

        assert(position_interface.get_interface_name() == "position");
        q_meas_[i] = position_interface.get_value();

        assert(state_interfaces_.at(3 * i + 1).get_interface_name() == "velocity");
        dq_meas_[i] = state_interfaces_.at(3 * i + 1).get_value();

        assert(state_interfaces_.at(3 * i + 2).get_interface_name() == "effort");
        tau_meas_[i] = state_interfaces_.at(3 * i + 2).get_value();
    }
}


// Function to compute eigenvalues
std::array<double, 7> JointVelocityRCMController::computeEigenvalues(const std::array<double, 49>& mass_matrix) {
    // Convert std::array to Eigen matrix
    Eigen::Matrix<double, 7, 7> matrix;
    for (size_t i = 0; i < 7; ++i) {
        for (size_t j = 0; j < 7; ++j) {
            matrix(i, j) = mass_matrix[i * 7 + j];
        }
    }

    // Perform eigendecomposition
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 7, 7>> solver(matrix);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Eigendecomposition failed!");
    }

    // Extract eigenvalues
    Eigen::VectorXd eigenvalues = solver.eigenvalues();
    std::array<double, 7> eigenvalues_array;
    for (size_t i = 0; i < 7; ++i) {
        eigenvalues_array[i] = eigenvalues(i);
    }

    return eigenvalues_array;
}


}  // namespace new_controllers
#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(new_controllers::JointVelocityRCMController, controller_interface::ControllerInterface)