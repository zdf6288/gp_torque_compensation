// Copyright 2016 Open Source Robotics Foundation, Inc.
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

#include <maxon_epos_ethercat_sdk/Maxon.hpp>

#include "endowrist_control/ethercat_device_configurator/EthercatDeviceConfigurator.hpp"
// System libraries
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <functional>
#include <iostream>
#include <memory>
#include <string>

#include "custom_msgs/msg/lambda_command.hpp"
#include "custom_msgs/msg/motor_command.hpp"
#include "custom_msgs/msg/motor_measurement.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

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

using std::placeholders::_1;

class MotorSubscriber : public rclcpp::Node {
   public:
    MotorSubscriber() : Node("MotorSubscriber") {
        subscription_ = this->create_subscription<custom_msgs::msg::LambdaCommand>("/TwistRight", 10, std::bind(&MotorSubscriber::lambda_callback, this, _1));


        std::string config_file = "./src/endowrist_control/example_config/endoWrist.yaml";
        configurator_ = std::make_shared<EthercatDeviceConfigurator>(config_file);

        /*
        ** Add callbacks to the devices that support them.
        ** If you don't want to use callbacks this part can simply be left out.
        ** configurator->getSlavesOfType is another way of extracting only the evices
        ** of a ceratin type.

        ** Start all masters.
        ** There is exactly one bus per master which is also started.
        ** All online (i.e. SDO) configuration is done during this call.
        ** The EtherCAT interface is active afterwards, all drives are in Operational
        ** EtherCAT state and PDO communication may begin.
        */
        for (auto& master : configurator_->getMasters()) {
            if (!master->startup()) {
                std::cerr << "Startup not successful." << std::endl;
                // TODO: TERMINATE
            }
        }

        rtSuccess = true;
        for (const auto& master : configurator_->getMasters()) {
            rtSuccess &= master->setRealtimePriority(99);
        }
        std::cout << "Setting RT Priority: " << (rtSuccess ? "successful." : "not successful. Check user privileges.") << std::endl;

        maxonEnabledAfterStartup = false;

        worker_thread_ = std::thread(&MotorSubscriber::worker, this);

        /*
        ** Wait for a few PDO cycles to pass.
        ** Set anydrives into to ControlOp state (internal state machine, not EtherCAT states)
        */
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        for (auto& slave : configurator_->getSlaves()) {
            std::cout << " " << slave->getName() << ": " << slave->getAddress() << std::endl;
        }

        std::cout << "Startup finished" << std::endl;
    }

    ~MotorSubscriber() {  // TODO: signal handler of ros2
        for (const auto& master : configurator_->getMasters()) {
            master->preShutdown();
        }
        abrt = true;
        worker_thread_.join();
        for (const auto& master : configurator_->getMasters()) {
            master->shutdown();
        }
    }

   private:
    std::thread worker_thread_;
    bool abrt = false;
    bool rtSuccess = false;
    bool maxonEnabledAfterStartup = false;

    double dt = 0.001;

    std::array<double, 4> measured_torque_;
    std::array<double, 4> motor_position_ = {0., 0., 0., 0.};
    std::array<double, 4> motor_integral_position_ = {0., 0., 0., 0.};

    double w_x = 0.;
    double w_y = 0.;
    double w_z = 0.;
    double v_gripper = 0.;
    bool enable_backlash_comp;

    double gripper_gain = 5.0;
    std::array<double, 16> T_lambda_motor = {1.0, 0.0, 0.0, 0.0,
                                             0.0, 1.0, 0.0, 0.0, 
                                             0.0, 0.52, 1.0, gripper_gain,
                                             0.0, -0.52, -1.0, gripper_gain}; 

    EthercatDeviceConfigurator::SharedPtr configurator_;

    rclcpp::Subscription<custom_msgs::msg::LambdaCommand>::SharedPtr subscription_;

    void lambda_callback(const custom_msgs::msg::LambdaCommand::SharedPtr msg) {
        custom_msgs::msg::LambdaCommand lambda_command_ = *msg;

        w_x = std ::max(MIN_RAD_PER_SEC, std::min(MAX_RAD_PER_SEC, lambda_command_.angular.x));
        w_y = std ::max(MIN_RAD_PER_SEC, std::min(MAX_RAD_PER_SEC, lambda_command_.angular.y));
        w_z = -std ::max(MIN_RAD_PER_SEC, std::min(MAX_RAD_PER_SEC, lambda_command_.angular.z));
        v_gripper = -(std ::max(MIN_RAD_PER_SEC, std::min(MAX_RAD_PER_SEC, lambda_command_.v_gripper / (MOTOR_ENDOWRIST_PULLEY_RADIUS))));
        enable_backlash_comp = lambda_command_.enable_backlash_compensation;
    }

    void worker() {
        while (!abrt) {
            // std::cout << "worker while loop \n";
            auto start_time = std::chrono::steady_clock::now();
            /*
            ** Update each master.
            ** This sends tha last staged commands and reads the latest readings over EtherCAT.
            ** The StandaloneEnforceRate update mode is used.
            ** This means that average update rate will be close to the target rate (if possible).
            */
            for (const auto& master : configurator_->getMasters()) {
                master->update(ecat_master::UpdateMode::StandaloneEnforceRate);  // TODO fix the rate compensation (Elmo
                                                                                 // reliability problem)!!
            }

            /*** Convert lambda message to motor commands ***/

            /* motor backlash compensation */
            if (enable_backlash_comp) {
                if (abs(measured_torque_[2]) + abs(measured_torque_[3]) < 0.001) {
                    v_gripper = -0.1;
                } else {
                    v_gripper = 0.0;
                }
            }


            /*
            ** Do things with the attached devices.
            ** Your lowlevel control input / measurement logic goes here.
            ** Different logic can be implemented for each device.
            */
            size_t slave_id = 0;
            for (const auto& slave : configurator_->getSlaves()) {
                // std::cout << "worker for loop \n";
                // Anydrive

                // Maxon
                if (configurator_->getInfoForSlave(slave).type == EthercatDeviceConfigurator::EthercatSlaveType::Maxon) {
                    std::shared_ptr<maxon::Maxon> maxon_slave_ptr = std::dynamic_pointer_cast<maxon::Maxon>(slave);

                    if (!maxonEnabledAfterStartup) {
                        // Set maxons to operation enabled state, do not block the call!
                        maxon_slave_ptr->setDriveStateViaPdo(maxon::DriveState::OperationEnabled, false);
                    }

                    // set commands if we can
                    if (maxon_slave_ptr->lastPdoStateChangeSuccessful() && maxon_slave_ptr->getReading().getDriveState() == maxon::DriveState::OperationEnabled) {

                        maxon::Command command;
                        // command.setModeOfOperation(maxon::ModeOfOperationEnum::CyclicSynchronousTorqueMode);

                        command.setModeOfOperation(maxon::ModeOfOperationEnum::CyclicSynchronousVelocityMode);
                        auto reading = maxon_slave_ptr->getReading();
                        double actualVelocity = reading.getActualVelocity();
                        double actualTorque = reading.getActualTorque();
                        measured_torque_[slave_id] = actualTorque;

                        motor_integral_position_[slave_id] = motor_integral_position_[slave_id] + actualVelocity * dt;

                        double motor_command = w_x * T_lambda_motor[4*slave_id+0] + w_y * T_lambda_motor[4*slave_id+1] + 
                                               w_z * T_lambda_motor[4*slave_id+2] + v_gripper * T_lambda_motor[4*slave_id+3];

                        if ((slave_id == 0 || slave_id == 1) && (motor_integral_position_[slave_id] < MIN_POSITION_SHAFT || motor_integral_position_[slave_id] > MAX_POSITION_SHAFT)) {
                            motor_command = 0.0;
                        }

                        if (slave_id == 2 || slave_id == 3) {
                            motor_position_[slave_id] = motor_position_[slave_id] + v_gripper * T_lambda_motor[4*slave_id+3] * dt;
                            motor_command += ((motor_position_[slave_id] - motor_integral_position_[slave_id]) * 0);
                        }

                        /* Send motor command */
                        command.setTargetVelocity(motor_command);
                        maxon_slave_ptr->stageCommand(command);

                        // if (robot_stop_flag) abrt = true; //TODO

                    } else {
                        MELO_WARN_STREAM("Maxon '" << maxon_slave_ptr->getName() << "': " << maxon_slave_ptr->getReading().getDriveState());
                    }

                    // publisher_->publish(motor_measurement_);
                    // std::cout << "motor measurement: " << motor_measurement_.vel[slave_id] << "\n";
                    // std::this_thread::sleep_until(start_time + std::chrono::microseconds(1));
                    slave_id += 1;
                }
            }
            maxonEnabledAfterStartup = true;

            // Constant update rate
            std::this_thread::sleep_until(start_time + std::chrono::microseconds(940));
            auto const delta_time = std::chrono::steady_clock::now() - start_time;
        }
    }
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MotorSubscriber>());
    rclcpp::shutdown();
    return 0;
}