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

#include "endowrist_control/ethercat_device_configurator/EthercatDeviceConfigurator.hpp"

#include <maxon_epos_ethercat_sdk/Maxon.hpp>
// System libraries
#include <functional>
#include <memory>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <csignal>
#include <functional>
#include <iostream>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "geometry_msgs/msg/twist.hpp"

using std::placeholders::_1;

class MotorSubscriber : public rclcpp::Node
{
public:
  MotorSubscriber()
  : Node("MotorSubscriber")
  { 

    std::string config_file="./src/endowrist_control/example_config/endoWrist.yaml";
    // TODO: remove hardcoded path and read from launch
 
    configurator = std::make_shared<EthercatDeviceConfigurator>(config_file);
    // Control variables
    abrt = false;
    counter = 0;

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
    for (auto& master : configurator->getMasters()) {
        if (!master->startup()) {
            std::cerr << "Startup not successful." << std::endl;
            // TODO: ROS-specific return
        }
    }

    for (auto& slave : configurator->getSlaves()) {
        std::cout << " " << slave->getName() << ": " << slave->getAddress() << std::endl;
    }

    ////
    rtSuccess = true;
    for (const auto& master : configurator->getMasters()) {
        rtSuccess &= master->setRealtimePriority(99);
    }
    std::cout << "Setting RT Priority: " << (rtSuccess ? "successful." : "not successful. Check user privileges.") << std::endl;

    // Flag to set the drive state for the elmos on first startup
    maxonEnabledAfterStartup = false;

    std::cout << "[motor_subscriber]: before initializing subscrition." << "\n";
    subscription_ = this->create_subscription<geometry_msgs::msg::Twist>(
      "motor_command", 10, std::bind(&MotorSubscriber::topic_callback, this, _1));

    std::cout << "[motor_subscriber]: initialized." << "\n";
  }

private:
  // motor
  bool abrt;
  bool rtSuccess;
  unsigned int counter;
  bool maxonEnabledAfterStartup;

  std::shared_ptr<EthercatDeviceConfigurator> configurator;
  // ROS
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr subscription_;

  void topic_callback(const geometry_msgs::msg::Twist & msg)
  { 
    // std::cout<< "in topic_callback\n";
    /*
    ** Update each master.
    ** This sends tha last staged commands and reads the latest readings over EtherCAT.
    ** The StandaloneEnforceRate update mode is used.
    ** This means that average update rate will be close to the target rate (if possible).
    */
    for (const auto& master : configurator->getMasters()) {
        master->update(ecat_master::UpdateMode::StandaloneEnforceRate);  // TODO fix the rate compensation (Elmo
                                                                            // reliability problem)!!
    }

    /*
    ** Do things with the attached devices.
    ** Your lowlevel control input / measurement logic goes here.
    ** Different logic can be implemented for each device.
    */
    size_t slave_id = 0;
    for (const auto& slave : configurator->getSlaves()) {
        std::cout << "in for loop \n";
        // Maxon
        if (configurator->getInfoForSlave(slave).type == EthercatDeviceConfigurator::EthercatSlaveType::Maxon) {

            // Keep constant update rate
            // auto start_time = std::chrono::steady_clock::now();

            std::shared_ptr<maxon::Maxon> maxon_slave_ptr = std::dynamic_pointer_cast<maxon::Maxon>(slave);

            if (!maxonEnabledAfterStartup) {
                // Set maxons to operation enabled state, do not block the call!
                maxon_slave_ptr->setDriveStateViaPdo(maxon::DriveState::OperationEnabled, false);
                std::cout << "in maxonEnabledAfterStartup\n";
            }

            std::cout << "after maxonEnabledAfterStartup\n";

            // set commands if we can

            if (maxon_slave_ptr->lastPdoStateChangeSuccessful()){
                std::cout << "last state changed successfully\n";
            }

            std::cout << "DriveState: " << maxon_slave_ptr->getReading().   () << "\n";

            if (maxon_slave_ptr->lastPdoStateChangeSuccessful() && maxon_slave_ptr->getReading().getDriveState() == maxon::DriveState::OperationEnabled) {
                std::cout << "in reading \n";
                maxon::Command command;
                // command.setModeOfOperation(maxon::ModeOfOperationEnum::CyclicSynchronousTorqueMode);
                command.setModeOfOperation(maxon::ModeOfOperationEnum::CyclicSynchronousVelocityMode);
                auto reading = maxon_slave_ptr->getReading();
                double actualVelocity = reading.getActualVelocity();
                double actualPosition = reading.getActualPosition();
                double actualTorque = reading.getActualTorque();
                std::array<double, 4> motor_commands {msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.x};
                std::cout << "motor commands: " << motor_commands[0] << " " << motor_commands[1] << " "<< motor_commands[2] << " "<< motor_commands[3] << "\n";
                // double* w_vec = static_cast<double*>(region.get_address());
                // motor_register_result_flag = w_vec[4];  //*(w_vec + 5);
                // TODO: read only 4 doubles for w_vec and read one double for the motorRegisterration flag
                // std::cout << "buffer counter: " << vec_buffer_counter << "\n";
                // calculate the norm of the difference of buffer avcerage and w_vec
                // double difference_norm =
                //     std::sqrt(std::pow(buffer_average[0] - w_vec[0], 2) + std::pow(buffer_average[1] - w_vec[1], 2) + std::pow(buffer_average[2] - w_vec[2], 2) + std::pow(buffer_average[3] - w_vec[3], 2));
                // double w_vel_norm = std::sqrt(std::pow(w_vec[0], 2) + std::pow(w_vec[1], 2) + std::pow(w_vec[2], 2) + std::pow(w_vec[3], 2));

                // std::cout << "w_vel: " << w_vel_norm << "\ndifference_norm: " << difference_norm << "\nflag: " << motor_register_result_flag << std::endl;
                // if (w_vel_norm >= 1.e-2 && difference_norm <= 1.e-10 && motor_register_result_flag == 1.0) {
                //     robot_stop_flag = true;
                //     std::cout << "safety constraint triggered !!!"
                //               << "\n";
                //     w_vec[0] = 0.0;
                //     w_vec[1] = 0.0;
                //     w_vec[2] = 0.0;
                //     w_vec[3] = 0.0;
                // }

                if (maxon_slave_ptr->getName() == "Maxon0") {
                    command.setTargetVelocity(motor_commands[0]);
                    // std::memcpy(region.get_address() + MOTOR_0_ACTUAL_VELOCITY_OFFSET, &actualVelocity, 8);
                 
                    // double maxon0Error = *(w_vec)-reading.getActualVelocity();
                    // double maxon0NormalisedError = maxon0Error / (*(w_vec));
                    // std::cout << "actual position = reading.getActualPosition() = " <<
                    // reading.getActualPosition()
                    //           << "\n";
                    // std::cout << "actual torque = reading.getActualTorque() = " << reading.getActualTorque()
                    //           << "\n";
                    // std::cout << "actual velocity = reading.getActualVelocity() = " <<
                    // reading.getActualVelocity()
                    //           << "\n";
                    // std::cout << "target velocity = setTargetVelocity() = " << *(w_vec) << "\n";
                    // std::cout << "velocity error = setTargetVelocity() - reading.getActualVelocity() = "
                    //           << maxon0Error << "\n";
                    // std::cout << "normalised velocity error = " << maxon0NormalisedError << "\n";
                
                // } else if (maxon_slave_ptr->getName() == "Maxon1") {
                //     command.setTargetVelocity(*(w_vec + 1));
                //     std::memcpy(region.get_address() + MOTOR_1_ACTUAL_VELOCITY_OFFSET, &actualVelocity, 8);
                //     std::memcpy(region.get_address() + MOTOR_1_ACTUAL_POSITION_OFFSET, &actualPosition, 8);
                //     std::memcpy(region.get_address() + MOTOR_1_ACTUAL_TORQUE_OFFSET, &actualTorque, 8);
                // } else if (maxon_slave_ptrabrt->getName() == "Maxon2") {
                //     command.setTargetVelocity(*(w_vec + 2));
                //     std::memcpy(region.get_address() + MOTOR_2_ACTUAL_VELOCITY_OFFSET, &actualVelocity, 8);
                //     std::memcpy(region.get_address() + MOTOR_2_ACTUAL_POSITION_OFFSET, &actualPosition, 8);
                //     std::memcpy(region.get_address() + MOTOR_2_ACTUAL_TORQUE_OFFSET, &actualTorque, 8);
                // } else if (maxon_slave_ptr->getName() == "Maxon3") {
                //     command.setTargetVelocity(*(w_vec + 3));
                //     std::memcpy(region.get_address() + MOTOR_3_ACTUAL_VELOCITY_OFFSET, &actualVelocity, 8);
                //     std::memcpy(region.get_address() + MOTOR_3_ACTUAL_POSITION_OFFSET, &actualPosition, 8);
                //     std::memcpy(region.get_address() + MOTOR_3_ACTUAL_TORQUE_OFFSET, &actualTorque, 8);
                } else
                    std::cout << "Wrong slave name\n";
                // std::cout << "EndoWrist joint velocities: " << *w_vec << ", " << *(w_vec + 1) << ", " << *(w_vec + 2) << ", " << *(w_vec + 3) << std::endl;
                maxon_slave_ptr->stageCommand(command);

                // if (robot_stop_flag) abrt = true; // TODO

            } else {
                MELO_WARN_STREAM("Maxon '" << maxon_slave_ptr->getName() << "': " << maxon_slave_ptr->getReading().getDriveState());
            }
        }
    }
    maxonEnabledAfterStartup = true;
  }

};
// TODO: check the native signal handler in rclcpp
void signal_handler(int sig) {
    // Pre shutdown procedure.
    // The devices execute procedures (e.g. state changes) that are necessary for a
    // proper shutdown and that must be done with PDO communication.
    // The communication update loop (i.e. PDO loop) continues to run!
    // You might thus want to implement some logic that stages zero torque / velocity commands
    // or simliar safety measures at this point using e.g. atomic variables and checking them
    // in the communication update loop.

    // for (const auto& master : configurator->getMasters()) {
    //     master->preShutdown();
    // }
    // // stop the PDO communication at the next update of the communication loop
    // abrt = true; // TODO
    // rclcpp::shutdown();
    // // Completely halt the EtherCAT communication.
    // // No online communication is possible afterwards, including SDOs.
    // for (const auto& master : configurator->getMasters()) {
    //     master->shutdown();
    // }
    // Exit this executable
    std::cout << "Shutdown" << std::endl;
    exit(0);
}

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  std::signal(SIGINT, signal_handler);

//   auto node_options = rclcpp::NodeOptions().arguments({argv[1]}); // Pass command line arguments to node
//   auto motor_node = std::make_shared<MotorSubscriber>(node_options);
  auto motor_node = std::make_shared<MotorSubscriber>();
  rclcpp::spin(motor_node);
  rclcpp::shutdown();
  return 0;
}
