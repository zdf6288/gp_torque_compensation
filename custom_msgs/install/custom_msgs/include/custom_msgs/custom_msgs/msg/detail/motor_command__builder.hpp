// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from custom_msgs:msg/MotorCommand.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MSGS__MSG__DETAIL__MOTOR_COMMAND__BUILDER_HPP_
#define CUSTOM_MSGS__MSG__DETAIL__MOTOR_COMMAND__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "custom_msgs/msg/detail/motor_command__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace custom_msgs
{

namespace msg
{

namespace builder
{

class Init_MotorCommand_vel
{
public:
  explicit Init_MotorCommand_vel(::custom_msgs::msg::MotorCommand & msg)
  : msg_(msg)
  {}
  ::custom_msgs::msg::MotorCommand vel(::custom_msgs::msg::MotorCommand::_vel_type arg)
  {
    msg_.vel = std::move(arg);
    return std::move(msg_);
  }

private:
  ::custom_msgs::msg::MotorCommand msg_;
};

class Init_MotorCommand_header
{
public:
  Init_MotorCommand_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_MotorCommand_vel header(::custom_msgs::msg::MotorCommand::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_MotorCommand_vel(msg_);
  }

private:
  ::custom_msgs::msg::MotorCommand msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::custom_msgs::msg::MotorCommand>()
{
  return custom_msgs::msg::builder::Init_MotorCommand_header();
}

}  // namespace custom_msgs

#endif  // CUSTOM_MSGS__MSG__DETAIL__MOTOR_COMMAND__BUILDER_HPP_
