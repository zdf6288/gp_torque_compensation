// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from custom_msgs:msg/MotorMeasurement.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MSGS__MSG__DETAIL__MOTOR_MEASUREMENT__BUILDER_HPP_
#define CUSTOM_MSGS__MSG__DETAIL__MOTOR_MEASUREMENT__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "custom_msgs/msg/detail/motor_measurement__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace custom_msgs
{

namespace msg
{

namespace builder
{

class Init_MotorMeasurement_torque
{
public:
  explicit Init_MotorMeasurement_torque(::custom_msgs::msg::MotorMeasurement & msg)
  : msg_(msg)
  {}
  ::custom_msgs::msg::MotorMeasurement torque(::custom_msgs::msg::MotorMeasurement::_torque_type arg)
  {
    msg_.torque = std::move(arg);
    return std::move(msg_);
  }

private:
  ::custom_msgs::msg::MotorMeasurement msg_;
};

class Init_MotorMeasurement_pos
{
public:
  explicit Init_MotorMeasurement_pos(::custom_msgs::msg::MotorMeasurement & msg)
  : msg_(msg)
  {}
  Init_MotorMeasurement_torque pos(::custom_msgs::msg::MotorMeasurement::_pos_type arg)
  {
    msg_.pos = std::move(arg);
    return Init_MotorMeasurement_torque(msg_);
  }

private:
  ::custom_msgs::msg::MotorMeasurement msg_;
};

class Init_MotorMeasurement_vel
{
public:
  explicit Init_MotorMeasurement_vel(::custom_msgs::msg::MotorMeasurement & msg)
  : msg_(msg)
  {}
  Init_MotorMeasurement_pos vel(::custom_msgs::msg::MotorMeasurement::_vel_type arg)
  {
    msg_.vel = std::move(arg);
    return Init_MotorMeasurement_pos(msg_);
  }

private:
  ::custom_msgs::msg::MotorMeasurement msg_;
};

class Init_MotorMeasurement_header
{
public:
  Init_MotorMeasurement_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_MotorMeasurement_vel header(::custom_msgs::msg::MotorMeasurement::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_MotorMeasurement_vel(msg_);
  }

private:
  ::custom_msgs::msg::MotorMeasurement msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::custom_msgs::msg::MotorMeasurement>()
{
  return custom_msgs::msg::builder::Init_MotorMeasurement_header();
}

}  // namespace custom_msgs

#endif  // CUSTOM_MSGS__MSG__DETAIL__MOTOR_MEASUREMENT__BUILDER_HPP_
