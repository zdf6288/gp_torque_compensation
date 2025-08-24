// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from custom_msgs:msg/LambdaCommand.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MSGS__MSG__DETAIL__LAMBDA_COMMAND__BUILDER_HPP_
#define CUSTOM_MSGS__MSG__DETAIL__LAMBDA_COMMAND__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "custom_msgs/msg/detail/lambda_command__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace custom_msgs
{

namespace msg
{

namespace builder
{

class Init_LambdaCommand_enable_backlash_compensation
{
public:
  explicit Init_LambdaCommand_enable_backlash_compensation(::custom_msgs::msg::LambdaCommand & msg)
  : msg_(msg)
  {}
  ::custom_msgs::msg::LambdaCommand enable_backlash_compensation(::custom_msgs::msg::LambdaCommand::_enable_backlash_compensation_type arg)
  {
    msg_.enable_backlash_compensation = std::move(arg);
    return std::move(msg_);
  }

private:
  ::custom_msgs::msg::LambdaCommand msg_;
};

class Init_LambdaCommand_v_gripper
{
public:
  explicit Init_LambdaCommand_v_gripper(::custom_msgs::msg::LambdaCommand & msg)
  : msg_(msg)
  {}
  Init_LambdaCommand_enable_backlash_compensation v_gripper(::custom_msgs::msg::LambdaCommand::_v_gripper_type arg)
  {
    msg_.v_gripper = std::move(arg);
    return Init_LambdaCommand_enable_backlash_compensation(msg_);
  }

private:
  ::custom_msgs::msg::LambdaCommand msg_;
};

class Init_LambdaCommand_angular
{
public:
  explicit Init_LambdaCommand_angular(::custom_msgs::msg::LambdaCommand & msg)
  : msg_(msg)
  {}
  Init_LambdaCommand_v_gripper angular(::custom_msgs::msg::LambdaCommand::_angular_type arg)
  {
    msg_.angular = std::move(arg);
    return Init_LambdaCommand_v_gripper(msg_);
  }

private:
  ::custom_msgs::msg::LambdaCommand msg_;
};

class Init_LambdaCommand_linear
{
public:
  explicit Init_LambdaCommand_linear(::custom_msgs::msg::LambdaCommand & msg)
  : msg_(msg)
  {}
  Init_LambdaCommand_angular linear(::custom_msgs::msg::LambdaCommand::_linear_type arg)
  {
    msg_.linear = std::move(arg);
    return Init_LambdaCommand_angular(msg_);
  }

private:
  ::custom_msgs::msg::LambdaCommand msg_;
};

class Init_LambdaCommand_header
{
public:
  Init_LambdaCommand_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_LambdaCommand_linear header(::custom_msgs::msg::LambdaCommand::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_LambdaCommand_linear(msg_);
  }

private:
  ::custom_msgs::msg::LambdaCommand msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::custom_msgs::msg::LambdaCommand>()
{
  return custom_msgs::msg::builder::Init_LambdaCommand_header();
}

}  // namespace custom_msgs

#endif  // CUSTOM_MSGS__MSG__DETAIL__LAMBDA_COMMAND__BUILDER_HPP_
