// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from custom_msgs:msg/TaskSpaceCommand.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MSGS__MSG__DETAIL__TASK_SPACE_COMMAND__BUILDER_HPP_
#define CUSTOM_MSGS__MSG__DETAIL__TASK_SPACE_COMMAND__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "custom_msgs/msg/detail/task_space_command__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace custom_msgs
{

namespace msg
{

namespace builder
{

class Init_TaskSpaceCommand_ddx_des
{
public:
  explicit Init_TaskSpaceCommand_ddx_des(::custom_msgs::msg::TaskSpaceCommand & msg)
  : msg_(msg)
  {}
  ::custom_msgs::msg::TaskSpaceCommand ddx_des(::custom_msgs::msg::TaskSpaceCommand::_ddx_des_type arg)
  {
    msg_.ddx_des = std::move(arg);
    return std::move(msg_);
  }

private:
  ::custom_msgs::msg::TaskSpaceCommand msg_;
};

class Init_TaskSpaceCommand_dx_des
{
public:
  explicit Init_TaskSpaceCommand_dx_des(::custom_msgs::msg::TaskSpaceCommand & msg)
  : msg_(msg)
  {}
  Init_TaskSpaceCommand_ddx_des dx_des(::custom_msgs::msg::TaskSpaceCommand::_dx_des_type arg)
  {
    msg_.dx_des = std::move(arg);
    return Init_TaskSpaceCommand_ddx_des(msg_);
  }

private:
  ::custom_msgs::msg::TaskSpaceCommand msg_;
};

class Init_TaskSpaceCommand_x_des
{
public:
  explicit Init_TaskSpaceCommand_x_des(::custom_msgs::msg::TaskSpaceCommand & msg)
  : msg_(msg)
  {}
  Init_TaskSpaceCommand_dx_des x_des(::custom_msgs::msg::TaskSpaceCommand::_x_des_type arg)
  {
    msg_.x_des = std::move(arg);
    return Init_TaskSpaceCommand_dx_des(msg_);
  }

private:
  ::custom_msgs::msg::TaskSpaceCommand msg_;
};

class Init_TaskSpaceCommand_header
{
public:
  Init_TaskSpaceCommand_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_TaskSpaceCommand_x_des header(::custom_msgs::msg::TaskSpaceCommand::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_TaskSpaceCommand_x_des(msg_);
  }

private:
  ::custom_msgs::msg::TaskSpaceCommand msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::custom_msgs::msg::TaskSpaceCommand>()
{
  return custom_msgs::msg::builder::Init_TaskSpaceCommand_header();
}

}  // namespace custom_msgs

#endif  // CUSTOM_MSGS__MSG__DETAIL__TASK_SPACE_COMMAND__BUILDER_HPP_
