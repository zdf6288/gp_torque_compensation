// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from custom_msgs:msg/EffortCommand.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MSGS__MSG__DETAIL__EFFORT_COMMAND__BUILDER_HPP_
#define CUSTOM_MSGS__MSG__DETAIL__EFFORT_COMMAND__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "custom_msgs/msg/detail/effort_command__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace custom_msgs
{

namespace msg
{

namespace builder
{

class Init_EffortCommand_efforts
{
public:
  explicit Init_EffortCommand_efforts(::custom_msgs::msg::EffortCommand & msg)
  : msg_(msg)
  {}
  ::custom_msgs::msg::EffortCommand efforts(::custom_msgs::msg::EffortCommand::_efforts_type arg)
  {
    msg_.efforts = std::move(arg);
    return std::move(msg_);
  }

private:
  ::custom_msgs::msg::EffortCommand msg_;
};

class Init_EffortCommand_header
{
public:
  Init_EffortCommand_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_EffortCommand_efforts header(::custom_msgs::msg::EffortCommand::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_EffortCommand_efforts(msg_);
  }

private:
  ::custom_msgs::msg::EffortCommand msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::custom_msgs::msg::EffortCommand>()
{
  return custom_msgs::msg::builder::Init_EffortCommand_header();
}

}  // namespace custom_msgs

#endif  // CUSTOM_MSGS__MSG__DETAIL__EFFORT_COMMAND__BUILDER_HPP_
