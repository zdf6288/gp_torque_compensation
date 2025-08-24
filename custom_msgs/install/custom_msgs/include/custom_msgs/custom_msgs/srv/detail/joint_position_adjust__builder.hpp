// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from custom_msgs:srv/JointPositionAdjust.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MSGS__SRV__DETAIL__JOINT_POSITION_ADJUST__BUILDER_HPP_
#define CUSTOM_MSGS__SRV__DETAIL__JOINT_POSITION_ADJUST__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "custom_msgs/srv/detail/joint_position_adjust__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace custom_msgs
{

namespace srv
{

namespace builder
{

class Init_JointPositionAdjust_Request_dq_des
{
public:
  explicit Init_JointPositionAdjust_Request_dq_des(::custom_msgs::srv::JointPositionAdjust_Request & msg)
  : msg_(msg)
  {}
  ::custom_msgs::srv::JointPositionAdjust_Request dq_des(::custom_msgs::srv::JointPositionAdjust_Request::_dq_des_type arg)
  {
    msg_.dq_des = std::move(arg);
    return std::move(msg_);
  }

private:
  ::custom_msgs::srv::JointPositionAdjust_Request msg_;
};

class Init_JointPositionAdjust_Request_q_des
{
public:
  Init_JointPositionAdjust_Request_q_des()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_JointPositionAdjust_Request_dq_des q_des(::custom_msgs::srv::JointPositionAdjust_Request::_q_des_type arg)
  {
    msg_.q_des = std::move(arg);
    return Init_JointPositionAdjust_Request_dq_des(msg_);
  }

private:
  ::custom_msgs::srv::JointPositionAdjust_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::custom_msgs::srv::JointPositionAdjust_Request>()
{
  return custom_msgs::srv::builder::Init_JointPositionAdjust_Request_q_des();
}

}  // namespace custom_msgs


namespace custom_msgs
{

namespace srv
{

namespace builder
{

class Init_JointPositionAdjust_Response_message
{
public:
  explicit Init_JointPositionAdjust_Response_message(::custom_msgs::srv::JointPositionAdjust_Response & msg)
  : msg_(msg)
  {}
  ::custom_msgs::srv::JointPositionAdjust_Response message(::custom_msgs::srv::JointPositionAdjust_Response::_message_type arg)
  {
    msg_.message = std::move(arg);
    return std::move(msg_);
  }

private:
  ::custom_msgs::srv::JointPositionAdjust_Response msg_;
};

class Init_JointPositionAdjust_Response_success
{
public:
  Init_JointPositionAdjust_Response_success()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_JointPositionAdjust_Response_message success(::custom_msgs::srv::JointPositionAdjust_Response::_success_type arg)
  {
    msg_.success = std::move(arg);
    return Init_JointPositionAdjust_Response_message(msg_);
  }

private:
  ::custom_msgs::srv::JointPositionAdjust_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::custom_msgs::srv::JointPositionAdjust_Response>()
{
  return custom_msgs::srv::builder::Init_JointPositionAdjust_Response_success();
}

}  // namespace custom_msgs

#endif  // CUSTOM_MSGS__SRV__DETAIL__JOINT_POSITION_ADJUST__BUILDER_HPP_
