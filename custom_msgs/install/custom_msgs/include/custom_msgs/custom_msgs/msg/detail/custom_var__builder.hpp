// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from custom_msgs:msg/CustomVar.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MSGS__MSG__DETAIL__CUSTOM_VAR__BUILDER_HPP_
#define CUSTOM_MSGS__MSG__DETAIL__CUSTOM_VAR__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "custom_msgs/msg/detail/custom_var__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace custom_msgs
{

namespace msg
{

namespace builder
{

class Init_CustomVar_error_rcm
{
public:
  explicit Init_CustomVar_error_rcm(::custom_msgs::msg::CustomVar & msg)
  : msg_(msg)
  {}
  ::custom_msgs::msg::CustomVar error_rcm(::custom_msgs::msg::CustomVar::_error_rcm_type arg)
  {
    msg_.error_rcm = std::move(arg);
    return std::move(msg_);
  }

private:
  ::custom_msgs::msg::CustomVar msg_;
};

class Init_CustomVar_error_endo
{
public:
  explicit Init_CustomVar_error_endo(::custom_msgs::msg::CustomVar & msg)
  : msg_(msg)
  {}
  Init_CustomVar_error_rcm error_endo(::custom_msgs::msg::CustomVar::_error_endo_type arg)
  {
    msg_.error_endo = std::move(arg);
    return Init_CustomVar_error_rcm(msg_);
  }

private:
  ::custom_msgs::msg::CustomVar msg_;
};

class Init_CustomVar_ddq_computed
{
public:
  explicit Init_CustomVar_ddq_computed(::custom_msgs::msg::CustomVar & msg)
  : msg_(msg)
  {}
  Init_CustomVar_error_endo ddq_computed(::custom_msgs::msg::CustomVar::_ddq_computed_type arg)
  {
    msg_.ddq_computed = std::move(arg);
    return Init_CustomVar_error_endo(msg_);
  }

private:
  ::custom_msgs::msg::CustomVar msg_;
};

class Init_CustomVar_dq_meas
{
public:
  explicit Init_CustomVar_dq_meas(::custom_msgs::msg::CustomVar & msg)
  : msg_(msg)
  {}
  Init_CustomVar_ddq_computed dq_meas(::custom_msgs::msg::CustomVar::_dq_meas_type arg)
  {
    msg_.dq_meas = std::move(arg);
    return Init_CustomVar_ddq_computed(msg_);
  }

private:
  ::custom_msgs::msg::CustomVar msg_;
};

class Init_CustomVar_q_meas
{
public:
  explicit Init_CustomVar_q_meas(::custom_msgs::msg::CustomVar & msg)
  : msg_(msg)
  {}
  Init_CustomVar_dq_meas q_meas(::custom_msgs::msg::CustomVar::_q_meas_type arg)
  {
    msg_.q_meas = std::move(arg);
    return Init_CustomVar_dq_meas(msg_);
  }

private:
  ::custom_msgs::msg::CustomVar msg_;
};

class Init_CustomVar_dq_des
{
public:
  explicit Init_CustomVar_dq_des(::custom_msgs::msg::CustomVar & msg)
  : msg_(msg)
  {}
  Init_CustomVar_q_meas dq_des(::custom_msgs::msg::CustomVar::_dq_des_type arg)
  {
    msg_.dq_des = std::move(arg);
    return Init_CustomVar_q_meas(msg_);
  }

private:
  ::custom_msgs::msg::CustomVar msg_;
};

class Init_CustomVar_q_des
{
public:
  Init_CustomVar_q_des()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_CustomVar_dq_des q_des(::custom_msgs::msg::CustomVar::_q_des_type arg)
  {
    msg_.q_des = std::move(arg);
    return Init_CustomVar_dq_des(msg_);
  }

private:
  ::custom_msgs::msg::CustomVar msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::custom_msgs::msg::CustomVar>()
{
  return custom_msgs::msg::builder::Init_CustomVar_q_des();
}

}  // namespace custom_msgs

#endif  // CUSTOM_MSGS__MSG__DETAIL__CUSTOM_VAR__BUILDER_HPP_
