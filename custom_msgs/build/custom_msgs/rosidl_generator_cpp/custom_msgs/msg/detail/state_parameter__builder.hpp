// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from custom_msgs:msg/StateParameter.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MSGS__MSG__DETAIL__STATE_PARAMETER__BUILDER_HPP_
#define CUSTOM_MSGS__MSG__DETAIL__STATE_PARAMETER__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "custom_msgs/msg/detail/state_parameter__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace custom_msgs
{

namespace msg
{

namespace builder
{

class Init_StateParameter_zero_jacobian_flange
{
public:
  explicit Init_StateParameter_zero_jacobian_flange(::custom_msgs::msg::StateParameter & msg)
  : msg_(msg)
  {}
  ::custom_msgs::msg::StateParameter zero_jacobian_flange(::custom_msgs::msg::StateParameter::_zero_jacobian_flange_type arg)
  {
    msg_.zero_jacobian_flange = std::move(arg);
    return std::move(msg_);
  }

private:
  ::custom_msgs::msg::StateParameter msg_;
};

class Init_StateParameter_coriolis
{
public:
  explicit Init_StateParameter_coriolis(::custom_msgs::msg::StateParameter & msg)
  : msg_(msg)
  {}
  Init_StateParameter_zero_jacobian_flange coriolis(::custom_msgs::msg::StateParameter::_coriolis_type arg)
  {
    msg_.coriolis = std::move(arg);
    return Init_StateParameter_zero_jacobian_flange(msg_);
  }

private:
  ::custom_msgs::msg::StateParameter msg_;
};

class Init_StateParameter_mass
{
public:
  explicit Init_StateParameter_mass(::custom_msgs::msg::StateParameter & msg)
  : msg_(msg)
  {}
  Init_StateParameter_coriolis mass(::custom_msgs::msg::StateParameter::_mass_type arg)
  {
    msg_.mass = std::move(arg);
    return Init_StateParameter_coriolis(msg_);
  }

private:
  ::custom_msgs::msg::StateParameter msg_;
};

class Init_StateParameter_o_t_f
{
public:
  explicit Init_StateParameter_o_t_f(::custom_msgs::msg::StateParameter & msg)
  : msg_(msg)
  {}
  Init_StateParameter_mass o_t_f(::custom_msgs::msg::StateParameter::_o_t_f_type arg)
  {
    msg_.o_t_f = std::move(arg);
    return Init_StateParameter_mass(msg_);
  }

private:
  ::custom_msgs::msg::StateParameter msg_;
};

class Init_StateParameter_gravity
{
public:
  explicit Init_StateParameter_gravity(::custom_msgs::msg::StateParameter & msg)
  : msg_(msg)
  {}
  Init_StateParameter_o_t_f gravity(::custom_msgs::msg::StateParameter::_gravity_type arg)
  {
    msg_.gravity = std::move(arg);
    return Init_StateParameter_o_t_f(msg_);
  }

private:
  ::custom_msgs::msg::StateParameter msg_;
};

class Init_StateParameter_effort_measured
{
public:
  explicit Init_StateParameter_effort_measured(::custom_msgs::msg::StateParameter & msg)
  : msg_(msg)
  {}
  Init_StateParameter_gravity effort_measured(::custom_msgs::msg::StateParameter::_effort_measured_type arg)
  {
    msg_.effort_measured = std::move(arg);
    return Init_StateParameter_gravity(msg_);
  }

private:
  ::custom_msgs::msg::StateParameter msg_;
};

class Init_StateParameter_velocity
{
public:
  explicit Init_StateParameter_velocity(::custom_msgs::msg::StateParameter & msg)
  : msg_(msg)
  {}
  Init_StateParameter_effort_measured velocity(::custom_msgs::msg::StateParameter::_velocity_type arg)
  {
    msg_.velocity = std::move(arg);
    return Init_StateParameter_effort_measured(msg_);
  }

private:
  ::custom_msgs::msg::StateParameter msg_;
};

class Init_StateParameter_position
{
public:
  explicit Init_StateParameter_position(::custom_msgs::msg::StateParameter & msg)
  : msg_(msg)
  {}
  Init_StateParameter_velocity position(::custom_msgs::msg::StateParameter::_position_type arg)
  {
    msg_.position = std::move(arg);
    return Init_StateParameter_velocity(msg_);
  }

private:
  ::custom_msgs::msg::StateParameter msg_;
};

class Init_StateParameter_header
{
public:
  Init_StateParameter_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_StateParameter_position header(::custom_msgs::msg::StateParameter::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_StateParameter_position(msg_);
  }

private:
  ::custom_msgs::msg::StateParameter msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::custom_msgs::msg::StateParameter>()
{
  return custom_msgs::msg::builder::Init_StateParameter_header();
}

}  // namespace custom_msgs

#endif  // CUSTOM_MSGS__MSG__DETAIL__STATE_PARAMETER__BUILDER_HPP_
