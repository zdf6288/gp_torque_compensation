// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from custom_msgs:msg/LambdaCommand.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MSGS__MSG__DETAIL__LAMBDA_COMMAND__TRAITS_HPP_
#define CUSTOM_MSGS__MSG__DETAIL__LAMBDA_COMMAND__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "custom_msgs/msg/detail/lambda_command__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"
// Member 'linear'
// Member 'angular'
#include "geometry_msgs/msg/detail/vector3__traits.hpp"

namespace custom_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const LambdaCommand & msg,
  std::ostream & out)
{
  out << "{";
  // member: header
  {
    out << "header: ";
    to_flow_style_yaml(msg.header, out);
    out << ", ";
  }

  // member: linear
  {
    out << "linear: ";
    to_flow_style_yaml(msg.linear, out);
    out << ", ";
  }

  // member: angular
  {
    out << "angular: ";
    to_flow_style_yaml(msg.angular, out);
    out << ", ";
  }

  // member: v_gripper
  {
    out << "v_gripper: ";
    rosidl_generator_traits::value_to_yaml(msg.v_gripper, out);
    out << ", ";
  }

  // member: enable_backlash_compensation
  {
    out << "enable_backlash_compensation: ";
    rosidl_generator_traits::value_to_yaml(msg.enable_backlash_compensation, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const LambdaCommand & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: header
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "header:\n";
    to_block_style_yaml(msg.header, out, indentation + 2);
  }

  // member: linear
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "linear:\n";
    to_block_style_yaml(msg.linear, out, indentation + 2);
  }

  // member: angular
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "angular:\n";
    to_block_style_yaml(msg.angular, out, indentation + 2);
  }

  // member: v_gripper
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "v_gripper: ";
    rosidl_generator_traits::value_to_yaml(msg.v_gripper, out);
    out << "\n";
  }

  // member: enable_backlash_compensation
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "enable_backlash_compensation: ";
    rosidl_generator_traits::value_to_yaml(msg.enable_backlash_compensation, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const LambdaCommand & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace custom_msgs

namespace rosidl_generator_traits
{

[[deprecated("use custom_msgs::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const custom_msgs::msg::LambdaCommand & msg,
  std::ostream & out, size_t indentation = 0)
{
  custom_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use custom_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const custom_msgs::msg::LambdaCommand & msg)
{
  return custom_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<custom_msgs::msg::LambdaCommand>()
{
  return "custom_msgs::msg::LambdaCommand";
}

template<>
inline const char * name<custom_msgs::msg::LambdaCommand>()
{
  return "custom_msgs/msg/LambdaCommand";
}

template<>
struct has_fixed_size<custom_msgs::msg::LambdaCommand>
  : std::integral_constant<bool, has_fixed_size<geometry_msgs::msg::Vector3>::value && has_fixed_size<std_msgs::msg::Header>::value> {};

template<>
struct has_bounded_size<custom_msgs::msg::LambdaCommand>
  : std::integral_constant<bool, has_bounded_size<geometry_msgs::msg::Vector3>::value && has_bounded_size<std_msgs::msg::Header>::value> {};

template<>
struct is_message<custom_msgs::msg::LambdaCommand>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // CUSTOM_MSGS__MSG__DETAIL__LAMBDA_COMMAND__TRAITS_HPP_
