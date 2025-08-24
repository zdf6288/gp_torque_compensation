// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from custom_msgs:msg/CustomVar.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MSGS__MSG__DETAIL__CUSTOM_VAR__TRAITS_HPP_
#define CUSTOM_MSGS__MSG__DETAIL__CUSTOM_VAR__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "custom_msgs/msg/detail/custom_var__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'q_des'
// Member 'dq_des'
// Member 'q_meas'
// Member 'dq_meas'
// Member 'ddq_computed'
// Member 'error_endo'
// Member 'error_rcm'
#include "std_msgs/msg/detail/float64_multi_array__traits.hpp"

namespace custom_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const CustomVar & msg,
  std::ostream & out)
{
  out << "{";
  // member: q_des
  {
    out << "q_des: ";
    to_flow_style_yaml(msg.q_des, out);
    out << ", ";
  }

  // member: dq_des
  {
    out << "dq_des: ";
    to_flow_style_yaml(msg.dq_des, out);
    out << ", ";
  }

  // member: q_meas
  {
    out << "q_meas: ";
    to_flow_style_yaml(msg.q_meas, out);
    out << ", ";
  }

  // member: dq_meas
  {
    out << "dq_meas: ";
    to_flow_style_yaml(msg.dq_meas, out);
    out << ", ";
  }

  // member: ddq_computed
  {
    out << "ddq_computed: ";
    to_flow_style_yaml(msg.ddq_computed, out);
    out << ", ";
  }

  // member: error_endo
  {
    out << "error_endo: ";
    to_flow_style_yaml(msg.error_endo, out);
    out << ", ";
  }

  // member: error_rcm
  {
    out << "error_rcm: ";
    to_flow_style_yaml(msg.error_rcm, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const CustomVar & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: q_des
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "q_des:\n";
    to_block_style_yaml(msg.q_des, out, indentation + 2);
  }

  // member: dq_des
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "dq_des:\n";
    to_block_style_yaml(msg.dq_des, out, indentation + 2);
  }

  // member: q_meas
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "q_meas:\n";
    to_block_style_yaml(msg.q_meas, out, indentation + 2);
  }

  // member: dq_meas
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "dq_meas:\n";
    to_block_style_yaml(msg.dq_meas, out, indentation + 2);
  }

  // member: ddq_computed
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "ddq_computed:\n";
    to_block_style_yaml(msg.ddq_computed, out, indentation + 2);
  }

  // member: error_endo
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "error_endo:\n";
    to_block_style_yaml(msg.error_endo, out, indentation + 2);
  }

  // member: error_rcm
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "error_rcm:\n";
    to_block_style_yaml(msg.error_rcm, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const CustomVar & msg, bool use_flow_style = false)
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
  const custom_msgs::msg::CustomVar & msg,
  std::ostream & out, size_t indentation = 0)
{
  custom_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use custom_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const custom_msgs::msg::CustomVar & msg)
{
  return custom_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<custom_msgs::msg::CustomVar>()
{
  return "custom_msgs::msg::CustomVar";
}

template<>
inline const char * name<custom_msgs::msg::CustomVar>()
{
  return "custom_msgs/msg/CustomVar";
}

template<>
struct has_fixed_size<custom_msgs::msg::CustomVar>
  : std::integral_constant<bool, has_fixed_size<std_msgs::msg::Float64MultiArray>::value> {};

template<>
struct has_bounded_size<custom_msgs::msg::CustomVar>
  : std::integral_constant<bool, has_bounded_size<std_msgs::msg::Float64MultiArray>::value> {};

template<>
struct is_message<custom_msgs::msg::CustomVar>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // CUSTOM_MSGS__MSG__DETAIL__CUSTOM_VAR__TRAITS_HPP_
