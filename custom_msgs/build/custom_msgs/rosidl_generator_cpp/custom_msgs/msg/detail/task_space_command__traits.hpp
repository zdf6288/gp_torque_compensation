// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from custom_msgs:msg/TaskSpaceCommand.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MSGS__MSG__DETAIL__TASK_SPACE_COMMAND__TRAITS_HPP_
#define CUSTOM_MSGS__MSG__DETAIL__TASK_SPACE_COMMAND__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "custom_msgs/msg/detail/task_space_command__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"

namespace custom_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const TaskSpaceCommand & msg,
  std::ostream & out)
{
  out << "{";
  // member: header
  {
    out << "header: ";
    to_flow_style_yaml(msg.header, out);
    out << ", ";
  }

  // member: x_des
  {
    if (msg.x_des.size() == 0) {
      out << "x_des: []";
    } else {
      out << "x_des: [";
      size_t pending_items = msg.x_des.size();
      for (auto item : msg.x_des) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: dx_des
  {
    if (msg.dx_des.size() == 0) {
      out << "dx_des: []";
    } else {
      out << "dx_des: [";
      size_t pending_items = msg.dx_des.size();
      for (auto item : msg.dx_des) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: ddx_des
  {
    if (msg.ddx_des.size() == 0) {
      out << "ddx_des: []";
    } else {
      out << "ddx_des: [";
      size_t pending_items = msg.ddx_des.size();
      for (auto item : msg.ddx_des) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const TaskSpaceCommand & msg,
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

  // member: x_des
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.x_des.size() == 0) {
      out << "x_des: []\n";
    } else {
      out << "x_des:\n";
      for (auto item : msg.x_des) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: dx_des
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.dx_des.size() == 0) {
      out << "dx_des: []\n";
    } else {
      out << "dx_des:\n";
      for (auto item : msg.dx_des) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: ddx_des
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.ddx_des.size() == 0) {
      out << "ddx_des: []\n";
    } else {
      out << "ddx_des:\n";
      for (auto item : msg.ddx_des) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const TaskSpaceCommand & msg, bool use_flow_style = false)
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
  const custom_msgs::msg::TaskSpaceCommand & msg,
  std::ostream & out, size_t indentation = 0)
{
  custom_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use custom_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const custom_msgs::msg::TaskSpaceCommand & msg)
{
  return custom_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<custom_msgs::msg::TaskSpaceCommand>()
{
  return "custom_msgs::msg::TaskSpaceCommand";
}

template<>
inline const char * name<custom_msgs::msg::TaskSpaceCommand>()
{
  return "custom_msgs/msg/TaskSpaceCommand";
}

template<>
struct has_fixed_size<custom_msgs::msg::TaskSpaceCommand>
  : std::integral_constant<bool, has_fixed_size<std_msgs::msg::Header>::value> {};

template<>
struct has_bounded_size<custom_msgs::msg::TaskSpaceCommand>
  : std::integral_constant<bool, has_bounded_size<std_msgs::msg::Header>::value> {};

template<>
struct is_message<custom_msgs::msg::TaskSpaceCommand>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // CUSTOM_MSGS__MSG__DETAIL__TASK_SPACE_COMMAND__TRAITS_HPP_
