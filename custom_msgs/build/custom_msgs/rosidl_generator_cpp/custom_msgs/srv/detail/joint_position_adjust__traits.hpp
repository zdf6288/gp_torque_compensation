// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from custom_msgs:srv/JointPositionAdjust.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MSGS__SRV__DETAIL__JOINT_POSITION_ADJUST__TRAITS_HPP_
#define CUSTOM_MSGS__SRV__DETAIL__JOINT_POSITION_ADJUST__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "custom_msgs/srv/detail/joint_position_adjust__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

namespace custom_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const JointPositionAdjust_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: q_des
  {
    if (msg.q_des.size() == 0) {
      out << "q_des: []";
    } else {
      out << "q_des: [";
      size_t pending_items = msg.q_des.size();
      for (auto item : msg.q_des) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: dq_des
  {
    if (msg.dq_des.size() == 0) {
      out << "dq_des: []";
    } else {
      out << "dq_des: [";
      size_t pending_items = msg.dq_des.size();
      for (auto item : msg.dq_des) {
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
  const JointPositionAdjust_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: q_des
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.q_des.size() == 0) {
      out << "q_des: []\n";
    } else {
      out << "q_des:\n";
      for (auto item : msg.q_des) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: dq_des
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.dq_des.size() == 0) {
      out << "dq_des: []\n";
    } else {
      out << "dq_des:\n";
      for (auto item : msg.dq_des) {
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

inline std::string to_yaml(const JointPositionAdjust_Request & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace custom_msgs

namespace rosidl_generator_traits
{

[[deprecated("use custom_msgs::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const custom_msgs::srv::JointPositionAdjust_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  custom_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use custom_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const custom_msgs::srv::JointPositionAdjust_Request & msg)
{
  return custom_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<custom_msgs::srv::JointPositionAdjust_Request>()
{
  return "custom_msgs::srv::JointPositionAdjust_Request";
}

template<>
inline const char * name<custom_msgs::srv::JointPositionAdjust_Request>()
{
  return "custom_msgs/srv/JointPositionAdjust_Request";
}

template<>
struct has_fixed_size<custom_msgs::srv::JointPositionAdjust_Request>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<custom_msgs::srv::JointPositionAdjust_Request>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<custom_msgs::srv::JointPositionAdjust_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace custom_msgs
{

namespace srv
{

inline void to_flow_style_yaml(
  const JointPositionAdjust_Response & msg,
  std::ostream & out)
{
  out << "{";
  // member: success
  {
    out << "success: ";
    rosidl_generator_traits::value_to_yaml(msg.success, out);
    out << ", ";
  }

  // member: message
  {
    out << "message: ";
    rosidl_generator_traits::value_to_yaml(msg.message, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const JointPositionAdjust_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: success
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "success: ";
    rosidl_generator_traits::value_to_yaml(msg.success, out);
    out << "\n";
  }

  // member: message
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "message: ";
    rosidl_generator_traits::value_to_yaml(msg.message, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const JointPositionAdjust_Response & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace custom_msgs

namespace rosidl_generator_traits
{

[[deprecated("use custom_msgs::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const custom_msgs::srv::JointPositionAdjust_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  custom_msgs::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use custom_msgs::srv::to_yaml() instead")]]
inline std::string to_yaml(const custom_msgs::srv::JointPositionAdjust_Response & msg)
{
  return custom_msgs::srv::to_yaml(msg);
}

template<>
inline const char * data_type<custom_msgs::srv::JointPositionAdjust_Response>()
{
  return "custom_msgs::srv::JointPositionAdjust_Response";
}

template<>
inline const char * name<custom_msgs::srv::JointPositionAdjust_Response>()
{
  return "custom_msgs/srv/JointPositionAdjust_Response";
}

template<>
struct has_fixed_size<custom_msgs::srv::JointPositionAdjust_Response>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<custom_msgs::srv::JointPositionAdjust_Response>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<custom_msgs::srv::JointPositionAdjust_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<custom_msgs::srv::JointPositionAdjust>()
{
  return "custom_msgs::srv::JointPositionAdjust";
}

template<>
inline const char * name<custom_msgs::srv::JointPositionAdjust>()
{
  return "custom_msgs/srv/JointPositionAdjust";
}

template<>
struct has_fixed_size<custom_msgs::srv::JointPositionAdjust>
  : std::integral_constant<
    bool,
    has_fixed_size<custom_msgs::srv::JointPositionAdjust_Request>::value &&
    has_fixed_size<custom_msgs::srv::JointPositionAdjust_Response>::value
  >
{
};

template<>
struct has_bounded_size<custom_msgs::srv::JointPositionAdjust>
  : std::integral_constant<
    bool,
    has_bounded_size<custom_msgs::srv::JointPositionAdjust_Request>::value &&
    has_bounded_size<custom_msgs::srv::JointPositionAdjust_Response>::value
  >
{
};

template<>
struct is_service<custom_msgs::srv::JointPositionAdjust>
  : std::true_type
{
};

template<>
struct is_service_request<custom_msgs::srv::JointPositionAdjust_Request>
  : std::true_type
{
};

template<>
struct is_service_response<custom_msgs::srv::JointPositionAdjust_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // CUSTOM_MSGS__SRV__DETAIL__JOINT_POSITION_ADJUST__TRAITS_HPP_
