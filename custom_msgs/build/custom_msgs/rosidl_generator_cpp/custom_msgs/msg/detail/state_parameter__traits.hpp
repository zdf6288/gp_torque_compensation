// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from custom_msgs:msg/StateParameter.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MSGS__MSG__DETAIL__STATE_PARAMETER__TRAITS_HPP_
#define CUSTOM_MSGS__MSG__DETAIL__STATE_PARAMETER__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "custom_msgs/msg/detail/state_parameter__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"

namespace custom_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const StateParameter & msg,
  std::ostream & out)
{
  out << "{";
  // member: header
  {
    out << "header: ";
    to_flow_style_yaml(msg.header, out);
    out << ", ";
  }

  // member: position
  {
    if (msg.position.size() == 0) {
      out << "position: []";
    } else {
      out << "position: [";
      size_t pending_items = msg.position.size();
      for (auto item : msg.position) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: velocity
  {
    if (msg.velocity.size() == 0) {
      out << "velocity: []";
    } else {
      out << "velocity: [";
      size_t pending_items = msg.velocity.size();
      for (auto item : msg.velocity) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: effort_measured
  {
    if (msg.effort_measured.size() == 0) {
      out << "effort_measured: []";
    } else {
      out << "effort_measured: [";
      size_t pending_items = msg.effort_measured.size();
      for (auto item : msg.effort_measured) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: gravity
  {
    if (msg.gravity.size() == 0) {
      out << "gravity: []";
    } else {
      out << "gravity: [";
      size_t pending_items = msg.gravity.size();
      for (auto item : msg.gravity) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: o_t_f
  {
    if (msg.o_t_f.size() == 0) {
      out << "o_t_f: []";
    } else {
      out << "o_t_f: [";
      size_t pending_items = msg.o_t_f.size();
      for (auto item : msg.o_t_f) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: mass
  {
    if (msg.mass.size() == 0) {
      out << "mass: []";
    } else {
      out << "mass: [";
      size_t pending_items = msg.mass.size();
      for (auto item : msg.mass) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: coriolis
  {
    if (msg.coriolis.size() == 0) {
      out << "coriolis: []";
    } else {
      out << "coriolis: [";
      size_t pending_items = msg.coriolis.size();
      for (auto item : msg.coriolis) {
        rosidl_generator_traits::value_to_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
    out << ", ";
  }

  // member: zero_jacobian_flange
  {
    if (msg.zero_jacobian_flange.size() == 0) {
      out << "zero_jacobian_flange: []";
    } else {
      out << "zero_jacobian_flange: [";
      size_t pending_items = msg.zero_jacobian_flange.size();
      for (auto item : msg.zero_jacobian_flange) {
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
  const StateParameter & msg,
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

  // member: position
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.position.size() == 0) {
      out << "position: []\n";
    } else {
      out << "position:\n";
      for (auto item : msg.position) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: velocity
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.velocity.size() == 0) {
      out << "velocity: []\n";
    } else {
      out << "velocity:\n";
      for (auto item : msg.velocity) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: effort_measured
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.effort_measured.size() == 0) {
      out << "effort_measured: []\n";
    } else {
      out << "effort_measured:\n";
      for (auto item : msg.effort_measured) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: gravity
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.gravity.size() == 0) {
      out << "gravity: []\n";
    } else {
      out << "gravity:\n";
      for (auto item : msg.gravity) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: o_t_f
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.o_t_f.size() == 0) {
      out << "o_t_f: []\n";
    } else {
      out << "o_t_f:\n";
      for (auto item : msg.o_t_f) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: mass
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.mass.size() == 0) {
      out << "mass: []\n";
    } else {
      out << "mass:\n";
      for (auto item : msg.mass) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: coriolis
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.coriolis.size() == 0) {
      out << "coriolis: []\n";
    } else {
      out << "coriolis:\n";
      for (auto item : msg.coriolis) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "- ";
        rosidl_generator_traits::value_to_yaml(item, out);
        out << "\n";
      }
    }
  }

  // member: zero_jacobian_flange
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.zero_jacobian_flange.size() == 0) {
      out << "zero_jacobian_flange: []\n";
    } else {
      out << "zero_jacobian_flange:\n";
      for (auto item : msg.zero_jacobian_flange) {
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

inline std::string to_yaml(const StateParameter & msg, bool use_flow_style = false)
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
  const custom_msgs::msg::StateParameter & msg,
  std::ostream & out, size_t indentation = 0)
{
  custom_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use custom_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const custom_msgs::msg::StateParameter & msg)
{
  return custom_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<custom_msgs::msg::StateParameter>()
{
  return "custom_msgs::msg::StateParameter";
}

template<>
inline const char * name<custom_msgs::msg::StateParameter>()
{
  return "custom_msgs/msg/StateParameter";
}

template<>
struct has_fixed_size<custom_msgs::msg::StateParameter>
  : std::integral_constant<bool, has_fixed_size<std_msgs::msg::Header>::value> {};

template<>
struct has_bounded_size<custom_msgs::msg::StateParameter>
  : std::integral_constant<bool, has_bounded_size<std_msgs::msg::Header>::value> {};

template<>
struct is_message<custom_msgs::msg::StateParameter>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // CUSTOM_MSGS__MSG__DETAIL__STATE_PARAMETER__TRAITS_HPP_
