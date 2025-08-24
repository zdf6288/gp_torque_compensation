// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from custom_msgs:msg/StateParameter.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "custom_msgs/msg/detail/state_parameter__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace custom_msgs
{

namespace msg
{

namespace rosidl_typesupport_introspection_cpp
{

void StateParameter_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) custom_msgs::msg::StateParameter(_init);
}

void StateParameter_fini_function(void * message_memory)
{
  auto typed_message = static_cast<custom_msgs::msg::StateParameter *>(message_memory);
  typed_message->~StateParameter();
}

size_t size_function__StateParameter__position(const void * untyped_member)
{
  (void)untyped_member;
  return 7;
}

const void * get_const_function__StateParameter__position(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 7> *>(untyped_member);
  return &member[index];
}

void * get_function__StateParameter__position(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 7> *>(untyped_member);
  return &member[index];
}

void fetch_function__StateParameter__position(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__StateParameter__position(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__StateParameter__position(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__StateParameter__position(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__StateParameter__velocity(const void * untyped_member)
{
  (void)untyped_member;
  return 7;
}

const void * get_const_function__StateParameter__velocity(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 7> *>(untyped_member);
  return &member[index];
}

void * get_function__StateParameter__velocity(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 7> *>(untyped_member);
  return &member[index];
}

void fetch_function__StateParameter__velocity(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__StateParameter__velocity(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__StateParameter__velocity(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__StateParameter__velocity(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__StateParameter__effort_measured(const void * untyped_member)
{
  (void)untyped_member;
  return 7;
}

const void * get_const_function__StateParameter__effort_measured(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 7> *>(untyped_member);
  return &member[index];
}

void * get_function__StateParameter__effort_measured(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 7> *>(untyped_member);
  return &member[index];
}

void fetch_function__StateParameter__effort_measured(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__StateParameter__effort_measured(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__StateParameter__effort_measured(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__StateParameter__effort_measured(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__StateParameter__gravity(const void * untyped_member)
{
  (void)untyped_member;
  return 7;
}

const void * get_const_function__StateParameter__gravity(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 7> *>(untyped_member);
  return &member[index];
}

void * get_function__StateParameter__gravity(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 7> *>(untyped_member);
  return &member[index];
}

void fetch_function__StateParameter__gravity(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__StateParameter__gravity(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__StateParameter__gravity(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__StateParameter__gravity(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__StateParameter__o_t_f(const void * untyped_member)
{
  (void)untyped_member;
  return 16;
}

const void * get_const_function__StateParameter__o_t_f(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 16> *>(untyped_member);
  return &member[index];
}

void * get_function__StateParameter__o_t_f(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 16> *>(untyped_member);
  return &member[index];
}

void fetch_function__StateParameter__o_t_f(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__StateParameter__o_t_f(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__StateParameter__o_t_f(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__StateParameter__o_t_f(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__StateParameter__mass(const void * untyped_member)
{
  (void)untyped_member;
  return 49;
}

const void * get_const_function__StateParameter__mass(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 49> *>(untyped_member);
  return &member[index];
}

void * get_function__StateParameter__mass(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 49> *>(untyped_member);
  return &member[index];
}

void fetch_function__StateParameter__mass(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__StateParameter__mass(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__StateParameter__mass(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__StateParameter__mass(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__StateParameter__coriolis(const void * untyped_member)
{
  (void)untyped_member;
  return 7;
}

const void * get_const_function__StateParameter__coriolis(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 7> *>(untyped_member);
  return &member[index];
}

void * get_function__StateParameter__coriolis(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 7> *>(untyped_member);
  return &member[index];
}

void fetch_function__StateParameter__coriolis(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__StateParameter__coriolis(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__StateParameter__coriolis(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__StateParameter__coriolis(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__StateParameter__zero_jacobian_flange(const void * untyped_member)
{
  (void)untyped_member;
  return 42;
}

const void * get_const_function__StateParameter__zero_jacobian_flange(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 42> *>(untyped_member);
  return &member[index];
}

void * get_function__StateParameter__zero_jacobian_flange(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 42> *>(untyped_member);
  return &member[index];
}

void fetch_function__StateParameter__zero_jacobian_flange(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__StateParameter__zero_jacobian_flange(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__StateParameter__zero_jacobian_flange(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__StateParameter__zero_jacobian_flange(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember StateParameter_message_member_array[9] = {
  {
    "header",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<std_msgs::msg::Header>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(custom_msgs::msg::StateParameter, header),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "position",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    7,  // array size
    false,  // is upper bound
    offsetof(custom_msgs::msg::StateParameter, position),  // bytes offset in struct
    nullptr,  // default value
    size_function__StateParameter__position,  // size() function pointer
    get_const_function__StateParameter__position,  // get_const(index) function pointer
    get_function__StateParameter__position,  // get(index) function pointer
    fetch_function__StateParameter__position,  // fetch(index, &value) function pointer
    assign_function__StateParameter__position,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "velocity",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    7,  // array size
    false,  // is upper bound
    offsetof(custom_msgs::msg::StateParameter, velocity),  // bytes offset in struct
    nullptr,  // default value
    size_function__StateParameter__velocity,  // size() function pointer
    get_const_function__StateParameter__velocity,  // get_const(index) function pointer
    get_function__StateParameter__velocity,  // get(index) function pointer
    fetch_function__StateParameter__velocity,  // fetch(index, &value) function pointer
    assign_function__StateParameter__velocity,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "effort_measured",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    7,  // array size
    false,  // is upper bound
    offsetof(custom_msgs::msg::StateParameter, effort_measured),  // bytes offset in struct
    nullptr,  // default value
    size_function__StateParameter__effort_measured,  // size() function pointer
    get_const_function__StateParameter__effort_measured,  // get_const(index) function pointer
    get_function__StateParameter__effort_measured,  // get(index) function pointer
    fetch_function__StateParameter__effort_measured,  // fetch(index, &value) function pointer
    assign_function__StateParameter__effort_measured,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "gravity",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    7,  // array size
    false,  // is upper bound
    offsetof(custom_msgs::msg::StateParameter, gravity),  // bytes offset in struct
    nullptr,  // default value
    size_function__StateParameter__gravity,  // size() function pointer
    get_const_function__StateParameter__gravity,  // get_const(index) function pointer
    get_function__StateParameter__gravity,  // get(index) function pointer
    fetch_function__StateParameter__gravity,  // fetch(index, &value) function pointer
    assign_function__StateParameter__gravity,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "o_t_f",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    16,  // array size
    false,  // is upper bound
    offsetof(custom_msgs::msg::StateParameter, o_t_f),  // bytes offset in struct
    nullptr,  // default value
    size_function__StateParameter__o_t_f,  // size() function pointer
    get_const_function__StateParameter__o_t_f,  // get_const(index) function pointer
    get_function__StateParameter__o_t_f,  // get(index) function pointer
    fetch_function__StateParameter__o_t_f,  // fetch(index, &value) function pointer
    assign_function__StateParameter__o_t_f,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "mass",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    49,  // array size
    false,  // is upper bound
    offsetof(custom_msgs::msg::StateParameter, mass),  // bytes offset in struct
    nullptr,  // default value
    size_function__StateParameter__mass,  // size() function pointer
    get_const_function__StateParameter__mass,  // get_const(index) function pointer
    get_function__StateParameter__mass,  // get(index) function pointer
    fetch_function__StateParameter__mass,  // fetch(index, &value) function pointer
    assign_function__StateParameter__mass,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "coriolis",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    7,  // array size
    false,  // is upper bound
    offsetof(custom_msgs::msg::StateParameter, coriolis),  // bytes offset in struct
    nullptr,  // default value
    size_function__StateParameter__coriolis,  // size() function pointer
    get_const_function__StateParameter__coriolis,  // get_const(index) function pointer
    get_function__StateParameter__coriolis,  // get(index) function pointer
    fetch_function__StateParameter__coriolis,  // fetch(index, &value) function pointer
    assign_function__StateParameter__coriolis,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "zero_jacobian_flange",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    42,  // array size
    false,  // is upper bound
    offsetof(custom_msgs::msg::StateParameter, zero_jacobian_flange),  // bytes offset in struct
    nullptr,  // default value
    size_function__StateParameter__zero_jacobian_flange,  // size() function pointer
    get_const_function__StateParameter__zero_jacobian_flange,  // get_const(index) function pointer
    get_function__StateParameter__zero_jacobian_flange,  // get(index) function pointer
    fetch_function__StateParameter__zero_jacobian_flange,  // fetch(index, &value) function pointer
    assign_function__StateParameter__zero_jacobian_flange,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers StateParameter_message_members = {
  "custom_msgs::msg",  // message namespace
  "StateParameter",  // message name
  9,  // number of fields
  sizeof(custom_msgs::msg::StateParameter),
  StateParameter_message_member_array,  // message members
  StateParameter_init_function,  // function to initialize message memory (memory has to be allocated)
  StateParameter_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t StateParameter_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &StateParameter_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace custom_msgs


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<custom_msgs::msg::StateParameter>()
{
  return &::custom_msgs::msg::rosidl_typesupport_introspection_cpp::StateParameter_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, custom_msgs, msg, StateParameter)() {
  return &::custom_msgs::msg::rosidl_typesupport_introspection_cpp::StateParameter_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
