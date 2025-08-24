// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from custom_msgs:msg/MotorMeasurement.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "custom_msgs/msg/detail/motor_measurement__struct.hpp"
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

void MotorMeasurement_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) custom_msgs::msg::MotorMeasurement(_init);
}

void MotorMeasurement_fini_function(void * message_memory)
{
  auto typed_message = static_cast<custom_msgs::msg::MotorMeasurement *>(message_memory);
  typed_message->~MotorMeasurement();
}

size_t size_function__MotorMeasurement__vel(const void * untyped_member)
{
  (void)untyped_member;
  return 4;
}

const void * get_const_function__MotorMeasurement__vel(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 4> *>(untyped_member);
  return &member[index];
}

void * get_function__MotorMeasurement__vel(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 4> *>(untyped_member);
  return &member[index];
}

void fetch_function__MotorMeasurement__vel(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__MotorMeasurement__vel(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__MotorMeasurement__vel(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__MotorMeasurement__vel(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__MotorMeasurement__pos(const void * untyped_member)
{
  (void)untyped_member;
  return 4;
}

const void * get_const_function__MotorMeasurement__pos(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 4> *>(untyped_member);
  return &member[index];
}

void * get_function__MotorMeasurement__pos(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 4> *>(untyped_member);
  return &member[index];
}

void fetch_function__MotorMeasurement__pos(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__MotorMeasurement__pos(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__MotorMeasurement__pos(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__MotorMeasurement__pos(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__MotorMeasurement__torque(const void * untyped_member)
{
  (void)untyped_member;
  return 4;
}

const void * get_const_function__MotorMeasurement__torque(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 4> *>(untyped_member);
  return &member[index];
}

void * get_function__MotorMeasurement__torque(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 4> *>(untyped_member);
  return &member[index];
}

void fetch_function__MotorMeasurement__torque(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__MotorMeasurement__torque(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__MotorMeasurement__torque(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__MotorMeasurement__torque(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember MotorMeasurement_message_member_array[4] = {
  {
    "header",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<std_msgs::msg::Header>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(custom_msgs::msg::MotorMeasurement, header),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "vel",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    4,  // array size
    false,  // is upper bound
    offsetof(custom_msgs::msg::MotorMeasurement, vel),  // bytes offset in struct
    nullptr,  // default value
    size_function__MotorMeasurement__vel,  // size() function pointer
    get_const_function__MotorMeasurement__vel,  // get_const(index) function pointer
    get_function__MotorMeasurement__vel,  // get(index) function pointer
    fetch_function__MotorMeasurement__vel,  // fetch(index, &value) function pointer
    assign_function__MotorMeasurement__vel,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "pos",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    4,  // array size
    false,  // is upper bound
    offsetof(custom_msgs::msg::MotorMeasurement, pos),  // bytes offset in struct
    nullptr,  // default value
    size_function__MotorMeasurement__pos,  // size() function pointer
    get_const_function__MotorMeasurement__pos,  // get_const(index) function pointer
    get_function__MotorMeasurement__pos,  // get(index) function pointer
    fetch_function__MotorMeasurement__pos,  // fetch(index, &value) function pointer
    assign_function__MotorMeasurement__pos,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "torque",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    4,  // array size
    false,  // is upper bound
    offsetof(custom_msgs::msg::MotorMeasurement, torque),  // bytes offset in struct
    nullptr,  // default value
    size_function__MotorMeasurement__torque,  // size() function pointer
    get_const_function__MotorMeasurement__torque,  // get_const(index) function pointer
    get_function__MotorMeasurement__torque,  // get(index) function pointer
    fetch_function__MotorMeasurement__torque,  // fetch(index, &value) function pointer
    assign_function__MotorMeasurement__torque,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers MotorMeasurement_message_members = {
  "custom_msgs::msg",  // message namespace
  "MotorMeasurement",  // message name
  4,  // number of fields
  sizeof(custom_msgs::msg::MotorMeasurement),
  MotorMeasurement_message_member_array,  // message members
  MotorMeasurement_init_function,  // function to initialize message memory (memory has to be allocated)
  MotorMeasurement_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t MotorMeasurement_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &MotorMeasurement_message_members,
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
get_message_type_support_handle<custom_msgs::msg::MotorMeasurement>()
{
  return &::custom_msgs::msg::rosidl_typesupport_introspection_cpp::MotorMeasurement_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, custom_msgs, msg, MotorMeasurement)() {
  return &::custom_msgs::msg::rosidl_typesupport_introspection_cpp::MotorMeasurement_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
