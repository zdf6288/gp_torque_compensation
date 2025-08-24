// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from custom_msgs:msg/TaskSpaceCommand.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "custom_msgs/msg/detail/task_space_command__struct.hpp"
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

void TaskSpaceCommand_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) custom_msgs::msg::TaskSpaceCommand(_init);
}

void TaskSpaceCommand_fini_function(void * message_memory)
{
  auto typed_message = static_cast<custom_msgs::msg::TaskSpaceCommand *>(message_memory);
  typed_message->~TaskSpaceCommand();
}

size_t size_function__TaskSpaceCommand__x_des(const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * get_const_function__TaskSpaceCommand__x_des(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void * get_function__TaskSpaceCommand__x_des(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void fetch_function__TaskSpaceCommand__x_des(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__TaskSpaceCommand__x_des(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__TaskSpaceCommand__x_des(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__TaskSpaceCommand__x_des(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__TaskSpaceCommand__dx_des(const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * get_const_function__TaskSpaceCommand__dx_des(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void * get_function__TaskSpaceCommand__dx_des(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void fetch_function__TaskSpaceCommand__dx_des(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__TaskSpaceCommand__dx_des(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__TaskSpaceCommand__dx_des(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__TaskSpaceCommand__dx_des(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

size_t size_function__TaskSpaceCommand__ddx_des(const void * untyped_member)
{
  (void)untyped_member;
  return 6;
}

const void * get_const_function__TaskSpaceCommand__ddx_des(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void * get_function__TaskSpaceCommand__ddx_des(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<double, 6> *>(untyped_member);
  return &member[index];
}

void fetch_function__TaskSpaceCommand__ddx_des(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const double *>(
    get_const_function__TaskSpaceCommand__ddx_des(untyped_member, index));
  auto & value = *reinterpret_cast<double *>(untyped_value);
  value = item;
}

void assign_function__TaskSpaceCommand__ddx_des(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<double *>(
    get_function__TaskSpaceCommand__ddx_des(untyped_member, index));
  const auto & value = *reinterpret_cast<const double *>(untyped_value);
  item = value;
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember TaskSpaceCommand_message_member_array[4] = {
  {
    "header",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<std_msgs::msg::Header>(),  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(custom_msgs::msg::TaskSpaceCommand, header),  // bytes offset in struct
    nullptr,  // default value
    nullptr,  // size() function pointer
    nullptr,  // get_const(index) function pointer
    nullptr,  // get(index) function pointer
    nullptr,  // fetch(index, &value) function pointer
    nullptr,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "x_des",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(custom_msgs::msg::TaskSpaceCommand, x_des),  // bytes offset in struct
    nullptr,  // default value
    size_function__TaskSpaceCommand__x_des,  // size() function pointer
    get_const_function__TaskSpaceCommand__x_des,  // get_const(index) function pointer
    get_function__TaskSpaceCommand__x_des,  // get(index) function pointer
    fetch_function__TaskSpaceCommand__x_des,  // fetch(index, &value) function pointer
    assign_function__TaskSpaceCommand__x_des,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "dx_des",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(custom_msgs::msg::TaskSpaceCommand, dx_des),  // bytes offset in struct
    nullptr,  // default value
    size_function__TaskSpaceCommand__dx_des,  // size() function pointer
    get_const_function__TaskSpaceCommand__dx_des,  // get_const(index) function pointer
    get_function__TaskSpaceCommand__dx_des,  // get(index) function pointer
    fetch_function__TaskSpaceCommand__dx_des,  // fetch(index, &value) function pointer
    assign_function__TaskSpaceCommand__dx_des,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  },
  {
    "ddx_des",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    6,  // array size
    false,  // is upper bound
    offsetof(custom_msgs::msg::TaskSpaceCommand, ddx_des),  // bytes offset in struct
    nullptr,  // default value
    size_function__TaskSpaceCommand__ddx_des,  // size() function pointer
    get_const_function__TaskSpaceCommand__ddx_des,  // get_const(index) function pointer
    get_function__TaskSpaceCommand__ddx_des,  // get(index) function pointer
    fetch_function__TaskSpaceCommand__ddx_des,  // fetch(index, &value) function pointer
    assign_function__TaskSpaceCommand__ddx_des,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers TaskSpaceCommand_message_members = {
  "custom_msgs::msg",  // message namespace
  "TaskSpaceCommand",  // message name
  4,  // number of fields
  sizeof(custom_msgs::msg::TaskSpaceCommand),
  TaskSpaceCommand_message_member_array,  // message members
  TaskSpaceCommand_init_function,  // function to initialize message memory (memory has to be allocated)
  TaskSpaceCommand_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t TaskSpaceCommand_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &TaskSpaceCommand_message_members,
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
get_message_type_support_handle<custom_msgs::msg::TaskSpaceCommand>()
{
  return &::custom_msgs::msg::rosidl_typesupport_introspection_cpp::TaskSpaceCommand_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, custom_msgs, msg, TaskSpaceCommand)() {
  return &::custom_msgs::msg::rosidl_typesupport_introspection_cpp::TaskSpaceCommand_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
