// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from custom_msgs:msg/MotorCommand.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "custom_msgs/msg/detail/motor_command__rosidl_typesupport_introspection_c.h"
#include "custom_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "custom_msgs/msg/detail/motor_command__functions.h"
#include "custom_msgs/msg/detail/motor_command__struct.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/header.h"
// Member `header`
#include "std_msgs/msg/detail/header__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void custom_msgs__msg__MotorCommand__rosidl_typesupport_introspection_c__MotorCommand_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  custom_msgs__msg__MotorCommand__init(message_memory);
}

void custom_msgs__msg__MotorCommand__rosidl_typesupport_introspection_c__MotorCommand_fini_function(void * message_memory)
{
  custom_msgs__msg__MotorCommand__fini(message_memory);
}

size_t custom_msgs__msg__MotorCommand__rosidl_typesupport_introspection_c__size_function__MotorCommand__vel(
  const void * untyped_member)
{
  (void)untyped_member;
  return 4;
}

const void * custom_msgs__msg__MotorCommand__rosidl_typesupport_introspection_c__get_const_function__MotorCommand__vel(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * custom_msgs__msg__MotorCommand__rosidl_typesupport_introspection_c__get_function__MotorCommand__vel(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void custom_msgs__msg__MotorCommand__rosidl_typesupport_introspection_c__fetch_function__MotorCommand__vel(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    custom_msgs__msg__MotorCommand__rosidl_typesupport_introspection_c__get_const_function__MotorCommand__vel(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void custom_msgs__msg__MotorCommand__rosidl_typesupport_introspection_c__assign_function__MotorCommand__vel(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    custom_msgs__msg__MotorCommand__rosidl_typesupport_introspection_c__get_function__MotorCommand__vel(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

static rosidl_typesupport_introspection_c__MessageMember custom_msgs__msg__MotorCommand__rosidl_typesupport_introspection_c__MotorCommand_message_member_array[2] = {
  {
    "header",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(custom_msgs__msg__MotorCommand, header),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "vel",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    4,  // array size
    false,  // is upper bound
    offsetof(custom_msgs__msg__MotorCommand, vel),  // bytes offset in struct
    NULL,  // default value
    custom_msgs__msg__MotorCommand__rosidl_typesupport_introspection_c__size_function__MotorCommand__vel,  // size() function pointer
    custom_msgs__msg__MotorCommand__rosidl_typesupport_introspection_c__get_const_function__MotorCommand__vel,  // get_const(index) function pointer
    custom_msgs__msg__MotorCommand__rosidl_typesupport_introspection_c__get_function__MotorCommand__vel,  // get(index) function pointer
    custom_msgs__msg__MotorCommand__rosidl_typesupport_introspection_c__fetch_function__MotorCommand__vel,  // fetch(index, &value) function pointer
    custom_msgs__msg__MotorCommand__rosidl_typesupport_introspection_c__assign_function__MotorCommand__vel,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers custom_msgs__msg__MotorCommand__rosidl_typesupport_introspection_c__MotorCommand_message_members = {
  "custom_msgs__msg",  // message namespace
  "MotorCommand",  // message name
  2,  // number of fields
  sizeof(custom_msgs__msg__MotorCommand),
  custom_msgs__msg__MotorCommand__rosidl_typesupport_introspection_c__MotorCommand_message_member_array,  // message members
  custom_msgs__msg__MotorCommand__rosidl_typesupport_introspection_c__MotorCommand_init_function,  // function to initialize message memory (memory has to be allocated)
  custom_msgs__msg__MotorCommand__rosidl_typesupport_introspection_c__MotorCommand_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t custom_msgs__msg__MotorCommand__rosidl_typesupport_introspection_c__MotorCommand_message_type_support_handle = {
  0,
  &custom_msgs__msg__MotorCommand__rosidl_typesupport_introspection_c__MotorCommand_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_custom_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, custom_msgs, msg, MotorCommand)() {
  custom_msgs__msg__MotorCommand__rosidl_typesupport_introspection_c__MotorCommand_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Header)();
  if (!custom_msgs__msg__MotorCommand__rosidl_typesupport_introspection_c__MotorCommand_message_type_support_handle.typesupport_identifier) {
    custom_msgs__msg__MotorCommand__rosidl_typesupport_introspection_c__MotorCommand_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &custom_msgs__msg__MotorCommand__rosidl_typesupport_introspection_c__MotorCommand_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
