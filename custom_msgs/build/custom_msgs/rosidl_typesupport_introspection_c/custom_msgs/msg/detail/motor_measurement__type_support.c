// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from custom_msgs:msg/MotorMeasurement.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "custom_msgs/msg/detail/motor_measurement__rosidl_typesupport_introspection_c.h"
#include "custom_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "custom_msgs/msg/detail/motor_measurement__functions.h"
#include "custom_msgs/msg/detail/motor_measurement__struct.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/header.h"
// Member `header`
#include "std_msgs/msg/detail/header__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void custom_msgs__msg__MotorMeasurement__rosidl_typesupport_introspection_c__MotorMeasurement_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  custom_msgs__msg__MotorMeasurement__init(message_memory);
}

void custom_msgs__msg__MotorMeasurement__rosidl_typesupport_introspection_c__MotorMeasurement_fini_function(void * message_memory)
{
  custom_msgs__msg__MotorMeasurement__fini(message_memory);
}

size_t custom_msgs__msg__MotorMeasurement__rosidl_typesupport_introspection_c__size_function__MotorMeasurement__vel(
  const void * untyped_member)
{
  (void)untyped_member;
  return 4;
}

const void * custom_msgs__msg__MotorMeasurement__rosidl_typesupport_introspection_c__get_const_function__MotorMeasurement__vel(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * custom_msgs__msg__MotorMeasurement__rosidl_typesupport_introspection_c__get_function__MotorMeasurement__vel(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void custom_msgs__msg__MotorMeasurement__rosidl_typesupport_introspection_c__fetch_function__MotorMeasurement__vel(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    custom_msgs__msg__MotorMeasurement__rosidl_typesupport_introspection_c__get_const_function__MotorMeasurement__vel(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void custom_msgs__msg__MotorMeasurement__rosidl_typesupport_introspection_c__assign_function__MotorMeasurement__vel(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    custom_msgs__msg__MotorMeasurement__rosidl_typesupport_introspection_c__get_function__MotorMeasurement__vel(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t custom_msgs__msg__MotorMeasurement__rosidl_typesupport_introspection_c__size_function__MotorMeasurement__pos(
  const void * untyped_member)
{
  (void)untyped_member;
  return 4;
}

const void * custom_msgs__msg__MotorMeasurement__rosidl_typesupport_introspection_c__get_const_function__MotorMeasurement__pos(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * custom_msgs__msg__MotorMeasurement__rosidl_typesupport_introspection_c__get_function__MotorMeasurement__pos(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void custom_msgs__msg__MotorMeasurement__rosidl_typesupport_introspection_c__fetch_function__MotorMeasurement__pos(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    custom_msgs__msg__MotorMeasurement__rosidl_typesupport_introspection_c__get_const_function__MotorMeasurement__pos(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void custom_msgs__msg__MotorMeasurement__rosidl_typesupport_introspection_c__assign_function__MotorMeasurement__pos(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    custom_msgs__msg__MotorMeasurement__rosidl_typesupport_introspection_c__get_function__MotorMeasurement__pos(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

size_t custom_msgs__msg__MotorMeasurement__rosidl_typesupport_introspection_c__size_function__MotorMeasurement__torque(
  const void * untyped_member)
{
  (void)untyped_member;
  return 4;
}

const void * custom_msgs__msg__MotorMeasurement__rosidl_typesupport_introspection_c__get_const_function__MotorMeasurement__torque(
  const void * untyped_member, size_t index)
{
  const double * member =
    (const double *)(untyped_member);
  return &member[index];
}

void * custom_msgs__msg__MotorMeasurement__rosidl_typesupport_introspection_c__get_function__MotorMeasurement__torque(
  void * untyped_member, size_t index)
{
  double * member =
    (double *)(untyped_member);
  return &member[index];
}

void custom_msgs__msg__MotorMeasurement__rosidl_typesupport_introspection_c__fetch_function__MotorMeasurement__torque(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const double * item =
    ((const double *)
    custom_msgs__msg__MotorMeasurement__rosidl_typesupport_introspection_c__get_const_function__MotorMeasurement__torque(untyped_member, index));
  double * value =
    (double *)(untyped_value);
  *value = *item;
}

void custom_msgs__msg__MotorMeasurement__rosidl_typesupport_introspection_c__assign_function__MotorMeasurement__torque(
  void * untyped_member, size_t index, const void * untyped_value)
{
  double * item =
    ((double *)
    custom_msgs__msg__MotorMeasurement__rosidl_typesupport_introspection_c__get_function__MotorMeasurement__torque(untyped_member, index));
  const double * value =
    (const double *)(untyped_value);
  *item = *value;
}

static rosidl_typesupport_introspection_c__MessageMember custom_msgs__msg__MotorMeasurement__rosidl_typesupport_introspection_c__MotorMeasurement_message_member_array[4] = {
  {
    "header",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(custom_msgs__msg__MotorMeasurement, header),  // bytes offset in struct
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
    offsetof(custom_msgs__msg__MotorMeasurement, vel),  // bytes offset in struct
    NULL,  // default value
    custom_msgs__msg__MotorMeasurement__rosidl_typesupport_introspection_c__size_function__MotorMeasurement__vel,  // size() function pointer
    custom_msgs__msg__MotorMeasurement__rosidl_typesupport_introspection_c__get_const_function__MotorMeasurement__vel,  // get_const(index) function pointer
    custom_msgs__msg__MotorMeasurement__rosidl_typesupport_introspection_c__get_function__MotorMeasurement__vel,  // get(index) function pointer
    custom_msgs__msg__MotorMeasurement__rosidl_typesupport_introspection_c__fetch_function__MotorMeasurement__vel,  // fetch(index, &value) function pointer
    custom_msgs__msg__MotorMeasurement__rosidl_typesupport_introspection_c__assign_function__MotorMeasurement__vel,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "pos",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    4,  // array size
    false,  // is upper bound
    offsetof(custom_msgs__msg__MotorMeasurement, pos),  // bytes offset in struct
    NULL,  // default value
    custom_msgs__msg__MotorMeasurement__rosidl_typesupport_introspection_c__size_function__MotorMeasurement__pos,  // size() function pointer
    custom_msgs__msg__MotorMeasurement__rosidl_typesupport_introspection_c__get_const_function__MotorMeasurement__pos,  // get_const(index) function pointer
    custom_msgs__msg__MotorMeasurement__rosidl_typesupport_introspection_c__get_function__MotorMeasurement__pos,  // get(index) function pointer
    custom_msgs__msg__MotorMeasurement__rosidl_typesupport_introspection_c__fetch_function__MotorMeasurement__pos,  // fetch(index, &value) function pointer
    custom_msgs__msg__MotorMeasurement__rosidl_typesupport_introspection_c__assign_function__MotorMeasurement__pos,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "torque",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_DOUBLE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    4,  // array size
    false,  // is upper bound
    offsetof(custom_msgs__msg__MotorMeasurement, torque),  // bytes offset in struct
    NULL,  // default value
    custom_msgs__msg__MotorMeasurement__rosidl_typesupport_introspection_c__size_function__MotorMeasurement__torque,  // size() function pointer
    custom_msgs__msg__MotorMeasurement__rosidl_typesupport_introspection_c__get_const_function__MotorMeasurement__torque,  // get_const(index) function pointer
    custom_msgs__msg__MotorMeasurement__rosidl_typesupport_introspection_c__get_function__MotorMeasurement__torque,  // get(index) function pointer
    custom_msgs__msg__MotorMeasurement__rosidl_typesupport_introspection_c__fetch_function__MotorMeasurement__torque,  // fetch(index, &value) function pointer
    custom_msgs__msg__MotorMeasurement__rosidl_typesupport_introspection_c__assign_function__MotorMeasurement__torque,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers custom_msgs__msg__MotorMeasurement__rosidl_typesupport_introspection_c__MotorMeasurement_message_members = {
  "custom_msgs__msg",  // message namespace
  "MotorMeasurement",  // message name
  4,  // number of fields
  sizeof(custom_msgs__msg__MotorMeasurement),
  custom_msgs__msg__MotorMeasurement__rosidl_typesupport_introspection_c__MotorMeasurement_message_member_array,  // message members
  custom_msgs__msg__MotorMeasurement__rosidl_typesupport_introspection_c__MotorMeasurement_init_function,  // function to initialize message memory (memory has to be allocated)
  custom_msgs__msg__MotorMeasurement__rosidl_typesupport_introspection_c__MotorMeasurement_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t custom_msgs__msg__MotorMeasurement__rosidl_typesupport_introspection_c__MotorMeasurement_message_type_support_handle = {
  0,
  &custom_msgs__msg__MotorMeasurement__rosidl_typesupport_introspection_c__MotorMeasurement_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_custom_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, custom_msgs, msg, MotorMeasurement)() {
  custom_msgs__msg__MotorMeasurement__rosidl_typesupport_introspection_c__MotorMeasurement_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Header)();
  if (!custom_msgs__msg__MotorMeasurement__rosidl_typesupport_introspection_c__MotorMeasurement_message_type_support_handle.typesupport_identifier) {
    custom_msgs__msg__MotorMeasurement__rosidl_typesupport_introspection_c__MotorMeasurement_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &custom_msgs__msg__MotorMeasurement__rosidl_typesupport_introspection_c__MotorMeasurement_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
