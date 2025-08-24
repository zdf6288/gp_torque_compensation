// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from custom_msgs:msg/StateParameter.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MSGS__MSG__DETAIL__STATE_PARAMETER__STRUCT_H_
#define CUSTOM_MSGS__MSG__DETAIL__STATE_PARAMETER__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.h"

/// Struct defined in msg/StateParameter in the package custom_msgs.
typedef struct custom_msgs__msg__StateParameter
{
  std_msgs__msg__Header header;
  double position[7];
  double velocity[7];
  double effort_measured[7];
  double gravity[7];
  double o_t_f[16];
  double mass[49];
  double coriolis[7];
  double zero_jacobian_flange[42];
} custom_msgs__msg__StateParameter;

// Struct for a sequence of custom_msgs__msg__StateParameter.
typedef struct custom_msgs__msg__StateParameter__Sequence
{
  custom_msgs__msg__StateParameter * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} custom_msgs__msg__StateParameter__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // CUSTOM_MSGS__MSG__DETAIL__STATE_PARAMETER__STRUCT_H_
