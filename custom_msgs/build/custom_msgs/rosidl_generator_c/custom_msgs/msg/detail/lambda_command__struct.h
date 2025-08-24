// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from custom_msgs:msg/LambdaCommand.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MSGS__MSG__DETAIL__LAMBDA_COMMAND__STRUCT_H_
#define CUSTOM_MSGS__MSG__DETAIL__LAMBDA_COMMAND__STRUCT_H_

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
// Member 'linear'
// Member 'angular'
#include "geometry_msgs/msg/detail/vector3__struct.h"

/// Struct defined in msg/LambdaCommand in the package custom_msgs.
typedef struct custom_msgs__msg__LambdaCommand
{
  std_msgs__msg__Header header;
  geometry_msgs__msg__Vector3 linear;
  geometry_msgs__msg__Vector3 angular;
  double v_gripper;
  bool enable_backlash_compensation;
} custom_msgs__msg__LambdaCommand;

// Struct for a sequence of custom_msgs__msg__LambdaCommand.
typedef struct custom_msgs__msg__LambdaCommand__Sequence
{
  custom_msgs__msg__LambdaCommand * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} custom_msgs__msg__LambdaCommand__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // CUSTOM_MSGS__MSG__DETAIL__LAMBDA_COMMAND__STRUCT_H_
