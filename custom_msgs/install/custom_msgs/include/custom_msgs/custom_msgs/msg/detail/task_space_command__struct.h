// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from custom_msgs:msg/TaskSpaceCommand.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MSGS__MSG__DETAIL__TASK_SPACE_COMMAND__STRUCT_H_
#define CUSTOM_MSGS__MSG__DETAIL__TASK_SPACE_COMMAND__STRUCT_H_

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

/// Struct defined in msg/TaskSpaceCommand in the package custom_msgs.
typedef struct custom_msgs__msg__TaskSpaceCommand
{
  std_msgs__msg__Header header;
  double x_des[6];
  double dx_des[6];
  double ddx_des[6];
} custom_msgs__msg__TaskSpaceCommand;

// Struct for a sequence of custom_msgs__msg__TaskSpaceCommand.
typedef struct custom_msgs__msg__TaskSpaceCommand__Sequence
{
  custom_msgs__msg__TaskSpaceCommand * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} custom_msgs__msg__TaskSpaceCommand__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // CUSTOM_MSGS__MSG__DETAIL__TASK_SPACE_COMMAND__STRUCT_H_
