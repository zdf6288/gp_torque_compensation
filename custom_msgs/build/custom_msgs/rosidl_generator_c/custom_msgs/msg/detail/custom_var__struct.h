// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from custom_msgs:msg/CustomVar.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MSGS__MSG__DETAIL__CUSTOM_VAR__STRUCT_H_
#define CUSTOM_MSGS__MSG__DETAIL__CUSTOM_VAR__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'q_des'
// Member 'dq_des'
// Member 'q_meas'
// Member 'dq_meas'
// Member 'ddq_computed'
// Member 'error_endo'
// Member 'error_rcm'
#include "std_msgs/msg/detail/float64_multi_array__struct.h"

/// Struct defined in msg/CustomVar in the package custom_msgs.
/**
  * Custom message for two variables
 */
typedef struct custom_msgs__msg__CustomVar
{
  std_msgs__msg__Float64MultiArray q_des;
  std_msgs__msg__Float64MultiArray dq_des;
  std_msgs__msg__Float64MultiArray q_meas;
  std_msgs__msg__Float64MultiArray dq_meas;
  std_msgs__msg__Float64MultiArray ddq_computed;
  std_msgs__msg__Float64MultiArray error_endo;
  std_msgs__msg__Float64MultiArray error_rcm;
} custom_msgs__msg__CustomVar;

// Struct for a sequence of custom_msgs__msg__CustomVar.
typedef struct custom_msgs__msg__CustomVar__Sequence
{
  custom_msgs__msg__CustomVar * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} custom_msgs__msg__CustomVar__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // CUSTOM_MSGS__MSG__DETAIL__CUSTOM_VAR__STRUCT_H_
