// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from custom_msgs:srv/JointPositionAdjust.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MSGS__SRV__DETAIL__JOINT_POSITION_ADJUST__STRUCT_H_
#define CUSTOM_MSGS__SRV__DETAIL__JOINT_POSITION_ADJUST__STRUCT_H_

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
#include "rosidl_runtime_c/primitives_sequence.h"

/// Struct defined in srv/JointPositionAdjust in the package custom_msgs.
typedef struct custom_msgs__srv__JointPositionAdjust_Request
{
  rosidl_runtime_c__double__Sequence q_des;
  rosidl_runtime_c__double__Sequence dq_des;
} custom_msgs__srv__JointPositionAdjust_Request;

// Struct for a sequence of custom_msgs__srv__JointPositionAdjust_Request.
typedef struct custom_msgs__srv__JointPositionAdjust_Request__Sequence
{
  custom_msgs__srv__JointPositionAdjust_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} custom_msgs__srv__JointPositionAdjust_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'message'
#include "rosidl_runtime_c/string.h"

/// Struct defined in srv/JointPositionAdjust in the package custom_msgs.
typedef struct custom_msgs__srv__JointPositionAdjust_Response
{
  bool success;
  rosidl_runtime_c__String message;
} custom_msgs__srv__JointPositionAdjust_Response;

// Struct for a sequence of custom_msgs__srv__JointPositionAdjust_Response.
typedef struct custom_msgs__srv__JointPositionAdjust_Response__Sequence
{
  custom_msgs__srv__JointPositionAdjust_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} custom_msgs__srv__JointPositionAdjust_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // CUSTOM_MSGS__SRV__DETAIL__JOINT_POSITION_ADJUST__STRUCT_H_
