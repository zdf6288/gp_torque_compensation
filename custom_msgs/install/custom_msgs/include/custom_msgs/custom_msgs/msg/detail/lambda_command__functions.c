// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from custom_msgs:msg/LambdaCommand.idl
// generated code does not contain a copyright notice
#include "custom_msgs/msg/detail/lambda_command__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"
// Member `linear`
// Member `angular`
#include "geometry_msgs/msg/detail/vector3__functions.h"

bool
custom_msgs__msg__LambdaCommand__init(custom_msgs__msg__LambdaCommand * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    custom_msgs__msg__LambdaCommand__fini(msg);
    return false;
  }
  // linear
  if (!geometry_msgs__msg__Vector3__init(&msg->linear)) {
    custom_msgs__msg__LambdaCommand__fini(msg);
    return false;
  }
  // angular
  if (!geometry_msgs__msg__Vector3__init(&msg->angular)) {
    custom_msgs__msg__LambdaCommand__fini(msg);
    return false;
  }
  // v_gripper
  // enable_backlash_compensation
  return true;
}

void
custom_msgs__msg__LambdaCommand__fini(custom_msgs__msg__LambdaCommand * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // linear
  geometry_msgs__msg__Vector3__fini(&msg->linear);
  // angular
  geometry_msgs__msg__Vector3__fini(&msg->angular);
  // v_gripper
  // enable_backlash_compensation
}

bool
custom_msgs__msg__LambdaCommand__are_equal(const custom_msgs__msg__LambdaCommand * lhs, const custom_msgs__msg__LambdaCommand * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__are_equal(
      &(lhs->header), &(rhs->header)))
  {
    return false;
  }
  // linear
  if (!geometry_msgs__msg__Vector3__are_equal(
      &(lhs->linear), &(rhs->linear)))
  {
    return false;
  }
  // angular
  if (!geometry_msgs__msg__Vector3__are_equal(
      &(lhs->angular), &(rhs->angular)))
  {
    return false;
  }
  // v_gripper
  if (lhs->v_gripper != rhs->v_gripper) {
    return false;
  }
  // enable_backlash_compensation
  if (lhs->enable_backlash_compensation != rhs->enable_backlash_compensation) {
    return false;
  }
  return true;
}

bool
custom_msgs__msg__LambdaCommand__copy(
  const custom_msgs__msg__LambdaCommand * input,
  custom_msgs__msg__LambdaCommand * output)
{
  if (!input || !output) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__copy(
      &(input->header), &(output->header)))
  {
    return false;
  }
  // linear
  if (!geometry_msgs__msg__Vector3__copy(
      &(input->linear), &(output->linear)))
  {
    return false;
  }
  // angular
  if (!geometry_msgs__msg__Vector3__copy(
      &(input->angular), &(output->angular)))
  {
    return false;
  }
  // v_gripper
  output->v_gripper = input->v_gripper;
  // enable_backlash_compensation
  output->enable_backlash_compensation = input->enable_backlash_compensation;
  return true;
}

custom_msgs__msg__LambdaCommand *
custom_msgs__msg__LambdaCommand__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  custom_msgs__msg__LambdaCommand * msg = (custom_msgs__msg__LambdaCommand *)allocator.allocate(sizeof(custom_msgs__msg__LambdaCommand), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(custom_msgs__msg__LambdaCommand));
  bool success = custom_msgs__msg__LambdaCommand__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
custom_msgs__msg__LambdaCommand__destroy(custom_msgs__msg__LambdaCommand * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    custom_msgs__msg__LambdaCommand__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
custom_msgs__msg__LambdaCommand__Sequence__init(custom_msgs__msg__LambdaCommand__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  custom_msgs__msg__LambdaCommand * data = NULL;

  if (size) {
    data = (custom_msgs__msg__LambdaCommand *)allocator.zero_allocate(size, sizeof(custom_msgs__msg__LambdaCommand), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = custom_msgs__msg__LambdaCommand__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        custom_msgs__msg__LambdaCommand__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
custom_msgs__msg__LambdaCommand__Sequence__fini(custom_msgs__msg__LambdaCommand__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      custom_msgs__msg__LambdaCommand__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

custom_msgs__msg__LambdaCommand__Sequence *
custom_msgs__msg__LambdaCommand__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  custom_msgs__msg__LambdaCommand__Sequence * array = (custom_msgs__msg__LambdaCommand__Sequence *)allocator.allocate(sizeof(custom_msgs__msg__LambdaCommand__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = custom_msgs__msg__LambdaCommand__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
custom_msgs__msg__LambdaCommand__Sequence__destroy(custom_msgs__msg__LambdaCommand__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    custom_msgs__msg__LambdaCommand__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
custom_msgs__msg__LambdaCommand__Sequence__are_equal(const custom_msgs__msg__LambdaCommand__Sequence * lhs, const custom_msgs__msg__LambdaCommand__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!custom_msgs__msg__LambdaCommand__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
custom_msgs__msg__LambdaCommand__Sequence__copy(
  const custom_msgs__msg__LambdaCommand__Sequence * input,
  custom_msgs__msg__LambdaCommand__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(custom_msgs__msg__LambdaCommand);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    custom_msgs__msg__LambdaCommand * data =
      (custom_msgs__msg__LambdaCommand *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!custom_msgs__msg__LambdaCommand__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          custom_msgs__msg__LambdaCommand__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!custom_msgs__msg__LambdaCommand__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
