// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from custom_msgs:msg/StateParameter.idl
// generated code does not contain a copyright notice
#include "custom_msgs/msg/detail/state_parameter__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"

bool
custom_msgs__msg__StateParameter__init(custom_msgs__msg__StateParameter * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    custom_msgs__msg__StateParameter__fini(msg);
    return false;
  }
  // position
  // velocity
  // effort_measured
  // gravity
  // o_t_f
  // mass
  // coriolis
  // zero_jacobian_flange
  return true;
}

void
custom_msgs__msg__StateParameter__fini(custom_msgs__msg__StateParameter * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // position
  // velocity
  // effort_measured
  // gravity
  // o_t_f
  // mass
  // coriolis
  // zero_jacobian_flange
}

bool
custom_msgs__msg__StateParameter__are_equal(const custom_msgs__msg__StateParameter * lhs, const custom_msgs__msg__StateParameter * rhs)
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
  // position
  for (size_t i = 0; i < 7; ++i) {
    if (lhs->position[i] != rhs->position[i]) {
      return false;
    }
  }
  // velocity
  for (size_t i = 0; i < 7; ++i) {
    if (lhs->velocity[i] != rhs->velocity[i]) {
      return false;
    }
  }
  // effort_measured
  for (size_t i = 0; i < 7; ++i) {
    if (lhs->effort_measured[i] != rhs->effort_measured[i]) {
      return false;
    }
  }
  // gravity
  for (size_t i = 0; i < 7; ++i) {
    if (lhs->gravity[i] != rhs->gravity[i]) {
      return false;
    }
  }
  // o_t_f
  for (size_t i = 0; i < 16; ++i) {
    if (lhs->o_t_f[i] != rhs->o_t_f[i]) {
      return false;
    }
  }
  // mass
  for (size_t i = 0; i < 49; ++i) {
    if (lhs->mass[i] != rhs->mass[i]) {
      return false;
    }
  }
  // coriolis
  for (size_t i = 0; i < 7; ++i) {
    if (lhs->coriolis[i] != rhs->coriolis[i]) {
      return false;
    }
  }
  // zero_jacobian_flange
  for (size_t i = 0; i < 42; ++i) {
    if (lhs->zero_jacobian_flange[i] != rhs->zero_jacobian_flange[i]) {
      return false;
    }
  }
  return true;
}

bool
custom_msgs__msg__StateParameter__copy(
  const custom_msgs__msg__StateParameter * input,
  custom_msgs__msg__StateParameter * output)
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
  // position
  for (size_t i = 0; i < 7; ++i) {
    output->position[i] = input->position[i];
  }
  // velocity
  for (size_t i = 0; i < 7; ++i) {
    output->velocity[i] = input->velocity[i];
  }
  // effort_measured
  for (size_t i = 0; i < 7; ++i) {
    output->effort_measured[i] = input->effort_measured[i];
  }
  // gravity
  for (size_t i = 0; i < 7; ++i) {
    output->gravity[i] = input->gravity[i];
  }
  // o_t_f
  for (size_t i = 0; i < 16; ++i) {
    output->o_t_f[i] = input->o_t_f[i];
  }
  // mass
  for (size_t i = 0; i < 49; ++i) {
    output->mass[i] = input->mass[i];
  }
  // coriolis
  for (size_t i = 0; i < 7; ++i) {
    output->coriolis[i] = input->coriolis[i];
  }
  // zero_jacobian_flange
  for (size_t i = 0; i < 42; ++i) {
    output->zero_jacobian_flange[i] = input->zero_jacobian_flange[i];
  }
  return true;
}

custom_msgs__msg__StateParameter *
custom_msgs__msg__StateParameter__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  custom_msgs__msg__StateParameter * msg = (custom_msgs__msg__StateParameter *)allocator.allocate(sizeof(custom_msgs__msg__StateParameter), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(custom_msgs__msg__StateParameter));
  bool success = custom_msgs__msg__StateParameter__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
custom_msgs__msg__StateParameter__destroy(custom_msgs__msg__StateParameter * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    custom_msgs__msg__StateParameter__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
custom_msgs__msg__StateParameter__Sequence__init(custom_msgs__msg__StateParameter__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  custom_msgs__msg__StateParameter * data = NULL;

  if (size) {
    data = (custom_msgs__msg__StateParameter *)allocator.zero_allocate(size, sizeof(custom_msgs__msg__StateParameter), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = custom_msgs__msg__StateParameter__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        custom_msgs__msg__StateParameter__fini(&data[i - 1]);
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
custom_msgs__msg__StateParameter__Sequence__fini(custom_msgs__msg__StateParameter__Sequence * array)
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
      custom_msgs__msg__StateParameter__fini(&array->data[i]);
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

custom_msgs__msg__StateParameter__Sequence *
custom_msgs__msg__StateParameter__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  custom_msgs__msg__StateParameter__Sequence * array = (custom_msgs__msg__StateParameter__Sequence *)allocator.allocate(sizeof(custom_msgs__msg__StateParameter__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = custom_msgs__msg__StateParameter__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
custom_msgs__msg__StateParameter__Sequence__destroy(custom_msgs__msg__StateParameter__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    custom_msgs__msg__StateParameter__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
custom_msgs__msg__StateParameter__Sequence__are_equal(const custom_msgs__msg__StateParameter__Sequence * lhs, const custom_msgs__msg__StateParameter__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!custom_msgs__msg__StateParameter__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
custom_msgs__msg__StateParameter__Sequence__copy(
  const custom_msgs__msg__StateParameter__Sequence * input,
  custom_msgs__msg__StateParameter__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(custom_msgs__msg__StateParameter);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    custom_msgs__msg__StateParameter * data =
      (custom_msgs__msg__StateParameter *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!custom_msgs__msg__StateParameter__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          custom_msgs__msg__StateParameter__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!custom_msgs__msg__StateParameter__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
