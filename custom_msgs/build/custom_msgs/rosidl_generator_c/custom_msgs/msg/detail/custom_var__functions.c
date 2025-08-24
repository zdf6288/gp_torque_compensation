// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from custom_msgs:msg/CustomVar.idl
// generated code does not contain a copyright notice
#include "custom_msgs/msg/detail/custom_var__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `q_des`
// Member `dq_des`
// Member `q_meas`
// Member `dq_meas`
// Member `ddq_computed`
// Member `error_endo`
// Member `error_rcm`
#include "std_msgs/msg/detail/float64_multi_array__functions.h"

bool
custom_msgs__msg__CustomVar__init(custom_msgs__msg__CustomVar * msg)
{
  if (!msg) {
    return false;
  }
  // q_des
  if (!std_msgs__msg__Float64MultiArray__init(&msg->q_des)) {
    custom_msgs__msg__CustomVar__fini(msg);
    return false;
  }
  // dq_des
  if (!std_msgs__msg__Float64MultiArray__init(&msg->dq_des)) {
    custom_msgs__msg__CustomVar__fini(msg);
    return false;
  }
  // q_meas
  if (!std_msgs__msg__Float64MultiArray__init(&msg->q_meas)) {
    custom_msgs__msg__CustomVar__fini(msg);
    return false;
  }
  // dq_meas
  if (!std_msgs__msg__Float64MultiArray__init(&msg->dq_meas)) {
    custom_msgs__msg__CustomVar__fini(msg);
    return false;
  }
  // ddq_computed
  if (!std_msgs__msg__Float64MultiArray__init(&msg->ddq_computed)) {
    custom_msgs__msg__CustomVar__fini(msg);
    return false;
  }
  // error_endo
  if (!std_msgs__msg__Float64MultiArray__init(&msg->error_endo)) {
    custom_msgs__msg__CustomVar__fini(msg);
    return false;
  }
  // error_rcm
  if (!std_msgs__msg__Float64MultiArray__init(&msg->error_rcm)) {
    custom_msgs__msg__CustomVar__fini(msg);
    return false;
  }
  return true;
}

void
custom_msgs__msg__CustomVar__fini(custom_msgs__msg__CustomVar * msg)
{
  if (!msg) {
    return;
  }
  // q_des
  std_msgs__msg__Float64MultiArray__fini(&msg->q_des);
  // dq_des
  std_msgs__msg__Float64MultiArray__fini(&msg->dq_des);
  // q_meas
  std_msgs__msg__Float64MultiArray__fini(&msg->q_meas);
  // dq_meas
  std_msgs__msg__Float64MultiArray__fini(&msg->dq_meas);
  // ddq_computed
  std_msgs__msg__Float64MultiArray__fini(&msg->ddq_computed);
  // error_endo
  std_msgs__msg__Float64MultiArray__fini(&msg->error_endo);
  // error_rcm
  std_msgs__msg__Float64MultiArray__fini(&msg->error_rcm);
}

bool
custom_msgs__msg__CustomVar__are_equal(const custom_msgs__msg__CustomVar * lhs, const custom_msgs__msg__CustomVar * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // q_des
  if (!std_msgs__msg__Float64MultiArray__are_equal(
      &(lhs->q_des), &(rhs->q_des)))
  {
    return false;
  }
  // dq_des
  if (!std_msgs__msg__Float64MultiArray__are_equal(
      &(lhs->dq_des), &(rhs->dq_des)))
  {
    return false;
  }
  // q_meas
  if (!std_msgs__msg__Float64MultiArray__are_equal(
      &(lhs->q_meas), &(rhs->q_meas)))
  {
    return false;
  }
  // dq_meas
  if (!std_msgs__msg__Float64MultiArray__are_equal(
      &(lhs->dq_meas), &(rhs->dq_meas)))
  {
    return false;
  }
  // ddq_computed
  if (!std_msgs__msg__Float64MultiArray__are_equal(
      &(lhs->ddq_computed), &(rhs->ddq_computed)))
  {
    return false;
  }
  // error_endo
  if (!std_msgs__msg__Float64MultiArray__are_equal(
      &(lhs->error_endo), &(rhs->error_endo)))
  {
    return false;
  }
  // error_rcm
  if (!std_msgs__msg__Float64MultiArray__are_equal(
      &(lhs->error_rcm), &(rhs->error_rcm)))
  {
    return false;
  }
  return true;
}

bool
custom_msgs__msg__CustomVar__copy(
  const custom_msgs__msg__CustomVar * input,
  custom_msgs__msg__CustomVar * output)
{
  if (!input || !output) {
    return false;
  }
  // q_des
  if (!std_msgs__msg__Float64MultiArray__copy(
      &(input->q_des), &(output->q_des)))
  {
    return false;
  }
  // dq_des
  if (!std_msgs__msg__Float64MultiArray__copy(
      &(input->dq_des), &(output->dq_des)))
  {
    return false;
  }
  // q_meas
  if (!std_msgs__msg__Float64MultiArray__copy(
      &(input->q_meas), &(output->q_meas)))
  {
    return false;
  }
  // dq_meas
  if (!std_msgs__msg__Float64MultiArray__copy(
      &(input->dq_meas), &(output->dq_meas)))
  {
    return false;
  }
  // ddq_computed
  if (!std_msgs__msg__Float64MultiArray__copy(
      &(input->ddq_computed), &(output->ddq_computed)))
  {
    return false;
  }
  // error_endo
  if (!std_msgs__msg__Float64MultiArray__copy(
      &(input->error_endo), &(output->error_endo)))
  {
    return false;
  }
  // error_rcm
  if (!std_msgs__msg__Float64MultiArray__copy(
      &(input->error_rcm), &(output->error_rcm)))
  {
    return false;
  }
  return true;
}

custom_msgs__msg__CustomVar *
custom_msgs__msg__CustomVar__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  custom_msgs__msg__CustomVar * msg = (custom_msgs__msg__CustomVar *)allocator.allocate(sizeof(custom_msgs__msg__CustomVar), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(custom_msgs__msg__CustomVar));
  bool success = custom_msgs__msg__CustomVar__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
custom_msgs__msg__CustomVar__destroy(custom_msgs__msg__CustomVar * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    custom_msgs__msg__CustomVar__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
custom_msgs__msg__CustomVar__Sequence__init(custom_msgs__msg__CustomVar__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  custom_msgs__msg__CustomVar * data = NULL;

  if (size) {
    data = (custom_msgs__msg__CustomVar *)allocator.zero_allocate(size, sizeof(custom_msgs__msg__CustomVar), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = custom_msgs__msg__CustomVar__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        custom_msgs__msg__CustomVar__fini(&data[i - 1]);
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
custom_msgs__msg__CustomVar__Sequence__fini(custom_msgs__msg__CustomVar__Sequence * array)
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
      custom_msgs__msg__CustomVar__fini(&array->data[i]);
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

custom_msgs__msg__CustomVar__Sequence *
custom_msgs__msg__CustomVar__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  custom_msgs__msg__CustomVar__Sequence * array = (custom_msgs__msg__CustomVar__Sequence *)allocator.allocate(sizeof(custom_msgs__msg__CustomVar__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = custom_msgs__msg__CustomVar__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
custom_msgs__msg__CustomVar__Sequence__destroy(custom_msgs__msg__CustomVar__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    custom_msgs__msg__CustomVar__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
custom_msgs__msg__CustomVar__Sequence__are_equal(const custom_msgs__msg__CustomVar__Sequence * lhs, const custom_msgs__msg__CustomVar__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!custom_msgs__msg__CustomVar__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
custom_msgs__msg__CustomVar__Sequence__copy(
  const custom_msgs__msg__CustomVar__Sequence * input,
  custom_msgs__msg__CustomVar__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(custom_msgs__msg__CustomVar);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    custom_msgs__msg__CustomVar * data =
      (custom_msgs__msg__CustomVar *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!custom_msgs__msg__CustomVar__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          custom_msgs__msg__CustomVar__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!custom_msgs__msg__CustomVar__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
