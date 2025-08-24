// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from custom_msgs:msg/CustomVar.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MSGS__MSG__DETAIL__CUSTOM_VAR__STRUCT_HPP_
#define CUSTOM_MSGS__MSG__DETAIL__CUSTOM_VAR__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'q_des'
// Member 'dq_des'
// Member 'q_meas'
// Member 'dq_meas'
// Member 'ddq_computed'
// Member 'error_endo'
// Member 'error_rcm'
#include "std_msgs/msg/detail/float64_multi_array__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__custom_msgs__msg__CustomVar __attribute__((deprecated))
#else
# define DEPRECATED__custom_msgs__msg__CustomVar __declspec(deprecated)
#endif

namespace custom_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct CustomVar_
{
  using Type = CustomVar_<ContainerAllocator>;

  explicit CustomVar_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : q_des(_init),
    dq_des(_init),
    q_meas(_init),
    dq_meas(_init),
    ddq_computed(_init),
    error_endo(_init),
    error_rcm(_init)
  {
    (void)_init;
  }

  explicit CustomVar_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : q_des(_alloc, _init),
    dq_des(_alloc, _init),
    q_meas(_alloc, _init),
    dq_meas(_alloc, _init),
    ddq_computed(_alloc, _init),
    error_endo(_alloc, _init),
    error_rcm(_alloc, _init)
  {
    (void)_init;
  }

  // field types and members
  using _q_des_type =
    std_msgs::msg::Float64MultiArray_<ContainerAllocator>;
  _q_des_type q_des;
  using _dq_des_type =
    std_msgs::msg::Float64MultiArray_<ContainerAllocator>;
  _dq_des_type dq_des;
  using _q_meas_type =
    std_msgs::msg::Float64MultiArray_<ContainerAllocator>;
  _q_meas_type q_meas;
  using _dq_meas_type =
    std_msgs::msg::Float64MultiArray_<ContainerAllocator>;
  _dq_meas_type dq_meas;
  using _ddq_computed_type =
    std_msgs::msg::Float64MultiArray_<ContainerAllocator>;
  _ddq_computed_type ddq_computed;
  using _error_endo_type =
    std_msgs::msg::Float64MultiArray_<ContainerAllocator>;
  _error_endo_type error_endo;
  using _error_rcm_type =
    std_msgs::msg::Float64MultiArray_<ContainerAllocator>;
  _error_rcm_type error_rcm;

  // setters for named parameter idiom
  Type & set__q_des(
    const std_msgs::msg::Float64MultiArray_<ContainerAllocator> & _arg)
  {
    this->q_des = _arg;
    return *this;
  }
  Type & set__dq_des(
    const std_msgs::msg::Float64MultiArray_<ContainerAllocator> & _arg)
  {
    this->dq_des = _arg;
    return *this;
  }
  Type & set__q_meas(
    const std_msgs::msg::Float64MultiArray_<ContainerAllocator> & _arg)
  {
    this->q_meas = _arg;
    return *this;
  }
  Type & set__dq_meas(
    const std_msgs::msg::Float64MultiArray_<ContainerAllocator> & _arg)
  {
    this->dq_meas = _arg;
    return *this;
  }
  Type & set__ddq_computed(
    const std_msgs::msg::Float64MultiArray_<ContainerAllocator> & _arg)
  {
    this->ddq_computed = _arg;
    return *this;
  }
  Type & set__error_endo(
    const std_msgs::msg::Float64MultiArray_<ContainerAllocator> & _arg)
  {
    this->error_endo = _arg;
    return *this;
  }
  Type & set__error_rcm(
    const std_msgs::msg::Float64MultiArray_<ContainerAllocator> & _arg)
  {
    this->error_rcm = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    custom_msgs::msg::CustomVar_<ContainerAllocator> *;
  using ConstRawPtr =
    const custom_msgs::msg::CustomVar_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<custom_msgs::msg::CustomVar_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<custom_msgs::msg::CustomVar_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      custom_msgs::msg::CustomVar_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<custom_msgs::msg::CustomVar_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      custom_msgs::msg::CustomVar_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<custom_msgs::msg::CustomVar_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<custom_msgs::msg::CustomVar_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<custom_msgs::msg::CustomVar_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__custom_msgs__msg__CustomVar
    std::shared_ptr<custom_msgs::msg::CustomVar_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__custom_msgs__msg__CustomVar
    std::shared_ptr<custom_msgs::msg::CustomVar_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const CustomVar_ & other) const
  {
    if (this->q_des != other.q_des) {
      return false;
    }
    if (this->dq_des != other.dq_des) {
      return false;
    }
    if (this->q_meas != other.q_meas) {
      return false;
    }
    if (this->dq_meas != other.dq_meas) {
      return false;
    }
    if (this->ddq_computed != other.ddq_computed) {
      return false;
    }
    if (this->error_endo != other.error_endo) {
      return false;
    }
    if (this->error_rcm != other.error_rcm) {
      return false;
    }
    return true;
  }
  bool operator!=(const CustomVar_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct CustomVar_

// alias to use template instance with default allocator
using CustomVar =
  custom_msgs::msg::CustomVar_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace custom_msgs

#endif  // CUSTOM_MSGS__MSG__DETAIL__CUSTOM_VAR__STRUCT_HPP_
