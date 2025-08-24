// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from custom_msgs:msg/MotorMeasurement.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MSGS__MSG__DETAIL__MOTOR_MEASUREMENT__STRUCT_HPP_
#define CUSTOM_MSGS__MSG__DETAIL__MOTOR_MEASUREMENT__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__custom_msgs__msg__MotorMeasurement __attribute__((deprecated))
#else
# define DEPRECATED__custom_msgs__msg__MotorMeasurement __declspec(deprecated)
#endif

namespace custom_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct MotorMeasurement_
{
  using Type = MotorMeasurement_<ContainerAllocator>;

  explicit MotorMeasurement_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 4>::iterator, double>(this->vel.begin(), this->vel.end(), 0.0);
      std::fill<typename std::array<double, 4>::iterator, double>(this->pos.begin(), this->pos.end(), 0.0);
      std::fill<typename std::array<double, 4>::iterator, double>(this->torque.begin(), this->torque.end(), 0.0);
    }
  }

  explicit MotorMeasurement_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init),
    vel(_alloc),
    pos(_alloc),
    torque(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 4>::iterator, double>(this->vel.begin(), this->vel.end(), 0.0);
      std::fill<typename std::array<double, 4>::iterator, double>(this->pos.begin(), this->pos.end(), 0.0);
      std::fill<typename std::array<double, 4>::iterator, double>(this->torque.begin(), this->torque.end(), 0.0);
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _vel_type =
    std::array<double, 4>;
  _vel_type vel;
  using _pos_type =
    std::array<double, 4>;
  _pos_type pos;
  using _torque_type =
    std::array<double, 4>;
  _torque_type torque;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__vel(
    const std::array<double, 4> & _arg)
  {
    this->vel = _arg;
    return *this;
  }
  Type & set__pos(
    const std::array<double, 4> & _arg)
  {
    this->pos = _arg;
    return *this;
  }
  Type & set__torque(
    const std::array<double, 4> & _arg)
  {
    this->torque = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    custom_msgs::msg::MotorMeasurement_<ContainerAllocator> *;
  using ConstRawPtr =
    const custom_msgs::msg::MotorMeasurement_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<custom_msgs::msg::MotorMeasurement_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<custom_msgs::msg::MotorMeasurement_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      custom_msgs::msg::MotorMeasurement_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<custom_msgs::msg::MotorMeasurement_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      custom_msgs::msg::MotorMeasurement_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<custom_msgs::msg::MotorMeasurement_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<custom_msgs::msg::MotorMeasurement_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<custom_msgs::msg::MotorMeasurement_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__custom_msgs__msg__MotorMeasurement
    std::shared_ptr<custom_msgs::msg::MotorMeasurement_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__custom_msgs__msg__MotorMeasurement
    std::shared_ptr<custom_msgs::msg::MotorMeasurement_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const MotorMeasurement_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->vel != other.vel) {
      return false;
    }
    if (this->pos != other.pos) {
      return false;
    }
    if (this->torque != other.torque) {
      return false;
    }
    return true;
  }
  bool operator!=(const MotorMeasurement_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct MotorMeasurement_

// alias to use template instance with default allocator
using MotorMeasurement =
  custom_msgs::msg::MotorMeasurement_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace custom_msgs

#endif  // CUSTOM_MSGS__MSG__DETAIL__MOTOR_MEASUREMENT__STRUCT_HPP_
