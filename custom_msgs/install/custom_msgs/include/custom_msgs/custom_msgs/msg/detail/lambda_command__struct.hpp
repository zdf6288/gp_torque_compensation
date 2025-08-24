// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from custom_msgs:msg/LambdaCommand.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MSGS__MSG__DETAIL__LAMBDA_COMMAND__STRUCT_HPP_
#define CUSTOM_MSGS__MSG__DETAIL__LAMBDA_COMMAND__STRUCT_HPP_

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
// Member 'linear'
// Member 'angular'
#include "geometry_msgs/msg/detail/vector3__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__custom_msgs__msg__LambdaCommand __attribute__((deprecated))
#else
# define DEPRECATED__custom_msgs__msg__LambdaCommand __declspec(deprecated)
#endif

namespace custom_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct LambdaCommand_
{
  using Type = LambdaCommand_<ContainerAllocator>;

  explicit LambdaCommand_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init),
    linear(_init),
    angular(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->v_gripper = 0.0;
      this->enable_backlash_compensation = false;
    }
  }

  explicit LambdaCommand_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init),
    linear(_alloc, _init),
    angular(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->v_gripper = 0.0;
      this->enable_backlash_compensation = false;
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _linear_type =
    geometry_msgs::msg::Vector3_<ContainerAllocator>;
  _linear_type linear;
  using _angular_type =
    geometry_msgs::msg::Vector3_<ContainerAllocator>;
  _angular_type angular;
  using _v_gripper_type =
    double;
  _v_gripper_type v_gripper;
  using _enable_backlash_compensation_type =
    bool;
  _enable_backlash_compensation_type enable_backlash_compensation;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__linear(
    const geometry_msgs::msg::Vector3_<ContainerAllocator> & _arg)
  {
    this->linear = _arg;
    return *this;
  }
  Type & set__angular(
    const geometry_msgs::msg::Vector3_<ContainerAllocator> & _arg)
  {
    this->angular = _arg;
    return *this;
  }
  Type & set__v_gripper(
    const double & _arg)
  {
    this->v_gripper = _arg;
    return *this;
  }
  Type & set__enable_backlash_compensation(
    const bool & _arg)
  {
    this->enable_backlash_compensation = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    custom_msgs::msg::LambdaCommand_<ContainerAllocator> *;
  using ConstRawPtr =
    const custom_msgs::msg::LambdaCommand_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<custom_msgs::msg::LambdaCommand_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<custom_msgs::msg::LambdaCommand_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      custom_msgs::msg::LambdaCommand_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<custom_msgs::msg::LambdaCommand_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      custom_msgs::msg::LambdaCommand_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<custom_msgs::msg::LambdaCommand_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<custom_msgs::msg::LambdaCommand_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<custom_msgs::msg::LambdaCommand_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__custom_msgs__msg__LambdaCommand
    std::shared_ptr<custom_msgs::msg::LambdaCommand_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__custom_msgs__msg__LambdaCommand
    std::shared_ptr<custom_msgs::msg::LambdaCommand_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const LambdaCommand_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->linear != other.linear) {
      return false;
    }
    if (this->angular != other.angular) {
      return false;
    }
    if (this->v_gripper != other.v_gripper) {
      return false;
    }
    if (this->enable_backlash_compensation != other.enable_backlash_compensation) {
      return false;
    }
    return true;
  }
  bool operator!=(const LambdaCommand_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct LambdaCommand_

// alias to use template instance with default allocator
using LambdaCommand =
  custom_msgs::msg::LambdaCommand_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace custom_msgs

#endif  // CUSTOM_MSGS__MSG__DETAIL__LAMBDA_COMMAND__STRUCT_HPP_
