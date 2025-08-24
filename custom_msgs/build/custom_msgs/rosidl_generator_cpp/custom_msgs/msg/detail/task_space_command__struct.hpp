// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from custom_msgs:msg/TaskSpaceCommand.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MSGS__MSG__DETAIL__TASK_SPACE_COMMAND__STRUCT_HPP_
#define CUSTOM_MSGS__MSG__DETAIL__TASK_SPACE_COMMAND__STRUCT_HPP_

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
# define DEPRECATED__custom_msgs__msg__TaskSpaceCommand __attribute__((deprecated))
#else
# define DEPRECATED__custom_msgs__msg__TaskSpaceCommand __declspec(deprecated)
#endif

namespace custom_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct TaskSpaceCommand_
{
  using Type = TaskSpaceCommand_<ContainerAllocator>;

  explicit TaskSpaceCommand_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 6>::iterator, double>(this->x_des.begin(), this->x_des.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->dx_des.begin(), this->dx_des.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->ddx_des.begin(), this->ddx_des.end(), 0.0);
    }
  }

  explicit TaskSpaceCommand_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init),
    x_des(_alloc),
    dx_des(_alloc),
    ddx_des(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 6>::iterator, double>(this->x_des.begin(), this->x_des.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->dx_des.begin(), this->dx_des.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->ddx_des.begin(), this->ddx_des.end(), 0.0);
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _x_des_type =
    std::array<double, 6>;
  _x_des_type x_des;
  using _dx_des_type =
    std::array<double, 6>;
  _dx_des_type dx_des;
  using _ddx_des_type =
    std::array<double, 6>;
  _ddx_des_type ddx_des;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__x_des(
    const std::array<double, 6> & _arg)
  {
    this->x_des = _arg;
    return *this;
  }
  Type & set__dx_des(
    const std::array<double, 6> & _arg)
  {
    this->dx_des = _arg;
    return *this;
  }
  Type & set__ddx_des(
    const std::array<double, 6> & _arg)
  {
    this->ddx_des = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    custom_msgs::msg::TaskSpaceCommand_<ContainerAllocator> *;
  using ConstRawPtr =
    const custom_msgs::msg::TaskSpaceCommand_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<custom_msgs::msg::TaskSpaceCommand_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<custom_msgs::msg::TaskSpaceCommand_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      custom_msgs::msg::TaskSpaceCommand_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<custom_msgs::msg::TaskSpaceCommand_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      custom_msgs::msg::TaskSpaceCommand_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<custom_msgs::msg::TaskSpaceCommand_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<custom_msgs::msg::TaskSpaceCommand_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<custom_msgs::msg::TaskSpaceCommand_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__custom_msgs__msg__TaskSpaceCommand
    std::shared_ptr<custom_msgs::msg::TaskSpaceCommand_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__custom_msgs__msg__TaskSpaceCommand
    std::shared_ptr<custom_msgs::msg::TaskSpaceCommand_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const TaskSpaceCommand_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->x_des != other.x_des) {
      return false;
    }
    if (this->dx_des != other.dx_des) {
      return false;
    }
    if (this->ddx_des != other.ddx_des) {
      return false;
    }
    return true;
  }
  bool operator!=(const TaskSpaceCommand_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct TaskSpaceCommand_

// alias to use template instance with default allocator
using TaskSpaceCommand =
  custom_msgs::msg::TaskSpaceCommand_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace custom_msgs

#endif  // CUSTOM_MSGS__MSG__DETAIL__TASK_SPACE_COMMAND__STRUCT_HPP_
