// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from custom_msgs:srv/JointPositionAdjust.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MSGS__SRV__DETAIL__JOINT_POSITION_ADJUST__STRUCT_HPP_
#define CUSTOM_MSGS__SRV__DETAIL__JOINT_POSITION_ADJUST__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__custom_msgs__srv__JointPositionAdjust_Request __attribute__((deprecated))
#else
# define DEPRECATED__custom_msgs__srv__JointPositionAdjust_Request __declspec(deprecated)
#endif

namespace custom_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct JointPositionAdjust_Request_
{
  using Type = JointPositionAdjust_Request_<ContainerAllocator>;

  explicit JointPositionAdjust_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_init;
  }

  explicit JointPositionAdjust_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_init;
    (void)_alloc;
  }

  // field types and members
  using _q_des_type =
    std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>>;
  _q_des_type q_des;
  using _dq_des_type =
    std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>>;
  _dq_des_type dq_des;

  // setters for named parameter idiom
  Type & set__q_des(
    const std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> & _arg)
  {
    this->q_des = _arg;
    return *this;
  }
  Type & set__dq_des(
    const std::vector<double, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<double>> & _arg)
  {
    this->dq_des = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    custom_msgs::srv::JointPositionAdjust_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const custom_msgs::srv::JointPositionAdjust_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<custom_msgs::srv::JointPositionAdjust_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<custom_msgs::srv::JointPositionAdjust_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      custom_msgs::srv::JointPositionAdjust_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<custom_msgs::srv::JointPositionAdjust_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      custom_msgs::srv::JointPositionAdjust_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<custom_msgs::srv::JointPositionAdjust_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<custom_msgs::srv::JointPositionAdjust_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<custom_msgs::srv::JointPositionAdjust_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__custom_msgs__srv__JointPositionAdjust_Request
    std::shared_ptr<custom_msgs::srv::JointPositionAdjust_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__custom_msgs__srv__JointPositionAdjust_Request
    std::shared_ptr<custom_msgs::srv::JointPositionAdjust_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const JointPositionAdjust_Request_ & other) const
  {
    if (this->q_des != other.q_des) {
      return false;
    }
    if (this->dq_des != other.dq_des) {
      return false;
    }
    return true;
  }
  bool operator!=(const JointPositionAdjust_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct JointPositionAdjust_Request_

// alias to use template instance with default allocator
using JointPositionAdjust_Request =
  custom_msgs::srv::JointPositionAdjust_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace custom_msgs


#ifndef _WIN32
# define DEPRECATED__custom_msgs__srv__JointPositionAdjust_Response __attribute__((deprecated))
#else
# define DEPRECATED__custom_msgs__srv__JointPositionAdjust_Response __declspec(deprecated)
#endif

namespace custom_msgs
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct JointPositionAdjust_Response_
{
  using Type = JointPositionAdjust_Response_<ContainerAllocator>;

  explicit JointPositionAdjust_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
      this->message = "";
    }
  }

  explicit JointPositionAdjust_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : message(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
      this->message = "";
    }
  }

  // field types and members
  using _success_type =
    bool;
  _success_type success;
  using _message_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _message_type message;

  // setters for named parameter idiom
  Type & set__success(
    const bool & _arg)
  {
    this->success = _arg;
    return *this;
  }
  Type & set__message(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->message = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    custom_msgs::srv::JointPositionAdjust_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const custom_msgs::srv::JointPositionAdjust_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<custom_msgs::srv::JointPositionAdjust_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<custom_msgs::srv::JointPositionAdjust_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      custom_msgs::srv::JointPositionAdjust_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<custom_msgs::srv::JointPositionAdjust_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      custom_msgs::srv::JointPositionAdjust_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<custom_msgs::srv::JointPositionAdjust_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<custom_msgs::srv::JointPositionAdjust_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<custom_msgs::srv::JointPositionAdjust_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__custom_msgs__srv__JointPositionAdjust_Response
    std::shared_ptr<custom_msgs::srv::JointPositionAdjust_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__custom_msgs__srv__JointPositionAdjust_Response
    std::shared_ptr<custom_msgs::srv::JointPositionAdjust_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const JointPositionAdjust_Response_ & other) const
  {
    if (this->success != other.success) {
      return false;
    }
    if (this->message != other.message) {
      return false;
    }
    return true;
  }
  bool operator!=(const JointPositionAdjust_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct JointPositionAdjust_Response_

// alias to use template instance with default allocator
using JointPositionAdjust_Response =
  custom_msgs::srv::JointPositionAdjust_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace custom_msgs

namespace custom_msgs
{

namespace srv
{

struct JointPositionAdjust
{
  using Request = custom_msgs::srv::JointPositionAdjust_Request;
  using Response = custom_msgs::srv::JointPositionAdjust_Response;
};

}  // namespace srv

}  // namespace custom_msgs

#endif  // CUSTOM_MSGS__SRV__DETAIL__JOINT_POSITION_ADJUST__STRUCT_HPP_
