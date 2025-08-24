// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from custom_msgs:msg/StateParameter.idl
// generated code does not contain a copyright notice

#ifndef CUSTOM_MSGS__MSG__DETAIL__STATE_PARAMETER__STRUCT_HPP_
#define CUSTOM_MSGS__MSG__DETAIL__STATE_PARAMETER__STRUCT_HPP_

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
# define DEPRECATED__custom_msgs__msg__StateParameter __attribute__((deprecated))
#else
# define DEPRECATED__custom_msgs__msg__StateParameter __declspec(deprecated)
#endif

namespace custom_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct StateParameter_
{
  using Type = StateParameter_<ContainerAllocator>;

  explicit StateParameter_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 7>::iterator, double>(this->position.begin(), this->position.end(), 0.0);
      std::fill<typename std::array<double, 7>::iterator, double>(this->velocity.begin(), this->velocity.end(), 0.0);
      std::fill<typename std::array<double, 7>::iterator, double>(this->effort_measured.begin(), this->effort_measured.end(), 0.0);
      std::fill<typename std::array<double, 7>::iterator, double>(this->gravity.begin(), this->gravity.end(), 0.0);
      std::fill<typename std::array<double, 16>::iterator, double>(this->o_t_f.begin(), this->o_t_f.end(), 0.0);
      std::fill<typename std::array<double, 49>::iterator, double>(this->mass.begin(), this->mass.end(), 0.0);
      std::fill<typename std::array<double, 7>::iterator, double>(this->coriolis.begin(), this->coriolis.end(), 0.0);
      std::fill<typename std::array<double, 42>::iterator, double>(this->zero_jacobian_flange.begin(), this->zero_jacobian_flange.end(), 0.0);
    }
  }

  explicit StateParameter_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init),
    position(_alloc),
    velocity(_alloc),
    effort_measured(_alloc),
    gravity(_alloc),
    o_t_f(_alloc),
    mass(_alloc),
    coriolis(_alloc),
    zero_jacobian_flange(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 7>::iterator, double>(this->position.begin(), this->position.end(), 0.0);
      std::fill<typename std::array<double, 7>::iterator, double>(this->velocity.begin(), this->velocity.end(), 0.0);
      std::fill<typename std::array<double, 7>::iterator, double>(this->effort_measured.begin(), this->effort_measured.end(), 0.0);
      std::fill<typename std::array<double, 7>::iterator, double>(this->gravity.begin(), this->gravity.end(), 0.0);
      std::fill<typename std::array<double, 16>::iterator, double>(this->o_t_f.begin(), this->o_t_f.end(), 0.0);
      std::fill<typename std::array<double, 49>::iterator, double>(this->mass.begin(), this->mass.end(), 0.0);
      std::fill<typename std::array<double, 7>::iterator, double>(this->coriolis.begin(), this->coriolis.end(), 0.0);
      std::fill<typename std::array<double, 42>::iterator, double>(this->zero_jacobian_flange.begin(), this->zero_jacobian_flange.end(), 0.0);
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _position_type =
    std::array<double, 7>;
  _position_type position;
  using _velocity_type =
    std::array<double, 7>;
  _velocity_type velocity;
  using _effort_measured_type =
    std::array<double, 7>;
  _effort_measured_type effort_measured;
  using _gravity_type =
    std::array<double, 7>;
  _gravity_type gravity;
  using _o_t_f_type =
    std::array<double, 16>;
  _o_t_f_type o_t_f;
  using _mass_type =
    std::array<double, 49>;
  _mass_type mass;
  using _coriolis_type =
    std::array<double, 7>;
  _coriolis_type coriolis;
  using _zero_jacobian_flange_type =
    std::array<double, 42>;
  _zero_jacobian_flange_type zero_jacobian_flange;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__position(
    const std::array<double, 7> & _arg)
  {
    this->position = _arg;
    return *this;
  }
  Type & set__velocity(
    const std::array<double, 7> & _arg)
  {
    this->velocity = _arg;
    return *this;
  }
  Type & set__effort_measured(
    const std::array<double, 7> & _arg)
  {
    this->effort_measured = _arg;
    return *this;
  }
  Type & set__gravity(
    const std::array<double, 7> & _arg)
  {
    this->gravity = _arg;
    return *this;
  }
  Type & set__o_t_f(
    const std::array<double, 16> & _arg)
  {
    this->o_t_f = _arg;
    return *this;
  }
  Type & set__mass(
    const std::array<double, 49> & _arg)
  {
    this->mass = _arg;
    return *this;
  }
  Type & set__coriolis(
    const std::array<double, 7> & _arg)
  {
    this->coriolis = _arg;
    return *this;
  }
  Type & set__zero_jacobian_flange(
    const std::array<double, 42> & _arg)
  {
    this->zero_jacobian_flange = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    custom_msgs::msg::StateParameter_<ContainerAllocator> *;
  using ConstRawPtr =
    const custom_msgs::msg::StateParameter_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<custom_msgs::msg::StateParameter_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<custom_msgs::msg::StateParameter_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      custom_msgs::msg::StateParameter_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<custom_msgs::msg::StateParameter_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      custom_msgs::msg::StateParameter_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<custom_msgs::msg::StateParameter_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<custom_msgs::msg::StateParameter_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<custom_msgs::msg::StateParameter_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__custom_msgs__msg__StateParameter
    std::shared_ptr<custom_msgs::msg::StateParameter_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__custom_msgs__msg__StateParameter
    std::shared_ptr<custom_msgs::msg::StateParameter_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const StateParameter_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->position != other.position) {
      return false;
    }
    if (this->velocity != other.velocity) {
      return false;
    }
    if (this->effort_measured != other.effort_measured) {
      return false;
    }
    if (this->gravity != other.gravity) {
      return false;
    }
    if (this->o_t_f != other.o_t_f) {
      return false;
    }
    if (this->mass != other.mass) {
      return false;
    }
    if (this->coriolis != other.coriolis) {
      return false;
    }
    if (this->zero_jacobian_flange != other.zero_jacobian_flange) {
      return false;
    }
    return true;
  }
  bool operator!=(const StateParameter_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct StateParameter_

// alias to use template instance with default allocator
using StateParameter =
  custom_msgs::msg::StateParameter_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace custom_msgs

#endif  // CUSTOM_MSGS__MSG__DETAIL__STATE_PARAMETER__STRUCT_HPP_
