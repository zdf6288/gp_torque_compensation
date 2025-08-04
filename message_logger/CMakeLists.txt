cmake_minimum_required(VERSION 3.5)
project(message_logger)

# Standard CMake modules
include(CMakeDependentOption)
cmake_dependent_option(MELO_USE_COUT "Use std::cout" OFF
                       "DEFINED ENV{ROS_DISTRO}" ON)

# Use C++11
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find ament_cmake and any ROS 2 dependencies
find_package(ament_cmake REQUIRED)
if(NOT MELO_USE_COUT)
  find_package(rclcpp REQUIRED)
endif()

# Configure header file
message("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! message_logger !!!!!!!!!")
message("CMAKE_CURRENT_BINARY_DIR is ${CMAKE_CURRENT_BINARY_DIR}")
configure_file(
  include/message_logger/log/log_messages_backend_config.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/message_logger/log/log_messages_backend_config.hpp @ONLY
)
include_directories(include ${CMAKE_CURRENT_BINARY_DIR}/include)

# Compiler definitions
if(DEFINED MELO_FUNCTION_PRINTS)
  add_definitions(-DMELO_FUNCTION_PRINTS)
endif()

# Add library
add_library(${PROJECT_NAME} SHARED
  src/time/Time.cpp
  src/time/TimeStd.cpp
)

# Link libraries
if(NOT MELO_USE_COUT)
  target_link_libraries(${PROJECT_NAME} rclcpp::rclcpp)
endif()

# Install library and headers
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION include/${PROJECT_NAME}
  PATTERN "*.in" EXCLUDE
)
install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/include/message_logger/log/log_messages_backend_config.hpp"
  DESTINATION include/message_logger/log
)

install(TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Export information
ament_export_include_directories(
        include
)
ament_export_libraries(
        ${PROJECT_NAME}
)
if(NOT MELO_USE_COUT)
  ament_export_dependencies(rclcpp)
endif()

# Tests
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  # Add tests here
  # ament_add_gtest(test_${PROJECT_NAME} test/EmptyTests.cpp)
  # target_link_libraries(test_${PROJECT_NAME} ${PROJECT_NAME})
endif()

# Call ament_package at the end
ament_package(CONFIG_EXTRAS cmake/message_logger.cmake)