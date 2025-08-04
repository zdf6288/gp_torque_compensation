cmake_minimum_required(VERSION 3.5)
project(endowrist_control)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    message(STATUS "Using ccache")
  set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
  set(CMAKE_C_COMPILER_LAUNCHER   ${CCACHE_PROGRAM})
endif()

set(CMAKE_CXX_STANDARD 17)
add_definitions(-Wall )#-Werror)
add_definitions(-DMELO_USE_COUT)


find_package(yaml-cpp REQUIRED)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(custom_msgs)

find_package(maxon_epos_ethercat_sdk)
add_definitions(-D_MAXON_FOUND_)

include_directories(
  include
  ${maxon_epos_ethercat_sdk_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME} SHARED
  src/EthercatDeviceConfigurator.cpp
)

target_link_libraries(
  ${PROJECT_NAME}
  # ${maxon_epos_ethercat_sdk_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
  stdc++fs
)

ament_target_dependencies(${PROJECT_NAME}
  maxon_epos_ethercat_sdk
)

# message("maxon_epos_ethercat_sdk_lib_path is ${maxon_epos_ethercat_sdk_prefix}")
# message("CMAKE_INSTALL_PREFIX is ${CMAKE_INSTALL_PREFIX}")
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${CMAKE_INSTALL_RPATH}")

add_executable(
  endowrist_control_node
  src/endowrist_control_node.cpp
)

set_target_properties(endowrist_control_node PROPERTIES
  INSTALL_RPATH_USE_LINK_PATH TRUE
)

add_dependencies(
  endowrist_control_node
  ${PROJECT_NAME}
  ${maxon_epos_ethercat_sdk_EXPORTED_TARGETS}
)
target_link_libraries(
  endowrist_control_node
  ${PROJECT_NAME}
  ${maxon_epos_ethercat_sdk_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
  -pthread
  stdc++fs
)

ament_target_dependencies(endowrist_control_node rclcpp std_msgs geometry_msgs custom_msgs)

install(TARGETS ${PROJECT_NAME} 
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS endowrist_control_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/${PROJECT_NAME}
  DESTINATION include
)

install(DIRECTORY launch example_config
  DESTINATION share/${PROJECT_NAME}
)

ament_export_libraries(${PROJECT_NAME})

ament_export_include_directories( 
  include
)
ament_export_dependencies(rclcpp std_msgs geometry_msgs custom_msgs)

ament_package()


