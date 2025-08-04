# Copyright (2019-2020) Robotics Systems Lab - ETH Zurich:
# Markus Staeuble, Jonas Junger, Johannes Pankert, Philipp Leemann,
# Tom Lankhorst, Samuel Bachmann, Gabriel Hottiger, Lennert Nachtigall,
# Mario Mauerer, Remo Diethelm
#
# This file is part of the soem_interface.
#
# The soem_interface is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# The seom_interface is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with the soem_interface.  If not, see <https://www.gnu.org/licenses/>.

cmake_minimum_required(VERSION 3.5)
project(soem_interface)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(PACKAGE_DEPENDENCIES
  message_logger
  soem
)

find_package(message_logger REQUIRED)
find_package(soem REQUIRED)
find_package(ament_cmake REQUIRED
  COMPONENTS
    ${PACKAGE_DEPENDENCIES}
)

include_directories(
  include
)

add_library(
  ${PROJECT_NAME} SHARED
  src/${PROJECT_NAME}/common/ThreadSleep.cpp
  src/${PROJECT_NAME}/common/Macros.cpp
  src/${PROJECT_NAME}/EthercatSlaveBase.cpp
  src/${PROJECT_NAME}/EthercatBusManagerBase.cpp
  src/${PROJECT_NAME}/EthercatBusBase.cpp
)
# add_dependencies(
#   ${PROJECT_NAME}
#   ${catkin_EXPORTED_TARGETS}
# )
# target_link_libraries(
#   ${PROJECT_NAME}
#   message_logger soem
# )

ament_target_dependencies(${PROJECT_NAME}
  message_logger
  soem)

set_target_properties(${PROJECT_NAME} PROPERTIES
  INSTALL_RPATH_USE_LINK_PATH TRUE
)
install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/${PROJECT_NAME}
  DESTINATION include
)

ament_export_include_directories( 
  include)
ament_export_libraries(${PROJECT_NAME})
ament_export_dependencies(
        message_logger
        soem
)
ament_package()