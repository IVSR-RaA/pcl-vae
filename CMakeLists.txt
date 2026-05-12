cmake_minimum_required(VERSION 3.0.2)
project(pcl_vae)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)

find_package(catkin REQUIRED COMPONENTS
  basic
  cv_bridge
  geometry_msgs
  pcl_conversions
  pcl_ros
  roscpp
  rospy
  sensor_msgs
  std_msgs
  super_lio
  tf2
  tf2_ros
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS basic cv_bridge geometry_msgs pcl_conversions pcl_ros roscpp rospy sensor_msgs std_msgs super_lio tf2 tf2_ros
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

add_executable(range_image_to_octvox_node
  src/range_image_to_octvox_node.cpp
)
add_dependencies(range_image_to_octvox_node
  ${catkin_EXPORTED_TARGETS}
)
target_compile_definitions(range_image_to_octvox_node PRIVATE
  ROOT="${CMAKE_CURRENT_SOURCE_DIR}/"
)
target_link_libraries(range_image_to_octvox_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  tbb
)

catkin_install_python(PROGRAMS
  scripts/vae_encoder_node.py
  scripts/vae_decoder_node.py
  scripts/point_to_range_image_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS range_image_to_octvox_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY pcl_vae/inference/config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/pcl_vae/inference
)

install(DIRECTORY pcl_vae/train/config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/pcl_vae/train
)

install(DIRECTORY pcl_vae/weights
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/pcl_vae
)
