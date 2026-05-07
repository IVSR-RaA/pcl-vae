cmake_minimum_required(VERSION 3.0.2)
project(pcl_vae)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  rospy
  sensor_msgs
  std_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS cv_bridge rospy sensor_msgs std_msgs
)

catkin_install_python(PROGRAMS
  scripts/vae_encoder_node.py
  scripts/vae_decoder_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
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
