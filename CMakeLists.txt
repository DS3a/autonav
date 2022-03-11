cmake_minimum_required(VERSION 3.0.2)
project(autonav)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_python_setup()
catkin_package()
catkin_install_python(PROGRAMS
#src/lane_detection_lib/lane_detection_lib.py
scripts/lane_follow.py scripts/lane_follow.py
scripts/lane_mask_publisher.py scripts/lane_mask_publisher.py
DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
