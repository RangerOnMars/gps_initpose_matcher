cmake_minimum_required(VERSION 3.0.2)
project(gps_initpose_matcher)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
)

catkin_package(
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
catkin_install_python(PROGRAMS
  src/map_utm_matcher.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)