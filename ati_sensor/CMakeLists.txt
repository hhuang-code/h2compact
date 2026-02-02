cmake_minimum_required(VERSION 3.11)
project(ati_sensor)

set(JSD_PATH "/opt/jsd")

find_package(catkin REQUIRED COMPONENTS roscpp geometry_msgs)
catkin_package(CATKIN_DEPENDS roscpp geometry_msgs)

include_directories(include 
		    ${JSD_PATH}/include
		    ${JSD_PATH}/include/soem
		    ${catkin_INCLUDE_DIRS})

link_directories(${JSD_PATH}/lib)

add_executable(ati_sensor src/ati_sensor.cpp src/jsd_test_utils.cpp)
target_link_libraries(ati_sensor -ljsd-lib -lsoem -lpthread)

add_executable(ati_ros src/ati_ros.cpp)
target_link_libraries(ati_ros -ljsd-lib -lsoem -lpthread ${catkin_LIBRARIES})