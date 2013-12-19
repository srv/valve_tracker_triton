cmake_minimum_required(VERSION 2.8.3)
project(valve_tracker_triton)
find_package(catkin REQUIRED COMPONENTS 
	roscpp
	sensor_msgs
	image_transport
	image_geometry
	cv_bridge
	tf
	std_srvs)

find_package(OpenCV REQUIRED)
find_package(Eigen REQUIRED)
find_package(Boost REQUIRED COMPONENTS signals)

catkin_package()

include_directories(${Boost_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS} ${Eigen_INCLUDE_DIRS} include)

add_executable(valve_tracker_node src/node_tracker.cpp src/tracker.cpp)
target_link_libraries(valve_tracker_node ${Boost_LIBRARIES} ${OpenCV_LIBS} ${catkin_LIBRARIES})

add_executable(valve_trainer_node src/node_trainer.cpp src/trainer.cpp)
target_link_libraries(valve_trainer_node ${Boost_LIBRARIES} ${OpenCV_LIBS} ${catkin_LIBRARIES})

add_executable(camera2camera src/camera2camera.cpp)
target_link_libraries(camera2camera ${Boost_LIBRARIES} ${OpenCV_LIBS} ${catkin_LIBRARIES})