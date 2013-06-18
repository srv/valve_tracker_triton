#include "valve_tracker.h"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "valve_tracker_node");
  ros::NodeHandle nh("~");
  if (ros::names::remap("stereo") == "stereo") 
  {
    ROS_WARN("'stereo' has not been remapped! "
             "Example command-line usage:\n"
             "\t$ rosrun valve_tracker_triton valve_tracker_node"
             "stereo:=/stereo_down image:=image_rect");
  }

  if (ros::names::remap("image").find("rect") == std::string::npos) {
    ROS_WARN("This valve tracker needs rectified input images. "
             "The used image topic is '%s'. Are you sure the images are "
             "rectified?",
        ros::names::remap("image").c_str());
  }

  std::string transport = argc > 1 ? argv[1] : "raw";
  valve_tracker::ValveTracker vt_node(transport);
  ros::spin();
  return 0;
}

