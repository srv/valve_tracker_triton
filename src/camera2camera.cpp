#include "ros/ros.h"
#include "tf/transform_listener.h"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "cam_to_cam");
  ros::NodeHandle nh;

  tf::TransformListener listener;

  while (nh.ok()){
  	tf::StampedTransform forward_to_marker,down_to_marker;
  	tf::Transform down_to_forward;
    try{
      listener.lookupTransform("/stereo_forward_optical", "/marker_2",  
                               ros::Time(0), forward_to_marker);
    }
    catch (tf::TransformException ex){
      ROS_ERROR("%s",ex.what());
    }

    try{
      listener.lookupTransform("/stereo_down_optical", "/marker_1",  
                               ros::Time(0), down_to_marker);
    }
    catch (tf::TransformException ex){
      ROS_ERROR("%s",ex.what());
    }

    down_to_forward = down_to_marker * forward_to_marker.inverse();

    double x, y, z, roll, pitch, yaw;
	  down_to_forward.getBasis().getRPY(roll, pitch, yaw);
	  x = down_to_forward.getOrigin().x();
	  y = down_to_forward.getOrigin().y();
	  z = down_to_forward.getOrigin().z();
	  ROS_INFO_STREAM("<node pkg=\"tf\" type=\"static_transform_publisher\"" <<
                    " name=\"down_to_forward\" args=\"" <<
	                  " [" << x << " " << y << " " << z << 
	                  " " << yaw << " " << pitch << " " << roll << 
                    "stereo_down_optical stereo_forward_optical 100\" />");



    ros::spinOnce();
  }

  
  return 0;
}
