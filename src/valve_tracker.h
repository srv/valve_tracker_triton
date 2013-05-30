#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <image_geometry/stereo_camera_model.h>
#include <image_geometry/pinhole_camera_model.h>
#include <image_transport/image_transport.h>
#include <image_transport/camera_subscriber.h>
#include <tf/transform_listener.h>

#include "stereo_processor.h"

class ValveTracker : public StereoImageProcessor
{
public:


  ValveTracker(const std::string transport);
  bool process(cv::Mat left, cv::Mat right);

protected:
  std::string stereo_frame_id_;
  std::string base_link_frame_id_;
  image_transport::Publisher image_pub_;

private:

  cv::Mat left_;
  cv::Mat right_;

  cv::Mat processed_;

  image_geometry::StereoCameraModel stereo_model_;
  image_geometry::PinholeCameraModel projector_model_;

  void stereoImageCallback(
      const sensor_msgs::ImageConstPtr& l_image_msg,
      const sensor_msgs::ImageConstPtr& r_image_msg,
      const sensor_msgs::CameraInfoConstPtr& l_info_msg,
      const sensor_msgs::CameraInfoConstPtr& r_info_msg);

  boost::shared_ptr<ValveTracker> vt_;
};
