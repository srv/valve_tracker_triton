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
  bool process(cv::Mat img);

protected:
  std::string stereo_frame_id_;
  std::string base_link_frame_id_;
  image_transport::Publisher image_pub_;

private:

  cv::Mat left_;
  cv::Mat right_;

  cv::Mat processed_;

  image_geometry::StereoCameraModel stereo_model_;

  int threshold_h_low_;
  int threshold_h_hi_;
  int threshold_s_low_;
  int threshold_s_hi_;
  int threshold_v_low_;
  int threshold_v_hi_;

  int closing_element_size_;
  int opening_element_size_;

  int canny_first_threshold_;
  int canny_second_threshold_;

  void stereoImageCallback(
      const sensor_msgs::ImageConstPtr& l_image_msg,
      const sensor_msgs::ImageConstPtr& r_image_msg,
      const sensor_msgs::CameraInfoConstPtr& l_info_msg,
      const sensor_msgs::CameraInfoConstPtr& r_info_msg);

  cv::Mat createElement(int element_size);

  boost::shared_ptr<ValveTracker> vt_;
};
