/**
 * @file
 * @brief Valve tracker presentation.
 */

#ifndef VALVE_TRACKER_H
#define VALVE_TRACKER_H

#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <image_geometry/stereo_camera_model.h>
#include <image_geometry/pinhole_camera_model.h>
#include <image_transport/image_transport.h>
#include <image_transport/camera_subscriber.h>
#include <tf/transform_listener.h>
#include "valve_tracker_triton/stereo_processor.h"

namespace valve_tracker
{

class ValveTracker : public StereoImageProcessor
{

public:
  // Constructor
  ValveTracker(const std::string transport);

private:

  // Node parameters
  std::string stereo_frame_id_;
  std::string valve_frame_id_;
  image_transport::Publisher image_pub_;
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
  int epipolar_width_threshold_;
  int show_debug_images_;

  cv::Mat processed_;                               //!> Processed image
  image_geometry::StereoCameraModel stereo_model_;  //!> Camera model to compute the 3d world points

  std::vector< std::vector<cv::Point2d> > points_;  //!> 2D points
  std::vector<cv::Point3d> points3d_;               //!> 3D world points

  void stereoImageCallback(
      const sensor_msgs::ImageConstPtr& l_image_msg,
      const sensor_msgs::ImageConstPtr& r_image_msg,
      const sensor_msgs::CameraInfoConstPtr& l_info_msg,
      const sensor_msgs::CameraInfoConstPtr& r_info_msg); //!> Image callback

  void valveDetection(cv::Mat img, int type);             //!> Valve detection
  void triangulatePoints();                               //!> Valve points triangulization
  tf::Transform estimateTransform();                      //!> Transform estimation
  
};

} // namespace

#endif // VALVE_TRACKER_H
