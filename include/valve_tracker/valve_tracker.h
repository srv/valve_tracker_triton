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
#include <tf/transform_broadcaster.h>
#include "valve_tracker/stereo_processor.h"

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

  // Tracker parameters
  int closing_element_size_;
  int opening_element_size_;
  int binary_threshold_;
  int min_blob_size_;
  int max_blob_size_;
  int epipolar_width_threshold_;
  int mean_filter_size_;
  int min_value_threshold_;
  double max_tf_error_;
  std::string trained_model_path_;
  std::vector<cv::Point3d> valve_model_points_;
  cv::MatND trained_model_;
  bool show_debug_;

  tf::TransformBroadcaster tf_broadcaster_;         //!> Transform publisher
  tf::Transform camera_to_valve_;                   //!> Camera to valve transformation

  cv::Mat processed_;                               //!> Processed image
  image_geometry::StereoCameraModel stereo_model_;  //!> Camera model to compute the 3d world points

  void stereoImageCallback(
      const sensor_msgs::ImageConstPtr& l_image_msg,
      const sensor_msgs::ImageConstPtr& r_image_msg,
      const sensor_msgs::CameraInfoConstPtr& l_info_msg,
      const sensor_msgs::CameraInfoConstPtr& r_info_msg); //!> Image callback

  std::vector<cv::Point2d> valveDetection(
    cv::Mat img, bool debug);                             //!> Valve detection
  std::vector<cv::Point3d> triangulatePoints(
    std::vector< std::vector<cv::Point2d> > points_2d);   //!> Valve points triangulization
  bool estimateTransform(
    std::vector<cv::Point3d> points_3d,
    tf::Transform& output);                               //!> Transform estimation
  std::vector<cv::Point3d> matchTgtMdlPoints(
    std::vector<cv::Point3d> points_3d, bool inverse);    //!> Sort the target points
};

} // namespace

#endif // VALVE_TRACKER_H
