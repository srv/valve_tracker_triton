/**
 * @file
 * @brief Valve tracker presentation.
 */

#ifndef TRACKER_H
#define TRACKER_H

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

class Tracker : public StereoImageProcessor
{

public:
  // Constructors
  Tracker(cv::MatND trained_model, 
          std::vector<cv::Point3d> valve_model,
          image_geometry::StereoCameraModel stereo_model, 
          int epipolar_width_threshold);
  Tracker(const std::string transport);

  // Public functions
  std::vector<cv::Point2d> valveDetection(
    cv::Mat image, bool debug);                           //!> Valve detection
  std::vector<cv::Point2d> valveDetection(
    cv::Mat image, bool debug, 
    std::vector<int> &contours_size);                     //!> Valve detection
  std::vector<cv::Point3d> triangulatePoints(
    std::vector<cv::Point2d> l_points_2d, 
    std::vector<cv::Point2d> r_points_2d,
    std::vector<int> l_contours_size,
    std::vector<int> r_contours_size);                    //!> Valve points triangulization
  bool estimateTransform(
    std::vector<cv::Point3d> points_3d,
    tf::Transform& output,
    double &error);                                       //!> Transform estimation
  void showParameterSet();                                //!> Logs out the parameter set

  // Access specifiers
  void setParameter(std::string param_name, int param_value);

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
  double max_rot_diff_;
  double max_trans_diff_;
  std::string trained_model_path_;
  std::string tuning_gui_name_;
  std::vector<cv::Point3d> valve_model_points_;
  cv::MatND trained_model_;
  bool show_debug_;
  bool warning_on_;

  // TF filter
  int tf_filter_size_;
  std::vector<double> tf_x_, tf_y_, tf_z_;
  std::vector<double> tf_roll_, tf_pitch_, tf_yaw_;

  // TF
  tf::TransformBroadcaster tf_broadcaster_;         //!> Transform publisher
  tf::Transform camera_to_valve_;                   //!> Camera to valve transformation

  cv::Mat processed_;                               //!> Processed image
  image_geometry::StereoCameraModel stereo_model_;  //!> Camera model to compute the 3d world points

  cv::Point3d valve_symmetric_point_;               //>! Used to track the valve symmetric points         

  void stereoImageCallback(
      const sensor_msgs::ImageConstPtr& l_image_msg,
      const sensor_msgs::ImageConstPtr& r_image_msg,
      const sensor_msgs::CameraInfoConstPtr& l_info_msg,
      const sensor_msgs::CameraInfoConstPtr& r_info_msg); //!> Image callback

  std::vector<cv::Point3d> matchTgtMdlPoints(
    std::vector<cv::Point3d> points_3d, bool inverse);    //!> Sort the target points
  static void staticMouseCallback(int event, int x, 
    int y, int flags, void* param);                       //!> Mouse callback interface
  void mouseCallback( int event, int x, int y, 
    int flags, void* param);                              //!> Mouse interface
  void autotuning();                                      //!> Function for autotuning process
};

} // namespace

#endif // TRACKER_H
