/**
 * @file
 * @brief Valve trainer presentation.
 */

#ifndef TRAINER_H
#define TRAINER_H

#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/image_encodings.h>
#include <image_geometry/stereo_camera_model.h>

#include "valve_tracker/stereo_processor.h"
#include "valve_tracker/trainer.h"

#define LEFT 0
#define RIGHT 1

namespace valve_tracker
{

class Trainer : public StereoImageProcessor
{

public:

  // Trainer states
	enum Mode
	{
	  DISPLAY_VIDEO,
	  AWAITING_TRAINING_IMAGE,
	  SHOWING_TRAINING_IMAGE,
	  PAINTING,
	  ROI_SELECTED,
	  TRAINED
	};

	cv::Mat image; //!> showing image

  // Constructors
  Trainer(const std::string transport);

  // Destructor
  ~Trainer();

private:

  // Trainer HSV values
  std::string trained_model_path_;
  int num_hue_bins_;
  int num_sat_bins_;
  int num_val_bins_;

  int training_status_;               //!> Defines the trainer Mode

  cv::Mat training_image_;            //!> Image HSV used to train

  cv::Point roi_rectangle_origin_;    
  cv::Rect roi_rectangle_selection_;

  cv::MatND model_histogram_;         //!> Histogram of the trained model

  void stereoImageCallback(
      const sensor_msgs::ImageConstPtr& l_image_msg,
      const sensor_msgs::ImageConstPtr& r_image_msg,
      const sensor_msgs::CameraInfoConstPtr& l_info_msg,
      const sensor_msgs::CameraInfoConstPtr& r_info_msg); //!> Image callback
  cv::MatND train(const cv::Mat& image);
  int detect(const cv::MatND& hist, const cv::Mat& image);
  static void staticMouseCallback(int event, int x, int y, int flags, void* param);
  void mouseCallback(int event, int x, int y, int flags, void* param);
  cv::MatND calculateHistogram(const cv::Mat& image, 
	                             const int bins[],  
	                             const cv::Mat& mask);
  void showHSVHistogram(const cv::MatND& histogram,
                        const std::string& name_hs, 
                        const std::string& name_hv);
};

} // namespace

#endif // TRACKER_H
