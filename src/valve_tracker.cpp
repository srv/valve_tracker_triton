#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>

#include "valve_tracker.h"


ValveTracker::ValveTracker(const std::string transport) : StereoImageProcessor(transport)
{
  ROS_INFO_STREAM("Instantiating the Valve Tracker...");

  ros::NodeHandle nhp("~");
  nhp.param("stereo_frame_id", stereo_frame_id_, std::string("/stereo_down"));
  nhp.param("base_link_frame_id", base_link_frame_id_, std::string("/base_link"));

  ROS_INFO_STREAM("Valve Tracker Settings:" << std::endl <<
                  "  stereo_frame_id    = " << stereo_frame_id_ << std::endl <<
                  "  base_link_frame_id = " << base_link_frame_id_ << std::endl);

  image_transport::ImageTransport it(nhp);
  image_pub_  = it.advertise("image_detections", 1);

}
void ValveTracker::stereoImageCallback(
  const sensor_msgs::ImageConstPtr     & l_image_msg,
  const sensor_msgs::ImageConstPtr     & r_image_msg,
  const sensor_msgs::CameraInfoConstPtr& l_info_msg,
  const sensor_msgs::CameraInfoConstPtr& r_info_msg)
{
  cv_bridge::CvImagePtr l_cv_image_ptr;
  cv_bridge::CvImagePtr r_cv_image_ptr;

  try
  {
    l_cv_image_ptr = cv_bridge::toCvCopy(l_image_msg,
                                         sensor_msgs::image_encodings::BGR8);
    r_cv_image_ptr = cv_bridge::toCvCopy(r_image_msg,
                                         sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  stereo_model_.fromCameraInfo(l_info_msg, r_info_msg);

  if (!process(l_cv_image_ptr->image, r_cv_image_ptr->image))
  {
    ROS_ERROR("Cannot process stereo images. Skipping!");
    return;
  }
  if (image_pub_.getNumSubscribers() > 0)
  {
    cv_bridge::CvImagePtr cv_ptr(new cv_bridge::CvImage);
    cv_ptr->image    = processed_;
    cv_ptr->encoding = "mono8";
    image_pub_.publish(cv_ptr->toImageMsg());
  }
}

bool ValveTracker::process(cv::Mat left, cv::Mat right)
{
  cv::Mat img_red;
  img_red  = cv::Mat(left.size(), CV_8UC1);
  int from_to[] = { 2, 0 };
  cv::mixChannels(&left, 1, &img_red, 1, from_to, 1);

  // threshold green image
  int maxval = 255;
  int thd_val_hi = 200;
  int thd_val_low = 150;

  cv::threshold(img_red, processed_, thd_val_low, maxval, cv::THRESH_BINARY);

  return true;
}
