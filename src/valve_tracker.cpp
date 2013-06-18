#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>

#include "valve_tracker.h"

#define LEFT 0
#define RIGHT 1

bool sort_points_x(const cv::Point2d& p1,const cv::Point2d& p2)
{
  return (p1.x < p2.x);
}

ValveTracker::ValveTracker(const std::string transport) : StereoImageProcessor(transport)
{
  ROS_INFO_STREAM("Instantiating the Valve Tracker...");

  // get all the params out!
  ros::NodeHandle nhp("~");
  nhp.param("stereo_frame_id", stereo_frame_id_, std::string("/stereo_down"));
  nhp.param("base_link_frame_id", base_link_frame_id_, std::string("/base_link"));
  nhp.param("threshold_h_low", threshold_h_low_, 0);
  nhp.param("threshold_h_hi", threshold_h_hi_, 255);
  nhp.param("threshold_s_low", threshold_s_low_, 0);
  nhp.param("threshold_s_hi", threshold_s_hi_, 255);
  nhp.param("threshold_v_low", threshold_v_low_, 0);
  nhp.param("threshold_v_hi", threshold_v_hi_, 255);
  nhp.param("closing_element_size", closing_element_size_, 255);
  nhp.param("opening_element_size", opening_element_size_, 255);
  nhp.param("canny_first_threshold", canny_first_threshold_, 100);
  nhp.param("canny_second_threshold", canny_second_threshold_, 110);
  nhp.param("epipolar_width_threshold",epipolar_width_threshold_, 3);
  nhp.param("show_debug_images",show_debug_images_, 0);

  ROS_INFO_STREAM("Valve Tracker Settings:" << std::endl <<
                  "  stereo_frame_id    = " << stereo_frame_id_ << std::endl <<
                  "  base_link_frame_id = " << base_link_frame_id_ << std::endl);

  // image publisher for future debug
  image_transport::ImageTransport it(nhp);
  image_pub_  = it.advertise("image_detections", 1);

  // OpenCV image windows for debugging
  if(show_debug_images_){
    cv::namedWindow("Valve Tracker", 1);
    cv::namedWindow("hue", 1);
    cv::namedWindow("sat", 1);
    //cv::namedWindow("val", 1);
    cv::createTrackbar("H low", "Valve Tracker", &threshold_h_low_, 255);
    cv::createTrackbar("H hi", "Valve Tracker", &threshold_h_hi_, 255);
    cv::createTrackbar("S low", "Valve Tracker", &threshold_s_low_, 255);
    cv::createTrackbar("S hi", "Valve Tracker", &threshold_s_hi_, 255);
    //cv::createTrackbar("V low", "Valve Tracker", &threshold_v_low_, 255);
    //cv::createTrackbar("V hi", "Valve Tracker", &threshold_v_hi_, 255);
    cv::createTrackbar("C", "Valve Tracker", &closing_element_size_, 255);
    cv::createTrackbar("O", "Valve Tracker", &opening_element_size_, 255);
  }

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

  // reserve memory for points
  points_.clear();
  points_.resize(2);

  process(l_cv_image_ptr->image, LEFT);
  process(r_cv_image_ptr->image, RIGHT);
  
  triangulatePoints();
  if (image_pub_.getNumSubscribers() > 0)
  {
    cv_bridge::CvImagePtr cv_ptr(new cv_bridge::CvImage);
    cv_ptr->image    = processed_;
    cv_ptr->encoding = "mono8";
    image_pub_.publish(cv_ptr->toImageMsg());
  }
}

bool ValveTracker::process(cv::Mat img, int type)
{

  cv::Mat hsv_img(img.size(), CV_8UC3);
  
  // convert to HSV color space
  cv::cvtColor(img, hsv_img, CV_BGR2HSV);

  cv::Mat hue_img(img.size(), CV_8UC1);
  cv::Mat sat_img(img.size(), CV_8UC1);
  cv::Mat val_img(img.size(), CV_8UC1);

  // copy the channels to their respective cv::Mat array
  int from_h[] = {0, 0};
  int from_s[] = {1, 0};
  int from_v[] = {2, 0};
  cv::mixChannels(&img, 1, &hue_img, 1, from_h, 1);
  cv::mixChannels(&img, 1, &sat_img, 1, from_s, 1);
  cv::mixChannels(&img, 1, &val_img, 1, from_v, 1);

  cv::Mat hue_img_low, hue_img_hi;
  cv::Mat sat_img_low, sat_img_hi;
  cv::Mat val_img_low, val_img_hi;

  // threshold those spaces
  cv::threshold(hue_img, hue_img_low, threshold_h_low_, 255, cv::THRESH_BINARY); // threshold binary
  cv::threshold(sat_img, sat_img_low, threshold_s_low_, 255, cv::THRESH_BINARY);
  cv::threshold(val_img, val_img_low, threshold_v_low_, 255, cv::THRESH_BINARY);
  
  cv::threshold(hue_img, hue_img_hi, threshold_h_hi_, 255, cv::THRESH_BINARY_INV); // threshold binary
  cv::threshold(sat_img, sat_img_hi, threshold_s_hi_, 255, cv::THRESH_BINARY_INV);
  cv::threshold(val_img, val_img_hi, threshold_v_hi_, 255, cv::THRESH_BINARY_INV);

  cv::Mat output_img(img.size(), CV_8UC1);
  
  cv::bitwise_and(hue_img_low, hue_img_hi, hue_img);
  cv::bitwise_and(sat_img_low, sat_img_hi, sat_img);
  cv::bitwise_and(val_img_low, val_img_hi, val_img);

  cv::bitwise_and(hue_img, sat_img, output_img);
  //cv::bitwise_and(val_img, output_img, output_img);
  
  // morphology closing
  cv::Mat element_closing = createElement(closing_element_size_);
  cv::Mat element_opening = createElement(opening_element_size_);

  cv::morphologyEx(output_img, output_img, cv::MORPH_OPEN, element_opening);
  cv::morphologyEx(output_img, output_img, cv::MORPH_CLOSE, element_closing);

  // detect blobs in image
  cv::Mat canny_output;
  std::vector<std::vector<cv::Point> > contours;
  std::vector<cv::Vec4i> hierarchy;

  cv::Canny(output_img, canny_output, canny_first_threshold_, canny_second_threshold_);
  cv::findContours(canny_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
  
  for (size_t i = 0; i < contours.size(); i++)
  {
    //cv::Scalar color = cv::Scalar( 255*(i+1)/4, 255*(i+1)/4, 255*(i+1)/4 );
    //cv::drawContours( output_img, contours, i, color, 2, 8, hierarchy, 0, cv::Point() );
    //std::cout << "New row " << contours[i] << std::endl;

    // calculate mean points
    double u_mean = 0;
    double v_mean = 0;
    for (size_t j = 0; j < contours[i].size(); j++)
    {
      u_mean += contours[i][j].x;
      v_mean += contours[i][j].y;
    }
    u_mean /= contours[i].size();
    v_mean /= contours[i].size();
    cv::Point mean_point(u_mean, v_mean);
    
    //std::cout << "Mean point: ( " << u_mean << " , " << v_mean << " )" << std::endl;

    // draw mean points
    cv::circle( output_img, mean_point, 15, cv::Scalar(127,127,127), 2);

    points_[type].push_back(mean_point);

  }

  //std::cout << points_[type].size() << " points found!" << std::endl;

  if(show_debug_images_ && !type){
    cv::imshow("hue", hue_img);
    cv::imshow("sat", sat_img);
    //cv::imshow("val", val_img);
    cv::imshow("Valve Tracker", output_img);
    cv::waitKey(3);

    processed_ = output_img;
  }
  
  return 0;
}

cv::Mat ValveTracker::createElement(int element_size)
{
  cv::Mat element = cv::Mat::zeros(element_size, element_size, CV_8UC1);
  cv::circle(element, cv::Point(element_size / 2, element_size / 2), element_size / 2, cv::Scalar(255), -1);
  return element;
}

void ValveTracker::triangulatePoints(void)
{
  // look in y the ones in the same epipolar line
  for (size_t i = 0; i < points_[LEFT].size(); i++)
  { 
    std::vector<cv::Point2d> correspondences;
    cv::Point2d pl(points_[LEFT][i]);
    for (size_t j = 0; j < points_[RIGHT].size(); j++)
    {
      cv::Point2d pr(points_[RIGHT][j]);
      double dist_y = abs(pl.y - pr.y);
      if (dist_y < epipolar_width_threshold_)
        correspondences.push_back(pr);
      // find min distance to another point
    }
    // loop all correspondences and look for the
    // order of appearance
    if (correspondences.size() == 1)
    {
      cv::Point3d p;
      stereo_model_.projectDisparityTo3d(pl, pl.x-correspondences[0].x,p);
      points3d_.push_back(p);
      ROS_INFO("3d point added");
    }else{
      // get all the left points in the same epipolar
      std::vector<cv::Point2d> left_points;
      std::vector<cv::Point2d> right_points;
      for (size_t ii=0; ii<points_[LEFT].size(); ii++)
      { 
        cv::Point2d pl2(points_[LEFT][ii]);
        double dist_y = abs(pl.y - pl2.y);
        if (dist_y < epipolar_width_threshold_) 
        {
          left_points.push_back(pl2);
        }
      }

      // get all the right points in the same epipolar
      for (size_t jj=0; jj<points_[RIGHT].size(); jj++)
      { 
        cv::Point2d pr(points_[RIGHT][jj]);
        double dist_y = abs(pl.y - pr.y);
        if (dist_y < epipolar_width_threshold_)
        {
          right_points.push_back(pr);
        }
      }
      
      // sort them and assign correspondences
      std::sort(left_points.begin(), left_points.end(), sort_points_x);
      std::sort(right_points.begin(), right_points.end(), sort_points_x);

      std::vector<cv::Point2d>::iterator it;
      it = std::find(left_points.begin(), left_points.end(), pl);
      unsigned int idx = it - left_points.begin();
      if(idx < left_points.size())
      {
        // we've correctly found the item. Get the same position point
        // from right points
        cv::Point3d p;
        cv::Point2d pr(right_points[idx]);
        stereo_model_.projectDisparityTo3d(pl, pl.x-pr.x, p);
        points3d_.push_back(p);
        ROS_INFO("Correspondence solved!");
      }else{
        // it has not been found
        ROS_WARN("Correspondence could not be found.");
      }
    }
  }
}
