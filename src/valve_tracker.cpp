#include "valve_tracker/valve_tracker.h"
#include "valve_tracker/utils.h"
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <tf/transform_broadcaster.h>
#include <Eigen/Eigen>
#include "opencv2/core/core.hpp"

#define PI 3.14159265

/** \brief ValveTracker constructor
  * \param transport
  */
valve_tracker::ValveTracker::ValveTracker(const std::string transport) : StereoImageProcessor(transport)
{
  ROS_INFO_STREAM("[ValveTracker:] Instantiating the Valve Tracker...");

  // Get all the params out!
  ros::NodeHandle nhp("~");
  nhp.param("stereo_frame_id", stereo_frame_id_, std::string("/stereo_down"));
  nhp.param("base_link_frame_id", valve_frame_id_, std::string("/valve"));
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
  nhp.param("epipolar_width_threshold", epipolar_width_threshold_, 3);

  // Load the model
  XmlRpc::XmlRpcValue model;
  int cols, rows;
  nhp.getParam("model_matrix/cols", cols);
  nhp.getParam("model_matrix/rows", rows);
  nhp.getParam("model_matrix/data", model);
  ROS_ASSERT(model.getType() == XmlRpc::XmlRpcValue::TypeArray);
  ROS_ASSERT(model.size() == cols*rows);
  for (int i=0; i<model.size(); i=i+3)
  {
    ROS_ASSERT(model[i].getType() == XmlRpc::XmlRpcValue::TypeDouble);
    cv::Point3f p((double)model[i], (double)model[i+1], (double)model[i+2]);
    mdl_.push_back(p);
  }

  // Image publisher for future debug
  image_transport::ImageTransport it(nhp);
  image_pub_  = it.advertise("image_detections", 1);

  // Initialize the camera to valve transformation
  camera_to_valve_.setIdentity();
}

/** \brief Stereo Image Callback
  * \param l_image_msg message of the left image
  * \param r_image_msg message of the right image
  * \param l_info_msg information message of the left image
  * \param r_info_msg information message of the right image
  */
void valve_tracker::ValveTracker::stereoImageCallback(
  const sensor_msgs::ImageConstPtr     & l_image_msg,
  const sensor_msgs::ImageConstPtr     & r_image_msg,
  const sensor_msgs::CameraInfoConstPtr& l_info_msg,
  const sensor_msgs::CameraInfoConstPtr& r_info_msg)
{

  // Images to opencv
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
    ROS_ERROR("[ValveTracker:] cv_bridge exception: %s", e.what());
    return;
  }

  // Get the camera model
  stereo_model_.fromCameraInfo(l_info_msg, r_info_msg);

  // Detect valve in both images
  std::vector< std::vector<cv::Point2d> > points_2d;
  points_2d.push_back(valveDetection(l_cv_image_ptr->image)); // Left  points at idx=0
  points_2d.push_back(valveDetection(r_cv_image_ptr->image)); // Right points at idx=1

  // Valve is defined by 3 points
  if (points_2d[0].size() == 3 || points_2d[1].size() == 3)
  {
    // Triangulate the 3D points
    std::vector<cv::Point3d> points3d;
    points3d = triangulatePoints(points_2d);
    
    // Compute the 3D points of the valve
    if(points3d.size() == 3)
    {
      // Compute the transformation from camera to valve
      camera_to_valve_ = estimateTransform(points3d);
    }
  }
  else
  {
    ROS_WARN_STREAM("[ValveTracker:] Incorrect number of valve points found (" << 
                    points_2d[0].size() << " points) 3 needed.");
  }

  // Publish processed image
  if (image_pub_.getNumSubscribers() > 0)
  {
    cv_bridge::CvImagePtr cv_ptr(new cv_bridge::CvImage);
    cv_ptr->image    = processed_;
    cv_ptr->encoding = "mono8";
    image_pub_.publish(cv_ptr->toImageMsg());
  }

  // Publish last computed transform
  tf_broadcaster_.sendTransform(
      tf::StampedTransform(camera_to_valve_, l_image_msg->header.stamp,
      stereo_frame_id_, valve_frame_id_));
}

/** \brief Detect the valve into the image
  * @return vector with the detected valve points
  * \param image where the valve will be detected
  */
std::vector<cv::Point2d> 
valve_tracker::ValveTracker::valveDetection(cv::Mat img)
{
  // Initialize output
  std::vector<cv::Point2d> points;

  cv::Mat hsv_img(img.size(), CV_8UC3);
  
  // Convert to HSV color space
  cv::cvtColor(img, hsv_img, CV_BGR2HSV);

  cv::Mat hue_img(img.size(), CV_8UC1);
  cv::Mat sat_img(img.size(), CV_8UC1);
  cv::Mat val_img(img.size(), CV_8UC1);

  // Copy the channels to their respective cv::Mat array
  int from_h[] = {0, 0};
  int from_s[] = {1, 0};
  int from_v[] = {2, 0};
  cv::mixChannels(&img, 1, &hue_img, 1, from_h, 1);
  cv::mixChannels(&img, 1, &sat_img, 1, from_s, 1);
  cv::mixChannels(&img, 1, &val_img, 1, from_v, 1);

  cv::Mat hue_img_low, hue_img_hi;
  cv::Mat sat_img_low, sat_img_hi;
  cv::Mat val_img_low, val_img_hi;

  // Threshold those spaces
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
  
  // Morphology closing
  cv::Mat element_closing = valve_tracker::Utils::createElement(closing_element_size_);
  cv::Mat element_opening = valve_tracker::Utils::createElement(opening_element_size_);

  cv::morphologyEx(output_img, output_img, cv::MORPH_OPEN, element_opening);
  cv::morphologyEx(output_img, output_img, cv::MORPH_CLOSE, element_closing);

  // Detect blobs in image
  cv::Mat canny_output;
  std::vector<std::vector<cv::Point> > contours;
  std::vector<cv::Vec4i> hierarchy;

  cv::Canny(output_img, canny_output, canny_first_threshold_, canny_second_threshold_);
  cv::findContours(canny_output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
  
  for (size_t i = 0; i < contours.size(); i++)
  {
    // Calculate mean points
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

    // Draw mean points
    cv::circle( output_img, mean_point, 15, cv::Scalar(127,127,127), 2);

    points.push_back(mean_point);
  }

  processed_ = output_img;

  return points;
}

/** \brief Triangulate the 3D points of the valve
  * @return vector with the 3D points of the valve
  * \param vector of left and right valve point detection
  */
std::vector<cv::Point3d> valve_tracker::ValveTracker::triangulatePoints(
    std::vector< std::vector<cv::Point2d> > points_2d)
{
  // Initialize output
  std::vector<cv::Point3d> points3d;

  // Sanity check
  if (points_2d[0].size() != 3 || points_2d[1].size() != 3)
  {
    ROS_WARN_STREAM(  "[ValveTracker:] Incorrect number of valve points found (" << 
                      points_2d[0].size() << " points) 3 needed.");
    return points3d;
  }

  // Look in y the ones in the same epipolar line
  for (size_t i = 0; i < points_2d[0].size(); i++)
  {
    std::vector<cv::Point2d> correspondences;
    cv::Point2d pl(points_2d[0][i]);
    for (size_t j = 0; j < points_2d[1].size(); j++)
    {
      cv::Point2d pr(points_2d[1][j]);
      double dist_y = abs(pl.y - pr.y);
      if (dist_y <= epipolar_width_threshold_)
        correspondences.push_back(pr);
    }

    // Loop all correspondences and look for the
    // order of appearance
    if (correspondences.size() == 1)
    {
      cv::Point3d p;
      stereo_model_.projectDisparityTo3d(pl, pl.x-correspondences[0].x, p);
      points3d.push_back(p);
      ROS_DEBUG("[ValveTracker:] 3D points added!");
    }
    else if (correspondences.size() > 1)
    {
      // get all the left points in the same epipolar
      std::vector<cv::Point2d> left_points;
      std::vector<cv::Point2d> right_points;
      for (size_t ii=0; ii<points_2d[0].size(); ii++)
      { 
        cv::Point2d pl2(points_2d[0][ii]);
        double dist_y = abs(pl.y - pl2.y);
        if (dist_y <= epipolar_width_threshold_) 
        {
          left_points.push_back(pl2);
        }
      }

      // Get all the right points in the same epipolar
      for (size_t jj=0; jj<points_2d[1].size(); jj++)
      { 
        cv::Point2d pr(points_2d[1][jj]);
        double dist_y = abs(pl.y - pr.y);
        if (dist_y <= epipolar_width_threshold_)
        {
          right_points.push_back(pr);
        }
      }

      // Sort them and assign correspondences
      std::sort(left_points.begin(), left_points.end(), valve_tracker::Utils::sort_points_x);
      std::sort(right_points.begin(), right_points.end(), valve_tracker::Utils::sort_points_x);

      std::vector<cv::Point2d>::iterator it;
      it = std::find(left_points.begin(), left_points.end(), pl);
      unsigned int idx = it - left_points.begin();

      if(idx < left_points.size())
      {
        // We've correctly found the item. Get the same position point
        // from right points
        cv::Point3d p;
        cv::Point2d pr(right_points[idx]);
        stereo_model_.projectDisparityTo3d(pl, pl.x-pr.x, p);
        points3d.push_back(p);
        ROS_DEBUG("[ValveTracker:] Correspondence solved!");
      }
      else
      {
        // It has not been found
        ROS_WARN("[ValveTracker:] Correspondence could not be found.");
      }
    }
    else
    {
      // It has not been found
      ROS_WARN_STREAM("[ValveTracker:] 0 correspondences found between left and " <<
                "right images, consider increasing the parameter 'epipolar_width_threshold'.");
    }
  }

  return points3d;
}

/** \brief Detect the valve into the image
  * @return the transformation between the camera and valve.
  * \param 3D points of the valve.
  */
tf::Transform valve_tracker::ValveTracker::estimateTransform(
    std::vector<cv::Point3d> points_3d)
{
  // Sanity check
  if (points_3d.size() != 3)
  {
    ROS_WARN_STREAM(  "[ValveTracker:] Impossible to estimate the transformation " << 
                      "between camera and valve, wrong 3d correspondences size: " <<
                      points_3d.size());

    tf::Transform output;
    output.setIdentity();
    return output;
  }

  // Target point cloud
  std::vector<cv::Point3f> tgt;

  // Get target root point
  double distance = valve_tracker::Utils::euclideanDist(points_3d[0]);
  int idx_root = 0;
  for (unsigned int i=1; i<points_3d.size(); i++)
  {
    if(valve_tracker::Utils::euclideanDist(points_3d[i]) > distance)
    {
      distance = valve_tracker::Utils::euclideanDist(points_3d[i]);
      idx_root = i;
    }
  }

  // Compute target central point
  static const int arr[] = {0, 1, 2};
  std::vector<int> v (arr, arr + sizeof(arr) / sizeof(arr[0]));
  v.erase(v.begin() + idx_root);
  tgt.push_back(cv::Point3f( (points_3d[v[0]].x + points_3d[v[1]].x) / 2,
                             (points_3d[v[0]].y + points_3d[v[1]].y) / 2,
                             (points_3d[v[0]].z + points_3d[v[1]].z) / 2));

  // Find the valve symmetrical sides
  if (points_3d[v[0]].x < points_3d[v[1]].x)
  {
    tgt.push_back(cv::Point3f( points_3d[v[0]].x, points_3d[v[0]].y, points_3d[v[0]].z ));
    tgt.push_back(cv::Point3f( points_3d[v[1]].x, points_3d[v[1]].y, points_3d[v[1]].z ));
  }
  else
  {
    tgt.push_back(cv::Point3f( points_3d[v[1]].x, points_3d[v[1]].y, points_3d[v[1]].z ));
    tgt.push_back(cv::Point3f( points_3d[v[0]].x, points_3d[v[0]].y, points_3d[v[0]].z ));
  }

  tgt.push_back(cv::Point3f( points_3d[idx_root].x, points_3d[idx_root].y, points_3d[idx_root].z ));

  // Compute centroids
  Eigen::Vector3f centroid_mdl(0.0, 0.0, 0.0);
  Eigen::Vector3f centroid_tgt(0.0, 0.0, 0.0);
  for (unsigned int i=0; i<mdl_.size(); i++)
  {
    centroid_mdl(0) += mdl_[i].x;
    centroid_mdl(1) += mdl_[i].y;
    centroid_mdl(2) += mdl_[i].z;
    centroid_tgt(0) += tgt[i].x;
    centroid_tgt(1) += tgt[i].y;
    centroid_tgt(2) += tgt[i].z;
  }
  centroid_mdl /= mdl_.size();
  centroid_tgt /= tgt.size();

  // Compute H
  Eigen::Matrix3f H = Eigen::Matrix3f::Zero();
  for (unsigned int i=0; i<mdl_.size(); i++)
  {
    Eigen::Vector3f p_mdl(mdl_[i].x, mdl_[i].y, mdl_[i].z);
    Eigen::Vector3f p_tgt(tgt[i].x, tgt[i].y, tgt[i].z);

    Eigen::Vector3f v_mdl = p_mdl - centroid_mdl;
    Eigen::Vector3f v_tgt = p_tgt - centroid_tgt;

    H += v_mdl * v_tgt.transpose();
  }
 
  // Pseudo inverse
  Eigen::JacobiSVD<Eigen::Matrix3f> svdOfH(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3f u_svd = svdOfH.matrixU ();
  Eigen::Matrix3f v_svd = svdOfH.matrixV ();
  
  // Compute R = V * U'
  if (u_svd.determinant () * v_svd.determinant () < 0)
  {
    for (int x = 0; x < 3; ++x)
      v_svd (x, 2) *= -1;
  }
  Eigen::Matrix3f R = v_svd * u_svd.transpose ();

  // Compute translation
  Eigen::Vector3f t = -R * centroid_mdl + centroid_tgt;

  // Build rotation matrix
  tf::Matrix3x3 rot(R(0,0), R(0,1), R(0,2),
                    R(1,0), R(1,1), R(1,2),
                    R(2,0), R(2,1), R(2,2));
 
  // Build the tf
  tf::Vector3 trans(t(0), t(1), t(2));
  tf::Transform output(rot, trans);

  return output;
}
