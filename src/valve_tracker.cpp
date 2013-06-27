#include "valve_tracker/valve_tracker.h"
#include "valve_tracker/utils.h"
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <tf/transform_broadcaster.h>
#include <Eigen/Eigen>
#include "opencv2/core/core.hpp"

/** \brief ValveTracker constructor
  * \param transport
  */
valve_tracker::ValveTracker::ValveTracker(const std::string transport) : StereoImageProcessor(transport)
{
  ROS_INFO_STREAM("[ValveTracker:] Instantiating the Valve Tracker...");

  // Get all the params out!
  ros::NodeHandle nhp("~");
  nhp.param("stereo_frame_id", stereo_frame_id_, std::string("/stereo_down"));
  nhp.param("valve_frame_id", valve_frame_id_, std::string("/valve"));
  nhp.param("closing_element_size", closing_element_size_, 0);
  nhp.param("opening_element_size", opening_element_size_, 1);
  nhp.param("binary_threshold", binary_threshold_, 80);
  nhp.param("min_value_threshold", min_value_threshold_, 110);
  nhp.param("min_blob_size", min_blob_size_, 8);
  nhp.param("max_blob_size", max_blob_size_, 200);
  nhp.param("epipolar_width_threshold", epipolar_width_threshold_, 3);
  nhp.param("mean_filter_size", mean_filter_size_, 1);
  nhp.param("max_tf_error", max_tf_error_, 0.1);
  nhp.param("trained_model_path", trained_model_path_, 
      valve_tracker::Utils::getPackageDir() + std::string("/etc/trained_model.yml"));
  nhp.param("show_debug", show_debug_, true);

  ROS_INFO_STREAM("[ValveTracker:] Valve Tracker Settings:" << std::endl <<
                  "  stereo_frame_id            = " << stereo_frame_id_ << std::endl <<
                  "  valve_frame_id             = " << valve_frame_id_ << std::endl <<
                  "  closing_element_size       = " << closing_element_size_ << std::endl <<
                  "  opening_element_size       = " << opening_element_size_ << std::endl <<
                  "  binary_threshold           = " << binary_threshold_ << std::endl <<
                  "  min_value_threshold        = " << min_value_threshold_ << std::endl <<
                  "  min_blob_size              = " << min_blob_size_ << std::endl <<
                  "  max_blob_size              = " << max_blob_size_ << std::endl <<
                  "  epipolar_width_threshold   = " << epipolar_width_threshold_ << std::endl <<
                  "  mean_filter_size           = " << mean_filter_size_ << std::endl <<
                  "  max_tf_error               = " << max_tf_error_ << std::endl <<
                  "  trained_model_path         = " << trained_model_path_ << std::endl);

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
    cv::Point3d p((double)model[i], (double)model[i+1], (double)model[i+2]);
    valve_model_points_.push_back(p);
  }

  // Read the trained histogram
  cv::FileStorage fs(trained_model_path_, cv::FileStorage::READ);
  fs["model_histogram"] >> trained_model_;
  fs.release();

  // Image publisher for future debug
  image_transport::ImageTransport it(nhp);
  image_pub_  = it.advertise("image_detections", 1);

  // Initialize the camera to valve transformation
  camera_to_valve_.setIdentity();

  // Initialize the symmetric point tracker
  valve_symmetric_point_ = cv::Point3d(0.0, 0.0, 0.0);
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
  points_2d.push_back(valveDetection(l_cv_image_ptr->image, show_debug_));  // Left  points at idx=0
  points_2d.push_back(valveDetection(r_cv_image_ptr->image, false));        // Right points at idx=1

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
      tf::Transform camera_to_valve_tmp;
      bool success = estimateTransform(points3d, camera_to_valve_tmp);
      if (success)
        camera_to_valve_ = camera_to_valve_tmp;
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
std::vector<cv::Point2d> valve_tracker::ValveTracker::valveDetection(cv::Mat image, bool debug)
{
  // Initialize output
  std::vector<cv::Point2d> points;

  // Compute backprojection
  cv::Mat hsv_image;
  cv::cvtColor(image, hsv_image, CV_BGR2HSV);
  cv::Mat backprojection = valve_tracker::Utils::calculateBackprojection(hsv_image, trained_model_);

  // Used to draw the contours
  cv::Mat contour_image(backprojection.size(), CV_8UC3, cv::Scalar(0,0,0));
  cv::cvtColor(backprojection, contour_image, CV_GRAY2RGB);
  cv::Scalar color(0, 0, 255);

  // filter out noise
  if (mean_filter_size_ > 2 && mean_filter_size_ % 2 == 1)
  {
    cv::medianBlur(backprojection, backprojection, mean_filter_size_);
  }

  // perform thresholding
  cv::Mat binary;
  cv::threshold(backprojection, binary, binary_threshold_, 255, CV_THRESH_BINARY);

  // morphologicla operations
  cv::Mat binary_morphed = binary.clone();
  if (opening_element_size_ > 0)
  {
    cv::Mat element = cv::Mat::zeros(opening_element_size_, opening_element_size_, CV_8UC1);
    cv::circle(element, cv::Point(opening_element_size_ / 2, opening_element_size_ / 2), opening_element_size_ / 2, cv::Scalar(255), -1);
    cv::morphologyEx(binary_morphed, binary_morphed, cv::MORPH_OPEN, element);
  }
  if (closing_element_size_ > 0)
  {
    cv::Mat element = cv::Mat::zeros(closing_element_size_, closing_element_size_, CV_8UC1);
    cv::circle(element, cv::Point(closing_element_size_ / 2, closing_element_size_ / 2), closing_element_size_ / 2, cv::Scalar(255), -1);
    cv::morphologyEx(binary_morphed, binary_morphed, cv::MORPH_CLOSE, element);
  }  

  // create mask for ivalid values
  std::vector<cv::Mat> hsv_channels;
  cv::split(hsv_image, hsv_channels);
  cv::Mat value = hsv_channels[2];
  cv::Mat min_value_mask;
  cv::threshold(value, min_value_mask, min_value_threshold_, 255, CV_THRESH_BINARY);

  // mask out low values in binary image
  cv::bitwise_and(min_value_mask, binary_morphed, binary_morphed);

  // Detect blobs in image
  std::vector< std::vector<cv::Point> > contours;
  cv::Mat contour_output = binary_morphed.clone();
  cv::findContours(contour_output, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
  std::vector< std::vector<cv::Point> > contours_filtered;

  if (contours.size() < 3)
  {
    ROS_DEBUG_STREAM("[ValveTracker:] Not enought points detected: " << contours.size() << " (3 needed).");
  }
  else
  {
    // 3 or more blobs detected. Delete too big and too small blobs
    std::vector< std::vector<cv::Point> >::iterator iter = contours.begin();
    while (iter != contours.end())
    {
      if (iter->size() > (unsigned int)max_blob_size_ || 
          iter->size() < (unsigned int)min_blob_size_)
      {
        iter = contours.erase(iter);
      }
      else
      {
        ++iter;
      }
    }

    // Check that we keep having at least 3 contours
    if (contours.size() < 3)
    {
      ROS_DEBUG("[ValveTracker:] Blob filtering has removed too many blobs!");
    }
    else
    {
      // Sort the result by size
      std::sort(contours.begin(), contours.end(), valve_tracker::Utils::sort_vectors_by_size);

      // Get the 3 biggest blobs
      std::vector< std::vector<cv::Point> > contours_tmp(contours.begin(), contours.begin() + 3);
      contours_filtered = contours_tmp;

      // Sort from left to right and from top to bottom


      for (size_t i = 0; i < contours_filtered.size(); i++)
      {
        // Calculate mean points
        double u_mean = 0;
        double v_mean = 0;
        for (size_t j = 0; j < contours_filtered[i].size(); j++)
        {
          u_mean += contours_filtered[i][j].x;
          v_mean += contours_filtered[i][j].y;
        }
        u_mean /= contours_filtered[i].size();
        v_mean /= contours_filtered[i].size();
        cv::Point mean_point(u_mean, v_mean);

        points.push_back(mean_point);
      }
    }
  }

  // debug purposes
  if (debug)
  {
    for (size_t idx=0; idx<contours_filtered.size(); idx++)
    {
      cv::drawContours(contour_image, contours_filtered, idx, color, 2);
      cv::circle(contour_image, points[idx], 15, color, 2);
    }

    // Show images
    std::string model_name = "valve";
    cv::namedWindow(model_name + "-backprojection-contours", 0);
    cv::namedWindow(model_name + "-binary", 0);
    cv::namedWindow(model_name + "-binary-morphed", 0);

    cv::createTrackbar("mean_filter_size", model_name + "-binary",  &mean_filter_size_, 255);
    cv::createTrackbar("binary_threshold", model_name + "-binary",  &binary_threshold_, 255);

    cv::createTrackbar("closing_element_size", model_name + "-binary-morphed",  &closing_element_size_, 255);
    cv::createTrackbar("opening_element_size", model_name + "-binary-morphed",  &opening_element_size_, 255);
    cv::createTrackbar("min_value_threshold", model_name + "-binary-morphed",  &min_value_threshold_, 255);

    cv::createTrackbar("min_blob_size", model_name + "-backprojection-contours",  &min_blob_size_, 255);
    cv::createTrackbar("max_blob_size", model_name + "-backprojection-contours",  &max_blob_size_, 255);

    cv::imshow(model_name + "-backprojection-contours", contour_image);
    cv::imshow(model_name + "-binary", binary);
    cv::imshow(model_name + "-binary-morphed", binary_morphed);
    cv::waitKey(5);
  }
  
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
    ROS_DEBUG_STREAM("[ValveTracker:] Incorrect number of valve points found (L:" << 
                      points_2d[0].size() << " R:" << points_2d[0].size() << 
                      " points) 3 needed.");
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
      // Get all the right points in the same epipolar
      std::vector<cv::Point2d> right_points;
      for (size_t ii=0; ii<points_2d[1].size(); ii++)
      { 
        cv::Point2d pr(points_2d[1][ii]);
        double dist_y = abs(pl.y - pr.y);
        if (dist_y <= epipolar_width_threshold_)
        {
          right_points.push_back(pr);
        }
      }

      // Compute the left epipolar distances
      std::vector<double> left_epipolar;
      for (size_t jj=0; jj<points_2d[0].size(); jj++)
      { 
        cv::Point2d pl2(points_2d[0][jj]);
        left_epipolar.push_back(abs(pl.y - pl2.y));
      }

      // Get the left points sorted by epipolar distance
      std::vector<cv::Point2d> left_points = points_2d[0];
      std::vector<cv::Point2d> left_sorted;
      unsigned int offset = 0;
      while(offset < right_points.size())
      { 
        std::vector<double>::iterator it = std::min_element(left_epipolar.begin(), left_epipolar.end());
        int min_idx = std::distance(left_epipolar.begin(), it);
        left_sorted.push_back(left_points[min_idx]);
        left_epipolar.erase(left_epipolar.begin() + min_idx);
        left_points.erase(left_points.begin() + min_idx);
        offset++;
      }
      left_points = left_sorted;

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
  * @return true if transform could be obtained, false otherwise.
  * \param 3D points of the valve.
  * \param output transformation.
  */
bool valve_tracker::ValveTracker::estimateTransform(
    std::vector<cv::Point3d> valve_3d_points, tf::Transform& affineTf)
{
  affineTf.setIdentity();

  // Sanity check
  if (valve_3d_points.size() != 3)
  {
    ROS_WARN_STREAM("[ValveTracker:] Impossible to estimate the transformation " << 
                    "between camera and valve, wrong 3d correspondences size: " <<
                    valve_3d_points.size());
    return false;
  }

  // Match the real valve points with the model
  std::vector<cv::Point3d> valve_target_points = matchTgtMdlPoints(valve_3d_points, false);

  // Compute the transformation
  affineTf = valve_tracker::Utils::affine3Dtransformation(valve_model_points_, valve_target_points);

  // Compute error
  double error = valve_tracker::Utils::getTfError(affineTf,
                                                  valve_model_points_,
                                                  valve_target_points);

  // Apply a threshold over the error
  if (error > max_tf_error_)
  {
    affineTf.setIdentity();
    ROS_WARN_STREAM("[ValveTracker:] Affine transformation error is too big: " << error);
    return false;
  }

  return true;
}

/** \brief Match the target points to the model
  * @return a vector with the 3D valve points re-sorted.
  * \param input vector of valve points.
  * \param true to inverse the valve symmetrical sides correspondences.
  */
std::vector<cv::Point3d> valve_tracker::ValveTracker::matchTgtMdlPoints(
  std::vector<cv::Point3d> points_3d, bool inverse)
{
  // Target point cloud
  std::vector<cv::Point3d> tgt;

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
  tgt.push_back(cv::Point3d( (points_3d[v[0]].x + points_3d[v[1]].x) / 2,
                             (points_3d[v[0]].y + points_3d[v[1]].y) / 2,
                             (points_3d[v[0]].z + points_3d[v[1]].z) / 2));

  // Symmetrical sides tracker indices
  int idx_track = 0;
  int idx_no_track = 1;

  if (valve_symmetric_point_.x == 0.0 &&
      valve_symmetric_point_.y == 0.0 &&
      valve_symmetric_point_.z == 0.0)
  {
    // Tracker not initialized
    if ((points_3d[v[0]].x < points_3d[v[1]].x) != inverse)
    {
      idx_track = 0;
      idx_no_track = 1;
    }
    else
    {
      idx_track = 1;
      idx_no_track = 0;
    }
  }
  else
  {
    // Tracker initialized, search the closest point
    double dist1 = valve_tracker::Utils::euclideanDist(valve_symmetric_point_ - points_3d[v[0]]);
    double dist2 = valve_tracker::Utils::euclideanDist(valve_symmetric_point_ - points_3d[v[1]]);
    
    if(dist1 <= dist2)
    {
      idx_track = 0;
      idx_no_track = 1;
    }
    else
    {
      idx_track = 1;
      idx_no_track = 0;
    }
  }

  // Mount the target points
  tgt.push_back(cv::Point3d( points_3d[v[idx_track]].x, points_3d[v[idx_track]].y, points_3d[v[idx_track]].z ));
  tgt.push_back(cv::Point3d( points_3d[v[idx_no_track]].x, points_3d[v[idx_no_track]].y, points_3d[v[idx_no_track]].z ));
  tgt.push_back(cv::Point3d( points_3d[idx_root].x, points_3d[idx_root].y, points_3d[idx_root].z ));

  // Set the trakcer
  valve_symmetric_point_ = tgt[1];

  return tgt;
}