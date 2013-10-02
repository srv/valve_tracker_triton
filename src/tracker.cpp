#include "valve_tracker/tracker.h"
#include "valve_tracker/utils.h"
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <tf/transform_broadcaster.h>
#include <Eigen/Eigen>
#include <numeric>
#include "opencv2/core/core.hpp"

/** \brief Tracker constructor
  * \param transport
  */
valve_tracker::Tracker::Tracker(cv::MatND trained_model, 
                                std::vector<cv::Point3d> valve_model, 
                                image_geometry::StereoCameraModel stereo_model, 
                                int epipolar_width_threshold) : StereoImageProcessor()
{
  // Setup the basic parameters
  trained_model_ = trained_model;
  valve_model_points_ = valve_model;
  stereo_model_ = stereo_model;
  epipolar_width_threshold_ = epipolar_width_threshold;
}
valve_tracker::Tracker::Tracker(const std::string transport) : StereoImageProcessor(transport)
{
  ROS_INFO_STREAM("[Tracker:] Instantiating the Valve Tracker...");

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
  nhp.param("max_rot_diff", max_rot_diff_, 0.175);
  nhp.param("max_trans_diff", max_trans_diff_, 0.05);
  nhp.param("trained_model_path", trained_model_path_, 
      valve_tracker::Utils::getPackageDir() + std::string("/etc/trained_model.yml"));
  nhp.param("show_debug", show_debug_, false);
  nhp.param("warning_on", warning_on_, false);
  nhp.param("tf_filter_size", tf_filter_size_, 0);
  nhp.param("listen_services", listen_services_, false);   

  ROS_INFO_STREAM("[Tracker:] Valve Tracker Settings:" << std::endl <<
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
                  "  max_rot_diff               = " << max_rot_diff_ << std::endl <<
                  "  max_trans_diff             = " << max_trans_diff_ << std::endl <<
                  "  tf_filter_size             = " << tf_filter_size_ << std::endl <<
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

  // Read the trained model
  cv::FileStorage fs(trained_model_path_, cv::FileStorage::READ);
  fs["model_histogram"] >> trained_model_;
  fs.release();

  // Image publisher for future debug
  image_transport::ImageTransport it(nhp);
  image_pub_  = it.advertise("image_detections", 1);

  // Services to start or stop the valve detection
  start_service_ = nhp.advertiseService("start_valve_detection", &Tracker::startDetection, this);
  stop_service_ = nhp.advertiseService("stop_valve_detection", &Tracker::stopDetection, this);

  if (listen_services_)
    do_detection_ = false;
  else
    do_detection_ = true;

  // Initialize the camera to valve transformation
  camera_to_valve_.setIdentity();

  // Initialize the symmetric point tracker
  valve_symmetric_point_ = cv::Point3d(0.0, 0.0, 0.0);

  // Set the gui names
  tuning_gui_name_ = "Valve Tracker Tuning";
}

/** \brief Show the current parameter set for console.
  */
void valve_tracker::Tracker::showParameterSet()
{
  ROS_INFO_STREAM("[Tracker:] Parameter set: [" <<
                  mean_filter_size_ << ", " <<
                  binary_threshold_ << ", " <<
                  closing_element_size_ << ", " <<
                  opening_element_size_ << ", " <<
                  min_value_threshold_ << ", " <<
                  min_blob_size_ << ", " <<
                  max_blob_size_ << "]");
}

/** \brief Set a parameter
  * \param name of the parameter
  * \param value of the parameter
  */
void valve_tracker::Tracker::setParameter(std::string param_name, int param_value)
{
  if (param_name == "mean_filter_size") 
    mean_filter_size_ = param_value;
  else if (param_name == "binary_threshold") 
    binary_threshold_ = param_value;
  else if (param_name == "min_value_threshold")
    min_value_threshold_ = param_value;
  else if (param_name == "closing_element_size")
    closing_element_size_ = param_value;
  else if (param_name == "opening_element_size") 
    opening_element_size_ = param_value;
  else if (param_name == "min_blob_size")
    min_blob_size_ = param_value;
  else if (param_name == "max_blob_size")
    max_blob_size_ = param_value;
}

/** \brief Stereo Image Callback
  * \param l_image_msg message of the left image
  * \param r_image_msg message of the right image
  * \param l_info_msg information message of the left image
  * \param r_info_msg information message of the right image
  */
void valve_tracker::Tracker::stereoImageCallback(
  const sensor_msgs::ImageConstPtr     & l_image_msg,
  const sensor_msgs::ImageConstPtr     & r_image_msg,
  const sensor_msgs::CameraInfoConstPtr& l_info_msg,
  const sensor_msgs::CameraInfoConstPtr& r_info_msg)
{

  // Check if service is called or not
  if (listen_services_ && !do_detection_)
  {
    ROS_INFO("[Tracker:] Waiting for start service...");
    return;
  }

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
    ROS_ERROR("[Tracker:] cv_bridge exception: %s", e.what());
    return;
  }

  // For debuging
  processed_ = l_cv_image_ptr->image;

  // Get the camera model
  stereo_model_.fromCameraInfo(l_info_msg, r_info_msg);

  // Detect valve in both images
  std::vector<int> l_contours_size, r_contours_size;
  std::vector<cv::Point2d> l_points_2d = valveDetection(l_cv_image_ptr->image, show_debug_, l_contours_size);
  std::vector<cv::Point2d> r_points_2d = valveDetection(r_cv_image_ptr->image, false, r_contours_size);

  // Triangulate the 3D points
  std::vector<cv::Point3d> points3d;
  points3d = triangulatePoints(l_points_2d, r_points_2d, l_contours_size, r_contours_size);

  // Proceed depending on the number of object points detected
  if(points3d.size() == 2)
  {
    // Get the non-basis point
    std::vector<double> tmp;
    tmp.push_back(points3d[0].y);
    tmp.push_back(points3d[1].y);
    int max_y_idx = std::max_element(tmp.begin(), tmp.end())-tmp.begin();

    // Valve is rotated 90º. Add an additional point
    cv::Point3d hidden_point(points3d[max_y_idx].x, points3d[max_y_idx].y, points3d[max_y_idx].z+0.08);
    points3d.push_back(hidden_point);
  }
  
  // All valve points are visible. Compute the transformation from camera to valve
  tf::Transform curr_camera_to_valve;
  double error = -1.0;
  bool success = estimateTransform(points3d, curr_camera_to_valve, error);
  if (success)
  {
    // Validate transform
    bool valid_tf = false;

    // 1) Compute transformation and angles
    tf::Vector3 prev_trans = camera_to_valve_.getOrigin();
    tf::Vector3 curr_trans = curr_camera_to_valve.getOrigin();
    double trans_diff = valve_tracker::Utils::euclideanDist(prev_trans - curr_trans);
    double prev_roll, prev_pitch, prev_yaw;
    double curr_roll, curr_pitch, curr_yaw;
    camera_to_valve_.getBasis().getRPY(prev_roll, prev_pitch, prev_yaw);
    curr_camera_to_valve.getBasis().getRPY(curr_roll, curr_pitch, curr_yaw);

    // Set current roll positive
    if (curr_roll < 0)
      curr_roll = -curr_roll;
    tf::Quaternion rot;
    rot.setRPY(curr_roll, curr_pitch, curr_yaw);
    curr_camera_to_valve.setRotation(rot);

    // 2) Check maximum roll angle
    if ( (curr_roll*180/M_PI) < 190.0 && (curr_roll*180/M_PI) > 0.0 &&
         (curr_yaw*180/M_PI) < 110.0 && (curr_yaw*180/M_PI) > -20.0)
    {
      valid_tf = true;
    }
    else if (warning_on_)
    {
      ROS_WARN_STREAM("[Tracker:] Roll or Yaw rotation does not fit (-20,+110). Roll: " << 
                      curr_roll*180/M_PI << "deg. Yaw: " << curr_yaw*180/M_PI);
    }

    // 3) Check distance between this transformation and previous
    tf::Transform ident;
    ident.setIdentity();
    if ((trans_diff < max_trans_diff_ &&
       fabs(prev_roll - curr_roll) < max_rot_diff_ &&
       fabs(prev_pitch - curr_pitch) < max_rot_diff_ &&
       fabs(prev_yaw - curr_yaw) < max_rot_diff_ && valid_tf) ||
       camera_to_valve_ == ident)
    {
      valid_tf = true;
    }
    else if (warning_on_)
    {
      ROS_WARN_STREAM("[Tracker:] Rotation or translation errors between current and previous " <<
                      "transformation are out of bounds: Trans: " << trans_diff << "m. Rot (RPY): " <<
                      fabs(prev_roll - curr_roll) << ", " << fabs(prev_pitch - curr_pitch) <<
                      ", " << fabs(prev_yaw - curr_yaw));
    }

    // Update tf if valid
    if (valid_tf)
      camera_to_valve_ = curr_camera_to_valve;
    else if (warning_on_)
      ROS_WARN("[Tracker:] No valid transform");
  }

  // Publish processed image
  if (image_pub_.getNumSubscribers() > 0)
  {
    cv_bridge::CvImagePtr cv_ptr(new cv_bridge::CvImage);
    cv_ptr->image    = processed_;
    cv_ptr->encoding = "mono8";
    image_pub_.publish(cv_ptr->toImageMsg());
  }

  // Filter tf
  double x, y, z, roll, pitch, yaw;
  if (tf_filter_size_ > 0)
  {
    // Push back to the filter
    camera_to_valve_.getBasis().getRPY(roll, pitch, yaw);
    tf_x_.push_back(camera_to_valve_.getOrigin().x());
    tf_y_.push_back(camera_to_valve_.getOrigin().y());
    tf_z_.push_back(camera_to_valve_.getOrigin().z());
    tf_roll_.push_back(roll);
    tf_pitch_.push_back(pitch);
    tf_yaw_.push_back(yaw);

    if (tf_x_.size() > (unsigned int)tf_filter_size_)
    {
      tf_x_.erase(tf_x_.begin(), tf_x_.begin() + 1);
      tf_y_.erase(tf_y_.begin(), tf_y_.begin() + 1);
      tf_z_.erase(tf_z_.begin(), tf_z_.begin() + 1);
      tf_roll_.erase(tf_roll_.begin(), tf_roll_.begin() + 1);
      tf_pitch_.erase(tf_pitch_.begin(), tf_pitch_.begin() + 1);
      tf_yaw_.erase(tf_yaw_.begin(), tf_yaw_.begin() + 1);
    }

    double x_mean = std::accumulate(tf_x_.begin(), tf_x_.end(), 0.0) / tf_x_.size();
    double y_mean = std::accumulate(tf_y_.begin(), tf_y_.end(), 0.0) / tf_y_.size();
    double z_mean = std::accumulate(tf_z_.begin(), tf_z_.end(), 0.0) / tf_z_.size();
    double roll_mean = std::accumulate(tf_roll_.begin(), tf_roll_.end(), 0.0) / tf_roll_.size();
    double pitch_mean = std::accumulate(tf_pitch_.begin(), tf_pitch_.end(), 0.0) / tf_pitch_.size();
    double yaw_mean = std::accumulate(tf_yaw_.begin(), tf_yaw_.end(), 0.0) / tf_yaw_.size();

    tf::Quaternion rotation;
    rotation.setRPY(roll_mean, pitch_mean, yaw_mean);
    tf::Vector3 trans(x_mean, y_mean, z_mean);
    camera_to_valve_.setRotation(rotation);
    camera_to_valve_.setOrigin(trans);
  }

  // Log
  camera_to_valve_.getBasis().getRPY(roll, pitch, yaw);
  x = camera_to_valve_.getOrigin().x();
  y = camera_to_valve_.getOrigin().y();
  z = camera_to_valve_.getOrigin().z();
  ROS_INFO_STREAM("Camera to valve: [" << x << ", " << y << ", " << z << 
                  ", " << roll << ", " << pitch << ", " << yaw << "]");

  // Publish last computed transform from camera to valve
  tf_broadcaster_.sendTransform(
      tf::StampedTransform(camera_to_valve_, l_image_msg->header.stamp,
      stereo_frame_id_, valve_frame_id_));

  // Publish an unrotated transform
  tf::Transform camera_to_valve_no_rot;
  camera_to_valve_no_rot.setIdentity();
  tf::Quaternion rot;
  rot.setRPY(0.0, 0.0, pitch);
  camera_to_valve_no_rot.setRotation(rot);
  tf_broadcaster_.sendTransform(
      tf::StampedTransform(camera_to_valve_no_rot, l_image_msg->header.stamp,
      valve_frame_id_, valve_frame_id_+"_no_rot"));
}

/** \brief Detect the valve into the image
  * @return vector with the detected valve points
  * \param image where the valve will be detected
  */
std::vector<cv::Point2d> valve_tracker::Tracker::valveDetection(cv::Mat image, bool debug)
{
  std::vector<int> contours_size;
  return valveDetection(image, debug, contours_size);
}
std::vector<cv::Point2d> valve_tracker::Tracker::valveDetection(cv::Mat image, bool debug, 
                                                                std::vector<int> &contours_size)
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

  // morphological operations
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

  // create mask for invalid values
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

  if (contours.size() < 2)
  {
    ROS_DEBUG_STREAM("[Tracker:] Not enough points detected: " << contours.size() << " (3 needed).");
  }
  else
  {
    // 2 or more blobs detected. Delete too big and too small blobs
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

    // Check that we keep having at least 2 contours
    if (contours.size() < 2)
    {
      ROS_DEBUG("[Tracker:] Blob filtering has removed too many blobs!");
    }
    else
    {
      // Sort the result by size
      std::sort(contours.begin(), contours.end(), valve_tracker::Utils::sort_vectors_by_size);

      // Get the 2-3 biggest blobs
      int max_blobs_num = 3;
      if (contours.size() == 2)
        max_blobs_num = 2;

      std::vector< std::vector<cv::Point> > contours_tmp(contours.begin(), contours.begin() + max_blobs_num);
      contours_filtered = contours_tmp;

      // Compute the blob centroids
      for (size_t i = 0; i < contours_filtered.size(); i++)
      {
        // Calculate contours size
        contours_size.push_back(contours_filtered[i].size());

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
  if (debug && !contour_image.empty())
  {
    for (size_t idx=0; idx<contours_filtered.size(); idx++)
    {
      cv::drawContours(contour_image, contours_filtered, idx, color, 2);
      cv::circle(contour_image, points[idx], 15, color, 2);
    }

    // Show images. First, convert to color
    cv::Mat binary_color, binary_morphed_color;
    cv::cvtColor(binary, binary_color, CV_GRAY2RGB);
    cv::cvtColor(binary_morphed, binary_morphed_color, CV_GRAY2RGB);

    // Concatenate horizontaly the images
    cv::Mat display_image(contour_image.size(), CV_8UC3);
    cv::hconcat(contour_image,binary_color,display_image);
    cv::hconcat(display_image,binary_morphed_color,display_image);

    // Create the window and the trackbars
    cv::namedWindow(tuning_gui_name_, 0);
    cv::setWindowProperty(tuning_gui_name_, CV_WND_PROP_ASPECTRATIO, CV_WINDOW_KEEPRATIO);
    cv::createTrackbar("mean_filter_size", tuning_gui_name_,  &mean_filter_size_, 255);
    cv::createTrackbar("binary_threshold", tuning_gui_name_,  &binary_threshold_, 255);
    cv::createTrackbar("closing_element_size", tuning_gui_name_,  &closing_element_size_, 255);
    cv::createTrackbar("opening_element_size", tuning_gui_name_,  &opening_element_size_, 255);
    cv::createTrackbar("min_value_threshold", tuning_gui_name_,  &min_value_threshold_, 255);
    cv::createTrackbar("min_blob_size", tuning_gui_name_,  &min_blob_size_, 255);
    cv::createTrackbar("max_blob_size", tuning_gui_name_,  &max_blob_size_, 255);

    // Get some reference values to position text in the image
    cv::Size sz = contour_image.size();
    int W = sz.width; // image width
    //int H = sz.height; // image height
    int scale = 2; // text scale
    int thickness = 4; // text font thickness
    int color = 255; // text colour (blue)
    int W0 = 10; // initial width 
    int H0 = 50; // initital height
    cv::putText(display_image, "Contours", cv::Point(W0,H0), cv::FONT_HERSHEY_SIMPLEX, scale, color, thickness);
    cv::putText(display_image, "Binarized", cv::Point(W+W0,H0), cv::FONT_HERSHEY_SIMPLEX, scale, color, thickness);
    cv::putText(display_image, "Morph", cv::Point(2*W+W0,H0), cv::FONT_HERSHEY_SIMPLEX, scale, color, thickness);
    cv::imshow(tuning_gui_name_, display_image);
    cv::waitKey(5);
  }
  
  return points;
}

/** \brief Triangulate the 3D points of the valve
  * @return vector with the 3D points of the valve
  * \param vector of left and right valve point detection
  */
std::vector<cv::Point3d> valve_tracker::Tracker::triangulatePoints(
    std::vector<cv::Point2d> l_points_2d, std::vector<cv::Point2d> r_points_2d,
    std::vector<int> l_contours_size, std::vector<int> r_contours_size)
{
  // Initialize output
  std::vector<cv::Point3d> points3d;

  // Handle the vertical valve case
  if (l_points_2d.size() == 2 && r_points_2d.size() == 3)
  {
    r_points_2d.erase(r_points_2d.begin() + r_points_2d.size() - 1);
    r_contours_size.erase(r_contours_size.begin() + r_contours_size.size() - 1);
  }
  else if (l_points_2d.size() == 3 && r_points_2d.size() == 2)
  {
    l_points_2d.erase(l_points_2d.begin() + l_points_2d.size() - 1);
    l_contours_size.erase(l_contours_size.begin() + l_contours_size.size() - 1);
  }

  // Sanity check
  if ((l_points_2d.size() == 2 && r_points_2d.size() == 2) ||
      (l_points_2d.size() == 3 && r_points_2d.size() == 3))
  {
    // De-offset x points
    std::vector<int> l_x, r_x;
    for (size_t n = 0; n < l_points_2d.size(); n++)
    {
      l_x.push_back(l_points_2d[n].x);
      r_x.push_back(r_points_2d[n].x);
    }
    int l_x_min = *std::min_element(l_x.begin(), l_x.end());
    int r_x_min = *std::min_element(r_x.begin(), r_x.end());

    // Loop through left points
    std::vector<int> matchings;
    for (size_t i = 0; i < l_points_2d.size(); i++)
    {
      cv::Point2d pl(l_points_2d[i]);

      // Initialize the measurements
      std::vector<int> epipolar_dist, x_dist, blobs_size_diff;

      // Loop through right points
      for (size_t j = 0; j < r_points_2d.size(); j++)
      {
        cv::Point2d pr(r_points_2d[j]);

        // 1) Epipolar distance
        epipolar_dist.push_back(abs(pl.y - pr.y));

        // 2) X distance
        x_dist.push_back(abs( (pl.x-l_x_min) - (pr.x-r_x_min) ));

        // 3) Blob size difference
        blobs_size_diff.push_back(abs(l_contours_size[i] - r_contours_size[j]));
      }

      // Normalize vectors
      int min_epipolar_dist = *std::min_element(epipolar_dist.begin(), epipolar_dist.end());
      int max_epipolar_dist = *std::max_element(epipolar_dist.begin(), epipolar_dist.end());
      int min_x_dist = *std::min_element(x_dist.begin(), x_dist.end());
      int max_x_dist = *std::max_element(x_dist.begin(), x_dist.end());
      int min_blobs_size_diff = *std::min_element(blobs_size_diff.begin() ,blobs_size_diff.end());
      int max_blobs_size_diff = *std::max_element(blobs_size_diff.begin(), blobs_size_diff.end());

      std::vector<float> normalized_errors;
      for (size_t k = 0; k < r_points_2d.size(); k++)
      {
        float a = (float)(epipolar_dist[k] - min_epipolar_dist) / (float)(max_epipolar_dist - min_epipolar_dist);
        float b = (float)(x_dist[k] - min_x_dist) / (float)(max_x_dist - min_x_dist);
        float c = (float)(blobs_size_diff[k] - min_blobs_size_diff) / (float)(max_blobs_size_diff - min_blobs_size_diff);
        normalized_errors.push_back(a+b+c);
      }

      // Get the best matching
      int best_matching = std::min_element(normalized_errors.begin(), normalized_errors.end()) - normalized_errors.begin();
      matchings.push_back(best_matching);
    }

    // Check if correspondences are solved
    bool correspondences_solved = false;
    if (l_points_2d.size() == 2)
    {
      if (std::find(matchings.begin(), matchings.end(), (int)0) != matchings.end() && 
        std::find(matchings.begin(), matchings.end(), (int)1) != matchings.end())
        correspondences_solved = true;
    }
    else
    {
      if (std::find(matchings.begin(), matchings.end(), (int)0) != matchings.end() && 
        std::find(matchings.begin(), matchings.end(), (int)1) != matchings.end() && 
        std::find(matchings.begin(), matchings.end(), (int)2) != matchings.end())
        correspondences_solved = true;
    }

    // Check correspondences
    if (correspondences_solved)
    {
      // Compute the 3d points
      for (size_t h = 0; h < l_points_2d.size(); h++)
      {
        cv::Point3d p3d;
        cv::Point2d pl(l_points_2d[h]);
        cv::Point2d pr(r_points_2d[matchings[h]]);
        stereo_model_.projectDisparityTo3d(pl, pl.x-pr.x, p3d);
        points3d.push_back(p3d);
        ROS_DEBUG("[Tracker:] Correspondence solved!");
      }    
    }
    else if (warning_on_)
    {
      ROS_WARN("[Tracker:] Impossible to solve correspondences!");
    }
  }
  else if (warning_on_)
  {
    ROS_WARN_STREAM("[Tracker:] Incorrect number of valve points found (" << 
                    l_points_2d.size() << " points) 2-3 needed.");
  }

  return points3d;
}

/** \brief Detect the valve into the image
  * @return true if transform could be obtained, false otherwise.
  * \param 3D points of the valve.
  * \param output transformation.
  */
bool valve_tracker::Tracker::estimateTransform(
    std::vector<cv::Point3d> valve_3d_points, tf::Transform& affineTf, double &error)
{
  affineTf.setIdentity();

  // Sanity check
  if (valve_3d_points.size() != 3)
    return false;

  // Match the real valve points with the model
  std::vector<cv::Point3d> valve_target_points = matchTgtMdlPoints(valve_3d_points, false);

  // Compute the transformation
  affineTf = valve_tracker::Utils::affine3Dtransformation(valve_model_points_, valve_target_points);

  // Compute error
  error = valve_tracker::Utils::getTfError(affineTf,
                                           valve_model_points_,
                                           valve_target_points);

  double roll, pitch, yaw;
  affineTf.getBasis().getRPY(roll, pitch, yaw);

  // Apply a threshold over the error
  if (error > max_tf_error_)
  {
    affineTf.setIdentity();
    if (warning_on_)
      ROS_WARN_STREAM("[Tracker:] Affine transformation error is too big: " << error);
    return false;
  }

  return true;
}

/** \brief Match the target points to the model
  * @return a vector with the 3D valve points re-sorted.
  * \param input vector of valve points.
  * \param true to inverse the valve symmetrical sides correspondences.
  */
std::vector<cv::Point3d> valve_tracker::Tracker::matchTgtMdlPoints(
  std::vector<cv::Point3d> points_3d, bool inverse)
{
  // Target point cloud
  std::vector<cv::Point3d> tgt;

  // The root point (where the valve touches the panel), can be located
  // as the point with minimum Y coordinate.
  std::vector<double> points3d_y;
  points3d_y.push_back(points_3d[0].y);
  points3d_y.push_back(points_3d[1].y);
  points3d_y.push_back(points_3d[2].y);
  int idx_root = std::min_element(points3d_y.begin(), points3d_y.end()) - points3d_y.begin();

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

  // Set the tracker
  valve_symmetric_point_ = tgt[1];

  // debug purposes
  if (show_debug_)
  {
    cv::namedWindow("Valve points debugger",0);
    cv::Mat I(processed_);
    std::vector<cv::Scalar> color;
    color.push_back(cv::Scalar(0,255,0));
    color.push_back(cv::Scalar(255,0,0));
    color.push_back(cv::Scalar(0,0,255));
    color.push_back(cv::Scalar(255,255,255));

    for (size_t idx=0; idx<tgt.size(); idx++)
    {
      cv::Point2d p = stereo_model_.left().project3dToPixel(tgt[idx]);
      cv::circle(I, p, 15, color[idx], 2);

      cv::Point2d q = stereo_model_.right().project3dToPixel(tgt[idx]);
      cv::circle(I, q, 15, color[idx], 2);
    }

    cv::imshow("Valve points debugger",I);
    cv::waitKey(5);
  }

  return tgt;
}

bool valve_tracker::Tracker::startDetection(std_srvs::Empty::Request&, std_srvs::Empty::Response&)
{
  do_detection_ = true;
  return true;
}

bool valve_tracker::Tracker::stopDetection(std_srvs::Empty::Request&, std_srvs::Empty::Response&)
{
  do_detection_ = false;
  return true;
}