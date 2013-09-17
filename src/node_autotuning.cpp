#include "valve_tracker/tracker.h"
#include "valve_tracker/utils.h"
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <tf/transform_broadcaster.h>
#include <Eigen/Eigen>
#include <numeric>
#include "opencv2/core/core.hpp"
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <image_transport/subscriber_filter.h>
#include <image_geometry/stereo_camera_model.h>

namespace valve_tracker
{

class NodeAutotuning
{

public:

  /** \brief Autotuning constructor
    * \param transport
    */
  NodeAutotuning(ros::NodeHandle nh, ros::NodeHandle nhp) : nh_(nh), nh_priv_(nhp)
  {
    ROS_INFO_STREAM("[NodeAutotuning:] Instantiating the Autotuning node...");

    // Topics subscriptions
    std::string left_topic, right_topic, left_info_topic, right_info_topic;
    nhp.param("left_topic", left_topic, std::string("/left/image_rect_color"));
    nhp.param("right_topic", right_topic, std::string("/right/image_rect_color"));
    nhp.param("left_info_topic", left_info_topic, std::string("/left/camera_info"));
    nhp.param("right_info_topic", right_info_topic, std::string("/right/camera_info"));
    image_transport::ImageTransport it(nh);
    left_sub_ .subscribe(it, left_topic, 1);
    right_sub_.subscribe(it, right_topic, 1);
    left_info_sub_.subscribe(nh, left_info_topic, 1);
    right_info_sub_.subscribe(nh, right_info_topic, 1);

    // Callback syncronization
    exact_sync_.reset(new ExactSync(ExactPolicy(2),
                                    left_sub_, 
                                    right_sub_, 
                                    left_info_sub_, 
                                    right_info_sub_) );
    exact_sync_->registerCallback(boost::bind(&valve_tracker::NodeAutotuning::msgsCallback, 
                                              this, _1, _2, _3, _4));

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
    nhp.param("trained_model_path", trained_model_path_, 
      valve_tracker::Utils::getPackageDir() + std::string("/etc/trained_model.yml"));
    cv::FileStorage fs(trained_model_path_, cv::FileStorage::READ);
    fs["model_histogram"] >> trained_model_;
    fs.release();

    // Read the tuning intervals for the parameters
    XmlRpc::XmlRpcValue param_list;
    nhp.getParam("tuning_parameters", param_list);
    ROS_ASSERT(param_list.getType() == XmlRpc::XmlRpcValue::TypeArray);
    for (int i=0; i<param_list.size(); i++)
    {
      ROS_ASSERT(param_list[i].getType() == XmlRpc::XmlRpcValue::TypeStruct);
      parameter_names_.push_back(static_cast<std::string>(param_list[i]["name"]).c_str());
      int value = (int)param_list[i]["min_value"];
      int max_value = (int)param_list[i]["max_value"];
      int step = (int)param_list[i]["step"];
      std::vector<int> param_iters;
      while(value <= max_value)
      {
        param_iters.push_back(value);
        value += step;
      }
      parameters_table_.push_back(param_iters);
    }

    // Some other parameters
    nhp.param("epipolar_width_threshold", epipolar_width_threshold_, 3);
    nhp.param("maximum_allowed_error", maximum_allowed_error_, 0.1); 
    nhp.param("save_tuned_parameters", save_tuned_parameters_, true);

    // Create the table of all posible combinations
    std::vector<int> combinations;
    std::vector< std::vector<int> > result;
    valve_tracker::Utils::createCombinations(parameters_table_, 0, combinations, result);
    parameters_table_ = result;

    // Window to select the image for tuning
    selector_gui_name_ = "Select Image for Tuning";
    cv::namedWindow(selector_gui_name_, 0);
    cv::setWindowProperty(selector_gui_name_, CV_WND_PROP_ASPECTRATIO, CV_WINDOW_KEEPRATIO);
    cv::setMouseCallback(selector_gui_name_, &valve_tracker::NodeAutotuning::staticMouseCallback, this);

    // Initialize autotuning
    image_selected_ = false;
  }

  /** \brief Messages callback. This function is called when syncronized images are received.
    */
  void msgsCallback(const sensor_msgs::ImageConstPtr& l_img,
                    const sensor_msgs::ImageConstPtr& r_img,
                    const sensor_msgs::CameraInfoConstPtr& l_info,
                    const sensor_msgs::CameraInfoConstPtr& r_info)
  {
    // When image for tuning have been selected do nothing.
    if(image_selected_) return;

    // Get the camera model
    stereo_model_.fromCameraInfo(l_info, r_info);

    // Images to opencv
    cv_bridge::CvImagePtr l_cv_image_ptr;
    cv_bridge::CvImagePtr r_cv_image_ptr;
    try
    {
      l_cv_image_ptr = cv_bridge::toCvCopy(l_img,
                                           sensor_msgs::image_encodings::BGR8);
      r_cv_image_ptr = cv_bridge::toCvCopy(r_img,
                                           sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("[NodeAutotuning:] cv_bridge exception: %s", e.what());
      return;
    }

    l_image_ = l_cv_image_ptr->image.clone();
    r_image_ = r_cv_image_ptr->image.clone();

    // Show the image and wait until click
    cv::imshow(selector_gui_name_, l_image_);
    cv::waitKey(5);
  }

  /** \brief static function version of mouseCallback
    * \param mouse event (left button down/up...)
    * \param x mouse position
    * \param y mouse position
    * \param flags not used
    * \param input params. Need to be correctly re-caster to work
    */
  static void staticMouseCallback(int event, int x, int y, int flags, void* param)
  {
    // extract this pointer and call function on object
    NodeAutotuning* vt = reinterpret_cast<NodeAutotuning*>(param);
    assert(vt != NULL);
    vt->mouseCallback(event, x, y, flags, 0);
  }

  /** \brief Mouse callback for training
    * \param mouse event (left button down/up...)
    * \param x mouse position
    * \param y mouse position
    * \param flags not used
    * \param input params. Need to be correctly re-caster to work
    */
  void mouseCallback( int event, int x, int y, int flags, void* param)
  {
    // if we press on the image, freeze it
    if (event == CV_EVENT_LBUTTONDOWN)
    {
      ROS_INFO("[NodeAutotuning:] Tuning image selected.");
      image_selected_ = true;

      // Close window
      cv::destroyWindow(selector_gui_name_);

      // Launch the autotuning process in other thread
      boost::thread autoTuningThread(&valve_tracker::NodeAutotuning::autotuning, this);
    }
  }

  /** \brief Autotuning process
    */
  void autotuning()
  {
    // Create new instance of valve detection
    valve_tracker::Tracker tracker(trained_model_, 
                                   valve_model_points_, 
                                   stereo_model_, 
                                   epipolar_width_threshold_);

    // Loop through the table of combinations
    for (size_t i=0; i<parameters_table_.size(); i++)
    {
      // Sanity check
      ROS_ASSERT(parameters_table_[i].size() == parameter_names_.size());

      // Loop through every parameter for this combination
      for (size_t j=0; j<parameters_table_[i].size(); j++)
        tracker.setParameter(parameter_names_[j], parameters_table_[i][j]);

      // Log
      tracker.showParameterSet();

      // Detect the valve
      std::vector<int> l_contours_size, r_contours_size;   
      std::vector<cv::Point2d> l_points_2d = tracker.valveDetection(l_image_, true, l_contours_size);
      std::vector<cv::Point2d> r_points_2d = tracker.valveDetection(r_image_, false, r_contours_size);

      // Compute total contours size
      int mean_contours = 0;
      mean_contours = abs((std::accumulate(l_contours_size.begin(), l_contours_size.end(), 0) + 
                           std::accumulate(r_contours_size.begin(), r_contours_size.end(), 0))/2);
      contours_size_list_.push_back(mean_contours);

      if ((l_points_2d.size() == 2 && r_points_2d.size() == 2) ||
         (l_points_2d.size() == 3 && r_points_2d.size() == 3))
      {
        // Triangulate the 3D points
        std::vector<cv::Point3d> points3d;
        points3d = tracker.triangulatePoints(l_points_2d, r_points_2d, l_contours_size, r_contours_size);

        // Proceed depending on the number of object points detected
        if(points3d.size() == 2)
        {
          // Get the non-basis point
          std::vector<double> tmp;
          tmp.push_back(points3d[0].y);
          tmp.push_back(points3d[1].y);
          int max_y_idx = std::max_element(tmp.begin(), tmp.end())-tmp.begin();

          // Valve is rotated 90º. Add an additional point
          cv::Point3d hidden_point(points3d[max_y_idx].x, points3d[max_y_idx].y+0.08, points3d[max_y_idx].z);
          points3d.push_back(hidden_point);
        }

        // Initialize errors
        double error = std::numeric_limits<double>::max();
        
        // Compute the transformation from camera to valve
        tf::Transform camera_to_valve_tmp;
        tracker.estimateTransform(points3d, camera_to_valve_tmp, error);

        ROS_INFO_STREAM("[NodeAutotuning:] Blob contours size: " << mean_contours);
        ROS_INFO_STREAM("[NodeAutotuning:] Transformation error: " << error);
        error_list_.push_back(error);
      }      
    }

    // Give me all the indices with minimum errors
    std::vector<int> contours_list_red_;
    std::vector<int> contours_list_red_idx;
    for (size_t i=0; i<error_list_.size(); i++)
    {
      if (error_list_[i] < maximum_allowed_error_)
      {
        contours_list_red_.push_back(contours_size_list_[i]);
        contours_list_red_idx.push_back(i);
      }
    }

    if (contours_list_red_.size() == 0)
    {
      ROS_INFO_STREAM("[NodeAutotuning:] Autotuning could not find a set of parameters " <<
                      "to achive an error smaller than " << maximum_allowed_error_);
      return;
    }

    // Find max contour element idx
    int max_idx = std::max_element(contours_list_red_.begin(), contours_list_red_.end()) - contours_list_red_.begin();
    int best_idx = contours_list_red_idx[max_idx];

    // Show the images with the best set of parameters
    for (size_t i=0; i<parameter_names_.size(); i++)
      tracker.setParameter(parameter_names_[i], parameters_table_[best_idx][i]);
    tracker.valveDetection(l_image_, true);

    // Show the best value
    ROS_INFO("[NodeAutotuning:] **********************************************");
    ROS_INFO_STREAM("[NodeAutotuning:] Mimimum error achieved is: " << error_list_[best_idx]);
    ROS_INFO_STREAM("[NodeAutotuning:] With a countour size of: " << contours_size_list_[best_idx]);
    ROS_INFO("[NodeAutotuning:] Using the parameter set:");
    for (size_t i=0; i<parameter_names_.size(); i++)
    {
      ROS_INFO_STREAM("[NodeAutotuning:]   - " << parameter_names_[i] << ": " << 
                      parameters_table_[best_idx][i]);
    }
    ROS_INFO("[NodeAutotuning:] The result for this parameter set is displayed in the GUI.");

    if(save_tuned_parameters_)
    {
      for (size_t i=0; i<parameter_names_.size(); i++)
      {
        nh_.setParam("/valve_tracker/" + parameter_names_[i], parameters_table_[best_idx][i]);
      }
      ROS_INFO("[NodeAutotuning:] The best parameter set has been saved into parameter server.");
    }

    ROS_INFO("[NodeAutotuning:] **********************************************");
  }

protected:

    // Node handlers
    ros::NodeHandle nh_;
    ros::NodeHandle nh_priv_;

private:

  cv::Mat l_image_; //!> Showing image.
  cv::Mat r_image_; //!> Showing image.

  std::string trained_model_path_;                  //!> String for the path of trained model.
  std::vector< std::vector<int> > parameters_table_;//!> Table of possible combinations to be tuned.
  std::vector<std::string> parameter_names_;        //!> List of names to be tuned.
  std::string selector_gui_name_;                   //!> Name for the GUI of the selector window.
  std::string tuning_gui_name_;                     //!> Name for the GUI of the valve detection.
  bool image_selected_;                             //!> True when image for tuning have been selected.
  cv::MatND trained_model_;                         //!> Trained model.
  std::vector<cv::Point3d> valve_model_points_;     //!> 3D synthetic valve points.
  image_geometry::StereoCameraModel stereo_model_;  //!> Camera model to compute the 3d world points.
  std::vector<double> error_list_;                  //!> List of errors for every combination.
  std::vector<double> contours_size_list_;          //!> List of contours size for every combination.
  int epipolar_width_threshold_;                    //!> For epipolar threshold filtering.
  double maximum_allowed_error_;                    //!> Maximum allowed error.
  bool save_tuned_parameters_;                      //!> To save the tuned parameters into the parameters server.

  // Topic properties
  image_transport::SubscriberFilter left_sub_, right_sub_;
  message_filters::Subscriber<sensor_msgs::CameraInfo> left_info_sub_, right_info_sub_;

  // Topic sync properties
  typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, 
                                                    sensor_msgs::Image, 
                                                    sensor_msgs::CameraInfo, 
                                                    sensor_msgs::CameraInfo> ExactPolicy;
  typedef message_filters::Synchronizer<ExactPolicy> ExactSync;
  boost::shared_ptr<ExactSync> exact_sync_;

};

} // namespace

int main(int argc, char** argv)
{
  ros::init(argc, argv, "node_autotuning");

  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");
  valve_tracker::NodeAutotuning autotuning(nh,nh_private);

  ros::spin();
  return 0;
}

