#include <algorithm>
#include <cv_bridge/cv_bridge.h>
#include "valve_tracker/trainer.h"
#include "valve_tracker/utils.h"

using namespace std;

/** \brief Trainer constructor
  * \param transport
  */
valve_tracker::Trainer::Trainer(const std::string transport) : StereoImageProcessor(transport)
{
  ROS_INFO_STREAM("[Trainer:] Instantiating the Valve Trainer...");

  // Get all the params out!
  ros::NodeHandle nhp("~");

  nhp.param("num_hue_bins", num_hue_bins_, 16);
  nhp.param("num_sat_bins", num_sat_bins_, 16);
  nhp.param("num_val_bins", num_val_bins_, 1);
  nhp.param("trained_model_path", trained_model_path_, 
      valve_tracker::Utils::getPackageDir() + std::string("/etc/trained_model.yml"));

  ROS_INFO_STREAM("[Trainer:] Valve Trainer Settings:" << std::endl <<
                  "  trained_model_path   = " << trained_model_path_ << std::endl <<
                  "  num_hue_bins         = " << num_hue_bins_ << std::endl <<
                  "  num_sat_bins         = " << num_sat_bins_ << std::endl <<
                  "  num_val_bins         = " << num_val_bins_ << std::endl);

  // first status: show live video
  training_status_ = DISPLAY_VIDEO;

  // OpenCV image windows for debugging
  cv::namedWindow("Training GUI", 0);
  cv::setMouseCallback("Training GUI", &valve_tracker::Trainer::staticMouseCallback, this);
}

valve_tracker::Trainer::~Trainer()
{
  cv::destroyWindow("Training GUI");
}

/** \brief Stereo Image Callback
  * \param l_image_msg message of the left image
  * \param r_image_msg message of the right image
  * \param l_info_msg information message of the left image
  * \param r_info_msg information message of the right image
  */
void valve_tracker::Trainer::stereoImageCallback(
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
    ROS_ERROR("[Trainer:] cv_bridge exception: %s", e.what());
    return;
  }

  image = l_cv_image_ptr->image.clone();

  // check where we are
  switch(training_status_)
  {
    case DISPLAY_VIDEO:
      // show live video
      ROS_INFO_ONCE("Click on the image to extract one frame.");
      break;
    case AWAITING_TRAINING_IMAGE:
      // save the image for later.
      training_image_ = image.clone();
      training_status_ = SHOWING_TRAINING_IMAGE;
      break;
    case SHOWING_TRAINING_IMAGE:
      // waiting user interaction
      image = training_image_.clone();
      break;
    case PAINTING:
      // undergoing user interaction. nothing to do
      ROS_INFO_ONCE("Release the mouse button whenever you need.");
      image = training_image_.clone();
      cv::rectangle(image, roi_rectangle_selection_, cv::Scalar(255,255,255),3);
      break;
    case ROI_SELECTED:
      // user interaction ended. 
      ROS_INFO_ONCE("ROI has been selected. Performing training...");
      model_histogram_ = train(training_image_);
      ROS_INFO_ONCE("Training done!");
      training_status_ = TRAINED;
      break;
    case TRAINED:
      // save the model
      ROS_INFO_STREAM_ONCE("Saving the model into: " << trained_model_path_);
      cv::FileStorage fs(trained_model_path_, cv::FileStorage::WRITE);
      fs << "model_histogram" << model_histogram_;
      fs.release();
      training_status_ = DISPLAY_VIDEO;
      break;
  }

  cv::imshow("Training GUI", image);
  cv::waitKey(5);
}

/** \brief Detect the valve into the image
  * \param image where the valve will be detected
  * \param type indicates if the image belongs to the left or right camera frame
  */
cv::MatND valve_tracker::Trainer::train(const cv::Mat& image)
{

  cv::Mat hsv_image(image.size(), CV_8UC3);
  cv::cvtColor(image, hsv_image, CV_BGR2HSV);

  const int bins_hsv[] = {num_hue_bins_, num_sat_bins_, num_val_bins_};

  // Calculate histogram of the target
  cv::Mat roi(hsv_image, roi_rectangle_selection_);
  ROS_INFO("Extracting histogram from target...");
  cv::MatND target_hist = calculateHistogram(roi, bins_hsv, cv::Mat());

  // Calculate the histogram of the background
  cv::Mat maskroi;
  maskroi = cv::Mat::zeros(hsv_image.size(), CV_8UC1); 
  cv::rectangle(maskroi, roi_rectangle_selection_, 255);
  ROS_INFO("Extracting histogram from background...");
  cv::MatND background_hist = calculateHistogram(hsv_image, bins_hsv, maskroi);

  //cv::MatND model_histogram = target_hist / background_hist * 255;
  cv::MatND model_histogram = target_hist;
  //cv::MatND model_histogram(target_hist.size(), CV_32FC3);
  float epsilon = 1e-6;

  for( int h = 0; h < num_hue_bins_; h++ ){
    for( int s = 0; s < num_sat_bins_; s++ ){
      for( int v = 0; v < num_val_bins_; v++ ){
        float target_value = target_hist.at<float>(h,s,v);
        float background_value = background_hist.at<float>(h,s,v);
        if (target_value > epsilon){
          if (background_value > epsilon){
            model_histogram.at<float>(h,s,v) = target_value / background_value;
          }else{
            model_histogram.at<float>(h,s,v) = target_value;
          }
        }else{
          model_histogram.at<float>(h,s,v) = 0;
        }
      }
    }
  }

  cv::namedWindow("valve-HS-histogram", 0);
  cv::namedWindow("valve-HV-histogram", 0);
  showHSVHistogram(model_histogram,"valve-HS-histogram","valve-HV-histogram");

  //TODO: set small saturations and values to zero
  //TODO: keep only the upper region of the sat and val channels. we want clear and "almost" sharp colors

  return model_histogram;
}

/** \brief static function version of mouseCallback
  * \param mouse event (left button down/up...)
  * \param x mouse position
  * \param y mouse position
  * \param flags not used
  * \param input params. Need to be correctly re-caster to work
  */
void valve_tracker::Trainer::staticMouseCallback(int event, int x, int y, int flags, void* param)
{
  // extract this pointer and call function on object
  Trainer* vt = reinterpret_cast<Trainer*>(param);
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
void valve_tracker::Trainer::mouseCallback( int event, int x, int y, int flags, void* param)
{
  // check in which training status we are
  switch( training_status_ )
  {
    case DISPLAY_VIDEO:
      // if we press on the image, freeze it
      if (event == CV_EVENT_LBUTTONDOWN)
      {
        training_status_ = AWAITING_TRAINING_IMAGE; 
        ROS_INFO("[Trainer:] Training image selected! Please draw a rectangle on the image.");
      }
      break;
    case SHOWING_TRAINING_IMAGE:
      // if we press the left button means we're going to select the ROI.
      if (event == CV_EVENT_LBUTTONDOWN)
      {
        roi_rectangle_origin_ = cv::Point(x,y);
        roi_rectangle_selection_ = cv::Rect(x,y,0,0);
        training_status_ = PAINTING;
      }
      break;
    case PAINTING:

      // animate the growing rectangle
      roi_rectangle_selection_.x = std::min(x, roi_rectangle_origin_.x);
      roi_rectangle_selection_.y = std::min(y, roi_rectangle_origin_.y);
      roi_rectangle_selection_.width = std::abs(x - roi_rectangle_origin_.x);
      roi_rectangle_selection_.height = std::abs(y - roi_rectangle_origin_.y);
      roi_rectangle_selection_ &= cv::Rect(0, 0, image.cols, image.rows);

      // if we release the button means we've ended selecting the ROI.
      if (event == CV_EVENT_LBUTTONUP)
      {
        if (roi_rectangle_selection_.width > 0 && roi_rectangle_selection_.height > 0)
          training_status_ = ROI_SELECTED;
      }
      break;
  }

}

/** \brief Histogram calculation in HSV colorspace
  * \param input image 
  * \param number of bins where to classify the histogram
  * \param mask of the selected ROI
  */
cv::MatND valve_tracker::Trainer::calculateHistogram(const cv::Mat& image, 
                                                          const int bins[],  
                                                          const cv::Mat& mask)
{   
  // we assume that the image is a regular three channel image
  CV_Assert(image.type() == CV_8UC3);

  // channels for wich to compute the histogram (H, S and V)
  int channels[] = {0, 1, 2};

  // Ranges for the histogram
  float hue_ranges[] = {0, 180}; 
  float saturation_ranges[] = {0, 256};
  float value_ranges[] = {0, 256};
  const float* ranges_hsv[] = {hue_ranges, saturation_ranges, value_ranges};

  cv::MatND histogram;

  // calculation
  int num_arrays = 1;
  int dimensions = 3;
  cv::calcHist(&image, num_arrays, channels, mask, histogram, dimensions, bins, ranges_hsv);
  
  return histogram;
}

/** \brief Histogram plot in OpenCV namedWindow
  * \param input histogram 
  * \param name of the HS image window
  * \param name of the HV image window
  */
void valve_tracker::Trainer::showHSVHistogram(const cv::MatND& histogram,
                                                   const std::string& name_hs, 
                                                   const std::string& name_hv)
{
  int num_hue_bins = histogram.size[0];
  int num_saturation_bins = histogram.size[1];
  int num_value_bins = histogram.size[2];
  
  // find the maximum value to properly scale the histogram
  float max_val = 0;
  for( int h = 1; h < num_hue_bins + 1; h++ ){
    for( int s = 1; s < num_saturation_bins + 1; s++ ){
      for( int v = 1; v < num_value_bins + 1; v++ ){
        float val = histogram.at<float>(h,s,v);
        if (val>max_val && val<=255) max_val = val;
      }
    }
  }

  int scale = 4; // square size
  cv::Mat histogram_image_hs = cv::Mat::zeros((num_saturation_bins + 1) * scale, (num_hue_bins + 1) * scale, CV_8UC3);
  cv::Mat histogram_image_hv = cv::Mat::zeros((num_value_bins + 1) * scale, (num_hue_bins + 1) * scale, CV_8UC3);

  // X axis
  for( int h = 1; h < num_hue_bins + 1; h++ )
    cv::rectangle( histogram_image_hs, cv::Point(h*scale, 0),
                   cv::Point((h + 1)*scale - 1, scale - 1),
                   cv::Scalar(1.0 * (h - 1) / num_hue_bins * 180.0, 255, 255, 0),
                   CV_FILLED);

  // Y axis
  for( int s = 1; s < num_saturation_bins + 1; s++ )
    cv::rectangle( histogram_image_hs, cv::Point(0, s * scale),
                   cv::Point(scale - 1, (s + 1)*scale - 1),
                   cv::Scalar(180, 1.0 * (s - 1) / num_saturation_bins * 255.0, 255, 0),
                   CV_FILLED);
  // second X axis
  for( int h = 1; h < num_hue_bins + 1; h++ )
    cv::rectangle( histogram_image_hv, cv::Point(h*scale, 0),
                   cv::Point((h + 1)*scale - 1, scale - 1),
                   cv::Scalar(1.0 * (h - 1) / num_hue_bins * 180.0, 255, 255, 0),
                   CV_FILLED);

  // second Y axis
  for( int v = 1; v < num_value_bins + 1; v++ )
    cv::rectangle( histogram_image_hv, cv::Point(0, v * scale),
                   cv::Point(scale - 1, (v + 1)*scale - 1),
                   cv::Scalar(180, 255, 1.0 * (v - 1) / num_value_bins * 255.0,  0),
                   CV_FILLED);

  cv::Mat histogram_hs, histogram_hv;
  cv::cvtColor(histogram_image_hs, histogram_hs, CV_HSV2BGR);
  cv::cvtColor(histogram_image_hv, histogram_hv, CV_HSV2BGR);

  for( int h = 1; h < num_hue_bins + 1; h++ )
    for( int s = 1; s < num_saturation_bins + 1; s++ )
    {
      float binVal = 0;
      for( int v = 1; v < num_value_bins + 1; v++ )
        binVal += (float)histogram.at<float>(h - 1, s - 1, v - 1);
      binVal /= num_value_bins;
      int intensity = (int)(binVal * 255 / max_val);
      cv::rectangle( histogram_hs, cv::Point(h*scale, s*scale),
                     cv::Point( (h+1)*scale - 1, (s+1)*scale - 1),
                     cv::Scalar::all(intensity),
                     CV_FILLED );
     }
  for( int h = 1; h < num_hue_bins + 1; h++ )
    for( int v = 1; v < num_value_bins + 1; v++ )
    {
      float binVal = 0;
      for( int s = 1; s < num_saturation_bins + 1; s++ )
        binVal += (float)histogram.at<float>(h - 1, s - 1, v - 1);
      binVal /= num_saturation_bins;
      int intensity = cvRound(binVal * 255 / max_val);
      cv::rectangle( histogram_hv, cv::Point(h*scale, v*scale),
                     cv::Point( (h+1)*scale - 1, (v+1)*scale - 1),
                     cv::Scalar::all(intensity),
                     CV_FILLED );
     }
  cv::imshow( name_hs, histogram_hs );
  cv::imshow( name_hv, histogram_hv );
  cv::waitKey(5);
}