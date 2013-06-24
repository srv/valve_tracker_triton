#ifndef UTILS
#define UTILS

#include <string>
#include <ros/ros.h>
#include <ros/package.h>
#include <tf/transform_broadcaster.h>
#include "opencv2/core/core.hpp"

namespace valve_tracker
{

	class Utils
	{

	public:

	  /** \brief Show a tf::Transform in the command line
	    * @return 
	    * \param input is the tf::Transform to be shown
	    */
		static void showTf(tf::Transform input)
		{
			tf::Vector3 tran = input.getOrigin();
		  tf::Matrix3x3 rot = input.getBasis();
		  tf::Vector3 r0 = rot.getRow(0);
		  tf::Vector3 r1 = rot.getRow(1);
		  tf::Vector3 r2 = rot.getRow(2);
		  ROS_INFO_STREAM("[ValveTracker:]\n" << r0.x() << ", " << r0.y() << ", " << r0.z() << ", " << tran.x() <<
		                  "\n" << r1.x() << ", " << r1.y() << ", " << r1.z() << ", " << tran.y() <<
		                  "\n" << r2.x() << ", " << r2.y() << ", " << r2.z() << ", " << tran.z());
		}

		/** \brief Creates an opencv circle element.
		  * @return the circle element
		  * \param element_size is radius of the circle (in pixels).
		  */
		static cv::Mat createElement(int element_size)
		{
		  cv::Mat element = cv::Mat::zeros(element_size, element_size, CV_8UC1);
		  cv::circle(element, cv::Point(element_size / 2, element_size / 2), element_size / 2, cv::Scalar(255), -1);
		  return element;
		}

		/** \brief Computes the euclidean distance of a point
		  * @return point distance
		  * \param 3D point
		  */
		static double euclideanDist(cv::Point3d point)
		{
		  return sqrt(point.x*point.x + point.y*point.y + point.z*point.z);
		}

		/** \brief Converts a cv::Mat to tf::Transform
		  * @return the corresponding transform
		  * \param input is the cv::Mat to be converted
		  */
		static tf::Transform cvmat2tf(cv::Mat input)
		{
		  tf::Matrix3x3 rot(input.at<double>(0,0),
		                    input.at<double>(0,1),
		                    input.at<double>(0,2),
		                    input.at<double>(1,0),
		                    input.at<double>(1,1),
		                    input.at<double>(1,2),
		                    input.at<double>(2,0),
		                    input.at<double>(2,1),
		                    input.at<double>(2,2));
		  tf::Vector3 trans(input.at<double>(0,3), 
		                    input.at<double>(1,3), 
		                    input.at<double>(2,3));
		  tf::Transform output(rot, trans);
		  return output;   
		}

		/** \brief Sort 2 points
		  * @return true if point 1 is smaller than point 2
		  * \param point 1
		  * \param point 2
		  */
		static bool sort_points_x(const cv::Point2d& p1, const cv::Point2d& p2)
		{
			return (p1.x < p2.x);
		}

		/** \brief Sort 2 vectors by size
		  * @return true if vector 1 is smaller than vector 2
		  * \param vector 1
		  * \param vector 2
		  */
		static bool sort_vectors_by_size(const std::vector<cv::Point>& v1, const std::vector<cv::Point>& v2)
		{
			return (v1.size() > v2.size());
		}

		/** \brief Get the directory of the package
		  * @return string with the path of the ros package
		  */
		static std::string getPackageDir()
		{
		  return ros::package::getPath(ROS_PACKAGE_NAME);
		}

	};
} // namespace

#endif // UTILS
