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

		/** \brief Computes the euclidean distance of a point
		  * @return point distance
		  * \param 3D point
		  */
		static double euclideanDist(cv::Point3d point)
		{
		  return sqrt(point.x*point.x + point.y*point.y + point.z*point.z);
		}
		static double euclideanDist(tf::Vector3 point)
		{
		  return sqrt(point[0]*point[0] + point[1]*point[1] + point[2]*point[2]);
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
