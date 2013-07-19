#ifndef UTILS
#define UTILS

#include <string>
#include <ros/ros.h>
#include <ros/package.h>
#include <tf/transform_broadcaster.h>
#include <Eigen/Eigen>
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

    /** \brief Computes the mathematical transformation between 2 set of points
      * @return transformation matrix
      * \param input vector of real valve points.
      */
    static tf::Transform affine3Dtransformation(std::vector<cv::Point3d> pointcloud_1, 
                                                                                            std::vector<cv::Point3d> pointcloud_2)
    {
      // Compute centroids
      Eigen::Vector3f centroid_mdl(0.0, 0.0, 0.0);
      Eigen::Vector3f centroid_tgt(0.0, 0.0, 0.0);
      for (unsigned int i=0; i<pointcloud_1.size(); i++)
      {
        centroid_mdl(0) += pointcloud_1[i].x;
        centroid_mdl(1) += pointcloud_1[i].y;
        centroid_mdl(2) += pointcloud_1[i].z;
        centroid_tgt(0) += pointcloud_2[i].x;
        centroid_tgt(1) += pointcloud_2[i].y;
        centroid_tgt(2) += pointcloud_2[i].z;
      }
      centroid_mdl /= pointcloud_1.size();
      centroid_tgt /= pointcloud_2.size();

      // Compute H
      Eigen::Matrix3f H = Eigen::Matrix3f::Zero();
      for (unsigned int i=0; i<pointcloud_1.size(); i++)
      {
        Eigen::Vector3f p_mdl(pointcloud_1[i].x, pointcloud_1[i].y, pointcloud_1[i].z);
        Eigen::Vector3f p_tgt(pointcloud_2[i].x, pointcloud_2[i].y, pointcloud_2[i].z);

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
      tf::Transform tf_composed(rot, trans);

      return tf_composed;
    }

    /** \brief Computes the error of the affine transformation
      * @return the average error
      */
    static double getTfError(tf::Transform affineTf,
                                                     std::vector<cv::Point3d> pointcloud_1, 
                                                     std::vector<cv::Point3d> pointcloud_2)
    {
      double error = 0.0;
    for (unsigned int j=0; j<pointcloud_1.size(); j++)
    {
      tf::Vector3 p_mdl(pointcloud_1[j].x,
                        pointcloud_1[j].y,
                        pointcloud_1[j].z);
      tf::Vector3 p_tgt(pointcloud_2[j].x,
                        pointcloud_2[j].y,
                        pointcloud_2[j].z);
      tf::Vector3 p_computed = affineTf*p_mdl;
      error += valve_tracker::Utils::euclideanDist(p_computed - p_tgt);
    }
    // Average error
    error = error / pointcloud_1.size();

    return error;
    }

    /** \brief Calculates the backprojection of the input image given the histogram
      * @return the back-projected image
      * \param input image
      * \param input histogram
      */
    static cv::Mat calculateBackprojection(const cv::Mat& image,
                                           const cv::MatND& histogram)
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

      cv::Mat back_projection;
      int num_arrays = 1;
      cv::calcBackProject(&image, num_arrays, channels, histogram,
             back_projection, ranges_hsv);

      return back_projection;
    }

    /** \brief Recursive function to create the table of possible combinations
      * given an input vector of vectors.
      * \param vectors containing the posibles values for each parameter
      * \param index of the current iteration
      * \param the result
      */
    static void createCombinations(
                          const std::vector< std::vector<int> > &all_vecs, 
                          size_t vec_idx, std::vector<int> combinations,
                          std::vector< std::vector<int> > &result)
    {
      if (vec_idx >= all_vecs.size())
      {
        result.push_back(combinations);
        return;
      }
      
      for (size_t i=0; i<all_vecs[vec_idx].size(); i++)
      {
        std::vector<int> tmp = combinations;
        tmp.push_back(all_vecs[vec_idx][i]);
        valve_tracker::Utils::createCombinations(all_vecs, vec_idx+1, tmp, result);
      }
    }

  };
} // namespace

#endif // UTILS
