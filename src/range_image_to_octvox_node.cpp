#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/TransformStamped.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>
#include <tf2/exceptions.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "OctVoxMap/OctVoxMap.hpp"
#include "basic/alias.h"

namespace pcl_vae {

constexpr double kPi = 3.14159265358979323846;

class RangeImageToOctVoxNode {
 public:
  RangeImageToOctVoxNode()
      : pnh_("~"),
        tf_buffer_(),
        tf_listener_(tf_buffer_) {
    pnh_.param("h_fov", h_fov_, 2.0 * kPi);
    pnh_.param("v_fov", v_fov_, kPi / 6.0);
    pnh_.param("image_min_depth", min_depth_, 0.0);
    pnh_.param("image_max_depth", max_depth_, std::numeric_limits<double>::infinity());
    pnh_.param("invalid_pixel_value", invalid_pixel_value_, -1.0);
    pnh_.param("voxel_size", octvox_resolution_, 0.2);
    pnh_.param("octvox_resolution", octvox_resolution_, octvox_resolution_);
    pnh_.param("octvox_capacity", octvox_capacity_, static_cast<int>(1000000));
    pnh_.param("clear_before_insert", clear_before_insert_, true);
    pnh_.param("publish_backprojected_cloud", publish_backprojected_cloud_, true);
    pnh_.param("source_frame", source_frame_fallback_, std::string("sensor_frame"));
    pnh_.param("target_frame", target_frame_, std::string(""));
    pnh_.param("tf_timeout", tf_timeout_, 0.05);
    pnh_.param("queue_size", queue_size_, 1);

    octvox_ = std::make_shared<OctVoxMapType>(
        OctVoxMapType::Options{static_cast<float>(octvox_resolution_),
                               static_cast<std::size_t>(octvox_capacity_)});

    pub_octvox_cloud_ = pnh_.advertise<sensor_msgs::PointCloud2>(
        "output/octvox_cloud", queue_size_);
    pub_backprojected_cloud_ = pnh_.advertise<sensor_msgs::PointCloud2>(
        "output/backprojected_cloud", queue_size_);

    sub_range_image_ = pnh_.subscribe(
        "input/range_image", queue_size_, &RangeImageToOctVoxNode::rangeImageCallback, this);

    ROS_INFO_STREAM("[range_image_to_octvox_node] h_fov=" << h_fov_
                    << " v_fov=" << v_fov_
                    << " min_depth=" << min_depth_
                    << " max_depth=" << max_depth_
                    << " octvox_resolution=" << octvox_resolution_
                    << " octvox_capacity=" << octvox_capacity_
                    << " clear_before_insert=" << std::boolalpha << clear_before_insert_
                    << " target_frame='" << target_frame_ << "'");
  }

 private:
  using OctVoxMapType = LI2Sup::OctVoxMap<BASIC::V3, BASIC::scalar>;

  void rangeImageCallback(const sensor_msgs::ImageConstPtr& msg) {
    try {
      const cv_bridge::CvImageConstPtr cv_ptr =
          cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::TYPE_32FC1);

      const std::string source_frame = msg->header.frame_id.empty()
                                           ? source_frame_fallback_
                                           : msg->header.frame_id;
      std::string output_frame = source_frame;
      Eigen::Quaternionf rotation = Eigen::Quaternionf::Identity();
      Eigen::Vector3f translation = Eigen::Vector3f::Zero();

      if (!target_frame_.empty() && target_frame_ != source_frame) {
        if (!lookupTransform(*msg, source_frame, rotation, translation)) {
          return;
        }
        output_frame = target_frame_;
      } else if (!clear_before_insert_) {
        ROS_WARN_ONCE(
            "[range_image_to_octvox_node] Accumulating OctVox data without target_frame. "
            "This is only geometrically valid if all range images are already in one frame.");
      }

      BASIC::VV3 points_world;
      pcl::PointCloud<pcl::PointXYZI>::Ptr backprojected_cloud(
          new pcl::PointCloud<pcl::PointXYZI>());
      backprojectRangeImage(cv_ptr->image, rotation, translation, points_world,
                            backprojected_cloud);

      if (clear_before_insert_) {
        octvox_->clear();
      }
      octvox_->insert(points_world);

      if (publish_backprojected_cloud_) {
        publishCloud(backprojected_cloud, msg->header.stamp, output_frame,
                     pub_backprojected_cloud_);
      }
      publishOctVoxCloud(msg->header.stamp, output_frame);
    } catch (const cv_bridge::Exception& exc) {
      ROS_ERROR_STREAM("[range_image_to_octvox_node] Expected 32FC1 range image: "
                       << exc.what());
    } catch (const std::exception& exc) {
      ROS_ERROR_STREAM("[range_image_to_octvox_node] Failed to convert range image: "
                       << exc.what());
    }
  }

  bool lookupTransform(const sensor_msgs::Image& msg,
                       const std::string& source_frame,
                       Eigen::Quaternionf& rotation,
                       Eigen::Vector3f& translation) {
    try {
      const ros::Time lookup_stamp = msg.header.stamp.isZero() ? ros::Time(0) : msg.header.stamp;
      const geometry_msgs::TransformStamped transform =
          tf_buffer_.lookupTransform(target_frame_, source_frame, lookup_stamp,
                                     ros::Duration(tf_timeout_));
      const auto& q = transform.transform.rotation;
      const auto& t = transform.transform.translation;
      rotation = Eigen::Quaternionf(
          static_cast<float>(q.w), static_cast<float>(q.x),
          static_cast<float>(q.y), static_cast<float>(q.z));
      rotation.normalize();
      translation = Eigen::Vector3f(
          static_cast<float>(t.x), static_cast<float>(t.y), static_cast<float>(t.z));
      return true;
    } catch (const tf2::TransformException& exc) {
      ROS_WARN_STREAM_THROTTLE(
          2.0, "[range_image_to_octvox_node] Cannot transform range image from '"
                   << source_frame << "' to '" << target_frame_ << "': " << exc.what());
      return false;
    }
  }

  void backprojectRangeImage(const cv::Mat& range_image,
                             const Eigen::Quaternionf& rotation,
                             const Eigen::Vector3f& translation,
                             BASIC::VV3& points_world,
                             pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) const {
    const int height = range_image.rows;
    const int width = range_image.cols;
    points_world.reserve(static_cast<std::size_t>(height) * static_cast<std::size_t>(width));
    cloud->points.reserve(points_world.capacity());

    const float h_fov = static_cast<float>(h_fov_);
    const float v_fov = static_cast<float>(v_fov_);
    const float h_half = 0.5f * h_fov;
    const float v_half = 0.5f * v_fov;
    const float width_den = static_cast<float>(std::max(width - 1, 1));
    const float height_den = static_cast<float>(std::max(height - 1, 1));

    for (int v = 0; v < height; ++v) {
      const float vertical_angle =
          (static_cast<float>(v) / height_den) * v_fov - v_half;
      const float cos_vertical = std::cos(vertical_angle);
      const float sin_vertical = std::sin(vertical_angle);

      for (int u = 0; u < width; ++u) {
        const float range = range_image.at<float>(v, u);
        if (!isValidRange(range)) {
          continue;
        }

        const float horizontal_angle =
            (static_cast<float>(u) / width_den) * h_fov - h_half;
        Eigen::Vector3f point(
            range * cos_vertical * std::cos(horizontal_angle),
            -range * cos_vertical * std::sin(horizontal_angle),
            -range * sin_vertical);
        point = rotation * point + translation;

        points_world.emplace_back(point.x(), point.y(), point.z());

        pcl::PointXYZI pcl_point;
        pcl_point.x = point.x();
        pcl_point.y = point.y();
        pcl_point.z = point.z();
        pcl_point.intensity = range;
        cloud->points.push_back(pcl_point);
      }
    }

    cloud->width = static_cast<std::uint32_t>(cloud->points.size());
    cloud->height = 1;
    cloud->is_dense = true;
  }

  bool isValidRange(float range) const {
    if (!std::isfinite(range)) {
      return false;
    }
    if (range == static_cast<float>(invalid_pixel_value_)) {
      return false;
    }
    return range >= min_depth_ && range <= max_depth_;
  }

  void publishOctVoxCloud(const ros::Time& stamp, const std::string& frame_id) const {
    std::vector<float> octvox_points;
    octvox_->getMap(octvox_points);

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());
    cloud->points.reserve(octvox_points.size() / 3);

    for (std::size_t i = 0; i + 2 < octvox_points.size(); i += 3) {
      pcl::PointXYZI point;
      point.x = octvox_points[i];
      point.y = octvox_points[i + 1];
      point.z = octvox_points[i + 2];
      point.intensity = 1.0f;
      cloud->points.push_back(point);
    }

    publishCloud(cloud, stamp, frame_id, pub_octvox_cloud_);
  }

  void publishCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
                    const ros::Time& stamp,
                    const std::string& frame_id,
                    const ros::Publisher& publisher) const {
    cloud->width = static_cast<std::uint32_t>(cloud->points.size());
    cloud->height = 1;
    cloud->is_dense = true;

    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*cloud, cloud_msg);
    cloud_msg.header.stamp = stamp;
    cloud_msg.header.frame_id = frame_id;
    publisher.publish(cloud_msg);
  }

  ros::NodeHandle pnh_;
  ros::Subscriber sub_range_image_;
  ros::Publisher pub_octvox_cloud_;
  ros::Publisher pub_backprojected_cloud_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  std::shared_ptr<OctVoxMapType> octvox_;

  double h_fov_ = 2.0 * kPi;
  double v_fov_ = kPi / 6.0;
  double min_depth_ = 0.0;
  double max_depth_ = std::numeric_limits<double>::infinity();
  double invalid_pixel_value_ = -1.0;
  double octvox_resolution_ = 0.2;
  int octvox_capacity_ = 1000000;
  bool clear_before_insert_ = true;
  bool publish_backprojected_cloud_ = true;
  std::string source_frame_fallback_ = "sensor_frame";
  std::string target_frame_;
  double tf_timeout_ = 0.05;
  int queue_size_ = 1;
};

}  // namespace pcl_vae

int main(int argc, char** argv) {
  ros::init(argc, argv, "range_image_to_octvox_node");
  pcl_vae::RangeImageToOctVoxNode node;
  ros::spin();
  return 0;
}
