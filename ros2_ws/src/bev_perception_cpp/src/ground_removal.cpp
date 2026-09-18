#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>

#include <pcl_conversions/pcl_conversions.h>

class GroundRemoval : public rclcpp::Node
{
public:
  GroundRemoval()
  : Node("ground_removal")
  {
    subscription_ =
      this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/lidar/points_filtered",
        10,
        std::bind(
          &GroundRemoval::callback,
          this,
          std::placeholders::_1
        )
      );

    publisher_ =
      this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/lidar/points_noground",
        10
      );
  }

private:
  void callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    auto cloud =
      std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();

    pcl::fromROSMsg(*msg, *cloud);

    pcl::SACSegmentation<pcl::PointXYZI> seg;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.2);

    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.empty())
    {
      RCLCPP_WARN(this->get_logger(), "No ground plane found");
      return;
    }

    auto no_ground =
      std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();

    pcl::ExtractIndices<pcl::PointXYZI> extract;

    extract.setInputCloud(cloud);
    extract.setIndices(inliers);
    extract.setNegative(true);

    extract.filter(*no_ground);

    sensor_msgs::msg::PointCloud2 output;
    pcl::toROSMsg(*no_ground, output);

    output.header = msg->header;

    publisher_->publish(output);

    RCLCPP_INFO(
      this->get_logger(),
      "Input: %zu | Ground: %zu | Remaining: %zu",
      cloud->size(),
      inliers->indices.size(),
      no_ground->size()
    );
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GroundRemoval>());
  rclcpp::shutdown();

  return 0;
}