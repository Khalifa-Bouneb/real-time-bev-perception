#include <memory>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/common.h>

#include <pcl_conversions/pcl_conversions.h>

class LidarClustering : public rclcpp::Node
{
public:
  LidarClustering()
  : Node("lidar_clustering")
  {
    subscription_ =
      this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/lidar/points_noground",
        10,
        std::bind(
          &LidarClustering::callback,
          this,
          std::placeholders::_1
        )
      );

    marker_publisher_ =
      this->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/lidar/cluster_boxes",
        10
      );
  }

private:
  void callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    auto cloud =
      std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();

    pcl::fromROSMsg(*msg, *cloud);

    if (cloud->empty())
    {
      RCLCPP_WARN(this->get_logger(), "Received empty point cloud");
      return;
    }

    auto tree =
      std::make_shared<pcl::search::KdTree<pcl::PointXYZI>>();

    tree->setInputCloud(cloud);

    std::vector<pcl::PointIndices> cluster_indices;

    pcl::EuclideanClusterExtraction<pcl::PointXYZI> clustering;

    clustering.setClusterTolerance(0.5);
    clustering.setMinClusterSize(20);
    clustering.setMaxClusterSize(5000);

    clustering.setSearchMethod(tree);
    clustering.setInputCloud(cloud);

    clustering.extract(cluster_indices);

    visualization_msgs::msg::MarkerArray marker_array;

    int marker_id = 0;
    int valid_clusters = 0;

    for (const auto & cluster : cluster_indices)
    {
      pcl::PointCloud<pcl::PointXYZI> cluster_cloud;

      cluster_cloud.reserve(cluster.indices.size());

      for (const auto & index : cluster.indices)
      {
        cluster_cloud.push_back((*cloud)[index]);
      }

      pcl::PointXYZI min_point;
      pcl::PointXYZI max_point;

      pcl::getMinMax3D(
        cluster_cloud,
        min_point,
        max_point
      );

      const float dx = max_point.x - min_point.x;
      const float dy = max_point.y - min_point.y;
      const float dz = max_point.z - min_point.z;

      // Reject very large merged structures
      if (dx > 8.0f || dy > 4.0f || dz > 4.0f)
      {
        continue;
      }

      // Reject very tiny clusters / noise
      if (dx < 0.2f || dy < 0.2f || dz < 0.2f)
      {
        continue;
      }

      visualization_msgs::msg::Marker marker;

      marker.header = msg->header;

      marker.ns = "clusters";
      marker.id = marker_id++;

      marker.type =
        visualization_msgs::msg::Marker::CUBE;

      marker.action =
        visualization_msgs::msg::Marker::ADD;

      marker.pose.position.x =
        (min_point.x + max_point.x) / 2.0;

      marker.pose.position.y =
        (min_point.y + max_point.y) / 2.0;

      marker.pose.position.z =
        (min_point.z + max_point.z) / 2.0;

      marker.pose.orientation.x = 0.0;
      marker.pose.orientation.y = 0.0;
      marker.pose.orientation.z = 0.0;
      marker.pose.orientation.w = 1.0;

      marker.scale.x = dx;
      marker.scale.y = dy;
      marker.scale.z = dz;

      marker.color.r = 0.0f;
      marker.color.g = 1.0f;
      marker.color.b = 0.0f;
      marker.color.a = 0.35f;

      marker.lifetime =
        rclcpp::Duration::from_seconds(0.2);

      marker_array.markers.push_back(marker);

      valid_clusters++;
    }

    marker_publisher_->publish(marker_array);

    RCLCPP_INFO(
      this->get_logger(),
      "Raw clusters: %zu | Valid clusters: %d",
      cluster_indices.size(),
      valid_clusters
    );
  }

  rclcpp::Subscription<
    sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;

  rclcpp::Publisher<
    visualization_msgs::msg::MarkerArray>::SharedPtr marker_publisher_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  rclcpp::spin(
    std::make_shared<LidarClustering>()
  );

  rclcpp::shutdown();

  return 0;
}