#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/voxel_grid.h>

#include <pcl_conversions/pcl_conversions.h>

class LidarPreprocessing : public rclcpp::Node
{
public:
  LidarPreprocessing()
  : Node("lidar_preprocessing")
  {
    subscription_ =
      this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/lidar/points",
        10,
        std::bind(
          &LidarPreprocessing::callback,
          this,
          std::placeholders::_1
        )
      );

    publisher_ =
      this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/lidar/points_filtered",
        10
      );
  }

private:
  void callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(
      new pcl::PointCloud<pcl::PointXYZI>
    );

    pcl::fromROSMsg(*msg, *cloud);

    pcl::PointCloud<pcl::PointXYZI>::Ptr roi_cloud(
      new pcl::PointCloud<pcl::PointXYZI>
    );

    pcl::CropBox<pcl::PointXYZI> crop;

    crop.setInputCloud(cloud);

    crop.setMin(Eigen::Vector4f(
      0.0f,
      -20.0f,
      -2.5f,
      1.0f
    ));

    crop.setMax(Eigen::Vector4f(
      50.0f,
      20.0f,
      3.0f,
      1.0f
    ));

    crop.filter(*roi_cloud);

    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloud(
      new pcl::PointCloud<pcl::PointXYZI>
    );

    pcl::VoxelGrid<pcl::PointXYZI> voxel;

    voxel.setInputCloud(roi_cloud);

    voxel.setLeafSize(
      0.2f,
      0.2f,
      0.2f
    );

    voxel.filter(*filtered_cloud);

    sensor_msgs::msg::PointCloud2 output;

    pcl::toROSMsg(*filtered_cloud, output);

    output.header = msg->header;

    publisher_->publish(output);

    RCLCPP_INFO(
      this->get_logger(),
      "Raw: %zu | ROI: %zu | Filtered: %zu",
      cloud->size(),
      roi_cloud->size(),
      filtered_cloud->size()
    );
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  rclcpp::spin(
    std::make_shared<LidarPreprocessing>()
  );

  rclcpp::shutdown();

  return 0;
}