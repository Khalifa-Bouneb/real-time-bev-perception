#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/point_field.hpp"

using namespace std::chrono_literals;
namespace fs = std::filesystem;

struct PointXYZI
{
  float x;
  float y;
  float z;
  float intensity;
};

class KittiLidarPublisher : public rclcpp::Node
{
public:
  KittiLidarPublisher()
  : Node("kitti_lidar_publisher"), current_frame_(0)
  {
    publisher_ =
      this->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/lidar/points", 10);

    data_dir_ =
      "/mnt/d/AI/real-time-bev-perception/data/kitti/velodyne_demo";

    load_file_list();

    timer_ = this->create_wall_timer(
      100ms,  // 10 Hz
      std::bind(&KittiLidarPublisher::publish_next_frame, this)
    );
  }

private:
  void load_file_list()
  {
    for (const auto & entry : fs::directory_iterator(data_dir_))
    {
      if (entry.path().extension() == ".bin")
      {
        files_.push_back(entry.path().string());
      }
    }

    std::sort(files_.begin(), files_.end());

    RCLCPP_INFO(
      this->get_logger(),
      "Found %zu KITTI frames",
      files_.size()
    );
  }

  void publish_next_frame()
  {
    if (files_.empty())
    {
      RCLCPP_ERROR(this->get_logger(), "No .bin files found");
      return;
    }

    const std::string & file_path = files_[current_frame_];

    std::ifstream input(file_path, std::ios::binary);

    if (!input.is_open())
    {
      RCLCPP_ERROR(
        this->get_logger(),
        "Could not open file: %s",
        file_path.c_str()
      );
      return;
    }

    std::vector<PointXYZI> points;
    PointXYZI point;

    while (input.read(
      reinterpret_cast<char *>(&point),
      sizeof(PointXYZI)))
    {
      points.push_back(point);
    }

    sensor_msgs::msg::PointCloud2 msg;

    msg.header.stamp = this->now();
    msg.header.frame_id = "lidar";

    msg.height = 1;
    msg.width = points.size();

    msg.is_bigendian = false;
    msg.is_dense = true;

    msg.fields.resize(4);

    msg.fields[0].name = "x";
    msg.fields[0].offset = 0;
    msg.fields[0].datatype =
      sensor_msgs::msg::PointField::FLOAT32;
    msg.fields[0].count = 1;

    msg.fields[1].name = "y";
    msg.fields[1].offset = 4;
    msg.fields[1].datatype =
      sensor_msgs::msg::PointField::FLOAT32;
    msg.fields[1].count = 1;

    msg.fields[2].name = "z";
    msg.fields[2].offset = 8;
    msg.fields[2].datatype =
      sensor_msgs::msg::PointField::FLOAT32;
    msg.fields[2].count = 1;

    msg.fields[3].name = "intensity";
    msg.fields[3].offset = 12;
    msg.fields[3].datatype =
      sensor_msgs::msg::PointField::FLOAT32;
    msg.fields[3].count = 1;

    msg.point_step = 16;
    msg.row_step = msg.point_step * msg.width;

    msg.data.resize(msg.row_step);

    std::memcpy(
      msg.data.data(),
      points.data(),
      msg.row_step
    );

    publisher_->publish(msg);

    RCLCPP_INFO(
      this->get_logger(),
      "Frame %zu/%zu | Published %zu points | %s",
      current_frame_ + 1,
      files_.size(),
      points.size(),
      fs::path(file_path).filename().c_str()
    );

    current_frame_++;

    if (current_frame_ >= files_.size())
    {
      current_frame_ = 0;
    }
  }

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::string data_dir_;
  std::vector<std::string> files_;
  size_t current_frame_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  rclcpp::spin(
    std::make_shared<KittiLidarPublisher>()
  );

  rclcpp::shutdown();

  return 0;
}
