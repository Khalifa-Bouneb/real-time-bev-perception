#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

struct PointXYZI
{
  float x;
  float y;
  float z;
  float intensity;
};

bool loadVelodyneCalibration(
  const std::string & path,
  Eigen::Matrix3f & R,
  Eigen::Vector3f & T)
{
  std::ifstream file(path);

  if (!file.is_open())
  {
    std::cerr << "Could not open: " << path << std::endl;
    return false;
  }

  std::string line;

  while (std::getline(file, line))
  {
    if (line.rfind("R:", 0) == 0)
    {
      std::stringstream ss(line.substr(2));

      for (int i = 0; i < 3; ++i)
      {
        for (int j = 0; j < 3; ++j)
        {
          ss >> R(i, j);
        }
      }
    }

    if (line.rfind("T:", 0) == 0)
    {
      std::stringstream ss(line.substr(2));

      ss >> T(0) >> T(1) >> T(2);
    }
  }

  return true;
}

bool loadCameraCalibration(
  const std::string & path,
  Eigen::Matrix3f & R_rect,
  Eigen::Matrix<float, 3, 4> & P_rect_02)
{
  std::ifstream file(path);

  if (!file.is_open())
  {
    std::cerr << "Could not open: " << path << std::endl;
    return false;
  }

  bool found_rect = false;
  bool found_projection = false;

  std::string line;

  while (std::getline(file, line))
  {
    if (line.rfind("R_rect_00:", 0) == 0)
    {
      std::stringstream ss(line.substr(10));

      for (int i = 0; i < 3; ++i)
      {
        for (int j = 0; j < 3; ++j)
        {
          ss >> R_rect(i, j);
        }
      }

      found_rect = true;
    }

    if (line.rfind("P_rect_02:", 0) == 0)
    {
      std::stringstream ss(line.substr(10));

      for (int i = 0; i < 3; ++i)
      {
        for (int j = 0; j < 4; ++j)
        {
          ss >> P_rect_02(i, j);
        }
      }

      found_projection = true;
    }
  }

  if (!found_rect)
  {
    std::cerr << "R_rect_00 not found" << std::endl;
  }

  if (!found_projection)
  {
    std::cerr << "P_rect_02 not found" << std::endl;
  }

  return found_rect && found_projection;
}

std::vector<PointXYZI> loadPointCloud(const std::string & path)
{
  std::ifstream input(path, std::ios::binary);

  std::vector<PointXYZI> points;

  if (!input.is_open())
  {
    std::cerr << "Could not open LiDAR file: " << path << std::endl;
    return points;
  }

  PointXYZI point;

  while (
    input.read(
      reinterpret_cast<char *>(&point),
      sizeof(PointXYZI)))
  {
    points.push_back(point);
  }

  return points;
}

int main()
{
  const std::string lidar_path =
    "/mnt/d/AI/real-time-bev-perception/data/kitti/velodyne/0000000010.bin";

  const std::string image_path =
    "/mnt/d/AI/real-time-bev-perception/data/kitti/image_02/0000000010.png";

  const std::string velo_calib_path =
    "/mnt/d/AI/real-time-bev-perception/data/kitti/calib/calib_velo_to_cam.txt";

  const std::string camera_calib_path =
    "/mnt/d/AI/real-time-bev-perception/data/kitti/calib/calib_cam_to_cam.txt";

  Eigen::Matrix3f R_velo_to_cam;
  Eigen::Vector3f T_velo_to_cam;

  Eigen::Matrix3f R_rect_00;
  Eigen::Matrix<float, 3, 4> P_rect_02;

  if (!loadVelodyneCalibration(
        velo_calib_path,
        R_velo_to_cam,
        T_velo_to_cam))
  {
    return 1;
  }

  if (!loadCameraCalibration(
        camera_calib_path,
        R_rect_00,
        P_rect_02))
  {
    return 1;
  }

  std::vector<PointXYZI> points =
    loadPointCloud(lidar_path);

  if (points.empty())
  {
    return 1;
  }

  cv::Mat image = cv::imread(image_path);

  if (image.empty())
  {
    std::cerr << "Could not load image" << std::endl;
    return 1;
  }

  std::cout
    << "Loaded " << points.size()
    << " LiDAR points"
    << std::endl;

  int projected_points = 0;

  for (const auto & point : points)
  {
    Eigen::Vector3f p_lidar(
      point.x,
      point.y,
      point.z
    );

    // 1) LiDAR frame -> reference camera frame
    Eigen::Vector3f p_cam =
      R_velo_to_cam * p_lidar +
      T_velo_to_cam;

    // 2) Reference camera frame -> rectified camera frame
    Eigen::Vector3f p_rect =
      R_rect_00 * p_cam;

    // Reject points behind the camera
    if (p_rect.z() <= 0.0f)
    {
      continue;
    }

    // Homogeneous 3D point
    Eigen::Vector4f p_rect_h;

    p_rect_h <<
      p_rect.x(),
      p_rect.y(),
      p_rect.z(),
      1.0f;

    // 3) Rectified camera coordinates -> image plane
    Eigen::Vector3f pixel =
      P_rect_02 * p_rect_h;

    const float u =
      pixel.x() / pixel.z();

    const float v =
      pixel.y() / pixel.z();

    if (
      u >= 0.0f &&
      u < image.cols &&
      v >= 0.0f &&
      v < image.rows)
    {
      const float distance =
        p_lidar.norm();

      const float normalized =
        std::min(distance / 50.0f, 1.0f);

      // OpenCV uses BGR
      cv::Scalar color(
        255.0f * (1.0f - normalized),
        255.0f * normalized,
        0.0f
      );

      cv::circle(
        image,
        cv::Point(
          static_cast<int>(u),
          static_cast<int>(v)
        ),
        1,
        color,
        -1
      );

      projected_points++;
    }
  }

  std::cout
    << "Projected " << projected_points
    << " points"
    << std::endl;

  cv::imwrite(
    "/mnt/d/AI/real-time-bev-perception/data/kitti/lidar_projection_rectified.png",
    image
  );

  cv::imshow(
    "Rectified LiDAR Camera Projection",
    image
  );

  cv::waitKey(0);

  return 0;
}