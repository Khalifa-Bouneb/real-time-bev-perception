cmake_minimum_required(VERSION 3.8)

project(bev_perception_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
# number_publisher
add_executable(
  number_publisher
  src/number_publisher.cpp
)

ament_target_dependencies(
  number_publisher
  rclcpp
  std_msgs
)

# number_subscriber
add_executable(
  number_subscriber
  src/number_subscriber.cpp
)

ament_target_dependencies(
  number_subscriber
  rclcpp
  std_msgs
)

# KITTI LiDAR publisher
add_executable(
  kitti_lidar_publisher
  src/kitti_lidar_publisher.cpp
)

ament_target_dependencies(
  kitti_lidar_publisher
  rclcpp
  sensor_msgs
)

# LiDAR preprocessing
add_executable(
  lidar_preprocessing
  src/lidar_preprocessing.cpp
)

ament_target_dependencies(
  lidar_preprocessing
  rclcpp
  sensor_msgs
  pcl_conversions
)

target_include_directories(
  lidar_preprocessing
  PRIVATE
  ${PCL_INCLUDE_DIRS}
)

target_link_libraries(
  lidar_preprocessing
  ${PCL_LIBRARIES}
)

add_executable(
  ground_removal
  src/ground_removal.cpp
)

ament_target_dependencies(
  ground_removal
  rclcpp
  sensor_msgs
  pcl_conversions
)

target_include_directories(
  ground_removal
  PRIVATE
  ${PCL_INCLUDE_DIRS}
)

target_link_libraries(
  ground_removal
  ${PCL_LIBRARIES}
)

add_executable(
  lidar_clustering
  src/lidar_clustering.cpp
)

ament_target_dependencies(
  lidar_clustering
  rclcpp
  sensor_msgs
  visualization_msgs
  pcl_conversions
)

target_include_directories(
  lidar_clustering
  PRIVATE
  ${PCL_INCLUDE_DIRS}
)

target_link_libraries(
  lidar_clustering
  ${PCL_LIBRARIES}
)
add_executable(
  lidar_camera_projection
  src/lidar_camera_projection.cpp
)

target_include_directories(
  lidar_camera_projection
  PRIVATE
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
  lidar_camera_projection
  Eigen3::Eigen
  ${OpenCV_LIBS}
)

# Install all executables
install(
  TARGETS
    number_publisher
    number_subscriber
    kitti_lidar_publisher
    lidar_preprocessing
    ground_removal
    lidar_clustering
    lidar_camera_projection
  DESTINATION lib/${PROJECT_NAME}
)

# Tests / linters
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)

  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)

  ament_lint_auto_find_test_dependencies()
endif()

ament_package()