# Real-Time BEV Perception

A modular **real-time LiDAR perception pipeline for autonomous-driving applications**, built with ROS2, C++, PointPillars, OpenPCDet, PyTorch, CUDA, PCL, and RViz.

The project starts from raw KITTI Velodyne scans and progressively integrates classical point-cloud processing, LiDAR-camera calibration, deep-learning-based 3D detection, object tracking, and runtime optimization.

---

## Current Demo

The current system supports:

- KITTI Velodyne LiDAR streaming through ROS2
- ROS2 `PointCloud2` processing
- ROI filtering and voxel downsampling
- RANSAC ground removal
- Euclidean clustering
- LiDAR-camera calibration and projection
- PointPillars 3D object detection with OpenPCDet
- Car / Pedestrian / Cyclist detection
- Confidence-score visualization
- Persistent object IDs using lightweight tracking
- Live 3D bounding-box visualization in RViz
- Runtime latency and FPS benchmarking

---

## System Architecture

```text
KITTI Velodyne LiDAR
        │
        ▼
ROS2 C++ LiDAR Publisher
        │
        ▼
/lidar/points
        │
        ▼
PointCloud2 → NumPy
        │
        ▼
OpenPCDet Preprocessing
        │
        ▼
PointPillars
GPU Inference
        │
        ▼
3D Object Detection
        │
        ▼
Object Tracking
        │
        ▼
ROS2 MarkerArray
        │
        ▼
RViz Visualization
```

---

## Performance

Benchmark environment:

- **GPU:** NVIDIA GeForce RTX 2050
- **Dataset:** KITTI Velodyne
- **Point cloud:** ~121K points/frame
- **OS:** Ubuntu 24.04 under WSL2
- **ROS2:** Jazzy
- **Inference:** PyTorch + CUDA

Representative runtime:

| Stage | Runtime |
|---|---:|
| PointCloud2 → NumPy | ~0.2 ms |
| Preprocessing | ~12–17 ms |
| PointPillars inference | ~57–59 ms |
| End-to-end latency | ~70–79 ms |
| P95 latency | ~85 ms |
| Average throughput | **~13.5 FPS** |

### Optimization Highlight

The initial ROS2 implementation converted the point cloud point-by-point using a Python loop.

For approximately **121,000 LiDAR points per frame**, this conversion alone required hundreds of milliseconds and limited the complete pipeline to roughly:

```text
~1 FPS
```

The conversion stage was replaced with vectorized `PointCloud2 → NumPy` processing.

Result:

```text
PointCloud conversion:
hundreds of ms → ~0.2 ms

Full pipeline:
~1 FPS → 13+ FPS
```

This shifted the main runtime cost back to the actual neural-network inference rather than ROS message conversion.

---

## PointPillars 3D Detection

The deep-learning branch uses a pretrained **PointPillars** detector through OpenPCDet.

Supported KITTI classes:

- Car
- Pedestrian
- Cyclist

Each predicted 3D box contains:

```text
[x, y, z, dx, dy, dz, yaw]
```

along with:

```text
class
confidence score
tracking ID
```

Example RViz labels:

```text
Car #4 | 0.92
Cyclist #7 | 0.81
Pedestrian #9 | 0.76
```

---

## ROS2 Packages

### `bev_perception_cpp`

C++ ROS2 package responsible for the classical LiDAR and calibration pipeline.

Implemented nodes include:

```text
kitti_lidar_publisher
lidar_preprocessing
ground_removal
lidar_clustering
lidar_camera_projection
```

### `pointpillars_ros2`

Python ROS2 package responsible for:

- PointCloud2 → NumPy conversion
- OpenPCDet preprocessing
- PointPillars GPU inference
- Confidence filtering
- Lightweight object tracking
- RViz 3D bounding boxes
- Class / confidence / tracking-ID labels
- Runtime benchmarking

---

## Main ROS2 Topics

### LiDAR

```text
/lidar/points
```

Type:

```text
sensor_msgs/msg/PointCloud2
```

### PointPillars Detections

```text
/pointpillars/boxes
```

Type:

```text
visualization_msgs/msg/MarkerArray
```

---

## Classical LiDAR Pipeline

Before integrating PointPillars, a classical LiDAR perception pipeline was implemented using PCL.

```text
Raw LiDAR
   │
   ▼
ROI Filtering
   │
   ▼
Voxel Downsampling
   │
   ▼
RANSAC Ground Removal
   │
   ▼
Euclidean Clustering
   │
   ▼
3D Visualization
```

This provides a geometry-based perception baseline before deep-learning-based 3D detection.

---

## LiDAR-Camera Calibration

KITTI calibration files are used to project Velodyne LiDAR points into the camera image.

The projection chain is:

```text
LiDAR Coordinates
        │
        ▼
T_velo_to_cam
        │
        ▼
Camera Coordinates
        │
        ▼
Rectification
        │
        ▼
Camera Projection Matrix
        │
        ▼
Image Pixel Coordinates
```

This calibration pipeline provides the geometric foundation for future camera-LiDAR fusion.

---

## Object Tracking

The current implementation adds persistent IDs to PointPillars detections.

Example:

```text
Frame t

Car #3
Car #7
Cyclist #8

        ↓

Frame t+1

Car #3
Car #7
Cyclist #8
```

The current tracker uses lightweight class-aware spatial association.

Planned tracking improvements include:

- Kalman filtering
- Hungarian data association
- Velocity estimation
- Track confidence
- Trajectory visualization
- Robust handling of missed detections

---

## Tech Stack

### Robotics

- ROS2 Jazzy
- RViz
- PCL
- Eigen

### 3D Perception / AI

- PointPillars
- OpenPCDet
- PyTorch
- CUDA
- NumPy

### Computer Vision

- OpenCV
- KITTI calibration

### Languages

- C++20
- Python

### Platform

- Ubuntu 24.04
- WSL2
- NVIDIA RTX 2050

---

## Repository Structure

```text
real-time-bev-perception/
│
├── ros2_ws/
│   └── src/
│       │
│       ├── bev_perception_cpp/
│       │   ├── src/
│       │   │   ├── kitti_lidar_publisher.cpp
│       │   │   ├── lidar_preprocessing.cpp
│       │   │   ├── ground_removal.cpp
│       │   │   ├── lidar_clustering.cpp
│       │   │   └── lidar_camera_projection.cpp
│       │   │
│       │   ├── CMakeLists.txt
│       │   └── package.xml
│       │
│       └── pointpillars_ros2/
│           ├── pointpillars_ros2/
│           │   └── pointpillars_node.py
│           ├── resource/
│           ├── setup.py
│           ├── setup.cfg
│           └── package.xml
│
├── third_party/
│   └── OpenPCDet/          # ignored by Git
│
├── data/                   # ignored by Git
│
├── .gitignore
└── README.md
```

---

## Build

Source ROS2:

```bash
source /opt/ros/jazzy/setup.bash
```

Build the workspace:

```bash
cd ros2_ws

colcon build \
  --packages-select bev_perception_cpp pointpillars_ros2 \
  --symlink-install
```

Source the workspace:

```bash
source install/setup.bash
```

---

## Running the Pipeline

### 1. Start the KITTI LiDAR Publisher

```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash

ros2 run bev_perception_cpp kitti_lidar_publisher
```

The publisher sends sequential KITTI Velodyne frames to:

```text
/lidar/points
```

---

### 2. Start PointPillars

Activate the ROS2-compatible Python environment:

```bash
source .venv-ros2/bin/activate
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
```

Run:

```bash
ros2 run pointpillars_ros2 pointpillars_node
```

Example runtime output:

```text
Points=121121
Raw=21
Tracked=8
Conv=0.2 ms
Prep=12.8 ms
Infer=57.0 ms
Total=73.3 ms
FPS=13.64
AVG FPS=13.51
P95=85.0 ms
```

---

### 3. Start RViz

```bash
rviz2
```

Configure:

```text
Fixed Frame:
lidar
```

Add a `PointCloud2` display:

```text
Topic:
/lidar/points
```

Add a `MarkerArray` display:

```text
Topic:
/pointpillars/boxes
```

RViz then displays:

- LiDAR point cloud
- 3D object boxes
- Object class
- Confidence score
- Persistent tracking ID

---

## Dataset

The current implementation uses the **KITTI Vision Benchmark Suite**.

Dataset files are intentionally excluded from the repository.

Expected local structure:

```text
data/
└── kitti/
    ├── velodyne/
    ├── calib/
    └── image_02/
```

---

## OpenPCDet

OpenPCDet is used as the 3D detection framework.

The external repository and pretrained model weights are intentionally excluded from this repository.

Expected local structure:

```text
third_party/
└── OpenPCDet/
    ├── pcdet/
    ├── tools/
    └── checkpoints/
        └── pointpillar_7728.pth
```

---

## Roadmap

### Completed

- [x] ROS2 C++ workspace
- [x] KITTI LiDAR streaming
- [x] RViz point-cloud visualization
- [x] ROI filtering
- [x] Voxel downsampling
- [x] RANSAC ground removal
- [x] Euclidean clustering
- [x] LiDAR-camera calibration
- [x] LiDAR-camera projection
- [x] Standalone PointPillars inference
- [x] OpenPCDet integration
- [x] ROS2 live PointPillars inference
- [x] 3D bounding-box visualization
- [x] Class and confidence labels
- [x] Persistent tracking IDs
- [x] Runtime FPS / latency benchmarking
- [x] Vectorized PointCloud2 conversion
- [x] ~13.5 FPS live pipeline

### Next

- [ ] Kalman Filter tracking
- [ ] Hungarian data association
- [ ] Object velocity estimation
- [ ] Trajectory visualization
- [ ] Camera BEV perception
- [ ] Camera-LiDAR fusion
- [ ] Multi-sensor state estimation
- [ ] SLAM integration
- [ ] TensorRT optimization
- [ ] Extended KITTI evaluation

---

## Long-Term Goal

The goal is to progressively build a modular autonomous-driving perception stack:

```text
LiDAR
  +
Camera
  ↓
3D Perception
  ↓
BEV Representation
  ↓
Object Tracking
  ↓
Sensor Fusion
  ↓
State Estimation
  ↓
SLAM
```

with emphasis on:

- real-time performance
- ROS2 modularity
- 3D geometry
- deep-learning deployment
- multi-sensor perception
- measurable system-level optimization

---

## Status

🚧 **Active development**

Current milestone:

**Real-time ROS2 LiDAR perception with PointPillars 3D detection, persistent IDs, RViz visualization, and ~13.5 FPS average throughput on an RTX 2050.**