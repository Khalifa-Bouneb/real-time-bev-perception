import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs_py import point_cloud2


# ============================================================
# OpenPCDet
# ============================================================

OPENPCDET_ROOT = Path(
    "/mnt/d/AI/real-time-bev-perception/third_party/OpenPCDet"
)

sys.path.insert(0, str(OPENPCDET_ROOT))

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


CLASS_NAMES = [
    "Car",
    "Pedestrian",
    "Cyclist",
]


# ============================================================
# OpenPCDet live dataset
# ============================================================

class LiveDataset(DatasetTemplate):

    def __init__(
        self,
        dataset_cfg,
        class_names,
        logger
    ):
        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=False,
            root_path=None,
            logger=logger
        )

    def prepare_points(self, points):

        input_dict = {
            "points": points,
            "frame_id": 0
        }

        return self.prepare_data(
            data_dict=input_dict
        )


# ============================================================
# Track object
# ============================================================

class Track:

    def __init__(
        self,
        track_id,
        class_name,
        center,
        score
    ):

        self.track_id = track_id
        self.class_name = class_name

        self.center = np.asarray(
            center,
            dtype=np.float32
        )

        self.score = float(score)

        self.age = 1
        self.missed = 0


# ============================================================
# Simple class-aware 3D tracker
# ============================================================

class SimpleTracker:

    def __init__(
        self,
        max_distance=4.0,
        max_missed=3
    ):

        self.max_distance = max_distance
        self.max_missed = max_missed

        self.next_track_id = 1

        self.tracks = {}

    def update(
        self,
        detections
    ):
        """
        detections:
        [
            {
                "center": [x, y, z],
                "class_name": "Car",
                "score": 0.91
            },
            ...
        ]

        returns:
            list of track IDs corresponding to detections
        """

        detection_track_ids = [
            None
        ] * len(detections)

        matched_tracks = set()
        matched_detections = set()

        # ----------------------------------------------------
        # Generate association candidates
        # ----------------------------------------------------

        candidates = []

        for track_id, track in self.tracks.items():

            for det_index, detection in enumerate(
                detections
            ):

                # Only associate same class
                if (
                    track.class_name
                    != detection["class_name"]
                ):
                    continue

                detection_center = np.asarray(
                    detection["center"],
                    dtype=np.float32
                )

                # XY distance for BEV tracking
                distance = np.linalg.norm(
                    track.center[:2]
                    - detection_center[:2]
                )

                if distance <= self.max_distance:

                    candidates.append(
                        (
                            float(distance),
                            track_id,
                            det_index
                        )
                    )

        # ----------------------------------------------------
        # Greedy nearest-neighbour association
        # ----------------------------------------------------

        candidates.sort(
            key=lambda item: item[0]
        )

        for (
            distance,
            track_id,
            det_index
        ) in candidates:

            if track_id in matched_tracks:
                continue

            if det_index in matched_detections:
                continue

            track = self.tracks[
                track_id
            ]

            detection = detections[
                det_index
            ]

            track.center = np.asarray(
                detection["center"],
                dtype=np.float32
            )

            track.score = float(
                detection["score"]
            )

            track.age += 1
            track.missed = 0

            detection_track_ids[
                det_index
            ] = track_id

            matched_tracks.add(
                track_id
            )

            matched_detections.add(
                det_index
            )

        # ----------------------------------------------------
        # Increase missed counter
        # ----------------------------------------------------

        for track_id, track in self.tracks.items():

            if track_id not in matched_tracks:
                track.missed += 1

        # ----------------------------------------------------
        # Create tracks for unmatched detections
        # ----------------------------------------------------

        for det_index, detection in enumerate(
            detections
        ):

            if det_index in matched_detections:
                continue

            track_id = self.next_track_id

            self.next_track_id += 1

            new_track = Track(
                track_id=track_id,
                class_name=detection["class_name"],
                center=detection["center"],
                score=detection["score"]
            )

            self.tracks[
                track_id
            ] = new_track

            detection_track_ids[
                det_index
            ] = track_id

        # ----------------------------------------------------
        # Remove dead tracks
        # ----------------------------------------------------

        dead_tracks = [

            track_id

            for track_id, track
            in self.tracks.items()

            if track.missed > self.max_missed
        ]

        for track_id in dead_tracks:

            del self.tracks[
                track_id
            ]

        return detection_track_ids


# ============================================================
# PointPillars ROS2 node
# ============================================================

class PointPillarsNode(Node):

    def __init__(self):

        super().__init__(
            "pointpillars_node"
        )

        self.get_logger().info(
            "Starting PointPillars ROS2 node..."
        )

        # ====================================================
        # Configuration
        # ====================================================

        cfg_file = (
            OPENPCDET_ROOT
            / "tools"
            / "cfgs"
            / "kitti_models"
            / "pointpillar.yaml"
        )

        checkpoint = (
            OPENPCDET_ROOT
            / "checkpoints"
            / "pointpillar_7728.pth"
        )

        os.chdir(
            OPENPCDET_ROOT / "tools"
        )

        self.get_logger().info(
            f"OpenPCDet working directory: "
            f"{os.getcwd()}"
        )

        self.get_logger().info(
            "Loading PointPillars configuration..."
        )

        cfg_from_yaml_file(
            str(cfg_file),
            cfg
        )

        self.pcdet_logger = (
            common_utils.create_logger()
        )

        self.get_logger().info(
            "Configuration loaded."
        )

        # ====================================================
        # Dataset
        # ====================================================

        self.get_logger().info(
            "Creating OpenPCDet dataset helper..."
        )

        self.dataset = LiveDataset(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            logger=self.pcdet_logger
        )

        # ====================================================
        # Model
        # ====================================================

        self.get_logger().info(
            "Building PointPillars network..."
        )

        self.model = build_network(
            model_cfg=cfg.MODEL,
            num_class=len(cfg.CLASS_NAMES),
            dataset=self.dataset
        )

        self.get_logger().info(
            f"Loading checkpoint: {checkpoint}"
        )

        self.model.load_params_from_file(
            filename=str(checkpoint),
            logger=self.pcdet_logger,
            to_cpu=True
        )

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA is required."
            )

        self.get_logger().info(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        self.model.cuda()
        self.model.eval()

        torch.cuda.synchronize()

        self.get_logger().info(
            "PointPillars model ready."
        )

        # ====================================================
        # Threshold
        # ====================================================

        self.confidence_threshold = 0.50

        # ====================================================
        # Tracker
        # ====================================================

        self.tracker = SimpleTracker(
            max_distance=4.0,
            max_missed=3
        )

        self.get_logger().info(
            "3D object tracker ready."
        )

        # ====================================================
        # ROS
        # ====================================================

        self.subscription = (
            self.create_subscription(
                PointCloud2,
                "/lidar/points",
                self.pointcloud_callback,
                1
            )
        )

        self.marker_publisher = (
            self.create_publisher(
                MarkerArray,
                "/pointpillars/boxes",
                10
            )
        )

        self.get_logger().info(
            "Subscribed to: /lidar/points"
        )

        self.get_logger().info(
            "Publishing to: /pointpillars/boxes"
        )

        self.get_logger().info(
            "Waiting for LiDAR frames..."
        )

        # ====================================================
        # Marker state
        # ====================================================

        self.previous_marker_count = 0

        # ====================================================
        # Benchmark
        # ====================================================

        self.latency_history = deque(
            maxlen=100
        )

        self.fps_history = deque(
            maxlen=100
        )

        self.inference_history = deque(
            maxlen=100
        )

        self.preprocessing_history = deque(
            maxlen=100
        )

        self.conversion_history = deque(
            maxlen=100
        )

        self.frame_count = 0

    # ========================================================
    # ROS PointCloud2 -> NumPy
    # ========================================================

    def ros_pointcloud_to_numpy(
        self,
        msg
    ):

        points = (
            point_cloud2.read_points_numpy(
                msg,
                field_names=(
                    "x",
                    "y",
                    "z",
                    "intensity"
                ),
                skip_nans=True
            )
        )

        if (
            points is None
            or len(points) == 0
        ):
            return None

        return np.asarray(
            points,
            dtype=np.float32
        )

    # ========================================================
    # Publish markers
    # ========================================================

    def publish_markers(
        self,
        boxes,
        scores,
        labels,
        header
    ):

        marker_array = MarkerArray()

        # ----------------------------------------------------
        # Filter detections
        # ----------------------------------------------------

        valid_detections = []

        for box, score, label in zip(
            boxes,
            scores,
            labels
        ):

            if score < self.confidence_threshold:
                continue

            label_index = (
                int(label) - 1
            )

            if (
                label_index < 0
                or label_index
                >= len(CLASS_NAMES)
            ):
                continue

            class_name = (
                CLASS_NAMES[
                    label_index
                ]
            )

            x, y, z, dx, dy, dz, yaw = box

            valid_detections.append(
                {
                    "box": box,
                    "score": float(score),
                    "class_name": class_name,
                    "center": [
                        float(x),
                        float(y),
                        float(z)
                    ]
                }
            )

        # ----------------------------------------------------
        # Tracking
        # ----------------------------------------------------

        track_ids = (
            self.tracker.update(
                valid_detections
            )
        )

        marker_id = 0

        # ----------------------------------------------------
        # Draw detections
        # ----------------------------------------------------

        for (
            detection,
            track_id
        ) in zip(
            valid_detections,
            track_ids
        ):

            box = detection["box"]

            score = detection["score"]

            class_name = (
                detection["class_name"]
            )

            x, y, z, dx, dy, dz, yaw = box

            # =================================================
            # BOX
            # =================================================

            box_marker = Marker()

            box_marker.header = header

            box_marker.ns = (
                "pointpillars_boxes"
            )

            box_marker.id = marker_id

            box_marker.type = (
                Marker.CUBE
            )

            box_marker.action = (
                Marker.ADD
            )

            # position
            box_marker.pose.position.x = (
                float(x)
            )

            box_marker.pose.position.y = (
                float(y)
            )

            box_marker.pose.position.z = (
                float(z)
            )

            # orientation
            box_marker.pose.orientation.z = (
                float(
                    np.sin(
                        yaw / 2.0
                    )
                )
            )

            box_marker.pose.orientation.w = (
                float(
                    np.cos(
                        yaw / 2.0
                    )
                )
            )

            # dimensions
            box_marker.scale.x = float(dx)
            box_marker.scale.y = float(dy)
            box_marker.scale.z = float(dz)

            # ------------------------------------------------
            # Colors
            # ------------------------------------------------

            if class_name == "Car":

                box_marker.color.r = 0.0
                box_marker.color.g = 1.0
                box_marker.color.b = 0.0

            elif class_name == "Pedestrian":

                box_marker.color.r = 1.0
                box_marker.color.g = 1.0
                box_marker.color.b = 0.0

            else:

                box_marker.color.r = 0.0
                box_marker.color.g = 1.0
                box_marker.color.b = 1.0

            box_marker.color.a = 0.30

            box_marker.lifetime.sec = 0

            box_marker.lifetime.nanosec = (
                300_000_000
            )

            marker_array.markers.append(
                box_marker
            )

            # =================================================
            # TEXT
            # =================================================

            text_marker = Marker()

            text_marker.header = header

            text_marker.ns = (
                "pointpillars_labels"
            )

            text_marker.id = marker_id

            text_marker.type = (
                Marker.TEXT_VIEW_FACING
            )

            text_marker.action = (
                Marker.ADD
            )

            text_marker.pose.position.x = (
                float(x)
            )

            text_marker.pose.position.y = (
                float(y)
            )

            text_marker.pose.position.z = (
                float(
                    z
                    + dz / 2.0
                    + 0.5
                )
            )

            text_marker.pose.orientation.w = (
                1.0
            )

            text_marker.scale.z = 0.55

            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0

            # ------------------------------------------------
            # NEW: class + track ID + confidence
            # ------------------------------------------------

            text_marker.text = (
                f"{class_name} "
                f"#{track_id} "
                f"{score:.2f}"
            )

            text_marker.lifetime.sec = 0

            text_marker.lifetime.nanosec = (
                300_000_000
            )

            marker_array.markers.append(
                text_marker
            )

            marker_id += 1

        # ----------------------------------------------------
        # Delete stale RViz markers
        # ----------------------------------------------------

        for old_id in range(
            marker_id,
            self.previous_marker_count
        ):

            delete_box = Marker()

            delete_box.header = header
            delete_box.ns = (
                "pointpillars_boxes"
            )
            delete_box.id = old_id

            delete_box.action = (
                Marker.DELETE
            )

            marker_array.markers.append(
                delete_box
            )

            delete_text = Marker()

            delete_text.header = header

            delete_text.ns = (
                "pointpillars_labels"
            )

            delete_text.id = old_id

            delete_text.action = (
                Marker.DELETE
            )

            marker_array.markers.append(
                delete_text
            )

        self.previous_marker_count = (
            marker_id
        )

        self.marker_publisher.publish(
            marker_array
        )

        return marker_id

    # ========================================================
    # Callback
    # ========================================================

    def pointcloud_callback(
        self,
        msg
    ):

        total_start = (
            time.perf_counter()
        )

        self.frame_count += 1

        # ====================================================
        # Conversion
        # ====================================================

        conversion_start = (
            time.perf_counter()
        )

        points = (
            self.ros_pointcloud_to_numpy(
                msg
            )
        )

        if points is None:
            return

        conversion_time = (
            time.perf_counter()
            - conversion_start
        )

        # ====================================================
        # Preprocessing
        # ====================================================

        preprocessing_start = (
            time.perf_counter()
        )

        data_dict = (
            self.dataset.prepare_points(
                points
            )
        )

        data_dict = (
            self.dataset.collate_batch(
                [data_dict]
            )
        )

        load_data_to_gpu(
            data_dict
        )

        preprocessing_time = (
            time.perf_counter()
            - preprocessing_start
        )

        # ====================================================
        # Inference
        # ====================================================

        torch.cuda.synchronize()

        inference_start = (
            time.perf_counter()
        )

        with torch.no_grad():

            pred_dicts, _ = (
                self.model.forward(
                    data_dict
                )
            )

        torch.cuda.synchronize()

        inference_time = (
            time.perf_counter()
            - inference_start
        )

        pred = pred_dicts[0]

        boxes = (
            pred["pred_boxes"]
            .detach()
            .cpu()
            .numpy()
        )

        scores = (
            pred["pred_scores"]
            .detach()
            .cpu()
            .numpy()
        )

        labels = (
            pred["pred_labels"]
            .detach()
            .cpu()
            .numpy()
        )

        # ====================================================
        # Tracking + RViz
        # ====================================================

        valid_detections = (
            self.publish_markers(
                boxes,
                scores,
                labels,
                msg.header
            )
        )

        # ====================================================
        # Benchmark
        # ====================================================

        total_time = (
            time.perf_counter()
            - total_start
        )

        fps = (
            1.0 / total_time
            if total_time > 0
            else 0.0
        )

        conversion_ms = (
            conversion_time * 1000.0
        )

        preprocessing_ms = (
            preprocessing_time * 1000.0
        )

        inference_ms = (
            inference_time * 1000.0
        )

        total_ms = (
            total_time * 1000.0
        )

        self.conversion_history.append(
            conversion_ms
        )

        self.preprocessing_history.append(
            preprocessing_ms
        )

        self.inference_history.append(
            inference_ms
        )

        self.latency_history.append(
            total_ms
        )

        self.fps_history.append(
            fps
        )

        avg_latency = float(
            np.mean(
                self.latency_history
            )
        )

        avg_fps = float(
            np.mean(
                self.fps_history
            )
        )

        p95_latency = float(
            np.percentile(
                self.latency_history,
                95
            )
        )

        # ====================================================
        # Live log
        # ====================================================

        self.get_logger().info(

            f"Frame={self.frame_count} | "

            f"Points={len(points)} | "

            f"Raw={len(boxes)} | "

            f"Tracked={valid_detections} | "

            f"ActiveTracks="
            f"{len(self.tracker.tracks)} | "

            f"Conv="
            f"{conversion_ms:.1f} ms | "

            f"Prep="
            f"{preprocessing_ms:.1f} ms | "

            f"Infer="
            f"{inference_ms:.1f} ms | "

            f"Total="
            f"{total_ms:.1f} ms | "

            f"FPS="
            f"{fps:.2f} | "

            f"AVG FPS="
            f"{avg_fps:.2f} | "

            f"P95="
            f"{p95_latency:.1f} ms"
        )

        # ====================================================
        # Benchmark every 25 frames
        # ====================================================

        if self.frame_count % 25 == 0:

            avg_conversion = float(
                np.mean(
                    self.conversion_history
                )
            )

            avg_preprocess = float(
                np.mean(
                    self.preprocessing_history
                )
            )

            avg_inference = float(
                np.mean(
                    self.inference_history
                )
            )

            self.get_logger().info(

                "=== Rolling Benchmark === | "

                f"Frames={self.frame_count} | "

                f"Avg Conversion="
                f"{avg_conversion:.2f} ms | "

                f"Avg Preprocess="
                f"{avg_preprocess:.2f} ms | "

                f"Avg Inference="
                f"{avg_inference:.2f} ms | "

                f"Avg Total="
                f"{avg_latency:.2f} ms | "

                f"P95="
                f"{p95_latency:.2f} ms | "

                f"Avg FPS="
                f"{avg_fps:.2f}"
            )


# ============================================================
# Main
# ============================================================

def main(args=None):

    rclpy.init(
        args=args
    )

    node = (
        PointPillarsNode()
    )

    try:

        rclpy.spin(
            node
        )

    except KeyboardInterrupt:

        node.get_logger().info(
            "Stopping PointPillars node..."
        )

    finally:

        node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":
    main()