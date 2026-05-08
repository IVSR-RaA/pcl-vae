#!/usr/bin/env python3
"""
ROS 1 node that converts a point cloud into a metric range image.

This is intended as a bridge from raw LiDAR topics (for example the LiDAR input
used by Super-LIO) into the `pcl_vae` encoder, which expects a `32FC1`
`sensor_msgs/Image`.
"""

import os
import sys

import numpy as np
import yaml

import rospy
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2

try:
    from pcl_vae.path_utils import get_validation_config_path
except ImportError:
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, _REPO_ROOT)
    from pcl_vae.path_utils import get_validation_config_path


class PointToRangeImageNode:
    """Projects a sensor-frame point cloud into a dense range image grid."""

    def __init__(self):
        rospy.init_node("point_to_range_image_node", anonymous=False)

        robot_type_override = rospy.get_param("~robot_type", "ground")
        config_path = rospy.get_param(
            "~config_path",
            get_validation_config_path(robot_type_override),
        )
        queue_size = rospy.get_param("~queue_size", 1)

        with open(config_path, "r") as fh:
            cfg = yaml.safe_load(fh)

        if robot_type_override:
            cfg["robot_type"] = robot_type_override
        elif "robot_type" not in cfg:
            rospy.logfatal(
                "[point_to_range_image_node] 'robot_type' missing from config and not "
                "provided as a ROS parameter."
            )
            raise SystemExit(1)

        self._image_height = int(cfg["image_height"])
        self._image_width = int(cfg["image_width"])
        self._max_depth = float(cfg["image_max_depth"])
        self._min_depth = float(cfg["image_min_depth"])
        self._invalid_pixel_value = float(cfg.get("invalid_pixel_value", -1.0))
        self._h_fov = float(cfg["h_fov"])
        self._v_fov = float(cfg["v_fov"])

        self._bridge = CvBridge()
        self._pub = rospy.Publisher(
            "~output/range_image",
            Image,
            queue_size=queue_size,
        )

        rospy.Subscriber(
            "~input/points",
            PointCloud2,
            self._pointcloud_callback,
            queue_size=queue_size,
        )

        rospy.loginfo(
            "[point_to_range_image_node] robot_type=%s image=%dx%d min_depth=%.3f "
            "max_depth=%.3f",
            cfg["robot_type"],
            self._image_height,
            self._image_width,
            self._min_depth,
            self._max_depth,
        )
        rospy.loginfo(
            "[point_to_range_image_node] Ready - waiting for PointCloud2 on ~input/points."
        )
        rospy.spin()

    def _pointcloud_callback(self, msg: PointCloud2) -> None:
        """Project a PointCloud2 into a metric range image and publish it."""
        try:
            if msg.header.frame_id.lower() == "world":
                rospy.logwarn_throttle(
                    5.0,
                    "[point_to_range_image_node] Received point cloud in frame 'world'. "
                    "The VAE expects a sensor-frame scan, so prefer the raw LiDAR topic "
                    "or a body-frame scan instead of /lio/cloud_world.",
                )

            range_image = self._pointcloud_to_range_image(msg)
            img_msg = self._bridge.cv2_to_imgmsg(range_image, encoding="32FC1")
            img_msg.header = msg.header
            self._pub.publish(img_msg)
        except Exception as exc:  # noqa: BLE001
            rospy.logerr(f"[point_to_range_image_node] Failed to convert point cloud: {exc}")

    def _pointcloud_to_range_image(self, msg: PointCloud2) -> np.ndarray:
        """Convert a PointCloud2 message to an `(H, W)` metric range image."""
        points = np.array(
            list(
                pc2.read_points(
                    msg,
                    field_names=("x", "y", "z"),
                    skip_nans=True,
                )
            ),
            dtype=np.float32,
        )

        range_image = np.full(
            (self._image_height, self._image_width),
            self._invalid_pixel_value,
            dtype=np.float32,
        )
        if points.size == 0:
            return range_image

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        radius = np.sqrt(x * x + y * y + z * z)
        valid = (radius >= self._min_depth) & (radius <= self._max_depth)
        if not np.any(valid):
            return range_image

        x = x[valid]
        y = y[valid]
        z = z[valid]
        radius = radius[valid]

        horizontal = np.arctan2(-y, x)
        vertical = np.arctan2(-z, np.sqrt(x * x + y * y))

        h_half = self._h_fov * 0.5
        v_half = self._v_fov * 0.5
        in_fov = (
            (horizontal >= -h_half)
            & (horizontal <= h_half)
            & (vertical >= -v_half)
            & (vertical <= v_half)
        )
        if not np.any(in_fov):
            return range_image

        horizontal = horizontal[in_fov]
        vertical = vertical[in_fov]
        radius = radius[in_fov]

        cols = np.rint(
            (horizontal + h_half) / self._h_fov * (self._image_width - 1)
        ).astype(np.int32)
        rows = np.rint(
            (vertical + v_half) / self._v_fov * (self._image_height - 1)
        ).astype(np.int32)

        cols = np.clip(cols, 0, self._image_width - 1)
        rows = np.clip(rows, 0, self._image_height - 1)

        for row, col, depth in zip(rows, cols, radius):
            current = range_image[row, col]
            if current == self._invalid_pixel_value or depth < current:
                range_image[row, col] = depth

        return range_image


if __name__ == "__main__":
    try:
        PointToRangeImageNode()
    except rospy.ROSInterruptException:
        pass
