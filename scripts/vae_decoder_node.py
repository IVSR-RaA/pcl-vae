#!/usr/bin/env python3
"""
ROS 1 node that decodes a VAE latent vector into a reconstructed range image.
"""

import os
import sys

import numpy as np
import yaml

import rospy
from cv_bridge import CvBridge
from pcl_vae.msg import LatentVectorStamped
from sensor_msgs.msg import Image
from std_msgs.msg import (
    Float32MultiArray,
    Int32MultiArray,
    MultiArrayDimension,
    MultiArrayLayout,
)

try:
    from pcl_vae.inference.scripts.vae_ros_interface import VAERosInterface
    from pcl_vae.path_utils import get_validation_config_path
except ImportError:
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, _REPO_ROOT)
    from pcl_vae.inference.scripts.vae_ros_interface import VAERosInterface
    from pcl_vae.path_utils import get_validation_config_path


class VAEDecoderNode:
    """Decodes incoming latent vectors and publishes reconstructed range images."""

    def __init__(self):
        rospy.init_node("vae_decoder_node", anonymous=False)

        robot_type_override = rospy.get_param("~robot_type", "ground")
        config_path = rospy.get_param(
            "~config_path",
            get_validation_config_path(robot_type_override),
        )
        self._publish_occ = rospy.get_param("~publish_occupancy_map", False)
        self._publish_norm = rospy.get_param("~publish_normalised_image", False)
        queue_size = rospy.get_param("~queue_size", 1)

        with open(config_path, "r") as fh:
            cfg = yaml.safe_load(fh)

        if robot_type_override:
            cfg["robot_type"] = robot_type_override
        elif "robot_type" not in cfg:
            rospy.logfatal(
                "[vae_decoder_node] 'robot_type' missing from config and not "
                "provided as a ROS parameter."
            )
            raise SystemExit(1)

        self._latent_dim = cfg["latent_space"]
        self._max_depth = cfg["image_max_depth"]
        self._image_height = cfg["image_height"]
        self._image_width = cfg["image_width"]

        rospy.loginfo(
            f"[vae_decoder_node] robot_type={cfg['robot_type']} "
            f"latent_dim={self._latent_dim} device={cfg['device']} "
            f"publish_occupancy_map={self._publish_occ}"
        )

        self._iface = VAERosInterface.from_config(cfg)
        if self._publish_occ:
            self._iface.ensure_occupancy_support()
        rospy.loginfo("[vae_decoder_node] VAE model loaded.")

        self._bridge = CvBridge()
        self._pub_range = rospy.Publisher(
            "~output/range_image",
            Image,
            queue_size=queue_size,
        )

        self._pub_norm = None
        if self._publish_norm:
            self._pub_norm = rospy.Publisher(
                "~output/range_image_norm",
                Image,
                queue_size=queue_size,
            )

        self._pub_occ = None
        if self._publish_occ:
            self._pub_occ = rospy.Publisher(
                "~output/occupancy_map",
                Int32MultiArray,
                queue_size=queue_size,
            )

        rospy.Subscriber(
            "~input/latent_vector",
            Float32MultiArray,
            self._latent_callback,
            queue_size=queue_size,
        )
        rospy.Subscriber(
            "~input/latent_vector_stamped",
            LatentVectorStamped,
            self._latent_stamped_callback,
            queue_size=queue_size,
        )

        rospy.loginfo("[vae_decoder_node] Ready - waiting for latent vectors.")
        rospy.spin()

    def _latent_callback(self, msg: Float32MultiArray) -> None:
        """Decode a latent vector and publish the reconstructed range image."""
        self._decode_and_publish(
            msg,
            stamp=rospy.Time.now(),
            frame_id=rospy.get_param("~default_frame_id", "sensor_frame"),
        )

    def _latent_stamped_callback(self, msg: LatentVectorStamped) -> None:
        """Decode a stamped latent vector and preserve its image header."""
        stamp = msg.header.stamp
        if stamp.is_zero():
            stamp = rospy.Time.now()
        frame_id = msg.header.frame_id or rospy.get_param(
            "~default_frame_id",
            "sensor_frame",
        )
        self._decode_and_publish(msg.latent_vector, stamp=stamp, frame_id=frame_id)

    def _decode_and_publish(
        self,
        msg: Float32MultiArray,
        stamp: rospy.Time,
        frame_id: str,
    ) -> None:
        """Decode a latent vector and publish range-image outputs."""
        try:
            data_offset = int(msg.layout.data_offset or 0)
            if data_offset > 0 and data_offset < len(msg.data):
                rospy.logwarn_throttle(
                    5.0,
                    "[vae_decoder_node] Applying non-zero MultiArray data_offset=%d.",
                    data_offset,
                )
                latent_data = msg.data[data_offset:]
            elif data_offset >= len(msg.data) and len(msg.data) > 0:
                rospy.logwarn_throttle(
                    5.0,
                    "[vae_decoder_node] Ignoring legacy data_offset=%d because it "
                    "does not describe padding inside data[].",
                    data_offset,
                )
                latent_data = msg.data
            else:
                latent_data = msg.data

            latent = np.array(latent_data, dtype=np.float32)
            expected_dim = (
                int(msg.layout.dim[0].size)
                if msg.layout.dim and msg.layout.dim[0].size
                else len(latent)
            )
            if expected_dim and int(expected_dim) != self._latent_dim:
                rospy.logwarn(
                    f"[vae_decoder_node] Received latent_dim={int(expected_dim)} "
                    f"but model expects {self._latent_dim}. Proceeding anyway."
                )
            if latent.size == 0:
                rospy.logwarn("[vae_decoder_node] Skipping empty latent vector.")
                return

            recon_norm = self._iface.decode_latent_to_range_image(latent)
            recon_metric = (recon_norm * self._max_depth).astype(np.float32)

            self._pub_range.publish(
                self._to_image_msg(recon_metric, stamp, frame_id=frame_id)
            )

            if self._publish_norm and self._pub_norm is not None:
                self._pub_norm.publish(
                    self._to_image_msg(
                        recon_norm.astype(np.float32), stamp, frame_id=frame_id
                    )
                )

            if self._publish_occ and self._pub_occ is not None:
                self._pub_occ.publish(self._build_occupancy_msg(recon_norm))
        except Exception as exc:  # noqa: BLE001
            rospy.logerr(f"[vae_decoder_node] Failed to decode latent vector: {exc}")

    def _to_image_msg(
        self, np_img: np.ndarray, stamp, frame_id: str = "sensor_frame"
    ) -> Image:
        """Convert an ``(H, W)`` float32 numpy array to ``sensor_msgs/Image``."""
        msg = self._bridge.cv2_to_imgmsg(np_img, encoding="32FC1")
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        return msg

    def _build_occupancy_msg(self, recon_norm: np.ndarray) -> Int32MultiArray:
        """Convert a reconstructed range image to a flat occupancy map message."""
        occ_map = self._iface.reconstructed_norm_to_occupancy_map(
            recon_norm,
            img_height=self._image_height,
            img_width=self._image_width,
        )
        x_size, y_size, z_size = occ_map.shape

        dims = [
            MultiArrayDimension(label="x", size=x_size, stride=x_size * y_size * z_size),
            MultiArrayDimension(label="y", size=y_size, stride=y_size * z_size),
            MultiArrayDimension(label="z", size=z_size, stride=z_size),
        ]
        layout = MultiArrayLayout(dim=dims, data_offset=0)

        return Int32MultiArray(
            layout=layout,
            data=occ_map.cpu().numpy().astype(np.int32).flatten().tolist(),
        )


if __name__ == "__main__":
    try:
        VAEDecoderNode()
    except rospy.ROSInterruptException:
        pass
