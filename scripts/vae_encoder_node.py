#!/usr/bin/env python3
"""
ROS 1 node that encodes a range image into a VAE latent vector.
"""

import os
import sys

import numpy as np
import torch
import yaml

import rospy
from cv_bridge import CvBridge
from pcl_vae.msg import LatentVectorStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout

try:
    from pcl_vae.inference.scripts.vae_ros_interface import VAERosInterface
    from pcl_vae.path_utils import get_validation_config_path
except ImportError:
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, _REPO_ROOT)
    from pcl_vae.inference.scripts.vae_ros_interface import VAERosInterface
    from pcl_vae.path_utils import get_validation_config_path


class VAEEncoderNode:
    """Encodes incoming range images and publishes latent vectors."""

    def __init__(self):
        rospy.init_node("vae_encoder_node", anonymous=False)

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
                "[vae_encoder_node] 'robot_type' missing from config and not "
                "provided as a ROS parameter."
            )
            raise SystemExit(1)

        self._latent_dim = cfg["latent_space"]

        rospy.loginfo(
            f"[vae_encoder_node] robot_type={cfg['robot_type']} "
            f"latent_dim={self._latent_dim} device={cfg['device']}"
        )

        self._iface = VAERosInterface.from_config(cfg)
        rospy.loginfo("[vae_encoder_node] VAE model loaded.")

        self._bridge = CvBridge()
        self._pub = rospy.Publisher(
            "~output/latent_vector",
            Float32MultiArray,
            queue_size=queue_size,
        )
        self._pub_stamped = rospy.Publisher(
            "~output/latent_vector_stamped",
            LatentVectorStamped,
            queue_size=queue_size,
        )

        rospy.Subscriber(
            "~input/range_image",
            Image,
            self._image_callback,
            queue_size=queue_size,
        )

        rospy.loginfo("[vae_encoder_node] Ready - waiting for range images.")
        rospy.spin()

    def _image_callback(self, msg: Image) -> None:
        """Convert incoming Image -> latent vector -> Float32MultiArray."""
        try:
            cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            raw_tensor = (
                torch.from_numpy(cv_img.astype(np.float32))
                .unsqueeze(0)
                .unsqueeze(0)
            )

            latent = self._iface.encode_range_image_to_latent(raw_tensor)
            latent_msg = self._build_latent_msg(latent)
            self._pub.publish(latent_msg)
            self._pub_stamped.publish(
                LatentVectorStamped(
                    header=msg.header,
                    latent_vector=latent_msg,
                )
            )
        except Exception as exc:  # noqa: BLE001
            rospy.logerr(f"[vae_encoder_node] Failed to encode image: {exc}")

    def _build_latent_msg(self, latent: np.ndarray) -> Float32MultiArray:
        """Wrap a 1-D latent numpy array in a Float32MultiArray."""
        dim = MultiArrayDimension(
            label="latent",
            size=len(latent),
            stride=len(latent),
        )
        layout = MultiArrayLayout(
            dim=[dim],
            data_offset=len(latent),
        )
        return Float32MultiArray(
            layout=layout,
            data=latent.astype(np.float32).tolist(),
        )


if __name__ == "__main__":
    try:
        VAEEncoderNode()
    except rospy.ROSInterruptException:
        pass
