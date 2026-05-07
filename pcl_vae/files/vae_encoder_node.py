#!/usr/bin/env python3
"""
vae_encoder_node.py
-------------------
ROS 1 node – VAE **Encoder / Publisher**

Subscribes
----------
~input/range_image  (sensor_msgs/Image, 32FC1)
    Raw range image in metres.  Expected shape matches the sensor config in
    vae_validation_config.yaml (e.g. 64×512 for aerial, 16×1800 for ground).

Publishes
---------
~output/latent_vector  (std_msgs/Float32MultiArray)
    Flattened latent vector z = mu  (inference_mode zeroes the noise term).
    The ``layout.data_offset`` field carries the latent dimensionality so the
    decoder can validate it without a second parameter channel.

Parameters (ROS)
----------------
~config_path   : str  – absolute path to vae_validation_config.yaml
~robot_type    : str  – "aerial" | "ground"  (overrides yaml if provided)
~queue_size    : int  – subscriber / publisher queue size  (default: 1)
"""

import os
import sys

import numpy as np
import torch
import yaml

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout

# ---------------------------------------------------------------------------
# Locate and import the shared interface (handle both installed-package and
# source-tree layouts gracefully).
# ---------------------------------------------------------------------------
try:
    from pcl_vae.inference.scripts.vae_ros_interface import VAERosInterface
except ImportError:
    _HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _HERE)
    from pcl_vae.inference.scripts.vae_ros_interface import VAERosInterface


class VAEEncoderNode:
    """Encodes incoming range images and publishes their latent vectors."""

    def __init__(self):
        rospy.init_node("vae_encoder_node", anonymous=False)

        # ------------------------------------------------------------------ #
        # Parameters
        # ------------------------------------------------------------------ #
        config_path = rospy.get_param("~config_path")
        robot_type_override = rospy.get_param("~robot_type", None)
        queue_size = rospy.get_param("~queue_size", 1)

        with open(config_path, "r") as fh:
            cfg = yaml.safe_load(fh)

        # Allow the ROS parameter to override the yaml field
        if robot_type_override:
            cfg["robot_type"] = robot_type_override
        elif "robot_type" not in cfg:
            rospy.logfatal(
                "[vae_encoder_node] 'robot_type' missing from config and not "
                "provided as a ROS parameter."
            )
            raise SystemExit(1)

        self._cfg = cfg
        self._latent_dim = cfg["latent_space"]
        self._max_depth = cfg["image_max_depth"]

        rospy.loginfo(
            f"[vae_encoder_node] robot_type={cfg['robot_type']}  "
            f"latent_dim={self._latent_dim}  device={cfg['device']}"
        )

        # ------------------------------------------------------------------ #
        # VAE interface
        # ------------------------------------------------------------------ #
        self._iface = VAERosInterface.from_config(cfg)
        rospy.loginfo("[vae_encoder_node] VAE model loaded.")

        # ------------------------------------------------------------------ #
        # ROS plumbing
        # ------------------------------------------------------------------ #
        self._bridge = CvBridge()

        self._pub = rospy.Publisher(
            "~output/latent_vector",
            Float32MultiArray,
            queue_size=queue_size,
        )

        rospy.Subscriber(
            "~input/range_image",
            Image,
            self._image_callback,
            queue_size=queue_size,
        )

        rospy.loginfo("[vae_encoder_node] Ready – waiting for range images.")
        rospy.spin()

    # ---------------------------------------------------------------------- #
    # Callback
    # ---------------------------------------------------------------------- #

    def _image_callback(self, msg: Image) -> None:
        """Convert incoming Image → latent vector → Float32MultiArray."""
        try:
            # ---- ROS Image → (1, 1, H, W) float32 tensor -----------------
            cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            raw_tensor = (
                torch.from_numpy(cv_img.astype(np.float32))
                .unsqueeze(0)   # H×W  → 1×H×W
                .unsqueeze(0)   # 1×H×W → 1×1×H×W
            )

            # ---- Encode ---------------------------------------------------
            latent = self._iface.encode_range_image_to_latent(raw_tensor)
            # latent : (latent_dim,) float32 numpy array

            # ---- Pack into Float32MultiArray ------------------------------
            out_msg = self._build_latent_msg(latent, msg.header)
            self._pub.publish(out_msg)

        except Exception as exc:  # noqa: BLE001
            rospy.logerr(f"[vae_encoder_node] Failed to encode image: {exc}")

    # ---------------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------------- #

    def _build_latent_msg(self, latent: np.ndarray, header) -> Float32MultiArray:
        """
        Wrap a 1-D latent numpy array in a Float32MultiArray.

        Layout
        ------
        layout.dim[0].label  = "latent"
        layout.dim[0].size   = latent_dim
        layout.dim[0].stride = latent_dim
        layout.data_offset   = latent_dim   (convenience: carries dim for decoder)
        """
        dim = MultiArrayDimension(
            label="latent",
            size=len(latent),
            stride=len(latent),
        )
        layout = MultiArrayLayout(
            dim=[dim],
            data_offset=len(latent),   # encoder stamps its latent_dim here
        )
        msg = Float32MultiArray(
            layout=layout,
            data=latent.astype(np.float32).tolist(),
        )
        return msg


# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    try:
        VAEEncoderNode()
    except rospy.ROSInterruptException:
        pass
