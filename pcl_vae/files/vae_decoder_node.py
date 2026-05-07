#!/usr/bin/env python3
"""
vae_decoder_node.py
-------------------
ROS 1 node – VAE **Decoder / Subscriber**

Subscribes
----------
~input/latent_vector  (std_msgs/Float32MultiArray)
    Latent vector published by vae_encoder_node.

Publishes
---------
~output/range_image          (sensor_msgs/Image, 32FC1)
    Reconstructed range image in *metric* metres  (de-normalised by max_depth).

~output/range_image_norm     (sensor_msgs/Image, 32FC1)   [optional]
    Normalised reconstruction  [0, 1]  – useful for debugging.

~output/occupancy_map        (std_msgs/Int32MultiArray)    [optional]
    Flattened 3-D occupancy map (VOXEL_FREE=0 / OCCUPIED=1 / UNKNOWN=2).
    Published only when ~publish_occupancy_map is True.
    ``layout.dim`` carries  [X, Y, Z]  grid dimensions.

Parameters (ROS)
----------------
~config_path              : str   – absolute path to vae_validation_config.yaml
~robot_type               : str   – "aerial" | "ground"  (overrides yaml if set)
~publish_occupancy_map    : bool  – enable occupancy map output  (default: False)
~publish_normalised_image : bool  – enable normalised image output (default: False)
~queue_size               : int   – subscriber / publisher queue size (default: 1)
"""

import os
import sys

import numpy as np
import torch
import yaml

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Int32MultiArray, MultiArrayDimension, MultiArrayLayout

try:
    from pcl_vae.inference.scripts.vae_ros_interface import VAERosInterface
except ImportError:
    _HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _HERE)
    from pcl_vae.inference.scripts.vae_ros_interface import VAERosInterface


class VAEDecoderNode:
    """Decodes incoming latent vectors and publishes reconstructed range images."""

    def __init__(self):
        rospy.init_node("vae_decoder_node", anonymous=False)

        # ------------------------------------------------------------------ #
        # Parameters
        # ------------------------------------------------------------------ #
        config_path = rospy.get_param("~config_path")
        robot_type_override = rospy.get_param("~robot_type", None)
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

        self._cfg = cfg
        self._latent_dim = cfg["latent_space"]
        self._max_depth = cfg["image_max_depth"]
        self._image_height = cfg["image_height"]
        self._image_width = cfg["image_width"]

        rospy.loginfo(
            f"[vae_decoder_node] robot_type={cfg['robot_type']}  "
            f"latent_dim={self._latent_dim}  device={cfg['device']}  "
            f"publish_occupancy_map={self._publish_occ}"
        )

        # ------------------------------------------------------------------ #
        # VAE interface
        # ------------------------------------------------------------------ #
        self._iface = VAERosInterface.from_config(cfg)
        rospy.loginfo("[vae_decoder_node] VAE model loaded.")

        # ------------------------------------------------------------------ #
        # ROS plumbing
        # ------------------------------------------------------------------ #
        self._bridge = CvBridge()

        self._pub_range = rospy.Publisher(
            "~output/range_image", Image, queue_size=queue_size
        )

        self._pub_norm = None
        if self._publish_norm:
            self._pub_norm = rospy.Publisher(
                "~output/range_image_norm", Image, queue_size=queue_size
            )

        self._pub_occ = None
        if self._publish_occ:
            self._pub_occ = rospy.Publisher(
                "~output/occupancy_map", Int32MultiArray, queue_size=queue_size
            )

        rospy.Subscriber(
            "~input/latent_vector",
            Float32MultiArray,
            self._latent_callback,
            queue_size=queue_size,
        )

        rospy.loginfo("[vae_decoder_node] Ready – waiting for latent vectors.")
        rospy.spin()

    # ---------------------------------------------------------------------- #
    # Callback
    # ---------------------------------------------------------------------- #

    def _latent_callback(self, msg: Float32MultiArray) -> None:
        """Decode a latent vector and publish the reconstructed range image."""
        try:
            latent = np.array(msg.data, dtype=np.float32)     # (latent_dim,)

            # Sanity-check against our loaded model
            expected_dim = msg.layout.data_offset              # encoder stamps this
            if expected_dim and int(expected_dim) != self._latent_dim:
                rospy.logwarn(
                    f"[vae_decoder_node] Received latent_dim={int(expected_dim)} "
                    f"but model expects {self._latent_dim}. Proceeding anyway."
                )

            # ---- Decode → normalised range image (H, W) float32 ----------
            recon_norm = self._iface.decode_latent_to_range_image(latent)
            # Returned by latent_space_decoded() already as (H, W) numpy

            # ---- De-normalise to metric -----------------------------------
            recon_metric = (recon_norm * self._max_depth).astype(np.float32)

            # ---- Publish metric range image --------------------------------
            stamp = rospy.Time.now()
            self._pub_range.publish(
                self._to_image_msg(recon_metric, stamp, frame_id="sensor_frame")
            )

            # ---- Publish normalised image (optional) ----------------------
            if self._publish_norm and self._pub_norm is not None:
                self._pub_norm.publish(
                    self._to_image_msg(
                        recon_norm.astype(np.float32), stamp, frame_id="sensor_frame"
                    )
                )

            # ---- Publish occupancy map (optional) -------------------------
            if self._publish_occ and self._pub_occ is not None:
                occ_msg = self._build_occupancy_msg(recon_norm)
                self._pub_occ.publish(occ_msg)

        except Exception as exc:  # noqa: BLE001
            rospy.logerr(f"[vae_decoder_node] Failed to decode latent vector: {exc}")

    # ---------------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------------- #

    def _to_image_msg(
        self, np_img: np.ndarray, stamp, frame_id: str = "sensor_frame"
    ) -> Image:
        """Convert (H, W) float32 numpy array → sensor_msgs/Image (32FC1)."""
        msg = self._bridge.cv2_to_imgmsg(np_img, encoding="32FC1")
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        return msg

    def _build_occupancy_msg(self, recon_norm: np.ndarray) -> Int32MultiArray:
        """
        Convert normalised reconstruction to occupancy map and pack it as
        a flat Int32MultiArray with shape info in ``layout.dim``.
        """
        # Wrap normalised image into tensor for the interface helper
        occ_map = self._iface.reconstructed_norm_to_occupancy_map(
            recon_norm,
            img_height=self._image_height,
            img_width=self._image_width,
        )
        # occ_map : (X, Y, Z) int32 tensor
        X, Y, Z = occ_map.shape

        dims = [
            MultiArrayDimension(label="x", size=X, stride=X * Y * Z),
            MultiArrayDimension(label="y", size=Y, stride=Y * Z),
            MultiArrayDimension(label="z", size=Z, stride=Z),
        ]
        layout = MultiArrayLayout(dim=dims, data_offset=0)

        flat = occ_map.cpu().numpy().astype(np.int32).flatten().tolist()
        return Int32MultiArray(layout=layout, data=flat)


# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    try:
        VAEDecoderNode()
    except rospy.ROSInterruptException:
        pass
