"""
vae_ros_interface.py
--------------------
Thin, reusable wrapper around the VAE and existing loss/conversion utilities.
Intended to be imported by both the encoder node and the decoder node so that
preprocessing, encoding, decoding and occupancy-map conversion are never
duplicated.

Reused unchanged:
  - VAE / VAE_2                               (vae.py)
  - VAENetworkInterfaceValidation.forward()   (VAENetworkInterfaceValidation.py)
  - VAENetworkInterfaceValidation.latent_space_decoded()
  - convert_range_image_to_pointcloud         (loss_functions.py)
  - convert_point_cloud_to_occupancy_map      (loss_functions.py)
  - convert_occupancy_map_to_range_image      (loss_functions.py)
  - range_image_reprojection                  (loss_functions.py)
"""

import cv2
import numpy as np
import torch

from pcl_vae.inference.scripts.VAENetworkInterfaceValidation import VAENetworkInterfaceValidation
from pcl_vae.networks.Loss.loss_functions import (
    convert_range_image_to_pointcloud,
    convert_point_cloud_to_occupancy_map,
    convert_occupancy_map_to_range_image,
    range_image_reprojection,
)

INVALID_PIXEL_VALUE = -1.0


class VAERosInterface:
    """
    High-level interface for encoder and decoder ROS nodes.

    Parameters
    ----------
    robot_type   : str   – "aerial" or "ground"
    model_name   : str   – weight filename (resolved inside pcl_vae package)
    latent_dim   : int   – latent space dimensionality
    device       : str   – "cuda" or "cpu"
    max_depth    : float – sensor max range (metres)
    min_depth    : float – sensor min range (metres)
    voxel_size   : float – voxel edge length for occupancy map (metres)
    h_fov        : float – horizontal FoV (radians)
    v_fov        : float – vertical   FoV (radians)
    """

    def __init__(
        self,
        robot_type: str,
        model_name: str,
        latent_dim: int,
        device: str,
        max_depth: float,
        min_depth: float,
        voxel_size: float,
        h_fov: float,
        v_fov: float,
    ):
        self.device = device
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.voxel_size = voxel_size
        self.h_fov = h_fov
        self.v_fov = v_fov

        # Reuse the existing inference wrapper (loads weights, moves model to device).
        self._net = VAENetworkInterfaceValidation(
            robot_type=robot_type,
            model_name=model_name,
            latent_space_dim=latent_dim,
            device=device,
        )

    # ------------------------------------------------------------------
    # Preprocessing  (mirrors process_for_validation in vae_node_validation.py)
    # ------------------------------------------------------------------

    def preprocess(self, raw_tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalise a raw range-image tensor and fill invalid pixels with inpainting.

        Parameters
        ----------
        raw_tensor : (1, 1, H, W) float32 tensor with metric depth values.

        Returns
        -------
        filled_tensor : (1, 1, H, W) float32 tensor, values in [0, 1]
                        with invalid pixels mapped to ``-2 * INVALID_PIXEL_VALUE``.
        """
        img = raw_tensor.clone()

        # Clip and normalise to [0, 1]; out-of-range values → INVALID_PIXEL_VALUE
        img[img > self.max_depth] = self.max_depth
        img[img < self.min_depth] = INVALID_PIXEL_VALUE
        img = img / self.max_depth
        img[img < 0] = INVALID_PIXEL_VALUE

        # Fill invalid pixels using OpenCV inpainting (CPU operation)
        np_img = img.cpu().numpy().squeeze()          # (H, W)
        mask = (np_img == INVALID_PIXEL_VALUE).astype(np.uint8)
        filled_np = cv2.inpaint(np_img, mask, inpaintRadius=3, flags=cv2.INPAINT_NS)

        filled = torch.from_numpy(filled_np).float().unsqueeze(0).unsqueeze(0)
        filled = filled.to(self.device)

        # Remap remaining negatives so the network never sees INVALID_PIXEL_VALUE
        filled[filled < 0] = -2 * INVALID_PIXEL_VALUE
        return filled

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_range_image_to_latent(self, raw_tensor: torch.Tensor) -> np.ndarray:
        """
        Preprocess a raw range image and encode it to a latent vector.

        Parameters
        ----------
        raw_tensor : (1, 1, H, W) float32 tensor with metric depth values.

        Returns
        -------
        latent : (latent_dim,) float32 numpy array  – the mean of the posterior
                 (inference_mode=True sets eps=0, so z == mu).
        """
        preprocessed = self.preprocess(raw_tensor)

        # forward() already runs in torch.no_grad() + eval mode.
        _recon, means, _logvar, _time = self._net.forward(preprocessed.cpu().numpy())

        # means shape: (1, latent_dim) → (latent_dim,)
        return means.cpu().numpy().squeeze(0)

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode_latent_to_range_image(self, latent: np.ndarray) -> np.ndarray:
        """
        Decode a latent vector back to a normalised range image.

        Parameters
        ----------
        latent : (latent_dim,) or (1, latent_dim) float32 numpy array.

        Returns
        -------
        range_image_norm : (H, W) float32 numpy array in [0, 1].
        """
        vec = np.asarray(latent, dtype=np.float32)
        if vec.ndim == 1:
            vec = vec[None, :]                          # → (1, latent_dim)

        latent_tensor = torch.from_numpy(vec).to(self.device)

        # latent_space_decoded() already wraps decode() in no_grad.
        # It returns a (H, W) numpy array (squeezed).
        return self._net.latent_space_decoded(latent_tensor)

    # ------------------------------------------------------------------
    # Occupancy-map helpers  (thin wrappers, no logic duplication)
    # ------------------------------------------------------------------

    def range_image_to_occupancy_map(
        self, range_image_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert a metric (un-normalised) range image to a 3-D occupancy map.

        Parameters
        ----------
        range_image_tensor : (1, 1, H, W) float32 tensor with metric depth values.

        Returns
        -------
        occupancy_map : 3-D int32 torch tensor  (VOXEL_FREE / OCCUPIED / UNKNOWN).
        """
        point_cloud = convert_range_image_to_pointcloud(
            range_image_tensor, self.h_fov, self.v_fov
        )
        return convert_point_cloud_to_occupancy_map(
            point_cloud, self.voxel_size, self.max_depth
        )

    def reconstructed_norm_to_occupancy_map(
        self,
        recon_norm: np.ndarray,
        img_height: int,
        img_width: int,
    ) -> torch.Tensor:
        """
        Go from a normalised reconstructed range image (decoder output) straight
        to an occupancy map, reusing range_image_reprojection.

        Parameters
        ----------
        recon_norm : (H, W) float32 numpy array in [0, 1].
        img_height : int
        img_width  : int

        Returns
        -------
        occupancy_map : 3-D int32 torch tensor.
        """
        # Wrap in the expected (1, 1, H, W) batch tensor
        recon_tensor = (
            torch.from_numpy(recon_norm)
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )

        # De-normalise to metric range image for point-cloud conversion
        metric_tensor = recon_tensor * self.max_depth

        point_cloud = convert_range_image_to_pointcloud(
            metric_tensor, self.h_fov, self.v_fov
        )
        return convert_point_cloud_to_occupancy_map(
            point_cloud, self.voxel_size, self.max_depth
        )

    # ------------------------------------------------------------------
    # Factory – build from a yaml config dict (matches vae_validation_config.yaml)
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: dict) -> "VAERosInterface":
        """
        Convenience constructor that accepts the same dictionary produced by
        yaml.safe_load(vae_validation_config.yaml).

        Example
        -------
        with open("vae_validation_config.yaml") as f:
            cfg = yaml.safe_load(f)
        iface = VAERosInterface.from_config(cfg)
        """
        return cls(
            robot_type=cfg["robot_type"],
            model_name=cfg["model_name"],
            latent_dim=cfg["latent_space"],
            device=cfg["device"],
            max_depth=cfg["image_max_depth"],
            min_depth=cfg["image_min_depth"],
            voxel_size=cfg["voxel_size"],
            h_fov=cfg["h_fov"],
            v_fov=cfg["v_fov"],
        )
