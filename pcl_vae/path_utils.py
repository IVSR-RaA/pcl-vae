"""Helpers for locating package resources in source and catkin layouts."""

import os

try:
    import rospkg
except ImportError:
    rospkg = None


def get_package_root() -> str:
    """Return the ROS package root when available, else fall back to source layout."""
    if rospkg is not None:
        try:
            return rospkg.RosPack().get_path("pcl_vae")
        except rospkg.ResourceNotFound:
            pass

    package_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(package_dir, os.pardir))


def get_package_data_path(*parts: str) -> str:
    """Return a path inside the package data tree under ``<pkg_root>/pcl_vae``."""
    return os.path.join(get_package_root(), "pcl_vae", *parts)


def get_validation_config_path(robot_type: str) -> str:
    """Return the validation config path for the selected robot type."""
    return get_package_data_path(
        "inference", "config", robot_type, "vae_validation_config.yaml"
    )


def get_training_config_path(robot_type: str) -> str:
    """Return the training config path for the selected robot type."""
    return get_package_data_path("train", "config", robot_type, "train_config.yaml")


def get_dataset_package_path() -> str:
    """Return the package directory that contains dataset assets."""
    return get_package_data_path("datasets")


def get_weights_path(model_name: str) -> str:
    """Return the path to a model checkpoint stored under ``pcl_vae/weights``."""
    return get_package_data_path("weights", model_name)
