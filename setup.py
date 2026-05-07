from setuptools import find_packages, setup

try:
    from catkin_pkg.python_setup import generate_distutils_setup
except ImportError:
    generate_distutils_setup = None


setup_kwargs = {
    "name": "pcl_vae",
    "version": "0.0.3",
    "author": "Angelos Zacharia",
    "author_email": "angelos.zacharia@ntnu.no",
    "description": "Package for training and evaluating task-driven point cloud compression VAEs",
    "license": "BSD-3-Clause",
    "packages": find_packages(),
    "package_data": {
        "pcl_vae": [
            "inference/config/*/*.yaml",
            "train/config/*/*.yaml",
            "weights/*",
            "weights/**/*",
        ],
    },
    "install_requires": [
        "numpy",
        "torch>=1.13",
    ],
}

if generate_distutils_setup is not None:
    catkin_kwargs = generate_distutils_setup(
        packages=setup_kwargs["packages"],
        package_dir={"": "."},
    )
    catkin_kwargs.update(setup_kwargs)
    setup_kwargs = catkin_kwargs

setup(**setup_kwargs)
