#  <div align="center">Task-Driven Point Cloud Compression for Volumetric Mapping</div>
<div align="center"> <a href="https://ntnu-arl.github.io/marsupial-collaborative-exploration/"><img src="https://img.shields.io/badge/Homepage-1E88E5?style=flat-square" alt="Webpage"></a>  <a href="https://ieeexplore.ieee.org/document/11159273"><img src="https://img.shields.io/badge/IEEE-00629B?style=flat-square" alt="IEEE"></a> <a href="https://arxiv.org/abs/2509.07655"><img src="https://img.shields.io/badge/arXiv-78909C?style=flat-square" alt="arXiv"></a> <a href="https://doi.org/10.5281/zenodo.18493894"><img src="https://img.shields.io/badge/Zenodo-0D98F2?style=flat-square" alt="Zenodo"></a> <a href="https://www.youtube.com/watch?v=VEYS5BjmZP8"><img src="https://img.shields.io/badge/YouTube-E57373?style=flat-square" alt="YouTube"></a> </div>

This repository contains the code for a bandwidth-efficient task-driven point cloud compression solution, developed as part of the paper titled _"Collaborative Exploration with a Marsupial Ground-Aerial Robot Team through Task-Driven Map Compression"_ .

![vae_model](img/pcl_vae_architecture.png)

We propose a bandwidth-efficient, task-driven point cloud compression method tailored for volumetric map reconstruction at mission-relevant resolutions. By emphasizing occupancy-relevant structure over raw point cloud fidelity, our approach achieves high compression rates while retaining the information essential for planning.

The proposed solution supports data gathered from two different LiDAR sensors:

* **Ouster OS0-64** (used by the aerial robot) 
* **Velodyne VLP-16** (used by the ground robot).


## Setup
#### Clone the repository
```bash
cd ~/all_ws/src
git clone git@github.com:ntnu-arl/pcl-vae.git
```

#### Install
To install the repository, run the following commands:
```bash
source /home/nlg/pcl-vae/env/bin/activate
cd ~/all_ws/src/pcl-vae
pip install -e .
```

#### ROS Noetic (catkin)
This repository can be used as a ROS 1 package named `pcl_vae` inside a catkin workspace.

Tested workspace layout:

- repo: `~/all_ws/src/pcl-vae`
- Python environment: `/home/nlg/pcl-vae/env`

Activate the environment and build from your workspace root:

```bash
source /home/nlg/pcl-vae/env/bin/activate
cd ~/all_ws
catkin build pcl_vae --force-cmake --cmake-args -DPYTHON_EXECUTABLE=/home/nlg/pcl-vae/env/bin/python3
source ~/all_ws/devel/setup.bash
```

`catkin` generates Python relay scripts for one interpreter. If you change the virtualenv later, rebuild with the same `-DPYTHON_EXECUTABLE=...` pattern so `roslaunch` uses the correct Python.

#### ROS Weights And Config
Place `.pth` files under:

- `~/all_ws/src/pcl-vae/pcl_vae/weights`

Then update the matching validation config:

- `pcl_vae/inference/config/aerial/vae_validation_config.yaml`
- `pcl_vae/inference/config/ground/vae_validation_config.yaml`

The following fields must match the selected weight:

- `robot_type`
- `model_name`
- `latent_space`
- `voxel_size`

Example:

```yaml
model_name: aerial_model_LD_32_epoch_20_batch_16_range_20_voxel_40.pth
robot_type: aerial
latent_space: 32
voxel_size: 0.4
```

Adding a new weight file does not require rebuilding the workspace.

#### ROS Topics
The ROS nodes use the following topics:

- `/point_to_range_image/input/points` : `sensor_msgs/PointCloud2`
- `/point_to_range_image/output/range_image` : `sensor_msgs/Image` with `32FC1` encoding
- `/vae_encoder/input/range_image` : `sensor_msgs/Image` with `32FC1` encoding
- `/vae_encoder/output/latent_vector` : `std_msgs/Float32MultiArray`
- `/vae_decoder/output/range_image` : `sensor_msgs/Image` with `32FC1` encoding
- `/vae_decoder/output/range_image_norm` : optional normalized reconstruction
- `/vae_decoder/output/occupancy_map` : optional `std_msgs/Int32MultiArray`

The launch file remaps the decoder latent input to the encoder latent output automatically.

## ROS Launch
Run the encoder/decoder pipeline when you already have a range image source:

```bash
source /home/nlg/pcl-vae/env/bin/activate
source ~/all_ws/devel/setup.bash
roslaunch pcl_vae vae_latent_communication.launch robot_type:=ground
```

Useful launch arguments:

- `robot_type:=aerial|ground`
- `publish_occ_map:=true|false`
- `publish_norm_image:=true|false`
- `queue_size:=1`
- `config_path:=/absolute/path/to/vae_validation_config.yaml`

Example with all decoder outputs enabled:

```bash
roslaunch pcl_vae vae_latent_communication.launch \
  robot_type:=aerial \
  publish_occ_map:=true \
  publish_norm_image:=true
```

`publish_occ_map:=true` requires the Python package `warp-lang`. The occupancy map is large, so leave it disabled unless you need it.

#### PointCloud2 To Range Image Bridge
If your upstream stack publishes raw LiDAR scans as `sensor_msgs/PointCloud2`, use the bridge node to project them into the range-image format expected by the VAE encoder.

Bridge only:

```bash
source /home/nlg/pcl-vae/env/bin/activate
source ~/all_ws/devel/setup.bash
roslaunch pcl_vae point_to_rangeimage.launch \
  robot_type:=ground \
  point_cloud_topic:=/mid/points
```

Full `PointCloud2 -> range image -> latent -> decoded range image` pipeline:

```bash
source /home/nlg/pcl-vae/env/bin/activate
source ~/all_ws/devel/setup.bash
roslaunch pcl_vae super_lio_to_vae.launch \
  robot_type:=ground \
  point_cloud_topic:=/mid/points
```

Useful arguments for the bridge launches:

- `robot_type:=aerial|ground`
- `point_cloud_topic:=/your/raw/pointcloud/topic`
- `config_path:=/absolute/path/to/vae_validation_config.yaml`
- `publish_occ_map:=true|false` for `super_lio_to_vae.launch`
- `publish_norm_image:=true|false` for `super_lio_to_vae.launch`

#### Super-LIO Integration
`pcl_vae` should consume the raw LiDAR scan topic used by `Super-LIO`, not the fused SLAM map outputs.

Use:

- raw sensor topic such as `/points_raw`, `/mid/points`, or another sensor-frame `PointCloud2`

Do not use:

- `/lio/cloud_world`
- `/lio/robo/cloud_world`
- any world-frame accumulated map or dense fused cloud

The VAE was trained on single-scan range images in the sensor frame. Feeding a fused world cloud changes the geometry and will not match the training distribution.

Examples:

- `Super-LIO` `M2DGR.yaml` uses `/mid/points` with `lidar_type: 4` (`VELO32`). You can bridge that topic with:

```bash
roslaunch pcl_vae super_lio_to_vae.launch \
  robot_type:=ground \
  point_cloud_topic:=/mid/points
```

- `Super-LIO` `MCD_ATH.yaml` uses `/livox/lidar` with `lidar_type: 1` (`LIVOX`). The current bridge node does not subscribe to `livox_ros_driver/CustomMsg`; it currently supports `sensor_msgs/PointCloud2` only.

#### Sensor Compatibility Notes
The provided pretrained `pcl_vae` models are for:

- aerial: `Ouster OS0-64`
- ground: `Velodyne VLP-16`

So there are two separate questions:

1. Can the ROS bridge run with your upstream sensor topic?
   Yes, if the input is `sensor_msgs/PointCloud2`.
2. Will a pretrained model give good compression quality on that sensor?
   Not necessarily.

Examples:

- `VELO32` data can be bridged and encoded, but model quality may be suboptimal because the pretrained ground model was trained for `VLP-16`, not `VELO32`.
- `Livox` data needs an additional adapter node, and in practice usually also needs retraining or fine-tuning because the scan pattern differs strongly from spinning lidars.

#### Direct Python Validation
If you want to validate a model without ROS:

```bash
source /home/nlg/pcl-vae/env/bin/activate
source ~/all_ws/devel/setup.bash
cd ~/all_ws/src/pcl-vae/pcl_vae/inference/src
python vae_node_validation.py --robot_type=aerial
```

## Folder Description
The folders contain the following:

- **datasets**: Contain scripts that utilize pytorch's dataset class to read from dataset files
- **inference**: Contains the scripts for running the pcl_vae node
- **launch**: ROS launch files for the VAE pipeline and point-cloud bridge
- **networks**: Contains the VAE networks, and the loss function for training the VAEs
- **scripts**: ROS nodes for encoding, decoding, and point-cloud to range-image conversion
- **weights**:  Contains the weights for the VAEs



## Datasets
For each robot, we provide a dataset containing both real and simulated range images from diverse environments, including caves, confined spaces, and complex buildings. Relevant files [here](https://figshare.com/s/ce4e4b87b3a28a75be27)

#### Download the datasets
```bash
cd ~/all_ws/src/pcl-vae/pcl_vae/datasets/
wget -O datasets.zip "https://ndownloader.figshare.com/files/53055530?private_link=ce4e4b87b3a28a75be27"
unzip datasets.zip 
```

##### Aerial Robot Dataset
- Range images (`64×512 (H×W)`) generated by OS0-64 LiDAR sensor. 
- Includes $\sim36,000$ range images ($\sim26,000$ simulated)
- Training set ($90 \%$) and Testing set ($10 \%$)

##### Ground Robot Dataset
- Range images (`16×1800 (H×W)`) generated by VLP-16 LiDAR sensor. 
- Includes  $\sim25,000$ ($\sim21,000$ real).
- Training set ($90$%) and Testing set ($10$%)


## Run a demo with pre-trained models
For each robot, we provide pre-trained models with varying latent space size `{32,64,128,256,512,1024}` and voxel size `{0.2,0.3,0.4}m`. Relevant files [here](https://figshare.com/s/ce4e4b87b3a28a75be27)

#### Download pre-trained models
For aerial robot models
```bash
cd ~/all_ws/src/pcl-vae/pcl_vae/weights/
wget -O aerial_robot_pcl_vae_models.zip "https://ndownloader.figshare.com/files/53083268?private_link=ce4e4b87b3a28a75be27"
unzip aerial_robot_pcl_vae_models.zip 
```

For ground robot models
```bash
cd ~/all_ws/src/pcl-vae/pcl_vae/weights/
wget -O ground_robot_pcl_vae_models.zip "https://ndownloader.figshare.com/files/53083163?private_link=ce4e4b87b3a28a75be27"
unzip ground_robot_pcl_vae_models.zip 
```

#### Configure
To validate a pre-trained model, you need to modify the parameters in the configuration file to match the desired settings. For both robot types, a `vae_validation_config.yaml` file exists under the `inference/config/<robot-type>` folder with `<robot-type> = {aerial, ground}`. A detailed description of all parameters is included in the configuration file.

**Note:** Set `model name`, `latent_space`, and `voxel_size` variables in `inference/config/<robot-type>/vae_validation_config.yaml`.

#### Run
Run the following command specifying the robot type. Replace `<robot-type>` with `aerial` or `ground` for validate the the pre-trained model for aerial or ground robot, respectively.
```bash
source /home/nlg/pcl-vae/env/bin/activate
source ~/all_ws/devel/setup.bash
cd ~/all_ws/src/pcl-vae/pcl_vae/inference/src
python vae_node_validation.py --robot_type=<robot-type>
```
An example of the output is shown in the figure below, which includes the raw range image input to the model, its voxel-aware representation, and the reconstructed range image. range_image_comparison
![range_image_comparison](img/range_images_comparison.png)

**Note:** To go through the entire dataset, press `n` to move from image to image.




## Training Models

#### Configure
To train a `pcl_vae` model, you need to modify the parameters in the configuration file to match the desired settings. For all robot types, a `train_config.yaml` file exists under the `train/config/<robot-type>` folder with `<robot-type> = {aerial, ground}`. A detailed description of all parameters is included in the configuration file.

#### Run
To start training the model, run the following command, specifying the robot type. Replace `<robot-type>` with aerial or ground for training a pcl_vae for aerial or ground robot, respectively.
```bash
source /home/nlg/pcl-vae/env/bin/activate
source ~/all_ws/devel/setup.bash
cd ~/all_ws/src/pcl-vae/pcl_vae/train
python train_pcl_vae.py --robot_type=<robot-type>
```
**Note:** The model is saved under the `train/weights/<robot-type>/<robot-type>_model` folder.



## Testing Models

#### Configure
To validate the trained model, you need to modify the parameters in the configuration file to match the desired settings. For both robot types, a `vae_validation_config.yaml` file exists under the `inference/config/<robot-type>` folder with `<robot-type> = {aerial, ground}`. A detailed description of all parameters is included in the configuration file.

**Note:** Move the trained model to `weights` folder, set `model name`, `latent_space`, and `voxel_size` variables in `inference/config/<robot-type>/vae_validation_config.yaml`.

#### Run
Run the following command specifying the robot type. Replace `<robot-type>` with `aerial` or `ground` for validate the the pre-trained model for aerial or ground robot, respectively.
```bash
source /home/nlg/pcl-vae/env/bin/activate
source ~/all_ws/devel/setup.bash
cd ~/all_ws/src/pcl-vae/pcl_vae/inference/src
python vae_node_validation.py --robot_type=<robot-type>
```


## Citation
If you use this work in your research, please cite the following publication:

```bibtex
@article{zacharia2025collaborative,
  title={Collaborative Exploration with a Marsupial Ground-Aerial Robot Team through Task-Driven Map Compression},
  author={Zacharia, Angelos and Dharmadhikari, Mihir and Alexis, Kostas},
  journal={IEEE Robotics and Automation Letters},
  year={2025},
  publisher={IEEE}
}
```



## License
Released under BSD-3-Clause.

## Acknowledgements
This open-source release is based upon work supported by the **European Commission** through:
- **Project SYNERGISE** under the **Horizon Europe Grant Agreement No. 101121321**
- **Project SPEAR** under the **Horizon Europe Grant Agreement No. 101119774**
- **Project DIGIFOREST** under the **Horizon Europe Grant Agreement No. 101070405** 


## Contact
For questions or support, reach out via [GitHub Issues](https://github.com/ntnu-arl/pcl-vae/issues) or contact authors:

* [Angelos Zacharia](mailto:angelos.zacharia@ntnu.no)
* [Mihir Dharmadhikari](mailto:mihir.dharmadhikari@ntnu.no)
* [Kostas Alexis](mailto:konstantinos.alexis@ntnu.no)

---
