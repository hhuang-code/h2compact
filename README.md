# H2-COMPACT: Human–Humanoid Co-Manipulation via Adaptive Contact Trajectory Policies

[![IsaacGym](https://img.shields.io/badge/IsaacGym-Preview4-b.svg)](https://developer.nvidia.com/isaac-gym) [![report](https://img.shields.io/badge/arxiv-report-red)](https://arxiv.org/abs/2505.17627) <a href="https://h2compact.github.io/h2compact/"><img alt="Project" src="https://img.shields.io/badge/-Project%20Page-lightgrey?logo=Google%20Chrome&color=informational&logoColor=white"></a> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

This guide provides detailed instructions for setting up and using the force sensor system, data collection, and robot control for the H2-COMPACT project.


---

## Force Sensor Setup

The force sensor system uses the JSD (Just 'SOEM' Drivers) library for EtherCAT communication with ATI Force-Torque sensors.

### Prerequisites

**Required:**
- SOEM (Simple Open EtherCAT Master)
- CMake 3.11 or later
- Posix threads

**Optional:**
```bash
sudo apt install libreadline-dev  # only used for test programs
```

### Building JSD from Source

1. Clone the JSD repository:
```bash
git clone git@github.com:nasa-jpl/jsd.git
cd jsd
```

2. Build the library:
```bash
mkdir build
cd build
cmake ..
make
```

3. Install JSD (installation will be done at `/opt/jsd`):
```bash
sudo make install
```

For detailed information, refer to the README.md file in the `force sensor installaion/jsd/` directory.

---

## ATI Sensor ROS Package Installation

After installing JSD, add the `ati_sensor` package to your catkin workspace:

1. Copy the `ati_sensor` package to your catkin workspace `src` directory:
```bash
cp -r ati_sensor ~/catkin_ws/src/
```

2. Build the package:
```bash
cd ~/catkin_ws
catkin_make
```

3. Source the workspace:
```bash
source ~/catkin_ws/devel/setup.bash
```

---

## Launching Force Sensors

### Step 1: Identify Network Interface Names

Before launching the sensors, you need to identify the network interface names for each force sensor:

1. Connect both force sensors (Mini 40 and Mini 45 (or the ATI sensor that your are using)) to your laptop via Ethernet.

2. Check available network interfaces:
```bash
ifconfig
```

3. Identify the interface names for each sensor. Look for interfaces like `enx00e04c6800dc` or `enx207bd2c36d89` (these are USB-to-Ethernet adapters).

### Step 2: Update Launch File

Edit the launch file `ati_sensor/launch/ati_dual_sensors.launch` and update the `ifname` parameters for each sensor:

- `mini40_ifname`: Set to the interface name for Mini 40 sensor
- `mini45_ifname`: Set to the interface name for Mini 45 sensor

Example:
```xml
<arg name="mini40_ifname" default="enx00e04c6800dc" doc="Network interface name for Mini 40"/>
<arg name="mini45_ifname" default="enx207bd2c36d89" doc="Network interface name for Mini 45"/>
```

### Step 3: Launch Both Sensors

Launch the dual sensors launch file:
```bash
roslaunch ati_sensor ati_dual_sensors.launch
```

**Note:** You may need to run with `sudo` privileges for EtherCAT access:
```bash
sudo -E bash -c "source ~/catkin_ws/devel/setup.bash && roslaunch ati_sensor ati_dual_sensors.launch"
```

### Step 4: Verify Sensor Streaming

Once launched, the sensors will publish Force-Torque data to the following topics:
- Mini 40: `/ati_ros_ati_mini40/ati_mini40/ft_meas`
- Mini 45: `/ati_ros_ati_mini45/ati_mini45/ft_meas`

You can verify the data is streaming using:
```bash
rostopic echo /ati_ros_ati_mini40/ati_mini40/ft_meas
rostopic echo /ati_ros_ati_mini45/ati_mini45/ft_meas
```

### Optional: Optimize EtherCAT Performance

For faster SOEM operation, you can optimize the network interface settings:
```bash
sudo ethtool -C <interface_name> rx-usecs 0 rx-frames 1 tx-usecs 0 tx-frames 1
```

Replace `<interface_name>` with the actual interface name (e.g., `enx00e04c6800dc`).

---

## Data Collection

For data collection, you need to run two scripts simultaneously:

### 1. ATI Zero Offset

This script performs zero-offset calibration for both force sensors and publishes zeroed force-torque data:

```bash
python3 ati_zero_offset.py
```

This script:
- Subscribes to raw force-torque measurements from both sensors
- Performs 3-second calibration to calculate offsets
- Publishes zeroed data to:
  - `/ati_ros_ati_mini40/ati_mini40/ft_meas_zeroed`
  - `/ati_ros_ati_mini45/ati_mini45/ft_meas_zeroed`

### 2. Camera Publisher

This script publishes RGB camera images from a RealSense camera:

```bash
python3 camera_publish.py
```

This script publishes RGB images to:
- `/camera/rgb/image_raw`

**Note:** You may need to adjust the camera device index in `camera_publish.py` (currently set to device index 4) if your camera uses a different index.

### Distributed ROS Setup

For data collection, we use a distributed ROS setup where:

1. **Force Sensor Device (Laptop/SBC)**: Force sensors are connected to a laptop or Single Board Computer (SBC) that moves with you. This device runs the force sensor launch file and acts as the ROS master/client (interchangable).

2. **Data Collection Device (Different Laptop)**: A separate laptop runs `ati_zero_offset.py` and `camera_publish.py` and subscribes to force topics over the network.

#### Setup Instructions

**Prerequisites:**
- Both devices must be on the same network (same WiFi)
- ROS must be installed on both devices
- The force sensor device must have the `ati_sensor` package built

**On Force Sensor Device (ROS Master):**

1. Find the IP address of this device:
```bash
ifconfig
# Note the IP address (e.g., 192.168.1.100)
```

2. Set ROS environment variables:
```bash
export ROS_MASTER_URI=http://<force_sensor_device_ip>:11311
export ROS_IP=<force_sensor_device_ip>
```

3. Launch the force sensors:
```bash
source ~/catkin_ws/devel/setup.bash
roslaunch ati_sensor ati_dual_sensors.launch
```

**On Data Collection Device (ROS Client):**

1. Find the IP address of this device:
```bash
ifconfig
# Note the IP address (e.g., 192.168.1.101)
```

2. Set ROS environment variables to connect to the master:
```bash
export ROS_MASTER_URI=http://<force_sensor_device_ip>:11311
export ROS_IP=<data_collection_device_ip>
```

Replace:
- `<force_sensor_device_ip>` with the IP address of the device running the force sensors
- `<data_collection_device_ip>` with the IP address of this device

3. Verify connection by checking available topics:
```bash
rostopic list
```

You should see the force sensor topics:
- `/ati_ros_ati_mini40/ati_mini40/ft_meas`
- `/ati_ros_ati_mini45/ati_mini45/ft_meas`

### Running Both Scripts

On the **Data Collection Device** (ROS client), open two separate terminals:

**Terminal 1:**
```bash
cd ~/h2compact
source ~/catkin_ws/devel/setup.bash
# Make sure ROS_MASTER_URI and ROS_IP are set (see Distributed ROS Setup above)
python3 ati_zero_offset.py
```

**Terminal 2:**
```bash
cd ~/h2compact
source ~/catkin_ws/devel/setup.bash
# Make sure ROS_MASTER_URI and ROS_IP are set (see Distributed ROS Setup above)
python3 camera_publish.py
```

**Note:** The `ati_zero_offset.py` script will subscribe to force topics from the force sensor device over the network, and `camera_publish.py` will publish camera images locally. Both topics will be available for recording with ROSBag.

---

## Data Recording with ROSBag

To record force-torque data and RGB camera images simultaneously, use ROSBag on the **Data Collection Device** (where `ati_zero_offset.py` and `camera_publish.py` are running):

### Record Force-Torque and Camera Topics

```bash
rosbag record /ati_ros_ati_mini40/ati_mini40/ft_meas_zeroed /ati_ros_ati_mini45/ati_mini45/ft_meas_zeroed /camera/rgb/image_raw -O <output_bag_name>.bag
```

Replace `<output_bag_name>` with your desired bag file name.



### Record with Compression (Recommended for Long Sessions)

To save disk space, use compression:

```bash
rosbag record /ati_ros_ati_mini40/ati_mini40/ft_meas_zeroed /ati_ros_ati_mini45/ati_mini45/ft_meas_zeroed /camera/rgb/image_raw -O <output_bag_name>.bag --bz2
```

### Playback Recorded Data

To playback and verify recorded data:

```bash
rosbag play <output_bag_name>.bag
```

### Check Bag File Contents

To inspect what topics are in a bag file:

```bash
rosbag info <output_bag_name>.bag
```

### Extract and Synchronize Data for Training

After recording data with ROSBag, use `sync_save_bag.py` to extract and synchronize force-torque data and RGB images for training. This script:

- Extracts force-torque measurements from both ATI sensors (mini40 and mini45) from topics ending with `ft_meas_zeroed`
- Extracts RGB images from `/camera/rgb/image_raw`
- Synchronizes all data by timestamp (aligns mini45 to mini40 timestamps, and images to force timestamps)
- Saves force data as CSV files and images as JPEG files

**Usage:**

Process a single bag file:
```bash
python3 sync_save_bag.py <path_to_bag_file>.bag
```

Process all bag files in a directory:
```bash
python3 sync_save_bag.py <path_to_directory_with_bags>/
```

**Example:**
```bash
python3 sync_save_bag.py data_collection_2024_01_15.bag
```

Or process multiple bags:
```bash
python3 sync_save_bag.py ./recorded_bags/
```

**Output Structure:**

The script creates the following directory structure:
```
<bag_name>/
├── forces/
│   ├── ati_mini40/
│   │   └── ati_mini40.txt  (CSV: timestamp,fx,fy,fz,tx,ty,tz)
│   └── ati_mini45/
│       └── ati_mini45.txt  (CSV: timestamp,fx,fy,fz,tx,ty,tz)
└── images/
    ├── <timestamp1>.jpg
    ├── <timestamp2>.jpg
    └── ...
```

**Notes:**
- Force data CSV files contain synchronized timestamps and 6 values per row: force (fx, fy, fz) and torque (tx, ty, tz)
- Images are saved with timestamps matching the nearest force measurement timestamp
- The script synchronizes mini45 force data to mini40 timestamps for alignment
- All timestamps are preserved for temporal alignment during training

---

## WHAM and Intent Inference Model

### Preparation
   1. Set up the conda environment and repo of WHAM: https://github.com/yohanshin/WHAM, refer to this [Installation](https://github.com/yohanshin/WHAM/blob/main/docs/INSTALL.md) section.
   2. Put the folders `data_collect`, `h2compact` under the root directory of WHAM
   3. Put the file `batch_demo.sh` under the root directory of WHAM and replace the original `demo.py`under WHAM with the one under the h2compact folder.

### Data Collection
   1. Follow the **Data Collection** section above to collect data. 
      - Preprocess the collected raw data to get `.txt` and `.jpg` files, and save the preprocessed data in `extracted_data` folder.  
      - NOTE: We provide our data [extracted_data](https://drive.google.com/file/d/15VXm2PtfuFKq3cJQ7oo_8kw-m7VabzNk/view?usp=sharing) so you can skip this step and start training.
   2. Run: ```bash batch_demo.sh``` to load WHAM to generate the SMPL models of human objects. 

### Training 
- Run: ```bash main.sh``` to train the `Intent Inference Model`.
- The checkpoints are saved into the `checkpoints` folder.

### Inference (Deployment)
- Run: ```bash test.sh``` to deploy the `Intent Inference Model`.
- The model generate high-level velocity command and will be converted to low-level ROS `cmd_vel` commands for movement control. 
- For detailed operations, follow the **Deployment** section below.

---

## Robot Training

This section covers robot policy training for dynamic load walking.

<!-- @Niraj: place holder
write somthing about this, maybe references or what chages you did or reward policy 
-->

---

## Deployment

After running the model, it publishes velocity commands to the `/cmd_vel` topic. To make the robot subscribe to these values and walk collaboratively, run the `g1_walk.py` script on the robot.

### Running g1_walk.py

The `g1_walk.py` script subscribes to `/cmd_vel` and sends motion commands to the Unitree G1 robot.

**Usage:**
```bash
python3 g1_walk.py <network_interface>
```

**Example:**
```bash
python3 g1_walk.py eth0
```

**Important Notes:**
- Replace `<network_interface>` with the actual network interface name used for robot communication (check with `ifconfig`)
- The script subscribes to `/cmd_vel` topic (geometry_msgs/Twist messages)
- It converts ROS cmd_vel commands to Unitree robot motion commands:
  - `linear.x`: Forward/backward motion
  - `linear.y`: Lateral (sideways) motion
  - `angular.z`: Rotational velocity

### Complete Workflow

1. **On the laptop/computer:**
   - Launch force sensors: `roslaunch ati_sensor ati_dual_sensors.launch`
   - Run zero offset: `python3 ati_zero_offset.py`
   - Run camera publisher: `python3 camera_publish.py`
   - Run your model (which publishes to `/cmd_vel`)

2. **On the robot:**
   - Run the walk script: `python3 g1_walk.py <network_interface>`

The robot will subscribe to `/cmd_vel` messages and execute collaborative walking based on the model's predictions.

---

## Troubleshooting

### Force Sensors Not Connecting
- Verify network interface names with `ifconfig` after connecting sensors
- Ensure sensors are powered on
- Check that the interface names in the launch file match the actual interfaces
- Try running with `sudo` privileges

### Camera Not Publishing
- Check camera device index in `camera_publish.py`
- Verify camera is connected: `ls /dev/video*`
- Adjust device index if needed (currently set to 4)

### ROSBag Recording Issues
- Ensure all topics are publishing before starting recording
- Check available disk space
- Verify topic names are correct: `rostopic list`

### Robot Not Responding to cmd_vel
- Verify network interface name is correct
- Check that `/cmd_vel` topic is publishing: `rostopic echo /cmd_vel`
- Ensure robot and computer are on the same network
- Check Unitree SDK installation and initialization

---

## Additional Resources

- JSD Library: https://github.com/nasa-jpl/jsd
- ATI Force-Torque Sensors Documentation
- Unitree G1 Robot SDK Documentation
- We also provide 3d printing files for box handles [here](https://drive.google.com/drive/folders/1m9Gnb3YexUC_VqL07D8ahP2rPpuELUrw?usp=sharing).
---

# Citation
If you find our work useful, please consider citing us:

```bibtex
@inproceedings{bethala2025h2,
  title={H2-COMPACT: Human-Humanoid Co-Manipulation via Adaptive Contact Trajectory Policies},
  author={Bethala, Geeta Chandra Raju and Huang, Hao and Pudasaini, Niraj and Ali, Abdullah Mohamed and Yuan, Shuaihang and Wen, Congcong and Tzes, Anthony and Fang, Yi},
  booktitle={2025 IEEE-RAS 24th International Conference on Humanoid Robots (Humanoids)}, 
  year={2025},
  pages={1004-1011}
}
```