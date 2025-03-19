# Multimodal Sensor Fusion

This is a curated list of resources, libraries, tools, frameworks, and practical implementations for **Multimodal sensor fusion**.

This collection is designed for developers, researchers, and industrial experts working in autonomous driving, robotics, and other perception-based applications.

## Table of Contents

- [Introduction to Sensor Fusion](#introduction-to-sensor-fusion)
- [Sensor Fusion Types and Approaches](#sensor-fusion-types-and-approaches)
- [Libraries and Frameworks](#libraries-and-frameworks)
- [Sensor-Specific Fusion Combinations](#sensor-specific-fusion-combinations)
- [Algorithms and Methods](#algorithms-and-methods)
- [Implementation Examples](#implementation-examples)
- [Datasets and Benchmarks](#datasets-and-benchmarks)
- [Tools and Utilities](#tools-and-utilities)
- [Research Papers and Publications](#research-papers-and-publications)
- [Courses and Tutorials](#courses-and-tutorials)
- [Industrial Applications](#industrial-applications)

## Introduction to Sensor Fusion

### What is Sensor Fusion?

Sensor fusion is the process of combining data from multiple sensors to achieve more accurate and reliable information than would be possible using individual sensors. In autonomous vehicles and robotics, perception refers to the processing and interpretation of sensor data to detect, identify, classify, and track objects.

### Key Concepts

- **Multi-Modal Sensing**: Combining different types of sensors (cameras, LiDAR, radar, etc.)
- **Temporal Fusion**: Combining multiple readings over time from the same sensor
- **Spatial Fusion**: Combining readings from sensors at different physical locations
- **Complementary Fusion**: Using sensors that observe different aspects of the environment
- **Competitive Fusion**: Using sensors that observe the same aspect for redundancy

## Sensor Fusion Types and Approaches

### By Processing Level

#### Low-Level (Early/Raw Data) Fusion
- **Description**: Merges raw sensor data before any feature extraction or object detection
- **Advantages**: Maximum information preservation, better SNR, can overcome single sensor faults
- **Disadvantages**: Computationally intensive, requires precise sensor calibration
- **Applications**: High-precision mapping, detailed environment reconstruction
- **Example**: LiDAR point cloud and camera image raw fusion before object detection

#### Mid-Level (Feature) Fusion
- **Description**: Extracts features from each sensor independently, then fuses these features
- **Advantages**: Balanced performance and efficiency, suitable for real-time applications
- **Disadvantages**: Requires feature compatibility between sensors
- **Applications**: Autonomous driving, real-time robotics tasks
- **Example**: Fusing CNN features from camera with geometric features from LiDAR

#### High-Level (Decision/Object) Fusion
- **Description**: Each sensor performs independent object detection, then results are fused
- **Advantages**: Computationally efficient, modular design, easier integration
- **Disadvantages**: May lose detailed information, relies on individual sensor performance
- **Applications**: Large-scale systems requiring quick decision-making, resource-constrained environments
- **Example**: Combining bounding boxes from camera, LiDAR, and radar detectors

### By Range Capability

| Fusion Type  | Range   | Applications                              | Strengths                        | Weaknesses                        |
|--------------|---------|-------------------------------------------|----------------------------------|-----------------------------------|
| Short-range  | <10m    | Parking assistance, robotic manipulation, close proximity operations | High precision, low latency      | Limited field of view             |
| Mid-range    | 10-50m  | Urban driving, indoor/outdoor robotics, drone operations            | Good balance of range and precision | Moderate computational requirements |
| Long-range   | >50m    | Highway driving, aviation, maritime navigation                     | Extended detection range         | Lower precision at extreme distances |

## Libraries and Frameworks

### Open Source Libraries

#### General Purpose

- [**libRSF**](https://github.com/TUC-ProAut/libRSF) - A Robust Sensor Fusion Library in C++ based on factor graphs, addressing non-Gaussian measurement errors.
  - **Language**: C++
  - **Features**: Factor graph optimization, robust against non-Gaussian outliers
  - **Applications**: SLAM, GNSS-based navigation
  - **Dependencies**: Ceres Solver

- [**Fusion**](https://github.com/xioTechnologies/Fusion) - Sensor fusion library for IMUs, optimized for embedded systems.
  - **Language**: C with Python bindings
  - **Features**: AHRS algorithm, supports various reference frames
  - **Applications**: IMU processing for orientation estimation
  - **Example**: [Simple example](https://github.com/xioTechnologies/Fusion/blob/main/Python/simple_example.py), [Advanced example](https://github.com/xioTechnologies/Fusion/blob/main/Python/advanced_example.py)

- [**robot_localization**](https://github.com/cra-ros-pkg/robot_localization) - ROS package for sensor fusion using EKF or UKF.
  - **Language**: C++
  - **Features**: Fuses arbitrary number of sensors, supports various message types
  - **Applications**: Robot localization, odometry integration
  - **Tutorial**: [Fusing Wheel Odometry and IMU](https://blog.abdurrosyid.com/2021/07/21/fusing-wheel-odometry-and-imu-data-using-robot_localization-in-ros/)

- [**SensorFusion**](https://github.com/simondlevy/SensorFusion) - Simple sensor fusion algorithms for inertial measurement units.
  - **Language**: C++, Python
  - **Features**: Mahony and Madgwick filters implementation
  - **Platforms**: Arduino, Raspberry Pi, general microcontrollers

#### Specific to Autonomous Driving

- [**Autoware.Auto**](https://github.com/autowarefoundation/autoware.auto) - Open-source autonomous driving stack with sensor fusion components.
  - **Language**: C++
  - **Features**: Point cloud and image fusion, 3D object detection
  - **ROS2 Compatible**: Yes

- [**Apollo**](https://github.com/ApolloAuto/apollo) - Baidu's open autonomous driving platform with multi-sensor fusion modules.
  - **Language**: C++
  - **Features**: Perception, planning, control with detailed sensor fusion components
  - **Documentation**: [Perception Module](https://github.com/ApolloAuto/apollo/blob/master/docs/specs/perception_apollo_5.0.md)

### Commercial/Industrial Libraries

- [**NXP Sensor Fusion Library**](https://www.nxp.com/docs/en/data-sheet/NSFK_DS.pdf) - Optimized for Kinetis MCUs.
  - **Features**: Low power consumption (1.4mA 9-axis, 0.4mA 6-axis at 40Hz)
  - **Platforms**: Kinetis ARM Cortex M4F devices
  - **Applications**: Notebooks, tablets, smartphones, gaming, motion control

- [**NVIDIA DriveWorks**](https://developer.nvidia.com/drive/driveworks) - SDK for autonomous vehicle development with sensor fusion components.
  - **Language**: C++
  - **Features**: Camera, LiDAR, radar fusion; accelerated on NVIDIA hardware
  - **Applications**: ADAS and autonomous vehicles

## Sensor-Specific Fusion Combinations

### Two-Sensor Fusion

| Fusion Type | Range | Fusion Level | Performance Insights | Efficiency Insights | Applications | Implementation Resources |
|-------------|-------|--------------|----------------------|---------------------|--------------|-------------------------|
| **imu-gps-fusion** | Long | Low | High accuracy positioning in dynamic environments | Requires significant processing for raw data | Navigation systems, drone flight controllers | [NMEA GPS-IMU example](https://github.com/GAVLab/nmea_gps_imu_driver) |
| **imu-lidar-fusion** | Short-Mid | Mid | Precise motion tracking and environmental mapping | Moderate computational load | Mobile robotics, precision mapping | [LIO-SAM](https://github.com/TixiaoShan/LIO-SAM) |
| **imu-radar-fusion** | Mid-Long | Mid | Reliable tracking in adverse weather | Efficient for long-range detection | Maritime navigation, autonomous vehicles | [RadarSLAM](https://github.com/tskuse/RadarSLAM) |
| **imu-camera-fusion** | Short-Mid | Mid | Enhanced orientation with visual context | Moderate efficiency | AR/VR applications, drone stabilization | [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion) |
| **gps-lidar-fusion** | Mid-Long | Low | Accurate navigation with detailed mapping | Computationally demanding | Surveying, autonomous driving | [GPS-LiDAR-SLAM](https://github.com/AbangLZU/multi_sensor_slam) |
| **gps-radar-fusion** | Long | Low | Reliable long-range detection | Efficient for navigation | Highway autonomous driving, aviation | [gps-radar](https://github.com/wangx1996/Radar-Localization) |
| **gps-camera-fusion** | Mid-Long | Mid | Spatial navigation with visual recognition | Moderate efficiency | Precision agriculture, traffic monitoring | [OpenSFM](https://github.com/mapillary/OpenSfM) |
| **lidar-radar-fusion** | Mid-Long | High | Robust detection in all weather conditions | Computationally intensive but reliable | Autonomous vehicles, military applications | [DeepFusion](https://github.com/hangqiu/DeepFusion) |
| **lidar-camera-fusion** | Short-Mid | High | High-resolution mapping with visual context | Computationally demanding | Object recognition, 3D reconstruction | [LiDAR-Camera Calibration](https://github.com/ankitdhall/lidar_camera_calibration) |
| **radar-camera-fusion** | Mid-Long | High | Reliable detection with visual context | Efficient radar + moderate camera demands | ADAS, maritime navigation | [RadarCameraFusion](https://github.com/TUMFTM/RadarCameraFusion) |
| **thermal-camera-fusion** | Short | Mid | Effective night vision and heat detection | Low computational demands | Night surveillance, firefighting | [Thermal-RGB Fusion](https://github.com/vlkniaz/EdgeFool) |
| **ultrasonic-camera-fusion** | Short | Low | Precise short-range obstacle detection | Highly efficient for close range | Parking assistance, industrial robotics | [Ultrasonic-Vision](https://github.com/mizmizo/sensor_fusion) |
| **infrared-camera-fusion** | Short-Mid | Mid | Enhanced low-light detection | Moderate efficiency | Security, wildlife monitoring | [IR-RGB Fusion](https://github.com/hanna-xu/infrared-visible-fusion) |

### Three-Sensor Fusion

| Fusion Type | Range | Fusion Level | Performance Insights | Efficiency Insights | Applications | Implementation Resources |
|-------------|-------|--------------|----------------------|---------------------|--------------|-------------------------|
| **imu-gps-lidar-fusion** | Mid-Long | Low | Accurate positioning, motion tracking, mapping | Computationally intensive | Autonomous vehicles, drone delivery | [LIO-Mapping](https://github.com/hyye/lio-mapping) |
| **imu-gps-radar-fusion** | Long | Low | Reliable navigation in challenging weather | Efficient radar processing balances GPS limitations | Maritime, aviation navigation | [GNSS-IMU-Radar](https://github.com/ethz-asl/kalibr) |
| **imu-lidar-camera-fusion** | Short-Mid | High | Precise motion + high-res mapping with context | Computationally demanding | Robotics, AR/VR, detailed mapping | [VINS-Fusion-GPU](https://github.com/pjrambo/VINS-Fusion-gpu) |
| **gps-lidar-radar-fusion** | Mid-Long | High | Comprehensive navigation and detection | Computationally intensive | Urban/highway autonomous driving | [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) |
| **gps-lidar-camera-fusion** | Mid-Long | High | Navigation with detailed mapping and recognition | Requires significant processing power | Precision agriculture, surveying | [LOAM-Livox](https://github.com/hku-mars/loam_livox) |
| **gps-radar-camera-fusion** | Mid-Long | High | Navigation with detection in adverse weather | Efficient radar + moderate camera demands | Autonomous vehicles in varied weather | [RadarNet](https://github.com/tudelft-iv/RTCnet) |
| **lidar-radar-camera-fusion** | Mid-Long | High | High-resolution mapping with reliable detection | Computationally intensive | ADAS, autonomous trucks, robotics | [PointPainting](https://github.com/AmrElsersy/PointPainting) |

## Algorithms and Methods

### Classical Approaches

- **Kalman Filter Variants**
  - Extended Kalman Filter (EKF)
    - **Implementation**: [robot_localization EKF](https://github.com/cra-ros-pkg/robot_localization)
    - **Applications**: Real-time state estimation for robotics
  - Unscented Kalman Filter (UKF)
    - **Implementation**: [robot_localization UKF](https://github.com/cra-ros-pkg/robot_localization)
    - **Applications**: Non-linear systems with strong non-linearities
  - Information Filter
    - **Implementation**: [InformationFilter](https://github.com/rlabbe/filterpy/blob/master/filterpy/kalman/information_filter.py)
    - **Applications**: Systems with many measurement inputs

- **Particle Filters**
  - **Implementation**: [ParticleFilter](https://github.com/rlabbe/filterpy/blob/master/filterpy/monte_carlo/particle_filter.py)
  - **Applications**: Highly non-linear systems, global localization problems

- **Complementary Filters**
  - **Implementation**: [Madgwick's AHRS](https://github.com/xioTechnologies/Fusion)
  - **Applications**: Orientation estimation from IMU data

### Modern/AI Approaches

- **Deep Learning Fusion**
  - **PointPainting**: [Implementation](https://github.com/AmrElsersy/PointPainting)
    - Fuses semantic segmentation from 2D images with 3D LiDAR points
  - **Frustum PointNets**: [Implementation](https://github.com/charlesq34/frustum-pointnets)
    - Uses 2D detection to guide 3D point cloud processing

- **Uncertainty-Aware Fusion**
  - **Cocoon**: Uncertainty-aware multimodal fusion framework for 3D object detection
    - **Features**: Object and feature-level uncertainty quantification
    - **Advantages**: Robust performance in normal and challenging conditions

- **Evidential Fusion**
  - **Dempster-Shafer Theory**: [Implementation](https://github.com/reroglu/pyds)
    - **Applications**: Decision-level fusion with uncertainty representation
    - **Advantages**: Handles imprecise, fuzzy, and incomplete information

## Implementation Examples

### ROS-Based Implementations

- **robot_localization**: [GitHub](https://github.com/cra-ros-pkg/robot_localization)
  - **Setup Guide**: [Fusing Wheel Odometry and IMU Tutorial](https://blog.abdurrosyid.com/2021/07/21/fusing-wheel-odometry-and-imu-data-using-robot_localization-in-ros/)[6]
  - **Configuration Example**:
    ```yaml
    # Sample ekf_localization.yaml configuration
    frequency: 30
    sensor_timeout: 0.1
    two_d_mode: true
    transform_time_offset: 0.0
    transform_timeout: 0.0
    print_diagnostics: true
    debug: false
    publish_tf: true
    publish_acceleration: false
    ```

- **LIO-SAM**: [GitHub](https://github.com/TixiaoShan/LIO-SAM)
  - **Features**: Tightly-coupled LiDAR inertial odometry via smoothing and mapping
  - **Requirements**: ROS, PCL, GTSAM

### Embedded Systems

- **Fusion for IMUs**: [GitHub](https://github.com/xioTechnologies/Fusion)
  - **Python Example**:
    ```python
    import numpy as np
    from imufusion import Fusion
    
    # Create fusion object
    fusion = Fusion()
    
    # Process sensor data
    for data in sensor_data:
        fusion.update_no_magnetometer(data.gyroscope, data.accelerometer, 1/100)  # 100 Hz sample rate
        
    # Get orientation as Euler angles
    euler = fusion.quaternion.to_euler()
    print(f"Roll: {euler[0]}, Pitch: {euler[1]}, Yaw: {euler[2]}")
    ```

- **NXP Sensor Fusion**: [Documentation](https://www.nxp.com/docs/en/data-sheet/NSFK_DS.pdf)
  - **Features**: Optimized for Kinetis MCUs, very low power consumption
  - **Implementation**: Available through Kinetis Software Development Kit (KSDK)

## Datasets and Benchmarks

### Autonomous Driving Datasets

- **KITTI**: [Website](http://www.cvlibs.net/datasets/kitti/)
  - **Sensors**: Stereo cameras, LiDAR, GPS/IMU
  - **Annotations**: 3D object detection, tracking, road/lane detection

- **nuScenes**: [Website](https://www.nuscenes.org/)
  - **Sensors**: 6 cameras, 1 LiDAR, 5 radars, GPS/IMU
  - **Annotations**: 3D bounding boxes, instance segmentation

- **Waymo Open Dataset**: [Website](https://waymo.com/open/)
  - **Sensors**: 5 LiDARs, 5 cameras
  - **Annotations**: 3D bounding boxes, semantic segmentation

### Indoor/Robotics Datasets

- **TUM RGB-D**: [Website](https://vision.in.tum.de/data/datasets/rgbd-dataset)
  - **Sensors**: RGB-D camera, motion capture
  - **Applications**: Visual odometry, SLAM benchmarking

- **EuRoC MAV**: [Website](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)
  - **Sensors**: Stereo camera, IMU
  - **Applications**: Visual-inertial odometry

## Tools and Utilities

### Calibration Tools

- **Kalibr**: [GitHub](https://github.com/ethz-asl/kalibr)
  - **Features**: Camera-IMU, multi-camera calibration
  - **Supported Sensors**: Cameras, IMUs, laser scanners

- **LiDAR-Camera Calibration**: [GitHub](https://github.com/ankitdhall/lidar_camera_calibration)
  - **Features**: Semi-automatic LiDAR-camera extrinsic calibration
  - **Requirements**: ROS, PCL

### Visualization

- **RViz**: [Documentation](http://wiki.ros.org/rviz)
  - **Features**: 3D visualization for ROS
  - **Applications**: Sensor data visualization, fusion results

- **Open3D**: [GitHub](https://github.com/intel-isl/Open3D)
  - **Features**: 3D visualization and processing
  - **Applications**: Point cloud processing and visualization

### Evaluation Metrics

- **Precision-Recall Curves**: Object detection evaluation
- **Intersection over Union (IoU)**: Bounding box accuracy
- **RMSE**: Trajectory accuracy for odometry/SLAM
- **ATE/RPE**: Absolute/Relative Pose Error metrics

## Research Papers and Publications

### Survey Papers

- **A Survey of Multisensor Fusion Techniques, Architectures and Methodologies** (2021)
  - **Authors**: R. Ashok, et al.
  - **Topics**: Comprehensive review of fusion architectures and methods

- **Multisensor Data Fusion: A Review of the State-of-the-Art** (2019)
  - **Authors**: B. Khaleghi, et al.
  - **Topics**: Uncertainty handling, conflict resolution, fusion algorithms

### Landmark Papers

- **Robust Multi-Modal Perception with Uncertainty-Aware Sensor Fusion** (2025)
  - **Authors**: Minkyoung Cho, et al.
  - **Impact**: Introduced uncertainty quantification for heterogeneous representations

- **Factor Graphs for Robot Perception** (2017)
  - **Authors**: F. Dellaert, M. Kaess
  - **Impact**: Formalized factor graph approach for SLAM and sensor fusion

### Recent Publications

- **Cocoon: Robust Multi-Modal Perception with Uncertainty-Aware Sensor Fusion** (2025)
  - **Authors**: Minkyoung Cho, et al.
  - **Conference**: ICLR 2025
  - **Key Innovation**: Uncertainty quantification for heterogeneous representations

## Courses and Tutorials

### University Courses

- **Sensor Fusion and Tracking for Autonomous Systems** - Stanford University
  - **Topics**: Kalman filtering, particle filtering, multi-target tracking
  - **Materials**: [Course Website](https://stanford.edu)

- **Probabilistic Robotics** - University of Washington
  - **Topics**: Bayes filters, Kalman filters, particle filters, SLAM
  - **Textbook**: Probabilistic Robotics by Thrun, Burgard, and Fox

### Online Tutorials

- **Sensor Fusion Nanodegree** - Udacity
  - **Topics**: Camera, radar, and LiDAR fusion for autonomous vehicles
  - **Link**: [Course Website](https://www.udacity.com/course/sensor-fusion-engineer-nanodegree--nd313)

- **Kalman and Bayesian Filters in Python** - Roger Labbe
  - **Topics**: Interactive tutorials on Bayesian filtering
  - **Link**: [GitHub Book](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python)

## Industrial Applications

### Autonomous Vehicles

- **Tesla Autopilot**: Vision-based approach with radar fusion
  - **Sensors**: Cameras, ultrasonics, radar
  - **Approach**: Neural networks for perception with sensor fusion

- **Waymo**: Multi-sensor fusion approach
  - **Sensors**: LiDAR, cameras, radar
  - **Approach**: ML-based fusion of multiple sensor modalities

### Robotics

- **Boston Dynamics Spot**: Advanced sensor fusion for navigation
  - **Sensors**: Stereo cameras, IMU, LiDAR
  - **Applications**: Industrial inspection, site monitoring

- **Skydio Drones**: Vision-based navigation with sensor fusion
  - **Sensors**: Cameras, IMU
  - **Approach**: Visual-inertial odometry for obstacle avoidance

### Industrial IoT

- **Predictive Maintenance**: Multi-sensor fusion for equipment monitoring
  - **Sensors**: Vibration, temperature, acoustic, power consumption
  - **Approach**: ML-based fusion for anomaly detection

---

## Contributing

Please feel free to contribute to this list by submitting a pull request. Make sure your contribution follows the guidelines below:

- Verify the accuracy of the information
- Provide links to relevant resources
- Follow the established format of the list
- Add a brief description of the resource
