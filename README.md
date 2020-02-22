## Team Members

Markus Glau (Team Lead)
glaumarkus1209@gmail.com

Bogdan Kovalchuk
bogdan.kovalchuk@globallogic.com

Jorve Kohls
jorve.kohls@tuhh.de

Chitra Chaudhari
chitraksonawane@gmail.com

Nihar Patel
patel.nihar2596@gmail.com

# Project overview

This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. For this project, we had to write ROS nodes to implement core functionality of the autonomous vehicle system, including traffic light detection, control, and waypoint following! You will test your code using a [Udacity Simulator]. As eplained in the walkthrough we divided the project in te following parts,
- Waypoint updater
- Drive By Wire node
- Traffic light detection

### Waypoint updater

### Drive By Wire node

Carla is equipped with Drive by Wire (DBW) technology, which makes possible controlling the throttle, steering and brakes electronically.
In particular, the DBW Node (in dbw_node.py) is resposible handling all the communications. It's subscribed to the current_vel,twist_cmd and dbw_enabled topics and it publishes the throttle_cmd, brake_cmd and steering_cmd topics.
Steering targets are generated by the YawController class inside yaw_controller.py, while Throttle and Brake commands use separate PID controllers found in pid.py.
The actual controls are implemented in the twist_controller.py file. In the current implementation is that brake and throttle commands are generated and published in separate branches of a conditional statement and reset when a competing signal is sent to the DWB node, and as such are not allowed to interfere with one another and allowing smoother transitions between one or the other.Throttle is controled by control algorithm (ref: https://ijssst.info/Vol-17/No-30/paper19.pdf).The positive value that returns this algorithm are the throttle values of the car.Then we smooth the final values of throttle. 
When the algorithm returns a negative value for throttle, it means the car needs to decelerate. Brake is the amount of torque that is applied to the brake system to decrease the car's speed. As we did for throttle, we smooth the final values of brake for the comfort .

### Traffic light detection

# Setup

Please use **one** of the two installation options, either native **or** docker installation.

### Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).

### Docker Installation
[Install Docker](https://docs.docker.com/engine/installation/)

Build the docker container
```bash
docker build . -t capstone
```

Run the docker file
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

### Port Forwarding
To set up port forwarding, please refer to the "uWebSocketIO Starter Guide" found in the classroom (see Extended Kalman Filter Project lesson).

### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### Real world testing
1. Download [training bag](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic_light_bag_file.zip) that was recorded on the Udacity self-driving car.
2. Unzip the file
```bash
unzip traffic_light_bag_file.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_file/traffic_light_training.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images

### Other library/driver information
Outside of `requirements.txt`, here is information on other driver/library versions used in the simulator and Carla:

Specific to these libraries, the simulator grader and Carla use the following:

|        | Simulator | Carla  |
| :-----------: |:-------------:| :-----:|
| Nvidia driver | 384.130 | 384.130 |
| CUDA | 8.0.61 | 8.0.61 |
| cuDNN | 6.0.21 | 6.0.21 |
| TensorRT | N/A | N/A |
| OpenCV | 3.2.0-dev | 2.4.8 |
| OpenMP | N/A | N/A |

