# ROS2 Real Time Classification Node
This is a ROS2 package for carrying out real time classification of images using PyTorch.

An ImageNet pre-trained SqueezeNet model is used for classification.

The package has been tested on NVIDIA Jetson Xavier AGX with Ubuntu 18.04, ROS Eloquent and PyTorch version 1.6.0
 
### Package Dependencies:

- Use either image_tools: https://github.com/ros2/demos/tree/eloquent/image_tools or usb_camera: https://github.com/klintan/ros2_usb_camera for obtaining the live stream of images from the webcam (make sure the name of this package is usb_camera_driver, rename the folder if needed.)

- vision_msgs: https://github.com/Kukanani/vision_msgs/tree/ros2

- cv_bridge: https://github.com/ros-perception/vision_opencv/tree/ros2/cv_bridge (May already be present, check by running ros2 pkg list)

Build these packages into your workspace


### Other Dependencies:

Pytorch and torchvision (if using Jetson, refer: https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-6-0-now-available/72048)
OpenCV


## Build and run

Make sure all the package dependencies are fulfilled and the packages are built in your workspace

Clone this package into your worskapce

Copy the imagenet_classes.txt into your home directory. This has the labels for the classification model.

Navigate into your worksapce. Run: colcon build --packages-select live_classifier

Next, open 2 terminals and navigate to your workspace. Run both these commands sequentially: 
source /opt/ros/eloquent/setup.bash
. install/setup.bash

Now, first begin streaming images from your webcam. In one of the terminals:

If using image_tools package: ros2 run image_tools cam2image
If using usb_camera package: ros2 run usb_camera_driver usb_camera_driver_node

In the second terminal:
ros2 run live_classifier live_classifier

The live_classifier node will subscribe to the image topic and will perform classification.
It will display the label and confidence for the image being classified.
Also, a small window will appear which will display the webcam image stream.

The results of the detection are published as Classification2D messages.
Open a new terminal and source them. Run: 
ros2 topic echo classification











