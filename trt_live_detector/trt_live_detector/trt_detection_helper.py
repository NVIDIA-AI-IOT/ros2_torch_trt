'''Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.'''


# ROS2 imports 
import rclpy
from rclpy.node import Node

# CV Bridge and message imports
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_msgs.msg import ObjectHypothesisWithPose, BoundingBox2D, Detection2D, Detection2DArray
from cv_bridge import CvBridge, CvBridgeError

from live_detection.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from live_detection.misc import Timer

import cv2
import numpy as np
import os

from torch2trt import torch2trt
from torch2trt import TRTModule
import torch

class TRTDetectionNode(Node):

    def __init__(self):
        super().__init__('trt_detection_node')

        # Create a subscriber to the Image topic
        self.subscription = self.create_subscription(Image, 'image', self.listener_callback, 10)
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()

        # Create a Detection 2D array topic to publish results on
        self.detection_publisher = self.create_publisher(Detection2DArray, 'trt_detection', 10)

        # Create an Image publisher for the results
        self.result_publisher = self.create_publisher(Image,'trt_detection_image',10)

        self.net_type = 'mb1-ssd'
        
        # Weights and labels locations
        self.label_path = os.getenv("HOME") + '/ros2_models/voc-model-labels.txt'
        trt_model_path = os.getenv("HOME") + '/ros2_models/mb1SSD_trt.pth'

        self.class_names = [name.strip() for name in open(self.label_path).readlines()]
        self.num_classes = len(self.class_names)
        
        if (os.path.isfile(trt_model_path)):
            print("TRT Module exists, loading..")        

        else:
            print("TRT Module does not exist, will create one...")
            self.create_TRT_module()
            
        self.net = TRTModule()
        self.net.load_state_dict(torch.load(trt_model_path))
        self.predictor = create_mobilenetv1_ssd_predictor(self.net, candidate_size=200)
            
        self.timer = Timer()

    def create_TRT_module(self):
        model_path = os.getenv("HOME")+ '/ros2_models/mobilenet-v1-ssd-mp-0_675.pth'
        model = create_mobilenetv1_ssd(len(self.class_names), is_test=True)
        model.load_state_dict(torch.load(model_path))
        model.eval().cuda()

        x = torch.ones((1,3,300,300)).cuda()

        print("Creating TRT version...........")
        model_trt = torch2trt(model, [x])
        print("Created TRT version.......")

        save_location = os.getenv("HOME") + '/ros2_models/mb1SSD_trt.pth'

        print("Saving TRT model......")
        torch.save(model_trt.state_dict(), save_location)
 

    def listener_callback(self, data):
        self.get_logger().info("Received an image! ")
        try:
          cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
          print(e)

        
        image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        self.timer.start()
        boxes, labels, probs = self.predictor.predict(image, 10, 0.4)
        interval = self.timer.end()
        print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))

        detection_array = Detection2DArray()
        
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            label = f"{self.class_names[labels[i]]}: {probs[i]:.2f}"
            print("Object: " + str(i) + " " + label)
            cv2.rectangle(cv_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)

            # Definition of 2D array message and ading all object stored in it.
            object_hypothesis_with_pose = ObjectHypothesisWithPose()
            object_hypothesis_with_pose.id = str(self.class_names[labels[i]])
            object_hypothesis_with_pose.score = float(probs[i])

            bounding_box = BoundingBox2D()
            bounding_box.center.x = float((box[0] + box[2])/2)
            bounding_box.center.y = float((box[1] + box[3])/2)
            bounding_box.center.theta = 0.0
            
            bounding_box.size_x = float(2*(bounding_box.center.x - box[0]))
            bounding_box.size_y = float(2*(bounding_box.center.y - box[1]))

            detection = Detection2D()
            detection.header = data.header
            detection.results.append(object_hypothesis_with_pose)
            detection.bbox = bounding_box

            detection_array.header = data.header
            detection_array.detections.append(detection)


            cv2.putText(cv_image, label,
                       (box[0]+20, box[1]+40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,  # font scale
                       (255, 0, 255), 2)  # line type
        # Displaying the predictions
        cv2.imshow('trt_object_detection', cv_image)
        # Publishing the results onto the the Detection2DArray vision_msgs format
        self.detection_publisher.publish(detection_array)
        
        ros_image = self.bridge.cv2_to_imgmsg(cv_image)
        ros_image.header.frame_id = data.header.frame_id
        self.result_publisher.publish(ros_image)
        cv2.waitKey(1)
        


