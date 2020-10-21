# This creates a node which will subscribe to the Image topic, 
# perform inference using PyTorch and send the results of the image
# on Classification2D in vision_msgs


# make the nedded ROS imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header, String
from vision_msgs.msg import Classification2D, ObjectHypothesis

# Import necessary PyTorch and related frameworks
import torch
import torchvision
from torchvision import models
from torchvision import transforms
import numpy as np
from timeit import default_timer as timer

import os

import cv2
from cv_bridge import CvBridge, CvBridgeError

from torch2trt import TRTModule

class WebcamClassifier(Node):

    def __init__(self):
        super().__init__('webcam_classification')
        # Create a subscriber to the Image topic
        self.image_subscriber = self.create_subscription(Image, 'image', self.listener_callback, 10)
        self.image_subscriber

        # create a publisher onto the vision_msgs 2D classification topic
        self.classification_publisher = self.create_publisher(Classification2D, 'classification', 10)
        # self.string_publisher = self.create_publisher(String, 'check_rate', 10)

        # Use the SqueezeNet model for classification
        self.squeezenet_trt = TRTModule()
        self.squeezenet_trt.load_state_dict(torch.load(os.getenv("HOME") + '/ros2_models/ros2_classification.pth'))

        # Use CV bridge to convert ROS Image to CV_image for visualizing in window
        self.bridge = CvBridge()

        # Find the location of the ImageNet labels text and open it
        with open(os.getenv("HOME") + '/ros2_models/imagenet_classes.txt') as f:
           self.labels = [line.strip() for line in f.readlines()]      
 
 
    def classify_image(self,img):
        
        transform = transforms.Compose([           
        transforms.Resize(256),                    
        transforms.CenterCrop(224),                
        transforms.ToTensor(),                     
        transforms.Normalize(                      
        mean=[0.485, 0.456, 0.406],                
        std=[0.229, 0.224, 0.225]                  
        )])
        tensor_to_image = transforms.ToPILImage()
        img = tensor_to_image(img)
        img_t = transform(img).cuda()
        batch_t = torch.unsqueeze(img_t, 0)
	
        # Classify the image
        start = timer() 
        out = self.squeezenet_trt(batch_t)
        end = timer()

        print("TRT Time: ", (end-start))

        _, index = torch.max(out, 1)

        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

        return self.labels[index[0]] , percentage[index[0]].item()
        

    def listener_callback(self, msg):
        
        img_data = np.asarray(msg.data)
        img = np.reshape(img_data,(msg.height, msg.width, 3))

        
        classified, confidence = self.classify_image(img)
        
  
        to_display = "Classified as: " + classified + " with confidence: " + str(confidence) 
        self.get_logger().info(to_display) 

        # Definition of Classification2D message
        classification = Classification2D()
        classification.header = msg.header
        result = ObjectHypothesis()
        result.id = classified
        result.score = confidence
        classification.results.append(result)

        # Publish Classification results
        self.classification_publisher.publish(classification)
       
        # Use OpenCV to visualize the images being classified from webcam 
        try:
          cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
          print(e)
        cv2.imshow('webcam_window', cv_image)
        cv2.waitKey(1)
       

def main(args=None):
    rclpy.init(args=args)

    webcam_classifier = WebcamClassifier()

    rclpy.spin(webcam_classifier)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    webcam_classifier.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
