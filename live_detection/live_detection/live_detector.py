import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError

from live_detection.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from live_detection.misc import Timer

from vision_msgs.msg import ObjectHypothesisWithPose, BoundingBox2D, Detection2D, Detection2DArray

import os


class DetectionNode(Node):

    def __init__(self):
        super().__init__('detection_node')

        # Create a subscriber to the Image topic
        self.subscription = self.create_subscription(Image, 'image', self.listener_callback, 10)
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()

        # self.publisher_ = self.create_publisher(Image, 'detection_results',10)
        self.detection_publisher = self.create_publisher(Detection2DArray, 'detection', 10)

        self.net_type = 'mb1-ssd'
        self.model_path = os.getenv("HOME")+ '/mobilenet-v1-ssd-mp-0_675.pth'
        self.label_path = os.getenv("HOME") + '/voc-model-labels.txt'

        self.class_names = [name.strip() for name in open(self.label_path).readlines()]
        self.num_classes = len(self.class_names)
        
        self.net = create_mobilenetv1_ssd(len(self.class_names), is_test=True)
        self.net.load(self.model_path)
        self.predictor = create_mobilenetv1_ssd_predictor(self.net, candidate_size=200)

        self.timer = Timer()

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
        cv2.imshow('object_detection', cv_image)
        self.detection_publisher.publish(detection_array)
        cv2.waitKey(1)
        


def main(args=None):
    rclpy.init(args=args)

    detection_node = DetectionNode()

    rclpy.spin(detection_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    detection_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
