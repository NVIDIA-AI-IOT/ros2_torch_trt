from live_classifier.live_classifier_helper import WebcamClassifier

import rclpy

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
