import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2

from ultralytics import YOLO

class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')

        qos = QoSProfile(depth=1)

        self.subscription_ = self.create_subscription(Image, 'image_topic',self.image_callback, qos)

        self.bridge_ = CvBridge()

        # Load the YOLO model
        self.model = YOLO('yolo26n.engine', task='detect')
        #self.model.to('cuda')

        self.get_logger().info('Yolo Node Started')

    def image_callback(self, msg):
        frame = self.bridge_.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        results = self.model(frame, imgsz=320, verbose=True)

        result = results[0]

        annotated_frame = result.plot()

        cv2.imshow("MK2 AI View", annotated_frame)
        cv2.waitKey(1)

        boxes = result.boxes

        if boxes is None:
            return
        
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = result.names[cls_id]

            if conf > 0.8: # Only log detections with confidence above 0.8
                self.get_logger().info(f"{label}: {conf:.2f}")

        self.get_logger().info(f'Received frame: {frame.shape}')

def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\nKeyboard interrupt received')
    finally:
        print('Shutting down Yolo Node')
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()
