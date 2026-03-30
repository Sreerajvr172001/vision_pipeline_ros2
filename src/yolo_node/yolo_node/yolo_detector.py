import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D

from cv_bridge import CvBridge

import cv2

from ultralytics import YOLO

class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')

        qos = QoSProfile(depth=1)

        self.subscription_ = self.create_subscription(Image, 'image_topic',self.image_callback, qos)

        self.bridge_ = CvBridge()

        # Publisher for detections
        self.detection_publisher_ = self.create_publisher(Detection2DArray, 'detections', 10)

        # Load the YOLO model
        self.model = YOLO('yolo26n.engine', task='detect')
        #self.model.to('cuda')

        self.get_logger().info('Yolo Node Started - publishing to /detections topic')

    def image_callback(self, msg):
        frame = self.bridge_.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        results = self.model(frame, imgsz=320, verbose=True)

        # Assuming results is a list of detections, we take the first one for processing
        result = results[0]
        
        # Annotate the frame with detection results
        annotated_frame = result.plot()
        # Display the annotated frame in a window
        cv2.imshow("MK2 AI View", annotated_frame)
        cv2.waitKey(1)

        # Create a Detection2DArray message to publish the detections
        detection_array_msg = Detection2DArray()
        detection_array_msg.header = msg.header

        boxes = result.boxes

        if boxes is None:
            return
        
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = result.names[cls_id]

            # Create a Detection2D message for each detection
            detection_msg = Detection2D()
            detection_msg.header = msg.header
            
            # Extract and Pack the bounding box information into the Detection2D message
            cx, cy, w, h = box.xywh[0]
            detection_msg.bbox = BoundingBox2D()
            detection_msg.bbox.center.position.x = float(cx)
            detection_msg.bbox.center.position.y = float(cy)
            detection_msg.bbox.size_x = float(w)
            detection_msg.bbox.size_y = float(h)

            # Create an ObjectHypothesisWithPose message for the detected object and add it to the Detection2D message
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(label)
            hyp.hypothesis.score = conf
            detection_msg.results.append(hyp)

            # Add the Detection2D message to the Detection2DArray message
            detection_array_msg.detections.append(detection_msg)
        
        # Publish the Detection2DArray message to the /detections topic
        self.detection_publisher_.publish(detection_array_msg)

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
