import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D

from cv_bridge import CvBridge

import cv2

from ultralytics import YOLO

import collections
import numpy as np

WARMUP_FRAMES = 10
MOVING_AVG_WINDOW = 30
REPORT_EVERY = 100

class YOLOLatencyTracker:
    def __init__(self):
        self.history_ = []
        self.window_ = collections.deque(maxlen=MOVING_AVG_WINDOW)
        self.frame_count_ = 0
    
    def record(self, total_latency_per_frame):
        self.frame_count_ += 1

        if self.frame_count_ <= WARMUP_FRAMES:
            return
        
        self.history_.append(total_latency_per_frame)
        self.window_.append(total_latency_per_frame)

    def get_moving_average(self):
        if len(self.window_) == 0:
            return None
        return np.mean(self.window_)
    
    def get_mean(self):
        if len(self.history_) == 0:
            return None
        return np.mean(self.history_)
    
    def get_P95(self):
        if len(self.history_) < 20:
            return None
        return float(np.percentile(self.history_, 95))
    
    def minimum(self):
        if len(self.history_) == 0:
            return None
        return np.min(self.history_)
    
    def maximum(self):
        if len(self.history_) == 0:
            return None
        return np.max(self.history_)
    
    def sample_count(self):
        return len(self.history_)
    
    def should_report(self):
        n = self.sample_count()
        return n > 0 and (n % REPORT_EVERY == 0)   
    

class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')

        qos = QoSProfile(depth=1)

        self.subscription_ = self.create_subscription(Image, 'image_topic',self.image_callback, qos)

        self.bridge_ = CvBridge()

        # Publisher for detections
        self.detection_publisher_ = self.create_publisher(Detection2DArray, 'detections', 10)

        # Load the YOLO model
        self.model_path = 'yolo26m.pt'  # Ensure this path is correct and the model file is present
        self.model_name = self.model_path.split('/')[-1]
        self.model = YOLO(self.model_path)
        self.model.to('cuda')

        self.tracker_ = YOLOLatencyTracker()

        self.get_logger().info('Yolo Node Started - publishing to /detections topic.\n'
                               f'Discarding first {WARMUP_FRAMES} warmup frames.\n'
                               f'Reporting latency every {REPORT_EVERY} samples after warmup.')

    def image_callback(self, msg):
        frame = self.bridge_.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        results = self.model(frame, imgsz=320, verbose=False)

        # Assuming results is a list of detections, we take the first one for processing
        result = results[0]

        pre_latency = result.speed['preprocess']
        infer_latency = result.speed['inference']
        post_latency = result.speed['postprocess']
        total_latency_per_frame = pre_latency + infer_latency + post_latency

        self.tracker_.record(total_latency_per_frame)

        mov_avg = self.tracker_.get_moving_average()
        if mov_avg is not None:
            print(f'pre: {pre_latency:.2f}ms  infer: {infer_latency:.2f}ms  post: {post_latency:.2f}ms  total: {total_latency_per_frame:.2f}ms | moving_avg({MOVING_AVG_WINDOW}): {mov_avg:.2f}ms')
        
        if self.tracker_.should_report():
            self.print_stats()
        
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
    def print_stats(self):
        n = self.tracker_.sample_count()
        mean_latency = self.tracker_.get_mean()
        p95_latency = self.tracker_.get_P95()
        min_latency = self.tracker_.minimum()
        max_latency = self.tracker_.maximum()

        p95_str = f'{p95_latency:.2f}ms' if p95_latency is not None else 'need at least 20 samples'
        print(f'\n--- Latency Stats after {n} samples (excluding warmup) (model:{self.model_name}) ---\n'
              f'| Mean Latency: {mean_latency:.2f}ms\n'
              f'| P95 Latency: {p95_str}\n'
              f'| Min Latency: {min_latency:.2f}ms\n'
              f'| Max Latency: {max_latency:.2f}ms\n'
              '-------------------------------------------------------------\n')

def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\nKeyboard interrupt received')
    finally:
        node.print_stats()
        print('Shutting down Yolo Node')
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()
