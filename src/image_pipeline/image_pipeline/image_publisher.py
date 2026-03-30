import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Header

from cv_bridge import CvBridge
import cv2
import numpy as np

import threading 

class ImagePublisher(Node):
    def __init__(self):
        super().__init__('image_publisher')

        self.publisher_ = self.create_publisher(Image, 'image_topic', 10)
        self.compressed_publisher_ = self.create_publisher(CompressedImage, 'image_topic/compressed', 10)

        self.timer_ = self.create_timer(0.033, self.publish_image)  # Publish at 30 Hz

        self.bridge_ = CvBridge()

        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2) # Open the default camera

        success_format = self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) # Set the video codec to MJPG for better performance

        success_width = self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320) # Set the desired width
        success_height = self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240) # Set the desired height
        success_fps = self.cap.set(cv2.CAP_PROP_FPS, 30) # Set the desired frame rate
        success_buffer = self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2) # Set the buffer size to 1 to get the latest frame
        

        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_cv2_fps = self.cap.get(cv2.CAP_PROP_FPS)

        if actual_width != 320 or actual_height != 240:
            self.get_logger().warning(f"HARDWARE REJECTED 320x240, Current: {int(actual_width)}x{int(actual_height)}")
        else:
            self.get_logger().info(f"SUCCESS: Camera resolution set to {int(actual_width)}x{int(actual_height)}")

        self.frame_count_ = 0
        self.start_time_ = self.get_clock().now()

        self.frame = None
        self.new_frame_available = False
        self.lock = threading.Lock()

        self.stop_event = threading.Event()

        self.capture_thread = threading.Thread(target=self.capture_loop)
        self.capture_thread.start() # Start the capture thread to continuously read frames from the camera

        self.get_logger().info('Threaded Image Publisher Node Started')
        self.get_logger().info(f"Camera resolution: {int(actual_width)}x{int(actual_height)}, CV2 FPS: {actual_cv2_fps}")
    
    def capture_loop(self):
        while not self.stop_event.is_set():
            start_time = self.get_clock().now()
            ret, frame = self.cap.read()
            end_time = self.get_clock().now()

            if self.stop_event.is_set(): # Check if stop event is set after reading the frame, if so,
                print('Stop event set, exiting capture loop')
                break

            if not ret or frame is None: # If the frame was not successfully captured, print an error message and break the loop
                self.get_logger().warn('Dropped frame or camera disconnected')
                continue

            with self.lock:
                self.frame = frame # Update the shared frame variable with the latest captured frame
                self.header = Header()
                self.header.stamp = self.get_clock().now().to_msg() # Update the header
                self.frame_id = 'camera_frame' # Set the frame ID for the header
                self.new_frame_available = True # Set the flag to indicate that a new frame is available for publishing
            
            average_brightness = np.mean(frame)
            if average_brightness < 100: # If the average brightness is very low, fps may be affected, so print a warning message
                self.get_logger().warn(f'Low brightness detected (average brightness: {average_brightness:.2f}), FPS may be affected')

            capture_time = (end_time - start_time).nanoseconds / 1e9
    
            self.frame_count_ += 1

            current_time = self.get_clock().now()
            elapsed_time = (current_time - self.start_time_).nanoseconds / 1e9

            if elapsed_time >= 1.0:
                fps = self.frame_count_ / elapsed_time
                print(f"FPS: {fps:.2f}, Capture Time: {capture_time*1000:.3f} ms")
                
                self.frame_count_ = 0
                self.start_time_ = current_time
        print('Exiting capture loop')

    def publish_image(self):
            with self.lock:
                if self.frame is None or not self.new_frame_available:
                    return
                frame = self.frame # Make a local copy of the frame to avoid holding the lock while publishing
                self.new_frame_available = False # Reset the flag after copying the frame

            try:
                msg = self.bridge_.cv2_to_imgmsg(frame, encoding='bgr8')
                msg.header = self.header
                self.publisher_.publish(msg)
                #self.get_logger().info('Published an image')
            except Exception as e:
                self.get_logger().error(f'Failed to publish image: {e}')

            if self.compressed_publisher_.get_subscription_count() > 0: # Only publish compressed image if there are subscribers to the compressed topic
                try:
                    success, compressed_frame = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80]) # Compress the frame using JPEG encoding with a quality of 80
                    if success:
                        compressed_msg = CompressedImage()
                        compressed_msg.header = self.header
                        compressed_msg.format = 'jpeg'
                        compressed_msg.data = np.array(compressed_frame).tobytes()
                        self.compressed_publisher_.publish(compressed_msg)
                        #self.get_logger().info('Published a compressed image')
                    else:
                        self.get_logger().warn('Failed to compress the image')
                except Exception as e:
                    self.get_logger().error(f'Failed to publish compressed image: {e}')

    def destroy_node(self):     
        self.stop_event.set() # Signal the capture thread to stop

        if self.capture_thread.is_alive():
            self.capture_thread.join() # Wait for the capture thread to finish
        
        if self.cap.isOpened():
            self.cap.release() # Release the camera resource
        
        #self.destroy_timer(self.timer_) # Stop the timer to prevent further calls to publish_image
        super().destroy_node() # Call the base class destroy_node to clean up any remaining resources

def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\nKeyboard interrupt received')
    finally:
        print('Shutting down Image Publisher Node')
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()




