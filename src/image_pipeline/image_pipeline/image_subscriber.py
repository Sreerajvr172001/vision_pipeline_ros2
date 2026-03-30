import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')

        self.subscription_ = self.create_subscription(Image, 'image_topic', self.listener_callback, 10)
        self.start_time = self.get_clock().now().nanoseconds / 1e9
        self.frame_count = 0


    def listener_callback(self, msg):
        now = self.get_clock().now()

        current_time = now.nanoseconds / 1e9

        msg_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        self.frame_count += 1

        elapsed_time = current_time - self.start_time
        latency = current_time - msg_time

        if elapsed_time >= 1:
            fps = self.frame_count / elapsed_time
            self.get_logger().info(f"latency: {latency*1000:.2f} ms, FPS: {fps:.2f}")
            self.frame_count = 0
            self.start_time = current_time
        

        
def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('Keyboard Interrupt Received')
    finally:
        print('Shutting down node')
        if rclpy.ok():
            node.destroy_node()
            try:
                 rclpy.shutdown()                
            except Exception:
                pass
           
if __name__ == '__main__':
    main()


