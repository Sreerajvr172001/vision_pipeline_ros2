import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')

        self.subscription_ = self.create_subscription(Image, 'image_topic', self.listener_callback, 10)

    def listener_callback(self, msg):
        now = self.get_clock().now().to_msg()

        current_time = now.sec + now.nanosec * 1e-9

        msg_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        latency = current_time - msg_time

        self.get_logger().info(f"latency: {latency*1000:.2f} ms")

def main(args=None):
    rclpy.init(args=args)

    node = ImageSubscriber()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


