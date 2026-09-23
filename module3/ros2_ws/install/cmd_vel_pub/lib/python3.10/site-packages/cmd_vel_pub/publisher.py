import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class CmdVelPublisher(Node):
    def __init__(self):
        super().__init__('cmd_vel_publisher')
        self.pub = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.on_timer)
        self.t0 = self.get_clock().now()
        self.get_logger().info('publishing to /turtle1/cmd_vel')

    def on_timer(self):
        elapsed = (self.get_clock().now() - self.t0).nanoseconds / 1e9
        msg = Twist()
        if elapsed < 3.0:
            msg.linear.x = 1.0
            msg.angular.z = 0.5
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = CmdVelPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()