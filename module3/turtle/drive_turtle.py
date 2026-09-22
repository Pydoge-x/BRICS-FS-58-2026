#!/usr/bin/env python3
import sys, time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

MAP = {
    "前进": (1.0, 0.0, 2.0),
    "后退": (-1.0, 0.0, 2.0),
    "左转": (0.0, 1.5, 1.5),
    "右转": (0.0, -1.5, 1.5),
    "停": (0.0, 0.0, 0.3),
}

class Driver(Node):
    def __init__(self):
        super().__init__("drive_turtle_cli")
        self.pub = self.create_publisher(Twist, "/turtle1/cmd_vel", 10)

    def run(self, text: str):
        key = text.strip()
        for k, v in MAP.items():
            if k in key:
                lx, az, dur = v
                break
        else:
            print("可用: 前进/后退/左转/右转/停")
            return
        msg = Twist()
        msg.linear.x, msg.angular.z = lx, az
        end = time.time() + dur
        while time.time() < end:
            self.pub.publish(msg)
            time.sleep(0.1)
        stop = Twist()
        for _ in range(5):
            self.pub.publish(stop)
            time.sleep(0.05)

def main():
    rclpy.init()
    node = Driver()
    node.run(" ".join(sys.argv[1:]) or "前进")
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
