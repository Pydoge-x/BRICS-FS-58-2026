#!/usr/bin/env python3
import sys, time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

MAP = {
    "前进": (1.0, 0.0, 2.0),
    "后退": (-1.0, 0.0, 2.0),
    "左转": (0.0, 1.5, 1.2),
    "右转": (0.0, -1.5, 1.2),
    "停": (0.0, 0.0, 0.3),
    "巡检": (0.8, 0.6, 6.0),
}

class Driver(Node):
    def __init__(self):
        super().__init__("m4_drive_turtle")
        self.pub = self.create_publisher(Twist, "/turtle1/cmd_vel", 10)

    def run(self, text: str):
        key = text.strip()
        for k, v in MAP.items():
            if k in key:
                lx, az, dur = v
                break
        else:
            print("unknown:", key); return
        msg = Twist(); msg.linear.x = lx; msg.angular.z = az
        end = time.time() + dur
        while time.time() < end:
            self.pub.publish(msg); time.sleep(0.1)
        stop = Twist()
        for _ in range(5):
            self.pub.publish(stop); time.sleep(0.05)

def main():
    rclpy.init()
    n = Driver(); n.run(" ".join(sys.argv[1:]) or "巡检")
    n.destroy_node(); rclpy.shutdown()

if __name__ == "__main__":
    main()
