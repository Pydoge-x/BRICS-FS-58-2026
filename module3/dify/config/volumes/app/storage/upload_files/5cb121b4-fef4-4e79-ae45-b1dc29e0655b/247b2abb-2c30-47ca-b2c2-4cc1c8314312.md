# turtlesim 操作手册

## 基本口令
- 前进：线速度沿 X 轴正方向
- 后退：线速度沿 X 轴负方向
- 左转 / 右转：角速度绕 Z 轴
- 停：线速度与角速度均为 0

## ROS 话题
- 速度指令：`/turtle1/cmd_vel`（geometry_msgs/Twist）
- 位姿：`/turtle1/pose`

## 安全规范
- 连续运动指令应设置超时并自动刹车
- 联调前确认 turtlesim 窗口已打开
