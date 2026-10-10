---
name: turtlesim-patrol
description: 控制 ROS2 turtlesim 海龟完成前进/转向/巡检
metadata:
  openclaw:
    emoji: "🐢"
---

# turtlesim 巡检技能

## 前置
- 已运行：`ros2 run turtlesim turtlesim_node`
- 脚本路径：`/home/ubuntu/lab-m4/scripts/drive_turtle.sh`

## 何时使用
用户提到：前进、后退、左转、右转、停、巡检、自动巡检、绕一圈。

## 如何执行
在终端执行（优先）：

```bash
/home/ubuntu/lab-m4/scripts/drive_turtle.sh "巡检"
```

其他口令把参数换成「前进」「左转」等。

## 注意
- 不要并行启动多个长时间控龟进程
- 执行后用一句话反馈结果
