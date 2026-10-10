# 模块3 配套软件（具身智能平台部署与运维）

对应实操手册下载路径：`$FILE_SERVER/module3/` 或 `$FILE_SERVER/金砖课程配套软件/module3/`

```text
module3/
├── scripts/
│   ├── install-ros2.sh
│   ├── install-dify.sh
│   └── install-openclaw.sh
├── ros2/
│   ├── ros.key
│   └── debs/…          # ROS 2 Humble 离线 deb
├── dify/
│   ├── images/dify-images.tar
│   ├── config/         # docker-compose + .env
│   └── load-and-start.sh
└── openclaw/
    └── openclaw-image.tar
```

适用镜像：`ydy-eia-jzbase`。勿放入 DeepSeek API Key。
