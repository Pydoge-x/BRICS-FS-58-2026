# 模块 B 备份版 · 推理专用

> 自 `module_b_behavior` 复制的**推理专用**版本：不含训练数据与训练脚本，仅保留两种分析能力。  
> **实操手册**：[模块B-人体行为识别-项目实操手册.md](./模块B-人体行为识别-项目实操手册.md)

## 两种场景

| 场景 | 技术栈 | 输入 |
|------|--------|------|
| **课堂** | YOLO-Pose + ByteTrack + 姿态规则（6 类） | 图片 / 视频 |
| **课外** | YOLO + ByteTrack + MMAction2 TSN（Kinetics-400） | 仅视频 |

## 入口

| 程序 | 用途 |
|------|------|
| `run_api.py` | Web 前端 + REST API（推荐） |
| `run_infer.py` | 命令行推理 |

## 安装

```powershell
cd C:\bricks\module_b_behavior_backup
python -m pip install -r requirements.txt
python scripts/setup_mmaction2.py --skip-pip
```

## Web 演示

```powershell
python run_api.py --port 8082
```

浏览器打开 http://localhost:8082

## 命令行

```powershell
# 课堂
python run_infer.py --media data/demo/classroom01_clip_8min_1min.mp4 --output output/classroom --scene classroom

# 课外（MMAction2）
python run_infer.py --media data/demo/extracurricular/sports.mp4 --output output/extra --scene extracurricular --max-frames 60
```

## 预置权重（推理用）

| 文件 | 用途 |
|------|------|
| `models/yolo/yolov8n-pose.pt` | 课堂姿态 |
| `models/yolo/yolov8n.pt` | 课外人体检测 |
| `models/mmaction2/tsn_imagenet-pretrained-r50_...pth` | 课外行为识别 |

## 与完整版区别

- 无 `run_train.py`、无 `data/raw/`（UCF101 等训练集）
- 无校园微调模型 `behavior_campus.pth`
- 课外仅 MMAction2 Kinetics-400 预训练，无 campus 归类模式

完整版（含训练）：`../module_b_behavior/`
