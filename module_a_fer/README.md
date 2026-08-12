# Face-FER — 人脸表情识别

基于 PyTorch 的人脸表情识别项目，支持四种模型架构（VGG-16 / ResNet-50 / MobileNetV3 / MobileViT），提供 Web 推理平台与 RESTful API。

**七种表情类别**：Angry · Disgust · Fear · Happy · Sad · Surprise · Neutral

---

## 项目结构

```
face-fer/
│
├── train.py                  # 单模型训练入口
├── train_all.py              # 批量训练四模型
│
├── run_api.py                # 启动 Flask RESTful 服务 + 前端
├── inference.py              # 命令行单张推理 / ONNX 导出
│
├── test/                     # 验证与评估脚本
│   ├── check_deps.py         #   依赖检查
│   ├── download_pretrained.py #  下载 ImageNet 预训练权重
│   ├── analyze_dataset.py    #   数据集统计分析
│   └── eval_models.py        #   多模型评估（混淆矩阵、F1）
│
├── configs/                  # 训练配置文件
│   ├── config.py             #   配置数据结构定义
│   ├── train_vgg_fer2013.json
│   ├── train_resnet_fer2013.json
│   ├── train_mobilenet_fer2013.json
│   └── train_mobilevit_fer2013.json
│
├── src/                      # 核心源码
│   ├── data/                 #   数据层（FERDataset, FER2013/RAF-DB 解析器）
│   ├── models/               #   模型层（VGG, ResNet, MobileNet, MobileViT）
│   ├── train/                #   训练层（FERTrainer, FocalLoss, Early Stopping）
│   ├── inference/            #   推理层（ONNX 推理器, 人脸检测, 模型导出）
│   └── api/                  #   服务层（Flask 路由, ModelManager）
│
├── frontend/                 # Web 前端（纯 HTML/CSS/JS）
│   ├── index.html
│   ├── app.js
│   └── style.css
│
├── checkpoints/              # 已训练的模型权重
│   ├── vgg_fer2013/best_model.pth
│   ├── resnet_fer2013/best_model.pth
│   ├── mobilenet_fer2013/best_model.pth
│   └── mobilevit_fer2013/best_model.pth
│
├── dataset_data/             # 数据集存放目录
├── pretrained-models/        # ImageNet 预训练权重
└── models/                   # ONNX 导出模型（启动服务时自动生成）
```

---

## 环境搭建

### 1. 创建虚拟环境

```powershell
conda create -y -n face-fer python=3.10
conda activate face-fer
```

### 2. 安装 PyTorch

```powershell
# CUDA 11.8（推荐）
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

### 3. 安装项目依赖

```powershell
cd face-fer
pip install -r requirements.txt
```

### 4. 验证环境

```powershell
python test/check_deps.py
```

---

## 启动推理服务与前端（快速体验）

项目提供**一体化 Flask 服务**，一个命令同时启动后端 API 和前端界面：

```powershell
# 确保在项目根目录
cd face-fer
conda activate face-fer

# 启动服务
python run_api.py
```

启动后访问：

| 地址 | 内容 |
|:---|:---|
| `http://localhost:5000` | 前端页面（上传图片 → 人脸检测 → 表情识别） |
| `http://localhost:5000/api/health` | 健康检查 |

**前端功能**：上传照片 → 自动检测人脸 → 显示表情标签与置信度 → 可切换 VGG / ResNet / MobileNet / MobileViT 四种模型对比结果。

### CLI 推理（单张图片）

不启动服务，直接用命令行识别：

```powershell
python inference.py --image test.jpg
```

### API 接口速览

| 方法 | 接口 | 说明 |
|:---|:---|:---|
| GET | `/api/health` | 健康检查 |
| GET | `/api/models` | 获取可用模型列表 |
| POST | `/api/model/switch` | 切换模型 `{"model_name": "mobilenet"}` |
| POST | `/api/recognize` | 单图表情识别（上传图片文件） |
| POST | `/api/detect` | 人脸检测 |
| GET | `/api/benchmark` | 性能基准测试 |

---

## 训练模型

### 单模型训练

```powershell
# 训练 MobileNetV3（推荐入门，最快）
python train.py --config configs/train_mobilenet_fer2013.json

# 训练 ResNet-50
python train.py --config configs/train_resnet_fer2013.json

# 训练 VGG-16（参数量最大）
python train.py --config configs/train_vgg_fer2013.json

# 训练 MobileViT-S（CNN+Transformer 混合）
python train.py --config configs/train_mobilevit_fer2013.json
```

### 批量训练四模型

```powershell
python train_all.py
```

依次训练 VGG → ResNet → MobileNet → MobileViT，完成后可自动对比评估。

### 训练输出

每个模型训练完成后，`checkpoints/` 下会生成：

```
checkpoints/{model}_fer2013/
├── best_model.pth          # 验证集准确率最高的权重
├── epoch_{N}.pth           # 各 epoch 快照
├── config.json             # 训练配置存档
└── training_history.json   # 训练历史（loss/acc 曲线数据）
```

### 模型评估

```powershell
python test/eval_models.py
```

输出四个模型的准确率、F1-Score 和混淆矩阵。

---

## 四种模型对比

| 模型 | 参数量 | 推理速度 | 准确率（FER-2013） | 适用场景 |
|:---|:---|:---|:---|:---|
| MobileNetV3-Large | ~5.4M | 快 | ~63% | 移动端 / 实时推理 |
| MobileViT-S | ~5.6M | 中 | ~61% | 移动端 + 全局特征 |
| ResNet-50 | ~25.6M | 中 | ~65% | 服务器端 / 平衡之选 |
| VGG-16 | ~138M | 慢 | ~68% | 精度优先 |

---

## 技术要点

- **数据增强**：RandomHorizontalFlip、RandomRotation、ColorJitter
- **损失函数**：Focal Loss（处理类别不平衡） + Label Smoothing
- **优化策略**：AdamW + Cosine Annealing + Warmup
- **训练技巧**：MixUp 增强、Early Stopping、混合精度训练（AMP）
- **推理部署**：PyTorch → ONNX 自动转换，ONNXRuntime 推理加速
- **人脸检测**：OpenCV（立即可用） + InsightFace（高精度，后台加载）

---

## 许可

教学用途，数据集来自 FER-2013 (CC0)。
