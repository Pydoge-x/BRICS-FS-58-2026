# Face-FER — 人脸表情识别

基于 PyTorch 的人脸表情识别项目，支持四种模型架构（VGG-16 / ResNet-50 / MobileNetV3 / MobileViT），提供 Web 推理平台与 RESTful API。

**七种表情类别**：Angry · Disgust · Fear · Happy · Sad · Surprise · Neutral

---

## 项目结构

```
module1/
│
├── train.py                 # [入口] 单模型训练：python train.py --config configs/xxx.json
├── train_all.py             # [入口] 批量训练：依次训练 VGG→ResNet→MobileNet→MobileViT
│
├── inference.py             # [入口] 命令行推理：单张图片 / 性能基准测试
├── run_api.py               # [入口] 启动 Flask API 服务
├── test_api.py              # [工具] API 接口测试脚本
│
├── test/                     # [验证] 中间过程验证脚本
│   ├── check_deps.py         #   依赖检查
│   ├── download_pretrained.py #  下载 ImageNet 预训练权重
│   ├── analyze_dataset.py    #   数据集统计分析
│   └── eval_models.py        #   多模型评估（混淆矩阵、F1）
│
├── configs/                 # [配置] 训练超参数
│   ├── config.py            #   配置数据结构定义（Config / DatasetConfig / ModelConfig / TrainingConfig）
│   ├── train_vgg_fer2013.json
│   ├── train_resnet_fer2013.json
│   ├── train_mobilenet_fer2013.json
│   └── train_mobilevit_fer2013.json
│
├── src/                     # [核心] 所有模块化代码
│   ├── data/                #   数据层
│   │   ├── dataset.py       #     数据集基类 FERDataset + 多数据集混合 MultiDataset
│   │   ├── fer2013.py       #     FER-2013 解析器（CSV / Kaggle 目录 / 标准目录）
│   │   ├── rafdb.py         #     RAF-DB 解析器
│   │   └── oulucasia.py     #     Oulu-CASIA 双模态（VIS+NIR）解析器
│   │
│   ├── models/              #   模型层
│   │   ├── factory.py      #     模型工厂：create_fer_model() 统一创建接口
│   │   ├── vgg.py           #     VGG-11/13/16/19
│   │   ├── resnet.py        #     ResNet-18/34/50/101/152
│   │   ├── mobilenet.py     #     MobileNetV2/V3-Small/V3-Large
│   │   └── mobilevit.py     #     MobileViT-XS/S/XXS（CNN+Transformer 混合）
│   │
│   ├── train/               #   训练层
│   │   ├── trainer.py       #     统一训练器 FERTrainer（含 early stopping / mixup / AMP）
│   │   └── losses.py        #     FocalLoss + LabelSmoothingLoss
│   │	└── optimizer.py	 # 	   优化器
│   │
│   ├── inference/           #   推理层
│   │   ├── inferrer.py      #     ONNX 推理器
│   │   ├── detector.py      #     人脸检测（InsightFace / OpenCV / MTCNN，默认为 InsightFace）
│   │   └── export.py        #     PyTorch → ONNX 导出
│   │
│   └── api/                 #   服务层
│       ├── app.py           #     Flask 应用创建
│       ├── routes.py        #     API 路由（/api/recognize, /api/models 等）
│       └── model_manager.py #     模型管理（自动扫描 checkpoints、ONNX 转换）
│
├── frontend/                # [前端] Web 推理测试平台
│   ├── index.html           #   页面结构
│   ├── style.css            #   样式
│   └── app.js               #   交互逻辑（上传、识别、结果展示）
│
├── models/                  # [模型] ONNX 推理模型 + InsightFace 人脸检测模型
│   ├── fer_vgg.onnx         #   VGG ONNX 导出
│   ├── fer_resnet.onnx      #   ResNet ONNX 导出
│   ├── fer_mobilenet.onnx   #   MobileNet ONNX 导出
│   ├── fer_mobilevit.onnx   #   MobileViT ONNX 导出
│   └── buffalo_l/           #   InsightFace 人脸检测模型（首次启动自动下载）
│
├── tools/                   # [工具] 辅助脚本
│   └── download_insightface_model.py  #   InsightFace 模型手动下载工具
│
├── dataset_data/            # [数据] 数据集文件
│   └── fer2013/             #   FER-2013（train/test 按类别分目录）
│       ├── train/
│       │   ├── angry/       (3995 张)
│       │   ├── disgust/     (436 张)
│       │   ├── fear/        (4097 张)
│       │   ├── happy/       (7215 张)
│       │   ├── sad/         (4830 张)
│       │   ├── surprise/    (3171 张)
│       │   └── neutral/     (4965 张)
│       └── test/
│           └── ...          (7178 张，同类结构)
│
├── checkpoints/             # [输出] 训练产出
│   ├── vgg_fer2013/
│   ├── resnet_fer2013/
│   ├── mobilenet_fer2013/
│   └── mobilevit_fer2013/
│       └── <timestamp>/     #   每次运行的独立目录
│           ├── best_model.pth       ← 最佳模型权重
│           ├── checkpoint_epoch_10.pth  ← 定期保存
│           ├── config.json          ← 训练配置存档
│           └── history.json         ← 训练历史（每轮 loss/acc/lr）
│
├── pretrained-models/       # [预训练] ImageNet 权重缓存
   ├── vgg16.pth
   ├── resnet50.pth
   ├── mobilenetv3_small.pth
   └── mobilevit_xs.pth
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


| 地址                               | 内容                                         |
| :----------------------------------- | :--------------------------------------------- |
| `http://localhost:5000`            | 前端页面（上传图片 → 人脸检测 → 表情识别） |
| `http://localhost:5000/api/health` | 健康检查                                     |

**前端功能**：上传照片 → 自动检测人脸 → 显示表情标签与置信度 → 可切换 VGG / ResNet / MobileNet / MobileViT 四种模型对比结果。

### CLI 推理（单张图片）

不启动服务，直接用命令行识别：

```powershell
python inference.py --image test.jpg
```

### API 接口速览


| 方法 | 接口                | 说明                                  |
| :----- | :-------------------- | :-------------------------------------- |
| GET  | `/api/health`       | 健康检查                              |
| GET  | `/api/models`       | 获取可用模型列表                      |
| POST | `/api/model/switch` | 切换模型`{"model_name": "mobilenet"}` |
| POST | `/api/recognize`    | 单图表情识别（上传图片文件）          |
| POST | `/api/detect`       | 人脸检测                              |
| GET  | `/api/benchmark`    | 性能基准测试                          |

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


| 模型              | 参数量 | 推理速度 | 准确率（FER-2013） | 适用场景            |
| :------------------ | :------- | :--------- | :------------------- | :-------------------- |
| MobileNetV3-Large | ~5.4M  | 快       | ~63%               | 移动端 / 实时推理   |
| MobileViT-S       | ~5.6M  | 中       | ~61%               | 移动端 + 全局特征   |
| ResNet-50         | ~25.6M | 中       | ~65%               | 服务器端 / 平衡之选 |
| VGG-16            | ~138M  | 慢       | ~68%               | 精度优先            |

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
