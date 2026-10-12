# Face-FER 学习指南 —— 从看懂到独立重写

> **适用对象**：拿到这个项目、想真正吃透它的人。
> **与实操手册的关系**：《模块A 情感计算之人脸表情识别-实操手册》讲"怎么操作"；本指南讲"怎么理解、怎么内化、怎么重现"。

---

## 两个终极检验

学没学会，不看"我读过"，看这两件事：


| 检验            | 标准                                                                      |
| ----------------- | --------------------------------------------------------------------------- |
| **① 空白重写** | 新建空文件夹，不抄原代码，独立写出"能训练 + 能推理 + 能开 Web 服务"的版本 |
| **② 白板讲解** | 合上电脑，讲清每个文件干什么、关键代码怎么工作、**为什么这么写**          |

如果现状是"能跑起来，但讲不出为什么"——很正常，这正是本指南要解决的。

---

## 1 方法论：三遍递进 + 输出驱动

### 1.1 三遍读码法

不要从头到尾"平推式"读代码（低效且容易迷失）。正确的姿势是递进三遍：


| 遍次         | 目标           | 动作                           | 禁止事项                 |
| -------------- | ---------------- | -------------------------------- | -------------------------- |
| 第一遍：建图 | 看清组成与流向 | 跑通所有命令 + 手画两条管线图  | 陷入细节                 |
| 第二遍：精读 | 逐层吃透       | 按依赖顺序读，每个文件答"五问" | 跳过实验                 |
| 第三遍：输出 | 内化成能力     | 写讲解稿、默写、空白重写       | 自我欺骗（"差不多懂了"） |

### 1.2 精读一个文件的"五问"

每读一个文件，先把这五个问题写进你的笔记：

1. **职责**：它解决什么问题？一句话说清。
2. **接口**：输入是什么、输出是什么？（函数签名 + 数据结构）
3. **数据流**：数据在这个文件里经历了怎样的变化？
4. **设计**：为什么这样写？换成别的写法会怎样？
5. **必要性**：删掉某一行/某一段会发生什么？

能流畅回答五问，这个文件才算"过关"。

### 1.3 五大核心技巧

**技巧 1：形状追踪法（深度学习项目最重要的技巧）**

深度学习代码的理解，本质是"张量形状的旅行"。在关键位置打印 shape，一路追踪。推理链路的形状旅行：

```
一张 224×224 的 RGB 图
→ np.array         : [224, 224, 3]（HWC，0~255）
→ 归一化 + 转置     : [3, 224, 224]（CHW）
→ 加 batch 维       : [1, 3, 224, 224]
→ 模型输出          : [1, 7]（7 个表情的 logits）
→ softmax          : [1, 7]（概率，和为 1）
```

每遇到"看不懂"的一行，先问自己：**此时张量形状是什么？**

**技巧 2：断点调试法**

在关键流程（如 recognize 接口、train_epoch）打断点，单步执行、观察变量。比反复"读"代码快一倍。

**技巧 3：修改实验法**

理解一个设计的最好方法是"改变它，观察结果"：改参数、注释代码、改错数值——运行后**解释现象**。每解释一次，理解就深一层。

**技巧 4：注释剥离法（检验真懂）**

把某个文件的注释全部删掉，自己重写一遍；进阶：只留函数名和 docstring，自己把函数体补全，再与原代码对比。

**技巧 5：费曼输出法**

假装给一个"完全不懂深度学习"的人讲这个文件。**卡壳的地方 = 没懂的地方**，回去补。

### 1.4 学习节奏与一个澄清

- 每天 2~4 小时，总计约 3 周；**输出时间（实验 + 复述 + 重写）不少于输入时间（读代码）**。
- 澄清"每行代码都懂"：目标不是背下每一行，而是——
  - **核心逻辑**（训练循环、预处理、Dataset、推理流程）：逐行掌握；
  - **防御性/兼容性代码**（多格式兼容、异常兜底）：知道"为什么需要它"即可。
  - 判断标准：**删掉这一行/改错它，会发生什么？** 能答出来，就到位了。

---

## 2 项目全景：先见森林

### 2.1 一句话理解项目

输入一张照片 → 人脸检测 → 对每张脸做 7 类表情分类（angry / disgust / fear / happy / sad / surprise / neutral）→ 通过 Web 服务返回结果。

围绕这条主线，项目完整覆盖了工程全流程：**数据 → 模型 → 训练 → 评估 → ONNX 导出 → API 服务 → 前端展示**。

### 2.2 两条管线（先记住这张图）

```
【训练管线 · 离线】
dataset_data/fer2013（按类别分目录的图片）
   │
   ▼
src/data/       读图 · 标签映射 · 数据增强 · 批量化
   │
   ▼
src/models/     VGG / ResNet / MobileNet / MobileViT（工厂统一创建）
   │
   ▼
src/train/      FERTrainer：训练循环 · 验证 · 调度 · 保存
   │
   ▼
checkpoints/<model>_fer2013/best_model.pth

【推理管线 · 在线】
用户上传图片（frontend/）
   │
   ▼
src/api/routes.py            接收图片 · 组织响应
   │
   ▼
src/inference/detector.py    人脸检测（InsightFace / OpenCV）
   │
   ▼
src/inference/inferrer.py    ONNX Runtime 推理 → 7 类概率
   │
   ▼
JSON 返回前端展示

【两条管线的连接点】
checkpoints/*.pth ──(src/inference/export.py 导出)──▶ models/*.onnx ──(供 inferrer 加载)
```

### 2.3 模块依赖层级（决定你的阅读顺序）

```
第 0 层（底座）：configs/config.py · src/models/pretrained_loader.py
第 1 层：src/data/*
第 2 层：src/models/*
第 3 层：src/train/*        （依赖 data + models）
第 4 层：src/inference/*    （依赖 models）
第 5 层：src/api/*          （依赖 inference）
第 6 层：入口脚本 train.py / inference.py / run_api.py / train_all.py（组装一切）
旁支：frontend/（只依赖 HTTP 接口）· test/（验证脚本）
```

为什么按这个顺序读：**每一层只依赖它下面一层**。读第 3 层时，它用到的东西你都已经懂了，不会悬空卡壳。

### 2.4 全文件速查表


| 文件                            | 行数 | 职责                                                  |  重要度  |
| --------------------------------- | -----: | ------------------------------------------------------- | :--------: |
| train.py                        |  361 | 训练入口：参数 → 配置 → 数据 → 训练器 → 训练/测试 |  ★★★  |
| train_all.py                    |  123 | 顺序训练四个模型（subprocess 调度）                   |    ★    |
| inference.py                    |  120 | 命令行推理：单图 / 性能测试 / 导出                    |   ★★   |
| run_api.py                      |    5 | 启动 API 服务                                         |    ★    |
| configs/config.py               |  154 | 配置数据结构（dataclass）+ 预置配置                   |  ★★★  |
| configs/train_*.json            |   — | 各模型的训练超参数                                    |   ★★   |
| src/data/dataset.py             |  317 | 数据集基类 + 多数据集 + DataLoader 工厂               |  ★★★  |
| src/data/fer2013.py             |  193 | FER-2013 解析器（CSV/目录/Kaggle 三格式自动识别）     |  ★★★  |
| src/data/rafdb.py               |  227 | RAF-DB 解析器（选学：对照差异）                       |    ★    |
| src/data/oulucasia.py           |  301 | Oulu-CASIA 双模态解析器（选学）                       |    ★    |
| src/models/pretrained_loader.py |   16 | 本地预训练权重优先加载                                |   ★★   |
| src/models/vgg.py               |  143 | VGG 系列（**模型文件入门首选**）                      |  ★★★  |
| src/models/resnet.py            |  154 | ResNet 系列                                           |   ★★   |
| src/models/mobilenet.py         |  146 | MobileNet 系列                                        |   ★★   |
| src/models/mobilevit.py         |  182 | MobileViT（timm + 手写 Block）                        |   ★★   |
| src/models/factory.py           |  247 | 模型工厂 + save/load checkpoint                       |  ★★★  |
| src/train/trainer.py            |  706 | **全项目最核心**：训练/验证/调度/保存                 | ★★★★ |
| src/train/losses.py             |  127 | Focal Loss + Label Smoothing                          |  ★★★  |
| src/train/optimizer.py          |  156 | 优化器/调度器工厂 + Warmup                            |   ★★   |
| src/inference/export.py         |  239 | PyTorch → ONNX 导出 + 校验                           |   ★★   |
| src/inference/inferrer.py       |  272 | ONNX Runtime 推理器                                   |  ★★★  |
| src/inference/detector.py       |  347 | 人脸检测（三种后端）                                  |   ★★   |
| src/api/app.py                  |  121 | Flask 应用创建 + 启动初始化                           |   ★★   |
| src/api/model_manager.py        |  245 | 模型扫描 / 转换 / 加载 / 切换                         |  ★★★  |
| src/api/routes.py               |  475 | 10 个 API 接口                                        |  ★★★  |
| frontend/index.html             |  127 | 页面结构                                              |    ★    |
| frontend/app.js                 |  569 | 前端交互（请求 / 渲染 / 画框）                        |   ★★   |
| frontend/style.css              |  566 | 样式（略读）                                          |    —    |
| test/eval_models.py             |  190 | 批量评估（Acc / F1 / 混淆矩阵）                       |   ★★   |
| test/analyze_dataset.py         |  217 | 数据集统计                                            |    ★    |
| test/check_deps.py              |   37 | 依赖自检                                              |    ★    |
| test/download_pretrained.py     |   79 | 下载预训练权重                                        |    ★    |
| src/models/check44.py           |   36 | 打印四模型参数量与 ONNX 体积                          |    ★    |

---

## 3 分阶段学习计划（总时长约 3 周）

### 阶段一：跑通 + 画地图（1~2 天）

**目标**：从"听说过"变成"亲手玩过"，并建立全局地图。

任务清单：

- [X]  `python test/check_deps.py` —— 确认依赖完整
- [X]  `python run_api.py` → 浏览器打开 `http://localhost:5000`，上传 `test/face-fers.jpg`，切换四个模型，观察结果与概率条
- [X]  命令行推理：`python inference.py --image test/face-fers.jpg --detect_face`
- [X]  性能测试：`python inference.py --benchmark 100`
- [X]  训练体验：把 `configs/train_mobilenet_fer2013.json` 的 `epochs` 改成 `1`，运行 `python train.py --config configs/train_mobilenet_fer2013.json`
- [X]  打开训练产物：看 `checkpoints/mobilenet_fer2013/` 下的 `config.json`、`history.json`
- [X]  **画图（最重要）**：白纸手绘两条管线（见 2.2），每个环节标注对应文件名

**通过标准**：合上资料，能口述"训练 / 推理 / 服务"三条命令分别触发了哪些模块。

### 阶段二：分层精读（7~10 天）

> 阅读总顺序：config.py → data → models → train → inference → api → frontend + 入口脚本。
> 每个文件都按"五问 + 实验"输出到你自己的笔记里。

#### 模块一：配置层 configs/（0.5 天）

**读什么**

- `config.py` 的四个 dataclass：`DatasetConfig` / `ModelConfig` / `TrainingConfig` / `Config`
- `Config.load()`：JSON → dataclass；`Config.save()` / `to_dict()`：反向导出
- 四个 `train_*.json` 与预置 `CONFIGS` 字典

**关键问题**

- 为什么用 dataclass 而不是 dict？（类型提示、默认值、IDE 补全、序列化方便）
- 命令行参数和 JSON 配置冲突时谁生效？（`train.py` 里的 `provided_args` 机制：**只有显式传入的命令行参数才覆盖 JSON**——去读那段代码并复述）
- 训练产物里的 `config.json` 有什么用？（实验可复现：任何一次训练都能追溯到完整配置）

**动手实验**：把 `batch_size` 从 32 改成 8，跑 1 个 epoch，观察速度变化。

**通过标准**：能画出"配置从 JSON / 命令行汇入 train.py，再分发给 data / models / train 三个模块"的流向图。

#### 模块二：数据层 src/data/（1.5 天）

**阅读顺序**：`dataset.py` → `fer2013.py` → `rafdb.py` / `oulucasia.py`（略读对照）

**dataset.py 精读要点**

- `__len__` 与 `__getitem__`：PyTorch Dataset 协议的两个必须方法，DataLoader 只认这两个接口
- `_load_samples()`：扫描"类别文件夹"，把 `(图片路径, 标签)` 存进 `self.samples`
- `EXPRESSION_LABELS` 与 `LABEL_MAPPING`：不同数据集的类别顺序不同（如 RAF-DB 第 6 类是 angry，而 FER-2013 第 0 类是 angry），映射表把它们统一到同一套 7 类体系——这是多数据集混合训练的前提
- `_get_transforms()`：`Resize → [增强：翻转/颜色抖动/旋转/平移] → ToTensor → Normalize`
  - 三个"为什么"：① 增强只对 train 开（验证/测试必须无随机性）；② `ToTensor` 做了三件事（PIL→Tensor、除以 255、HWC→CHW）；③ Normalize 用 ImageNet 的 mean/std（与预训练权重对齐——模型是在这个分布上学出来的）
- `MultiDataset`：组合多个数据集；`__getitem__` 返回**三元组** `(image, label, dataset_idx)`——记住这个细节，trainer 里会用到
- `create_dataloader`：`drop_last=True`（丢弃不足一个 batch 的尾巴，避免 BatchNorm 单样本问题）、`prefetch_factor` 的作用

**fer2013.py 精读要点**

- `_auto_detect_and_load()`：依次尝试 CSV → split 目录 → Kaggle 目录 → train/test/val 目录（同一数据集有多种流通格式，这是真实工程的常见处理）
- CSV 解析：`pixels` 是空格分隔的 2304 个灰度值 → `np.uint8` → `reshape(48,48)` → `Image(L)` → `convert('RGB')`
  - 为什么灰度图要转 RGB？（模型输入是 3 通道；与预训练权重对齐）

**动手实验**

- 取一个样本，打印 `img.shape`、`img.dtype`、`img.min()`、`img.max()`
- 开/关增强，对同一张图取 3 次，观察差异
- 统计类别分布（`python test/analyze_dataset.py`），找到样本最少的类别——体会"类别不平衡"（disgust 仅 436 张），这就是 Focal Loss 存在的理由

**通过标准**：能默写 Dataset 两个方法的签名；能说清增强流水线的顺序与三个"为什么"。

#### 模块三：模型层 src/models/（2 天）

**阅读顺序**：`pretrained_loader.py` → `vgg.py` → `resnet.py` → `mobilenet.py` → `mobilevit.py` → `factory.py` → `check44.py`

**统一套路（读懂一个 = 读懂四个）**

```
① 挑选骨干网络（torchvision / timm 提供）
② 加载权重：优先本地 pretrained-models/，失败再在线下载
③ 替换分类头：1000 类 → 7 类
④ forward：一行转发
⑤ 附带 create_xxx_model() 工厂函数 + XXX_CONFIG 配置字典
```

**逐文件要点**

- `pretrained_loader.py`：`load_local_pretrained()` 先查本地文件再决定是否回退——为什么？（离线部署、下载慢）
- `vgg.py`（**精读首选**）：分类头重建 `Linear(512*7*7, 4096) → ... → Linear(4096, 7)`。自问：`512×7×7` 怎么来的？（512 = 最后卷积层通道数；7×7 = 224 经 5 次下采样：224→112→56→28→14→7）
- `resnet.py`：`fc` 换成 `Sequential(Dropout, Linear)`；理解残差连接解决什么问题
- `mobilenet.py`：三个变体的分类头结构不同（V3-Small 是 `Linear→Hardswish→Dropout→Linear`）；理解深度可分离卷积为什么"轻"
- `mobilevit.py`：用 timm 一行创建；文件底部的手写 `MobileViTBlock`（局部卷积 + Transformer）——读懂它 = 理解"CNN 局部 + Transformer 全局"的混合范式
- `factory.py`：`create_fer_model()` 按字符串分发（工厂模式：调用方不需要知道任何具体类名）；`save_model()` / `load_model()` 的 checkpoint 字典包含什么？（`model_type`、`model_name`、`model_state_dict`、`epoch`、`best_acc`，可选 `optimizer_state_dict`）——这些元信息让 `load_model` 能自动重建正确结构

**动手实验**

- `python -m src.models.check44`（在项目根目录运行）：看四个模型参数量（2.5M ↔ 138M）与 ONNX 体积（约 5MB ↔ 500MB+）的巨大差距
- 创建一个 MobileNetV3，喂 `torch.randn(2,3,224,224)`，打印输出形状（应为 `[2,7]`）

**通过标准**：随机指一个模型文件，3 分钟内讲清它的五段式结构；能回答"512×7×7 怎么来的"。

#### 模块四：训练层 src/train/（2.5 天，核心中的核心）

**阅读顺序**：`losses.py` → `optimizer.py` → `trainer.py`（分块读，别一口气读）

**losses.py**

- FocalLoss 公式：`FL = -α(1-p_t)^γ log(p_t)`；γ 压低"已学会样本"的权重，把学习焦点放到难样本上（呼应 disgust 类只有 436 张的不平衡问题）
- FocalLoss 对 MixUp 的支持：targets 是 `(labels_a, labels_b, lam)` 时，两组标签分别算 loss 再加权
- LabelSmoothingLoss：把 one-hot 标签软化（防过拟合、防"过度自信"）

**optimizer.py**

- `create_optimizer` / `create_scheduler` / `WarmupScheduler` 三个组件
- 进阶观察：`trainer.py` 内部也有一套同类实现——对比两套代码，理解"公共模块 vs 训练器私有实现"的关系（这本身就是一个很好的读码练习）

**trainer.py 九块读法**

1. `TrainingConfig`：所有超参数一览（epochs / batch / lr / warmup / T_0 / mixup_alpha / early_stop_patience / use_amp ...）
2. `__init__`：初始化顺序——设备 → 模型 → 损失 → 优化器 → 调度器 → AMP scaler → 历史
3. `_create_criterion`：Focal Loss 或 CE + LabelSmoothing 二选一
4. `_create_optimizer` / `_create_scheduler`：分发
5. `_warmup_scheduler`：`lr * (epoch+1) / warmup_epochs` 的线性升温
6. `train_epoch`：**全项目最重要的一段代码**。逐行抠：数据解析（2 元组 / 3 元组）→ 训练五步 `zero_grad → forward → loss → backward → step` → 梯度裁剪 → AMP 分支（`autocast` + `scaler.scale / unscale_ / step / update`）→ MixUp 分支 → 统计口径
7. `validate`：与训练的三处区别（`model.eval()` / `torch.no_grad()` / 不更新参数）；各类别准确率统计
8. `train`：训练总编排——warmup 判断 → 训练 → 验证 → `scheduler.step()` → 记录 history → 保存 best → 早停计数 → 定期 checkpoint + 清理
9. `_cleanup_old_checkpoints` / `test`：保留最近 N 个；测试前加载 best_model

**必须能回答（选摘）**

- 为什么必须 `zero_grad()`？（PyTorch 梯度默认累加）
- `loss.backward()` 到底做了什么？（反向自动微分，算出每个参数的梯度）
- best_model 为什么 `include_optimizer=False`，而定期 checkpoint 是 `True`？（前者给推理用；后者为"恢复训练"服务）
- 早停计数器何时重置？（发现更优验证精度时清零）
- AMP 分支里 `unscale_` 为什么必须在梯度裁剪之前？（把梯度还原回真实尺度才能正确裁剪）

**动手实验**

- 在 `train_epoch` 加一行 `print(images.shape, labels.shape)`，跑 1 个 epoch
- 注释掉 `zero_grad()`，观察 loss 行为（跑完记得改回来）
- 打开 `history.json`，用 matplotlib 画 loss 曲线（可选）

**通过标准**：白板默写训练循环五步 + AMP 分支的调用顺序；能解释上面 5 个问题。

#### 模块五：推理层 src/inference/（1.5 天）

**阅读顺序**：`export.py` → `inferrer.py` → `detector.py`

**export.py**

- 为什么导出 ONNX？（脱离 PyTorch 依赖、跨语言、推理加速、图优化）
- 三要素：`dummy_input`（追踪计算图）、`opset_version`（算子集版本）、`dynamic_axes`（batch 维可变）
- 导出后：`onnxsim` 简化 + `verify_onnx_model` 对比 PyTorch 输出（阈值 1e-5）
- `_sdpa_compatible_export`：**真实工程案例**——MobileViT 的注意力算子无法直接导出 ONNX，代码用 monkey-patch 换成了手写注意力。读懂它，你就学会了"导出失败怎么排查"的思路

**inferrer.py**

- 生命周期：创建 session（线程数 / 图优化级别 / 执行提供者 CUDA→CPU）→ warmup（预热，为什么？）→ preprocess → predict
- `preprocess` 六个动作按顺序：PIL→numpy → 保证 RGB 三通道 → resize 到 224 → 除以 255 → 减 mean 除 std → CHW + 加 batch 维
  - **推理与训练的预处理必须一致**——结果不对时的第一个排查点
- `predict`：softmax（先减最大值保证数值稳定）→ argmax → 组织结果字典

**detector.py**

- 三个后端的降级链：InsightFace（首选，高精度，带 5 点关键点）→ OpenCV（兜底）→ MTCNN（备选）
- 颜色通道陷阱：OpenCV 是 BGR，InsightFace 要 RGB——代码里哪些地方做了 `cvtColor`？漏转的后果？
- `extract_face`：bbox 向外扩 `margin=0.2` 再裁剪、再 resize——为什么外扩？（避免割掉下巴/额头）

**动手实验**

- `python inference.py --benchmark 100` 测速；用 netron 打开 `models/fer_mobilenet.onnx` 看计算图
- 破坏实验：改掉 preprocess 里的 mean/std，重新推理同一张图，观察结果变化

**通过标准**：按顺序复述 preprocess 的六个动作；解释 ONNX、warmup、margin 三个"为什么"。

#### 模块六：服务层 src/api/（1.5 天）

**阅读顺序**：`app.py` → `model_manager.py` → `routes.py`

**app.py**

- 启动时做三件事：`pre_convert_all()` 预转换所有模型 ONNX → 预初始化人脸检测器（失败自动降级 OpenCV）→ 加载默认模型
- 为什么放"启动时"而不是"首次请求时"？（首次请求延迟对用户体验很差）
- 静态目录指向 `frontend/`：浏览器访问 `/` 就是前端页面

**model_manager.py**

- `_scan_models`：按 `checkpoints/<名称>_fer2013/best_model.pth` 约定发现模型
- `_pth_to_onnx`：用文件修改时间（mtime）比较避免重复转换
- `load_model` / `switch_model`：加载后缓存；切换只是指针切换

**routes.py**（10 个接口，重点三个）

- `GET /api/health`、`GET /api/models`、`POST /api/model/switch`
- `POST /api/recognize`：收图（文件或 base64）→ 检测人脸 → 逐脸裁剪 + 推理 → 组织响应
  - numpy → Python 原生类型转换（`jsonify` 不能序列化 numpy 类型——这类"坑"只有读代码才知道）
  - 无人脸兜底：整图直接推理
- 练习：把 10 个接口整理成"入参 / 返回 / 错误码"三列表格

**动手实验**

- `curl http://localhost:5000/api/health`；`curl -X POST http://localhost:5000/api/recognize -F "image=@test/face-fers.jpg"`
- 在 `recognize` 打断点，完整走一遍请求
- 传一个非图片文件，观察错误处理

**通过标准**：能讲清 `/api/recognize` 的完整处理流程（从收到请求到返回 JSON）；接口表格至少能讲 6 个。

#### 模块七：前端与入口脚本（1 天）

**frontend/**（前端只是"HTTP 客户端 + 界面渲染"，不要害怕）

- `app.js` 函数地图：`apiCall`（封装 fetch）→ `loadModels / switchModel / switchDetector / recognizeImage`；`handleImageUpload / runInference`；`redrawWithFaces`（canvas 画人脸框）/ `renderResults` / `renderProbBars`
- 两个重点：
  - `runInference`：上传图片后拿到的 JSON 如何变成结果卡片与概率条
  - `redrawWithFaces`：bbox 从"原图坐标"映射到"canvas 显示坐标"的缩放
- 读前端可以反推后端接口契约（字段名、结构）

**入口脚本**

- `train.py`：argparse → `load_config`（provided_args 技巧）→ `create_dataset` → DataLoader → FERTrainer → train/test
- `train_all.py`：用 subprocess 顺序训练四个模型（为什么用子进程：隔离失败、实时转发输出）
- `inference.py`：三种模式（单图 / benchmark / export）；`run_api.py`：5 行

**通过标准**：能讲清"一张图从前端上传到结果展示"的端到端调用链（frontend → api → detector → inferrer → 回前端）。

### 阶段三：动手改造实验（2 天）

原则：每个实验都是"**改动 → 观察 → 解释**"闭环。建议建一个实验记录表：`改动 | 现象 | 原因解释`。


| #  | 实验                                     | 建议观察点                |
| ---- | ------------------------------------------ | --------------------------- |
| 1  | 关闭数据增强训练 1 epoch                 | train/val 精度差异        |
| 2  | 模型换 resnet18 训练 1 epoch             | 速度与效果                |
| 3  | `use_focal_loss` 开 ↔ 关                | 训练曲线差异              |
| 4  | `batch_size` 8 ↔ 64                     | 速度 / 显存 / 稳定性      |
| 5  | 注释`zero_grad()`                        | loss 行为                 |
| 6  | 改错 normalize 的 std                    | 推理结果变化              |
| 7  | 在`train_epoch` 打印中间张量形状         | 验证你的"形状旅行"模型    |
| 8  | 导出 resnet 为 ONNX 并用 netron 查看     | 输入/输出节点名           |
| 9  | 给`/api/recognize` 加一个返回字段        | 在浏览器 Network 面板验证 |
| 10 | （进阶）同时关闭 Focal Loss 而开启 MixUp | 观察代码模块间的耦合关系  |

### 阶段四：从零重写（5~7 天）—— 决胜检验

**规则**

- 新建空目录（如 `my_fer/`）；**不看原项目源码**；允许查 PyTorch / Flask / onnxruntime 官方文档
- 卡住时：先想 15 分钟 → 查官方文档 → 仍无解才允许看原项目一眼（看完合上，凭记忆写）
- 每个里程碑"能跑出可见结果"才算通过


| 里程碑    | 内容                                                 | 验收（预期现象）                   |
| ----------- | ------------------------------------------------------ | ------------------------------------ |
| M1 数据   | Dataset + DataLoader                                 | 打印一个 batch：`[B, 3, 224, 224]` |
| M2 模型   | 任一模型 + 分类头替换                                | forward 输出`[B, 7]`               |
| M3 训练   | 最朴素训练循环（无 AMP/MixUp）                       | loss 下降、精度上升                |
| M4 训练器 | warmup + 调度 + LabelSmoothing + best 保存 + history | 产物结构与原项目一致               |
| M5 推理   | ONNX 导出 + ONNXRuntime 推理                         | 与 PyTorch 结果差 < 1e-4           |
| M6 服务   | Flask：health / models / recognize                   | 能直接对接原版`frontend/`          |
| M7 全链路 | 训练 → 导出 → 服务 → 前端出结果                   | 端到端走通                         |

七个里程碑都通过后，你与"独立写出这个项目"的距离，只剩工程细节（错误处理、降级链、缓存策略）——对照原项目补齐即可。

### 阶段五：讲解输出（持续）

- **讲解稿**：每个文件 200~500 字（职责 / 输入输出 / 关键设计 / 为什么）
- **录音试讲**：假装给一个不懂深度学习的同学讲 40 分钟"这个项目怎么工作"，回听找卡壳点
- **白板默写**：训练五步 / 推理预处理六步 / recognize 接口主流程
- **随机抽考**：从第 4 章题库抽 10 题，每题 90 秒内作答

---

## 4 自测题库

> 用法：学完一个模块，抽该模块的题自测；全部学完后随机抽 20 题。答不出的题，回到对应文件再读一遍。
> 挑战模式：让我扮演面试官随机抽题，你口头回答，我指出漏洞。

### A 档 · 概念理解（能说清）

1. 这个项目解决什么问题？输入什么、输出什么？
2. 画出训练管线和推理管线的完整数据流。
3. 7 个表情类别分别是哪些？
4. 四种模型架构的名字与大致参数量？
5. 什么是预训练权重？为什么它对小数据集训练有帮助？
6. 数据增强有哪些手段？为什么能提升泛化？
7. 什么是类别不平衡？本项目哪个环节处理它？
8. Focal Loss 的 γ、α 分别起什么作用？
9. Label Smoothing 是做什么的？
10. ONNX 是什么？为什么推理用它而不是直接跑 PyTorch？
11. 混合精度（AMP）是什么？带来什么收益？
12. warmup 与 cosine 调度分别是什么、为什么用？
13. 早停（early stopping）的原理？
14. 什么是 MixUp？它对损失计算有什么影响？
15. 为什么"人脸检测 + 表情分类"要分成两步？

### B 档 · 代码细节（能定位）

16. FERDataset 必须实现哪两个方法？各自作用？
17. 数据增强在哪个方法里构建？为什么验证集不走增强？
18. LABEL_MAPPING 的作用？不做映射会怎样？
19. FER2013Dataset 支持哪几种格式？如何自动判断？
20. ToTensor 做了哪三件事？
21. Normalize 的 mean/std 具体数值？为什么是这组值？
22. `drop_last` 什么情况下为 True？为什么？
23. VGG 分类头第一层 Linear 的输入维度？由什么决定？
24. 四个模型"替换分类头"的代码位置分别在哪儿？
25. 训练循环的标准五个步骤？
26. AMP 分支涉及哪些调用？顺序是？
27. best_model 与 checkpoint_epoch_N 的保存内容差异？
28. history.json 有哪些字段？
29. preprocess 按顺序做了哪些操作？
30. softmax 里为什么要先减去最大值？
31. detect() 返回的 bbox 是什么格式？
32. extract_face 的 margin 参数作用？
33. ModelManager 如何发现一个新训练的模型？
34. `_pth_to_onnx` 如何避免重复转换？
35. /api/recognize 支持哪两种图片提交方式？
36. 响应里为什么必须把 numpy 类型转成 Python 原生类型？

### C 档 · 设计决策（能讲"为什么"）

37. 为什么用 dataclass 定义配置而不是 dict？
38. train.py 为什么要检测"命令行是否显式传参"？不检测会怎样？
39. 为什么预训练权重优先从本地加载？
40. 为什么每个模型都提供 `create_xxx_model()` 工厂函数？
41. trainer 里为什么兼容 2 元组 / 3 元组两种数据？（提示：MultiDataset）
42. 为什么梯度裁剪放在 AMP 的 `unscale_` 之后？
43. 为什么验证阶段不需要 AMP 的 scaler 逻辑？
44. 为什么 best_model 不含优化器状态？
45. 早停计数器为什么"发现更优"时要清零？
46. 推理前为什么做 warmup？
47. ONNX 导出为什么需要 dummy_input？
48. dynamic_axes 解决什么问题？
49. MobileViT 导出 ONNX 遇到什么问题？如何解决？
50. 服务启动时为什么"预转换所有模型"？
51. 为什么缓存"已加载的推理器"？
52. 为什么 API 要做"无脸兜底"（整图推理）？
53. 前端为什么用 canvas 重绘人脸框，而不是直接用后端返回的图片？
54. 新增一种模型架构（如 EfficientNet）要改哪些文件？
55. 新增一个数据集要改哪些文件？

---

## 5 常见坑与调试技巧

- **Windows 多进程**：自己写实验脚本用 `num_workers>0` 时，需要 `if __name__ == '__main__':` 保护，否则报错或卡死
- **显存不足**：先降 `batch_size` → 换小模型 → 开 `use_amp`
- **"怎么识别都是同一类"**：按顺序排查——① normalize 的 mean/std 是否与训练一致；② 是否 RGB/BGR 弄反；③ 权重是否加载成功
- **eval_models.py 找不到 checkpoint**：脚本里的路径可能不是当前目录结构——读懂路径生成逻辑并修正（本身就是绝佳练习）
- **ONNX 导出失败**：检查 opset 与算子兼容性；参考 `export.py` 里 SDPA 的处理方案
- **端口占用**：改端口或结束占用进程
- **train_all.py 输出乱码**：脚本用 gbk 读取子进程输出，终端编码不匹配时可能乱码，通常不影响训练本身
- **调试 Flask**：`python run_api.py`（debug=False）直接打断点即可
- **善用形状追踪**：任何"看不懂"的张量操作，先 print shape

---

## 6 学习进度打卡表

- [ ]  阶段一：跑通全部命令 + 手绘两条管线图
- [ ]  模块一：配置层 —— 能画配置流向图
- [ ]  模块二：数据层 —— 能默写 Dataset 签名 + 三个"为什么"
- [ ]  模块三：模型层 —— 能讲清五段式 + "512×7×7"
- [ ]  模块四：训练层 —— 能默写训练五步 + AMP 分支
- [ ]  模块五：推理层 —— 能复述预处理六步 + 三个"为什么"
- [ ]  模块六：服务层 —— 能讲清 recognize 全流程
- [ ]  模块七：前端与入口 —— 能讲端到端调用链
- [ ]  阶段三：完成 10 个改造实验并记录
- [ ]  阶段四：七个重写里程碑全部通过
- [ ]  阶段五：完成一次 40 分钟完整试讲
- [ ]  终极检验：空白重写（M1~M7）+ 白板讲解（第 4 章题库答对 90%）

---

## 7 附：把 AI 用成你的私人助教

学习过程中随时可以用这几类请求（示例）：

- "解释 `src/train/trainer.py` 的 `train_epoch`，逐行讲，并说明每一行为什么存在"
- "针对 `src/data/dataset.py` 出 10 道题考我，从易到难"
- "我来复述一遍推理链路，你指出我讲错的地方"
- "帮我 review 我重写的训练循环，和原版对比，列出我漏掉的细节"
- "模拟面试官，随机抽 10 题问我，我逐题回答，你评分"
