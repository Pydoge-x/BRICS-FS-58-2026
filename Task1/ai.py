import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# =====================
# 配置参数
# =====================
TRAIN_DIR = "datasets/Fer-2013/train"
TEST_DIR = "datasets/Fer-2013/test"
BATCH_SIZE = 64
NUM_WORKERS = 4
IMAGE_SIZE = 224  # 预训练模型的标准输入尺寸

# ImageNet 标准化参数（预训练模型就是用这组均值/方差训练的）
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# =====================
# 1. 数据增强流水线
# =====================
# 训练集：增强 → 让模型见到更多"变化"，防止死记硬背（过拟合）
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # 灰度→3通道RGB
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    # TODO_1: 添加随机水平翻转（提示：transforms.Random________）
    transforms.______(p=0.5),
    # TODO_2: 添加轻微旋转，增强鲁棒性（提示：transforms.Random________）
    transforms.______(degrees=10),
    transforms.ToTensor(),                         # PIL图片 → Tensor，像素值0-255映射到0-1
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)  # 标准化
])

# 测试集：只做必要预处理，不做随机增强（保证评估结果可复现）
test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

# =====================
# 2. 加载数据集
# =====================
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)

print(f"训练集样本数: {len(train_dataset)}")
print(f"测试集样本数: {len(test_dataset)}")
print(f"类别映射: {train_dataset.class_to_idx}")

# =====================
# 3. 创建 DataLoader
# =====================
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=______,   # TODO_3: 训练时填 True 还是 False？
    num_workers=NUM_WORKERS,
    pin_memory=True   # GPU训练加速：数据预加载到锁页内存
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=______,   # TODO_4: 测试时呢？
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# =====================
# 4. 验证：取一个batch看看形状
# =====================
images, labels = next(iter(train_loader))
print(f"\n一个batch的图片形状: {images.shape}")   # 期望: [64, 3, 224, 224]
print(f"一个batch的标签: {labels[:10]}")
print(f"像素值范围: [{images.min():.3f}, {images.max():.3f}]")