import torch 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms 

# ============
# 配置参数
# ============

TRAIN_DIR = "../datasets/Fer-2013/train"
TEST_DIR =  "../datasets/Fer-2013/test"

IMAGE_SIZE = 224 # 预训练模型的标准输入尺寸

BATCH_SIZE = 64 # 每轮的传入图片数量
NUM_WORKS = 4 # CPU运行核的数量

# ImageNet 标准化参数（预训练模型就是用这组均值/方差训练的）
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229,0.224,0.225]

# 1. 数据增强，防止模型过拟合
# 训练集数据增强
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3), # 将原始灰度图片转换为三通道图片
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), # 统一成(224,224)维度的图片
    # 添加随机水平反转
    transforms.RandomHorizontalFlip(p=0.5),
    # 添加轻微旋转，增强鲁棒性
    transforms.RandomRotation(degrees=10),
    
    transforms.ToTensor(), # 转换成张量格式，将[0,255]转化成[0,1]之间的维度
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD) # 标准化
    ])

# 测试集之作必要的预处理，不做随机增强
test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

# 2. 加载数据集
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)


# 3. 创建DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKS,
    pin_memory=True
    )

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKS,
    pin_memory=True
    )

if __name__ == '__main__':
    print(f"训练集样本数：{len(train_dataset)}")
    print(f"测试集样本数：{len(test_dataset)}")
    print(f"类别映射：{train_dataset.class_to_idx}")
    # 4. 验证
    images, labels = next(iter(train_loader))
    print(f"\n一个batch的图片形状: {images.shape}")   # 期望: [64, 3, 224, 224]
    print(f"一个batch的标签: {labels[:10]}")
    print(f"像素值范围: [{images.min():.3f}, {images.max():.3f}]")