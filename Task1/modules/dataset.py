import torch 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms 

# ============
# 配置参数
# ============

# 传入训练集和测试集的数据路径
TRAIN_DIR = "../datasets/Fer-2013/train"
TEST_DIR =  "../datasets/Fer-2013/test"

# 输入尺寸
IMAGE_SIZE = 224 # 预训练模型的标准输入尺寸

# 每轮输入的图片数量以及CPU运行核的数量
BATCH_SIZE = 64 # 每轮的传入图片数量
NUM_WORKS = 8 # CPU运行核的数量

# ImageNet 标准化参数（预训练模型就是用这组均值/方差训练的）
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229,0.224,0.225]

# 1. 数据增强，防止模型过拟合
# 训练集数据增强

"""
    步骤：
    1. 由于Fer-2013的数据集原始图片为灰度图片只有单通道，因此第一步需要先转换为三通道图片
    2. 统一图片维度，使用配置参数中的输入尺寸
    3. 进行数据增强，考虑使用：几何变换、光度变换和随机擦除进行增强
"""
train_transform = transforms.Compose([
    
    transforms.Grayscale(num_output_channels=3), # 将原始灰度图片转换为三通道图片
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), # 统一成(224,224)维度的图片
    # 添加随机水平反转
    transforms.RandomHorizontalFlip(p=0.5),
    # 添加轻微旋转，增强鲁棒性
    transforms.RandomRotation(degrees=10),
    # 添加颜色抖动，由于是黑白图片，所以不需要进行色调和饱和度的调整
    transforms.ColorJitter(brightness=0.18, contrast=0.22, saturation=0,hue=0),
    
    
    transforms.ToTensor(), # 转换成张量格式，将[0,255]转化成[0,1]之间的维度
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD), # 标准化
    # 添加随机擦除 
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.20),ratio=(0.3, 3.3), value="random"),
    
    
    ])

# 测试集之作必要的预处理，不做随机增强
test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

# 2. 加载数据集并创建DataLoader（封装为函数，避免Windows多进程spawn问题）
def et_train_loader(train_dir=TRAIN_DIR, batch_size=BATCH_SIZE,
                     num_workers=NUM_WORKS, transform=train_transform):
    """创建训练集 DataLoader"""
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=True,  # 丢弃最后一个不完整batch，保证BatchNorm稳定
    )


def get_test_loader(test_dir=TEST_DIR, batch_size=BATCH_SIZE,
                    num_workers=NUM_WORKS, transform=test_transform):
    """创建测试集 DataLoader"""
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


if __name__ == '__main__':
    train_loader = get_train_loader()
    test_loader = get_test_loader()
    train_dataset = train_loader.dataset
    test_dataset = test_loader.dataset

    print(f"训练集样本数：{len(train_dataset)}")
    print(f"测试集样本数：{len(test_dataset)}")
    print(f"类别映射：{train_dataset.class_to_idx}")
    # 3. 验证
    images, labels = next(iter(train_loader))
    print(f"\n一个batch的图片形状: {images.shape}")   # 期望: [64, 3, 224, 224]
    print(f"一个batch的标签: {labels[:10]}")
    print(f"像素值范围: [{images.min():.3f}, {images.max():.3f}]")