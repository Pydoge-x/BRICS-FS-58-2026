"""
FER-2013 数据集处理
支持CSV格式、图像目录格式和Kaggle下载格式
"""

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from .dataset import DatasetConfig, EXPRESSION_LABELS


class FER2013Dataset(Dataset):
    """
    FER-2013 数据集类
    支持多种格式：
    - CSV格式 (原始fer2013.csv)
    - 图像目录格式 (train/angry/*.jpg)
    - Kaggle解压格式 (test/angry/PrivateTest_*.jpg)
    """
    
    # 表情类别
    LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        image_size: int = 224,
        use_augmentation: bool = True,
        csv_file: Optional[str] = None
    ):
        """
        初始化FER-2013数据集
        
        Args:
            root_dir: 数据集根目录
            split: 数据集划分 (train/val/test/PublicTest/PrivateTest)
            image_size: 图像尺寸
            use_augmentation: 是否使用数据增强
            csv_file: CSV文件路径（如果使用CSV格式）
        """
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.use_augmentation = use_augmentation
        
        # 检查是否有CSV文件
        self.csv_file = csv_file or os.path.join(root_dir, 'fer2013.csv')
        
        # 自动检测数据集格式
        self.samples = self._auto_detect_and_load()
        
        # 设置变换
        self.transform = self._get_transforms()
        
        print(f"FER-2013 {split}: {len(self.samples)} 张图像")
    
    def _auto_detect_and_load(self) -> list:
        """自动检测并加载数据集"""
        # 尝试CSV格式
        if os.path.exists(self.csv_file):
            return self._load_from_csv()
        
        # 尝试标准目录格式 (train/angry/*.jpg)
        standard_split_dir = os.path.join(self.root_dir, self.split)
        if os.path.exists(standard_split_dir):
            return self._load_from_directory(standard_split_dir)
        
        # 尝试Kaggle格式 (根目录下直接有angry/disgust等子目录)
        has_class_dirs = any(os.path.isdir(os.path.join(self.root_dir, c)) for c in self.LABELS)
        if has_class_dirs:
            return self._load_from_directory(self.root_dir)
        
        # 尝试test/train目录下的类别目录
        for dir_name in ['train', 'test', 'val']:
            dir_path = os.path.join(self.root_dir, dir_name)
            if os.path.exists(dir_path):
                return self._load_from_directory(dir_path)
        
        raise ValueError(f"无法检测数据集格式: {self.root_dir}")
    
    def _load_from_csv(self) -> list:
        """从CSV文件加载"""
        df = pd.read_csv(self.csv_file)
        
        samples = []
        
        # 根据split选择数据
        if self.split == 'train':
            usage_filter = 'Training'
        elif self.split in ['val', 'PublicTest']:
            usage_filter = 'PublicTest'
        elif self.split in ['test', 'PrivateTest']:
            usage_filter = 'PrivateTest'
        else:
            usage_filter = 'Training'
        
        df_filtered = df[df['Usage'] == usage_filter]
        
        for idx, row in df_filtered.iterrows():
            emotion = int(row['emotion'])
            pixels = row['pixels']
            
            pixel_array = np.array(pixels.split(), dtype=np.uint8)
            image = pixel_array.reshape(48, 48)
            
            samples.append((image, emotion))
        
        return samples
    
    def _load_from_directory(self, base_dir: str) -> list:
        """从目录结构加载"""
        samples = []
        
        for class_idx, class_name in enumerate(self.LABELS):
            class_dir = os.path.join(base_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        img_path = os.path.join(class_dir, img_name)
                        samples.append((img_path, class_idx))
        
        return samples
    
    def _get_transforms(self) -> transforms.Compose:
        """获取数据变换"""
        transform_list = []
        
        # 数据增强（仅训练集）
        if self.use_augmentation and self.split == 'train':
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            ])
        
        # 调整尺寸
        transform_list.append(transforms.Resize((self.image_size, self.image_size)))
        
        # 转换为Tensor
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return transforms.Compose(transform_list)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        
        if isinstance(sample[0], np.ndarray):
            # CSV格式：直接是numpy数组（灰度）
            image_array, label = sample
            image = Image.fromarray(image_array, mode='L')
            image = image.convert('RGB')  # 转为RGB
        else:
            # 目录格式：图像路径
            img_path, label = sample
            image = Image.open(img_path).convert('RGB')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self) -> dict:
        """获取类别分布"""
        distribution = {}
        for _, label in self.samples:
            distribution[label] = distribution.get(label, 0) + 1
        return distribution


def create_fer2013_config(root_dir: str, split: str = 'train') -> DatasetConfig:
    """创建FER-2013数据集配置"""
    return DatasetConfig(
        name='fer2013',
        root_dir=root_dir,
        split=split,
        image_size=224,
        use_augmentation=True if split == 'train' else False,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )