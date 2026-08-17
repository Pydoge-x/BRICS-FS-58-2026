"""
RAF-DB 数据集处理
真实世界人脸表情数据集
支持多种格式：标准RAF-DB格式和Kaggle下载格式
"""

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Tuple, Optional, List
from .dataset import DatasetConfig


class RAFDBDataset(Dataset):
    """
    RAF-DB 数据集类
    真实世界人脸表情识别数据集
    支持多种格式：
    - 标准RAF-DB格式 (Image/train/, label/train_label.txt)
    - Kaggle格式 (DATASET/train/1/*.jpg)
    - 目录格式 (train/surprise/*.jpg)
    """
    
    # RAF-DB表情类别（7类基本表情）
    LABELS = {
        1: 'surprise',
        2: 'fear',
        3: 'disgust',
        4: 'happy',
        5: 'sad',
        6: 'angry',
        7: 'neutral'
    }
    
    # RAF-DB到FER-2013标签映射
    # FER-2013: 0-angry, 1-disgust, 2-fear, 3-happy, 4-sad, 5-surprise, 6-neutral
    LABEL_MAPPING = {
        1: 5,   # Surprise -> surprise (5)
        2: 2,   # Fear -> fear (2)
        3: 1,   # Disgust -> disgust (1)
        4: 3,   # Happiness -> happy (3)
        5: 4,   # Sadness -> sad (4)
        6: 0,   # Anger -> angry (0)
        7: 6    # Neutral -> neutral (6)
    }
    
    # FER-2013类别名称到索引
    FER_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        image_size: int = 224,
        use_augmentation: bool = True,
        use_aligned: bool = True
    ):
        """
        初始化RAF-DB数据集
        
        Args:
            root_dir: 数据集根目录
            split: 数据集划分 (train/test)
            image_size: 图像尺寸
            use_augmentation: 是否使用数据增强
            use_aligned: 是否使用对齐后的图像
        """
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.use_augmentation = use_augmentation
        self.use_aligned = use_aligned
        
        # 加载样本
        self.samples = self._auto_detect_and_load()
        
        # 设置变换
        self.transform = self._get_transforms()
        
        print(f"RAF-DB {split}: {len(self.samples)} 张图像")
    
    def _auto_detect_and_load(self) -> List[Tuple[str, int]]:
        """自动检测并加载数据集"""
        # 尝试标准RAF-DB格式 (Image/train/, label/train_label.txt)
        standard_image_dir = os.path.join(self.root_dir, 'Image', self.split)
        standard_label_file = os.path.join(self.root_dir, 'label', f'{self.split}_label.txt')
        if os.path.exists(standard_image_dir) and os.path.exists(standard_label_file):
            return self._load_from_label_file(standard_image_dir, standard_label_file)
        
        # 尝试Kaggle格式 (DATASET/train/1/*.jpg)
        kaggle_dir = os.path.join(self.root_dir, 'DATASET', self.split)
        if os.path.exists(kaggle_dir):
            return self._load_from_kaggle_format(kaggle_dir)
        
        # 尝试简化目录格式 (train/ 直接在根目录)
        simple_dir = os.path.join(self.root_dir, self.split)
        if os.path.exists(simple_dir):
            # 检查是否有数字类别目录 (1, 2, 3...)
            has_num_dirs = any(os.path.isdir(os.path.join(simple_dir, str(i))) for i in range(1, 8))
            if has_num_dirs:
                return self._load_from_kaggle_format(simple_dir)
            
            # 检查是否有类别名称目录
            has_name_dirs = any(os.path.isdir(os.path.join(simple_dir, c)) for c in self.FER_LABELS)
            if has_name_dirs:
                return self._load_from_name_directory(simple_dir)
        
        # 尝试根目录直接有类别目录
        has_root_num_dirs = any(os.path.isdir(os.path.join(self.root_dir, str(i))) for i in range(1, 8))
        if has_root_num_dirs:
            return self._load_from_kaggle_format(self.root_dir)
        
        has_root_name_dirs = any(os.path.isdir(os.path.join(self.root_dir, c)) for c in self.FER_LABELS)
        if has_root_name_dirs:
            return self._load_from_name_directory(self.root_dir)
        
        raise ValueError(f"无法检测RAF-DB数据集格式: {self.root_dir}")
    
    def _load_from_label_file(self, image_dir: str, label_file: str) -> List[Tuple[str, int]]:
        """从标签文件加载（标准RAF-DB格式）"""
        samples = []
        
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_name = parts[0]
                    label = int(parts[1])
                    
                    # 查找图像路径
                    img_path = os.path.join(image_dir, img_name)
                    if not os.path.exists(img_path):
                        img_path = os.path.join(image_dir, 'aligned', img_name)
                    
                    if os.path.exists(img_path):
                        mapped_label = self.LABEL_MAPPING.get(label, label - 1)
                        samples.append((img_path, mapped_label))
        
        return samples
    
    def _load_from_kaggle_format(self, base_dir: str) -> List[Tuple[str, int]]:
        """从Kaggle格式加载 (目录名为数字1-7)"""
        samples = []
        
        for raf_label in range(1, 8):
            class_dir = os.path.join(base_dir, str(raf_label))
            if os.path.exists(class_dir):
                mapped_label = self.LABEL_MAPPING.get(raf_label, raf_label - 1)
                
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        img_path = os.path.join(class_dir, img_name)
                        samples.append((img_path, mapped_label))
        
        return samples
    
    def _load_from_name_directory(self, base_dir: str) -> List[Tuple[str, int]]:
        """从类别名称目录加载"""
        samples = []
        
        for fer_idx, class_name in enumerate(self.FER_LABELS):
            class_dir = os.path.join(base_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        img_path = os.path.join(class_dir, img_name)
                        samples.append((img_path, fer_idx))
        
        return samples
    
    def _get_transforms(self) -> transforms.Compose:
        """获取数据变换"""
        transform_list = [
            transforms.Resize((self.image_size, self.image_size)),
        ]
        
        # 数据增强（仅训练集）
        if self.use_augmentation and self.split == 'train':
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomRotation(degrees=15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            ])
        
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return transforms.Compose(transform_list)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        # 加载图像
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


def create_rafdb_config(root_dir: str, split: str = 'train') -> DatasetConfig:
    """创建RAF-DB数据集配置"""
    return DatasetConfig(
        name='rafdb',
        root_dir=root_dir,
        split=split,
        image_size=224,
        use_augmentation=True if split == 'train' else False,
        label_mapping=RAFDBDataset.LABEL_MAPPING
    )