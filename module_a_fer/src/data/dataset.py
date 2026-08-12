"""
表情识别数据集基类和多数据集支持
支持单个数据集或多个数据集组合训练
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import random


# 表情类别定义
EXPRESSION_LABELS = {
    'fer2013': ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'],
    'rafdb': ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral'],
    'oulucasia': ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise']
}

# 类别映射（统一到FER-2013标准）
LABEL_MAPPING = {
    'fer2013': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6},  # 直接映射
    'rafdb': {1: 5, 2: 2, 3: 1, 4: 3, 5: 4, 6: 0, 7: 6},    # RAF-DB到FER-2013
    'oulucasia': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}       # Oulu-CASIA到FER-2013（无neutral）
}


@dataclass
class DatasetConfig:
    """数据集配置"""
    name: str                          # 数据集名称
    root_dir: str                      # 数据集根目录
    split: str = 'train'               # 数据集划分 (train/val/test)
    image_size: int = 224              # 图像尺寸
    use_augmentation: bool = True      # 是否使用数据增强
    normalize: bool = True             # 是否标准化
    label_mapping: Optional[Dict] = None  # 标签映射
    
    # ImageNet标准化参数
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)


class FERDataset(Dataset):
    """
    表情识别数据集基类
    支持从目录结构加载图像
    """
    
    def __init__(self, config: DatasetConfig):
        """
        初始化数据集
        
        Args:
            config: 数据集配置
        """
        self.config = config
        self.root_dir = config.root_dir
        self.split = config.split
        self.image_size = config.image_size
        
        # 获取标签映射
        self.label_mapping = config.label_mapping or LABEL_MAPPING.get(config.name, {})
        
        # 加载图像路径和标签
        self.samples = self._load_samples()
        
        # 设置变换
        self.transform = self._get_transforms()
        
        print(f"加载 {config.name} 数据集 ({self.split}): {len(self.samples)} 张图像")
    
    def _load_samples(self) -> List[Tuple[str, int]]:
        """加载图像路径和标签"""
        samples = []
        split_dir = os.path.join(self.root_dir, self.split)
        
        if not os.path.exists(split_dir):
            # 尝试其他目录结构
            split_dir = self.root_dir
        
        # 检查目录结构
        if os.path.exists(split_dir):
            # 按类别目录组织
            for class_idx, class_name in enumerate(EXPRESSION_LABELS.get(self.config.name, [])):
                class_dir = os.path.join(split_dir, class_name)
                if os.path.exists(class_dir):
                    for img_name in os.listdir(class_dir):
                        if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            img_path = os.path.join(class_dir, img_name)
                            mapped_label = self.label_mapping.get(class_idx, class_idx)
                            samples.append((img_path, mapped_label))
        
        return samples
    
    def _get_transforms(self) -> transforms.Compose:
        """获取数据变换"""
        transform_list = [
            transforms.Resize((self.image_size, self.image_size)),
        ]
        
        # 数据增强（仅训练集）
        if self.config.use_augmentation and self.split == 'train':
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomRotation(degrees=10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            ])
        
        transform_list.extend([
            transforms.ToTensor(),
        ])
        
        # 标准化
        if self.config.normalize:
            transform_list.append(
                transforms.Normalize(mean=self.config.mean, std=self.config.std)
            )
        
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
    
    def get_class_distribution(self) -> Dict[int, int]:
        """获取类别分布"""
        distribution = {}
        for _, label in self.samples:
            distribution[label] = distribution.get(label, 0) + 1
        return distribution


class MultiDataset(Dataset):
    """
    多数据集组合类
    支持同时使用多个数据集进行训练
    """
    
    def __init__(self, configs: List[DatasetConfig], balance: bool = False):
        """
        初始化多数据集
        
        Args:
            configs: 数据集配置列表
            balance: 是否平衡各数据集样本数量
        """
        self.configs = configs
        self.balance = balance
        
        # 加载各数据集
        self.datasets = []
        self.dataset_indices = []  # 记录每个样本来自哪个数据集
        
        total_samples = 0
        for i, config in enumerate(configs):
            dataset = FERDataset(config)
            self.datasets.append(dataset)
            
            # 记录数据集索引
            for _ in range(len(dataset)):
                self.dataset_indices.append(i)
            
            total_samples += len(dataset)
            print(f"  - {config.name}: {len(dataset)} 张图像")
        
        # 合并样本
        self.all_samples = []
        for dataset in self.datasets:
            self.all_samples.extend(dataset.samples)
        
        # 平衡处理
        if balance:
            self._balance_datasets()
        
        print(f"多数据集总计: {len(self.all_samples)} 张图像")
    
    def _balance_datasets(self):
        """平衡各数据集样本数量"""
        # 找到最小数据集大小
        min_size = min(len(d) for d in self.datasets)
        
        # 对每个数据集进行采样
        balanced_samples = []
        for dataset in self.datasets:
            indices = random.sample(range(len(dataset)), min_size)
            for idx in indices:
                balanced_samples.append(dataset.samples[idx])
        
        self.all_samples = balanced_samples
        print(f"平衡后总计: {len(self.all_samples)} 张图像")
    
    def __len__(self) -> int:
        return len(self.all_samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        """
        返回图像、标签和数据集索引
        
        Returns:
            image: 图像张量
            label: 标签
            dataset_idx: 数据集索引
        """
        img_path, label = self.all_samples[idx]
        dataset_idx = self.dataset_indices[idx]
        
        # 使用对应数据集的变换
        dataset = self.datasets[dataset_idx]
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        
        # 应用变换
        if dataset.transform:
            image = dataset.transform(image)
        
        return image, label, dataset_idx
    
    def get_dataset_stats(self) -> Dict:
        """获取数据集统计信息"""
        stats = {}
        for i, config in enumerate(self.configs):
            dataset = self.datasets[i]
            stats[config.name] = {
                'total': len(dataset),
                'distribution': dataset.get_class_distribution()
            }
        return stats


def create_dataloader(
    dataset: Union[FERDataset, MultiDataset],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 4
) -> DataLoader:
    """
    创建数据加载器
    
    Args:
        dataset: 数据集
        batch_size: 批大小
        shuffle: 是否打乱
        num_workers: 工作进程数
        pin_memory: 是否固定内存
        prefetch_factor: 每个worker预取的batch数量
    
    Returns:
        DataLoader
    """
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True if shuffle else False,
    )
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = prefetch_factor
        pass  # persistent_workers not needed for num_workers=0
    return DataLoader(**loader_kwargs)


def create_multi_dataset(
    dataset_names: List[str],
    root_dirs: List[str],
    split: str = 'train',
    image_size: int = 224,
    use_augmentation: bool = True,
    balance: bool = False
) -> MultiDataset:
    """
    创建多数据集的便捷函数
    
    Args:
        dataset_names: 数据集名称列表 ['fer2013', 'rafdb', 'oulucasia']
        root_dirs: 数据集根目录列表
        split: 数据集划分
        image_size: 图像尺寸
        use_augmentation: 是否使用数据增强
        balance: 是否平衡数据集
    
    Returns:
        MultiDataset
    """
    configs = []
    for name, root_dir in zip(dataset_names, root_dirs):
        config = DatasetConfig(
            name=name,
            root_dir=root_dir,
            split=split,
            image_size=image_size,
            use_augmentation=use_augmentation
        )
        configs.append(config)
    
    return MultiDataset(configs, balance=balance)