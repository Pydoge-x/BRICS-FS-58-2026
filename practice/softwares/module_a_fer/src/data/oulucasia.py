"""
Oulu-CASIA 数据集处理
NIR-VIS跨模态表情数据集
支持多种格式：标准Oulu-CASIA格式和Kaggle下载格式
"""

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Tuple, Optional, List
from .dataset import DatasetConfig


class OuluCASIADataset(Dataset):
    """
    Oulu-CASIA 数据集类
    包含可见光(VIS)和近红外(NIR)两种模态
    支持多种格式：
    - 标准格式 (VIS/Strong/Normal/Weak/Subject/Expression/)
    - Kaggle格式 (Oulu_CASIA_NIR_VIS/NI/Dark/P001/Anger/)
    """
    
    # Oulu-CASIA表情类别（6类，无neutral）
    LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise']
    
    # Oulu-CASIA到FER-2013标签映射
    # FER-2013: 0-angry, 1-disgust, 2-fear, 3-happy, 4-sad, 5-surprise, 6-neutral
    LABEL_MAPPING = {
        0: 0,   # Angry -> angry
        1: 1,   # Disgust -> disgust
        2: 2,   # Fear -> fear
        3: 3,   # Happy -> happy
        4: 4,   # Sad -> sad
        5: 5    # Surprise -> surprise
    }
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        image_size: int = 224,
        use_augmentation: bool = True,
        modality: str = 'vis',  # 'vis' 或 'nir' 或 'both'
        illumination: str = 'normal'  # 'normal', 'strong', 'weak', 'dark', 'all'
    ):
        """
        初始化Oulu-CASIA数据集
        
        Args:
            root_dir: 数据集根目录
            split: 数据集划分 (train/test)
            image_size: 图像尺寸
            use_augmentation: 是否使用数据增强
            modality: 模态选择 (vis/nir/both)
            illumination: 光照条件 (normal/strong/weak/dark/all)
        """
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.use_augmentation = use_augmentation
        self.modality = modality
        self.illumination = illumination
        
        # 加载样本
        self.samples = self._auto_detect_and_load()
        
        # 设置变换
        self.transform = self._get_transforms()
        
        print(f"Oulu-CASIA {split} ({modality}, {illumination}): {len(self.samples)} 张图像")
    
    def _auto_detect_and_load(self) -> List[Tuple[str, int]]:
        """自动检测并加载数据集"""
        samples = []
        
        # 尝试Kaggle格式 (Oulu_CASIA_NIR_VIS/NI/Dark/P001/Anger/)
        kaggle_dir = os.path.join(self.root_dir, 'Oulu_CASIA_NIR_VIS')
        if os.path.exists(kaggle_dir):
            return self._load_kaggle_format(kaggle_dir)
        
        # 尝试标准格式 (VIS/Strong/Normal/Weak/)
        vis_dir = os.path.join(self.root_dir, 'VIS')
        nir_dir = os.path.join(self.root_dir, 'NIR')
        if os.path.exists(vis_dir) or os.path.exists(nir_dir):
            return self._load_standard_format(vis_dir, nir_dir)
        
        # 尝试直接在根目录查找
        has_label_dirs = any(os.path.isdir(os.path.join(self.root_dir, label)) for label in self.LABELS)
        if has_label_dirs:
            return self._load_simple_format(self.root_dir)
        
        raise ValueError(f"无法检测Oulu-CASIA数据集格式: {self.root_dir}")
    
    def _load_kaggle_format(self, base_dir: str) -> List[Tuple[str, int]]:
        """加载Kaggle格式数据集"""
        samples = []
        
        modalities = []
        if self.modality == 'vis' or self.modality == 'both':
            modalities.append('VI')  # Kaggle格式中VIS使用VI
        if self.modality == 'nir' or self.modality == 'both':
            modalities.append('NI')  # Kaggle格式中NIR使用NI
        
        # 光照条件映射
        illumination_mapping = {
            'normal': 'Normal',
            'strong': 'SAD',      # Strong illumination
            'weak': 'DARK',       # Dark/weak illumination
            'dark': 'DARK'
        }
        
        illuminations = []
        if self.illumination == 'all':
            illuminations = ['Normal', 'SAD', 'DARK']
        else:
            illum_name = illumination_mapping.get(self.illumination.lower())
            if illum_name:
                illuminations = [illum_name]
        
        for mod in modalities:
            mod_dir = os.path.join(base_dir, mod)
            if not os.path.exists(mod_dir):
                continue
            
            for illum in illuminations:
                illum_dir = os.path.join(mod_dir, illum)
                if not os.path.exists(illum_dir):
                    continue
                
                # 扫描Subject目录
                for subject in os.listdir(illum_dir):
                    subject_dir = os.path.join(illum_dir, subject)
                    if not os.path.isdir(subject_dir):
                        continue
                    
                    # 扫描表情类别目录
                    for class_idx, class_name in enumerate(self.LABELS):
                        class_dir = os.path.join(subject_dir, class_name)
                        if not os.path.exists(class_dir):
                            continue
                        
                        # 获取该表情下的所有图像
                        for img_name in os.listdir(class_dir):
                            if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                                img_path = os.path.join(class_dir, img_name)
                                mapped_label = self.LABEL_MAPPING.get(class_idx, class_idx)
                                samples.append((img_path, mapped_label))
        
        return self._split_samples(samples)
    
    def _load_standard_format(self, vis_dir: str, nir_dir: str) -> List[Tuple[str, int]]:
        """加载标准Oulu-CASIA格式"""
        samples = []
        
        modalities = []
        if self.modality == 'vis' or self.modality == 'both':
            modalities.append(('VIS', vis_dir))
        if self.modality == 'nir' or self.modality == 'both':
            modalities.append(('NIR', nir_dir))
        
        illuminations = []
        if self.illumination == 'all':
            illuminations = ['Strong', 'Normal', 'Weak']
        else:
            illuminations = [self.illumination.capitalize()]
        
        for mod, mod_dir in modalities:
            if not os.path.exists(mod_dir):
                continue
            
            if mod == 'VIS':
                for illum in illuminations:
                    illum_dir = os.path.join(mod_dir, illum)
                    if os.path.exists(illum_dir):
                        samples.extend(self._scan_standard_dir(illum_dir))
            elif mod == 'NIR':
                samples.extend(self._scan_standard_dir(mod_dir))
        
        return self._split_samples(samples)
    
    def _load_simple_format(self, base_dir: str) -> List[Tuple[str, int]]:
        """加载简化格式（直接按表情类别组织）"""
        samples = []
        
        for class_idx, class_name in enumerate(self.LABELS):
            class_dir = os.path.join(base_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(class_dir, img_name)
                    mapped_label = self.LABEL_MAPPING.get(class_idx, class_idx)
                    samples.append((img_path, mapped_label))
        
        return self._split_samples(samples)
    
    def _scan_standard_dir(self, directory: str) -> List[Tuple[str, int]]:
        """扫描标准格式目录"""
        samples = []
        
        for subject in os.listdir(directory):
            subject_dir = os.path.join(directory, subject)
            if not os.path.isdir(subject_dir):
                continue
            
            for class_idx, class_name in enumerate(self.LABELS):
                class_dir = os.path.join(subject_dir, class_name)
                if not os.path.exists(class_dir):
                    continue
                
                for sequence in os.listdir(class_dir):
                    seq_dir = os.path.join(class_dir, sequence)
                    if not os.path.isdir(seq_dir):
                        continue
                    
                    for img_name in os.listdir(seq_dir):
                        if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            img_path = os.path.join(seq_dir, img_name)
                            mapped_label = self.LABEL_MAPPING.get(class_idx, class_idx)
                            samples.append((img_path, mapped_label))
        
        return samples
    
    def _split_samples(self, samples: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        """划分训练集和测试集"""
        if self.split not in ['train', 'test']:
            return samples
        
        total = len(samples)
        train_size = int(total * 0.8)
        
        # 按标签排序后划分，确保类别分布均匀
        samples.sort(key=lambda x: x[1])
        
        if self.split == 'train':
            return samples[:train_size]
        else:
            return samples[train_size:]
    
    def _get_transforms(self) -> transforms.Compose:
        """获取数据变换"""
        transform_list = [
            transforms.Resize((self.image_size, self.image_size)),
        ]
        
        # 数据增强（仅训练集）
        if self.use_augmentation and self.split == 'train':
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.3, contrast=0.3),
                transforms.RandomRotation(degrees=10),
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


def create_oulucasia_config(
    root_dir: str, 
    split: str = 'train',
    modality: str = 'vis',
    illumination: str = 'normal'
) -> DatasetConfig:
    """创建Oulu-CASIA数据集配置"""
    return DatasetConfig(
        name='oulucasia',
        root_dir=root_dir,
        split=split,
        image_size=224,
        use_augmentation=True if split == 'train' else False,
        label_mapping=OuluCASIADataset.LABEL_MAPPING
    )