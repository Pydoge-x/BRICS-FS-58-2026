"""
训练配置文件
"""

import os
from dataclasses import dataclass, asdict
import json


@dataclass
class DatasetConfig:
    """数据集配置"""
    name: str = 'fer2013'
    root_dir: str = 'dataset_data/fer2013'
    split: str = 'train'
    image_size: int = 224
    use_augmentation: bool = True


@dataclass
class ModelConfig:
    """模型配置"""
    type: str = 'mobilevit'
    name: str = 'mobilevit_xs'
    num_classes: int = 7
    pretrained: bool = True
    dropout_rate: float = 0.2


@dataclass
class TrainingConfig:
    """训练配置"""
    epochs: int = 100
    batch_size: int = 64
    num_workers: int = 4
    
    # 优化器
    optimizer: str = 'AdamW'
    lr: float = 3e-4
    weight_decay: float = 1e-5
    
    # 学习率调度
    scheduler: str = 'CosineAnnealingWarmRestarts'
    warmup_epochs: int = 5
    T_0: int = 20
    T_mult: int = 2
    eta_min: float = 1e-6
    
    # 损失函数
    label_smoothing: float = 0.1
    use_focal_loss: bool = False
    mixup_alpha: float = 0.0
    
    # 早停
    early_stop_patience: int = 0  # 0=禁用
    
    # 其他
    seed: int = 42
    use_amp: bool = True
    grad_clip: float = 1.0
    save_dir: str = 'checkpoints'
    save_interval: int = 10  # 定期保存间隔(epoch)，VGG等大模型建议≥10
    keep_checkpoint_max: int = 5  # 最多保留定期checkpoint数量，超出自动清理


@dataclass
class Config:
    """总配置"""
    dataset: DatasetConfig = None
    model: ModelConfig = None
    training: TrainingConfig = None
    
    def __post_init__(self):
        if self.dataset is None:
            self.dataset = DatasetConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
    
    def to_dict(self):
        return {
            'dataset': asdict(self.dataset),
            'model': asdict(self.model),
            'training': asdict(self.training)
        }
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @staticmethod
    def load(path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        
        return Config(
            dataset=DatasetConfig(**data.get('dataset', {})),
            model=ModelConfig(**data.get('model', {})),
            training=TrainingConfig(**data.get('training', {}))
        )


# 预定义配置
CONFIGS = {
    'mobilevit_fer2013': Config(
        dataset=DatasetConfig(name='fer2013', root_dir='dataset_data/fer2013'),
        model=ModelConfig(type='mobilevit', name='mobilevit_xs'),
        training=TrainingConfig(epochs=100, batch_size=64, lr=3e-4)
    ),
    'resnet_fer2013': Config(
        dataset=DatasetConfig(name='fer2013', root_dir='dataset_data/fer2013'),
        model=ModelConfig(type='resnet', name='resnet50'),
        training=TrainingConfig(epochs=80, batch_size=32, lr=1e-4)
    ),
    'mobilenet_fer2013': Config(
        dataset=DatasetConfig(name='fer2013', root_dir='dataset_data/fer2013'),
        model=ModelConfig(type='mobilenet', name='mobilenetv3_small'),
        training=TrainingConfig(epochs=100, batch_size=64, lr=3e-4)
    ),
    'vgg_fer2013': Config(
        dataset=DatasetConfig(name='fer2013', root_dir='dataset_data/fer2013'),
        model=ModelConfig(type='vgg', name='vgg16'),
        training=TrainingConfig(epochs=50, batch_size=32, lr=1e-4)
    ),
    'mobilevit_rafdb': Config(
        dataset=DatasetConfig(name='rafdb', root_dir='dataset_data/RAF-DB'),
        model=ModelConfig(type='mobilevit', name='mobilevit_xs'),
        training=TrainingConfig(epochs=100, batch_size=64, lr=3e-4)
    ),
    'mobilevit_affectnet': Config(
        dataset=DatasetConfig(name='fer2013_affectnet', root_dir='dataset_data/fer2013_affectnet'),
        model=ModelConfig(type='mobilevit', name='mobilevit_xs'),
        training=TrainingConfig(epochs=100, batch_size=64, lr=3e-4)
    ),
    'mobilevit_kdef': Config(
        dataset=DatasetConfig(name='kdef', root_dir='dataset_data/kdef-dataset'),
        model=ModelConfig(type='mobilevit', name='mobilevit_xs'),
        training=TrainingConfig(epochs=100, batch_size=64, lr=3e-4)
    ),
    # 多数据集训练
    'mobilevit_multi': Config(
        dataset=DatasetConfig(name='multi', root_dir='dataset_data'),
        model=ModelConfig(type='mobilevit', name='mobilevit_xs'),
        training=TrainingConfig(epochs=100, batch_size=64, lr=3e-4)
    )
}


def get_config(name: str) -> Config:
    """获取预定义配置"""
    if name in CONFIGS:
        return CONFIGS[name]
    else:
        return Config()