"""
训练模块
统一训练器，支持所有模型
"""

from .trainer import FERTrainer, TrainingConfig
from .optimizer import create_optimizer, create_scheduler
from .losses import FocalLoss, LabelSmoothingLoss

__all__ = [
    'FERTrainer',
    'TrainingConfig',
    'create_optimizer',
    'create_scheduler',
    'FocalLoss',
    'LabelSmoothingLoss'
]