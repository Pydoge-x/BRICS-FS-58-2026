"""
优化器和学习率调度器模块
"""

import torch
import torch.optim as optim
from typing import Optional, Dict, Any


def create_optimizer(
    model: torch.nn.Module,
    optimizer_type: str = 'AdamW',
    lr: float = 3e-4,
    weight_decay: float = 1e-5,
    momentum: float = 0.9,
    **kwargs
) -> optim.Optimizer:
    """
    创建优化器
    
    Args:
        model: 模型
        optimizer_type: 优化器类型
        lr: 学习率
        weight_decay: 权重衰减
        momentum: 动量（SGD使用）
        **kwargs: 其他参数
    
    Returns:
        优化器
    """
    if optimizer_type == 'Adam':
        return optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_type == 'AdamW':
        return optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_type == 'SGD':
        return optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            **kwargs
        )
    else:
        raise ValueError(f"不支持的优化器: {optimizer_type}")


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str = 'CosineAnnealingWarmRestarts',
    **kwargs
) -> Optional[optim.lr_scheduler._LRScheduler]:
    """
    创建学习率调度器
    
    Args:
        optimizer: 优化器
        scheduler_type: 调度器类型
        **kwargs: 调度器参数
    
    Returns:
        学习率调度器
    """
    if scheduler_type == 'StepLR':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 20),
            gamma=kwargs.get('gamma', 0.1)
        )
    elif scheduler_type == 'MultiStepLR':
        return optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=kwargs.get('milestones', [50, 80]),
            gamma=kwargs.get('gamma', 0.1)
        )
    elif scheduler_type == 'ExponentialLR':
        return optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=kwargs.get('gamma', 0.95)
        )
    elif scheduler_type == 'CosineAnnealingLR':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', 100),
            eta_min=kwargs.get('eta_min', 1e-6)
        )
    elif scheduler_type == 'CosineAnnealingWarmRestarts':
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=kwargs.get('T_0', 20),
            T_mult=kwargs.get('T_mult', 2),
            eta_min=kwargs.get('eta_min', 1e-6)
        )
    elif scheduler_type == 'ReduceLROnPlateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 10),
            verbose=True
        )
    else:
        return None


class WarmupScheduler:
    """
    Warmup学习率调度器
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        warmup_lr: float,
        after_scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
    ):
        """
        Args:
            optimizer: 优化器
            warmup_epochs: warmup epoch数
            warmup_lr: warmup结束时的学习率
            after_scheduler: warmup后的调度器
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr
        self.after_scheduler = after_scheduler
        self.current_epoch = 0
    
    def step(self):
        """更新学习率"""
        if self.current_epoch < self.warmup_epochs:
            # Warmup阶段
            lr = self.warmup_lr * (self.current_epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # Warmup后
            if self.after_scheduler is not None:
                self.after_scheduler.step()
        
        self.current_epoch += 1
    
    def get_lr(self) -> float:
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']