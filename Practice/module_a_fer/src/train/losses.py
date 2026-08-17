"""
损失函数模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss
    用于处理类别不平衡问题
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        num_classes: int = 7,
        reduction: str = 'mean'
    ):
        """
        Args:
            alpha: 类别权重
            gamma: 聚焦参数
            num_classes: 类别数
            reduction: 归约方式
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets) -> torch.Tensor:
        """
        Args:
            inputs: 模型输出 logits [N, C]
            targets: 目标标签 [N] 或 Mixup 格式 (labels_a, labels_b, lam)
        
        Returns:
            loss: Focal Loss
        """
        # 处理 Mixup：targets 为 (labels_a, labels_b, lam) 时分别计算再加权
        if isinstance(targets, (tuple, list)):
            labels_a, labels_b, lam = targets
            loss_a = self._compute_focal_loss(inputs, labels_a)
            loss_b = self._compute_focal_loss(inputs, labels_b)
            return lam * loss_a + (1 - lam) * loss_b
        else:
            return self._compute_focal_loss(inputs, targets)
    
    def _compute_focal_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """计算单组标签的 Focal Loss"""
        # 计算交叉熵
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 计算概率
        pt = torch.exp(-ce_loss)
        
        # Focal权重
        focal_weight = (1 - pt) ** self.gamma
        
        # 类别权重
        if isinstance(self.alpha, (float, int)):
            alpha_t = self.alpha
        else:
            alpha_t = self.alpha[targets]
        
        # Focal Loss
        loss = alpha_t * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss
    """
    
    def __init__(
        self,
        num_classes: int = 7,
        smoothing: float = 0.1,
        reduction: str = 'mean'
    ):
        """
        Args:
            num_classes: 类别数
            smoothing: 平滑系数
            reduction: 归约方式
        """
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: 模型输出 logits [N, C]
            targets: 目标标签 [N]
        
        Returns:
            loss: Label Smoothing Loss
        """
        # 创建平滑标签
        with torch.no_grad():
            smooth_targets = torch.zeros_like(inputs)
            smooth_targets.fill_(self.smoothing / (self.num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)
        
        # 计算损失
        loss = -torch.sum(smooth_targets * F.log_softmax(inputs, dim=1), dim=1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss