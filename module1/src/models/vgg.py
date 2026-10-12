"""
VGG模型 - 表情识别
基于VGG-16/19的预训练模型
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional
from .pretrained_loader import load_local_pretrained


class FER_VGG(nn.Module):
    """
    人脸表情识别模型 - 基于VGG
    
    支持VGG-11/13/16/19
    """
    
    def __init__(
        self, 
        num_classes: int = 7, 
        vgg_type: str = 'vgg16',
        pretrained: bool = True, 
        dropout_rate: float = 0.5
    ):
        """
        初始化VGG模型
        
        Args:
            num_classes: 分类类别数
            vgg_type: VGG类型 (vgg11/vgg13/vgg16/vgg19)
            pretrained: 是否使用预训练权重
            dropout_rate: Dropout比例
        """
        super(FER_VGG, self).__init__()
        
        self.vgg_type = vgg_type
        self.num_classes = num_classes
        
        # 加载预训练VGG模型
        vgg_models = {
            'vgg11': models.vgg11,
            'vgg13': models.vgg13,
            'vgg16': models.vgg16,
            'vgg19': models.vgg19,
        }
        
        if vgg_type not in vgg_models:
            raise ValueError(f"不支持的VGG类型: {vgg_type}")
        
        # 加载预训练模型
        if pretrained:
            weights = {
                'vgg11': models.VGG11_Weights.IMAGENET1K_V1,
                'vgg13': models.VGG13_Weights.IMAGENET1K_V1,
                'vgg16': models.VGG16_Weights.IMAGENET1K_V1,
                'vgg19': models.VGG19_Weights.IMAGENET1K_V1,
            }
            self.vgg = vgg_models[vgg_type](weights=None); load_local_pretrained(self.vgg, vgg_type + ".pth") or self.vgg.load_state_dict(vgg_models[vgg_type](weights=weights[vgg_type]).state_dict())
        else:
            self.vgg = vgg_models[vgg_type](weights=None)
        
        # 修改分类器
        # VGG原始分类器: 4096 -> 4096 -> 1000
        # 修改为: 4096 -> 4096 -> num_classes
        in_features = self.vgg.classifier[6].in_features
        
        self.vgg.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.vgg(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征（不包含分类层）"""
        x = self.vgg.features(x)
        x = self.vgg.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.vgg.classifier[:4](x)  # 前4096维特征
        return x


def create_vgg_model(
    vgg_type: str = 'vgg16',
    num_classes: int = 7,
    pretrained: bool = True,
    dropout_rate: float = 0.5
) -> FER_VGG:
    """
    创建VGG表情识别模型
    
    Args:
        vgg_type: VGG类型
        num_classes: 分类类别数
        pretrained: 是否使用预训练权重
        dropout_rate: Dropout比例
    
    Returns:
        FER_VGG模型
    """
    return FER_VGG(
        num_classes=num_classes,
        vgg_type=vgg_type,
        pretrained=pretrained,
        dropout_rate=dropout_rate
    )


# VGG模型配置
VGG_CONFIG = {
    'vgg11': {
        'params': '~133M',
        'input_size': 224,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
    },
    'vgg13': {
        'params': '~133M',
        'input_size': 224,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
    },
    'vgg16': {
        'params': '~138M',
        'input_size': 224,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
    },
    'vgg19': {
        'params': '~144M',
        'input_size': 224,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
    }
}