"""
ResNet模型 - 表情识别
基于ResNet-18/34/50/101/152的预训练模型
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional
from .pretrained_loader import load_local_pretrained


class FER_ResNet(nn.Module):
    """
    人脸表情识别模型 - 基于ResNet
    
    支持ResNet-18/34/50/101/152
    """
    
    def __init__(
        self, 
        num_classes: int = 7, 
        resnet_type: str = 'resnet50',
        pretrained: bool = True, 
        dropout_rate: float = 0.3
    ):
        """
        初始化ResNet模型
        
        Args:
            num_classes: 分类类别数
            resnet_type: ResNet类型 (resnet18/34/50/101/152)
            pretrained: 是否使用预训练权重
            dropout_rate: Dropout比例
        """
        super(FER_ResNet, self).__init__()
        
        self.resnet_type = resnet_type
        self.num_classes = num_classes
        
        # ResNet模型字典
        resnet_models = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
            'resnet152': models.resnet152,
        }
        
        if resnet_type not in resnet_models:
            raise ValueError(f"不支持的ResNet类型: {resnet_type}")
        
        # 加载预训练模型
        if pretrained:
            weights = {
                'resnet18': models.ResNet18_Weights.IMAGENET1K_V1,
                'resnet34': models.ResNet34_Weights.IMAGENET1K_V1,
                'resnet50': models.ResNet50_Weights.IMAGENET1K_V1,
                'resnet101': models.ResNet101_Weights.IMAGENET1K_V1,
                'resnet152': models.ResNet152_Weights.IMAGENET1K_V1,
            }
            self.resnet = resnet_models[resnet_type](weights=None); load_local_pretrained(self.resnet, resnet_type + ".pth") or self.resnet.load_state_dict(resnet_models[resnet_type](weights=weights[resnet_type]).state_dict())
        else:
            self.resnet = resnet_models[resnet_type](weights=None)
        
        # 修改全连接层
        in_features = self.resnet.fc.in_features
        
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.resnet(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征（不包含分类层）"""
        # ResNet特征提取流程
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x


def create_resnet_model(
    resnet_type: str = 'resnet50',
    num_classes: int = 7,
    pretrained: bool = True,
    dropout_rate: float = 0.3
) -> FER_ResNet:
    """
    创建ResNet表情识别模型
    
    Args:
        resnet_type: ResNet类型
        num_classes: 分类类别数
        pretrained: 是否使用预训练权重
        dropout_rate: Dropout比例
    
    Returns:
        FER_ResNet模型
    """
    return FER_ResNet(
        num_classes=num_classes,
        resnet_type=resnet_type,
        pretrained=pretrained,
        dropout_rate=dropout_rate
    )


# ResNet模型配置
RESNET_CONFIG = {
    'resnet18': {
        'params': '~11M',
        'input_size': 224,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
    },
    'resnet34': {
        'params': '~21M',
        'input_size': 224,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
    },
    'resnet50': {
        'params': '~25M',
        'input_size': 224,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
    },
    'resnet101': {
        'params': '~44M',
        'input_size': 224,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
    },
    'resnet152': {
        'params': '~60M',
        'input_size': 224,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
    }
}