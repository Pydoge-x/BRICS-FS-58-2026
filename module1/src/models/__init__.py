"""
表情识别模型模块
支持VGG、ResNet、MobileNet、MobileViT四种模型
"""

from .vgg import FER_VGG, create_vgg_model
from .resnet import FER_ResNet, create_resnet_model
from .mobilenet import FER_MobileNet, create_mobilenet_model

# MobileViT depends on timm, conditional import to avoid blocking other models
try:
    from .mobilevit import FER_MobileViT, create_mobilevit_model
except ImportError:
    FER_MobileViT = None
    create_mobilevit_model = None

from .factory import create_fer_model, get_model_config

__all__ = [
    'FER_VGG',
    'FER_ResNet',
    'FER_MobileNet',
    'FER_MobileViT',
    'create_vgg_model',
    'create_resnet_model',
    'create_mobilenet_model',
    'create_mobilevit_model',
    'create_fer_model',
    'get_model_config'
]
