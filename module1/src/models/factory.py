"""
模型工厂函数
统一创建和管理所有模型
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Union

from .vgg import FER_VGG, create_vgg_model, VGG_CONFIG
from .resnet import FER_ResNet, create_resnet_model, RESNET_CONFIG
from .mobilenet import FER_MobileNet, create_mobilenet_model, MOBILENET_CONFIG
from .mobilevit import FER_MobileViT, create_mobilevit_model, MOBILEVIT_CONFIG


# 所有模型配置汇总
MODEL_CONFIGS = {
    'vgg': VGG_CONFIG,
    'resnet': RESNET_CONFIG,
    'mobilenet': MOBILENET_CONFIG,
    'mobilevit': MOBILEVIT_CONFIG
}


def create_fer_model(
    model_type: str,
    model_name: str = None,
    num_classes: int = 7,
    pretrained: bool = True,
    dropout_rate: float = 0.2,
    **kwargs
) -> nn.Module:
    """
    表情识别模型工厂函数
    
    Args:
        model_type: 模型类型 ('vgg', 'resnet', 'mobilenet', 'mobilevit')
        model_name: 具体模型名称 (如 'vgg16', 'resnet50', 'mobilenetv3_small', 'mobilevit_xs')
        num_classes: 分类类别数
        pretrained: 是否使用预训练权重
        dropout_rate: Dropout比例
        **kwargs: 其他参数
    
    Returns:
        PyTorch模型
    """
    
    # 默认模型名称
    default_names = {
        'vgg': 'vgg16',
        'resnet': 'resnet50',
        'mobilenet': 'mobilenetv3_small',
        'mobilevit': 'mobilevit_xs'
    }
    
    if model_name is None:
        model_name = default_names.get(model_type)
    
    # 创建模型
    if model_type == 'vgg':
        return create_vgg_model(
            vgg_type=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate
        )
    
    elif model_type == 'resnet':
        return create_resnet_model(
            resnet_type=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate
        )
    
    elif model_type == 'mobilenet':
        return create_mobilenet_model(
            mobilenet_type=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate
        )
    
    elif model_type == 'mobilevit':
        return create_mobilevit_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate
        )
    
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


def get_model_config(model_type: str, model_name: str = None) -> Dict:
    """
    获取模型配置
    
    Args:
        model_type: 模型类型
        model_name: 具体模型名称
    
    Returns:
        模型配置字典
    """
    configs = MODEL_CONFIGS.get(model_type)
    
    if configs is None:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    if model_name is None:
        # 返回默认配置
        default_names = {
            'vgg': 'vgg16',
            'resnet': 'resnet50',
            'mobilenet': 'mobilenetv3_small',
            'mobilevit': 'mobilevit_xs'
        }
        model_name = default_names.get(model_type)
    
    return configs.get(model_name, configs.get(list(configs.keys())[0]))


def get_available_models() -> Dict[str, list]:
    """
    获取所有可用模型列表
    
    Returns:
        模型类型和名称字典
    """
    return {
        'vgg': list(VGG_CONFIG.keys()),
        'resnet': list(RESNET_CONFIG.keys()),
        'mobilenet': list(MOBILENET_CONFIG.keys()),
        'mobilevit': list(MOBILEVIT_CONFIG.keys())
    }


def count_parameters(model: nn.Module) -> int:
    """
    计算模型参数量
    
    Args:
        model: PyTorch模型
    
    Returns:
        参数数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_model(checkpoint_path: str, model_type: str = None, model_name: str = None) -> nn.Module:
    """
    从检查点加载模型
    
    Args:
        checkpoint_path: 检查点文件路径
        model_type: 模型类型（如果检查点中未保存）
        model_name: 模型名称
    
    Returns:
        加载的模型
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 从检查点获取模型信息
    if 'model_type' in checkpoint:
        model_type = checkpoint['model_type']
    if 'model_name' in checkpoint:
        model_name = checkpoint['model_name']
    
    # 创建模型
    model = create_fer_model(
        model_type=model_type,
        model_name=model_name,
        pretrained=False
    )
    
    # 加载权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model


def save_model(
    model: nn.Module,
    save_path: str,
    model_type: str,
    model_name: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    best_acc: float = 0.0,
    include_optimizer: bool = False,
    **kwargs
):
    """
    保存模型检查点
    
    Args:
        model: 模型
        save_path: 保存路径
        model_type: 模型类型
        model_name: 模型名称
        optimizer: 优化器（仅在 include_optimizer=True 时保存其状态）
        epoch: 当前epoch
        best_acc: 最佳准确率
        include_optimizer: 是否保存优化器状态（用于恢复训练）
        **kwargs: 其他信息
    """
    checkpoint = {
        'model_type': model_type,
        'model_name': model_name,
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'best_acc': best_acc,
        **kwargs
    }
    
    if include_optimizer and optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    torch.save(checkpoint, save_path)


# 模型信息打印
def print_model_info(model: nn.Module, model_type: str, model_name: str):
    """
    打印模型信息
    
    Args:
        model: 模型
        model_type: 模型类型
        model_name: 模型名称
    """
    params = count_parameters(model)
    config = get_model_config(model_type, model_name)
    
    print(f"\n模型信息:")
    print(f"  类型: {model_type}")
    print(f"  名称: {model_name}")
    print(f"  参数量: {params / 1e6:.2f}M")
    print(f"  输入尺寸: {config['input_size']}x{config['input_size']}")
    print(f"  预处理: mean={config['mean']}, std={config['std']}")
