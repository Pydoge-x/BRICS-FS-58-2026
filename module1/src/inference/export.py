"""
ONNX模型导出模块
"""

import torch
import torch.onnx
import torch.nn.functional as F
import os
import math
from typing import Tuple, Optional
import numpy as np

from src.models.factory import create_fer_model, load_model


def _manual_scaled_dot_product_attention(query, key, value, attn_mask=None,
                                          dropout_p=0.0, is_causal=False, scale=None):
    """
    ONNX 兼容的手动注意力计算，用于替换 F.scaled_dot_product_attention。
    PyTorch 2.0 的 SDPA 在 ONNX tracing 中不受支持，需要分解为基础运算。
    """
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    if is_causal:
        mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(0)
        attn_weight = attn_weight.masked_fill(~mask, float('-inf'))
    if attn_mask is not None:
        attn_weight = attn_weight + attn_mask
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = F.dropout(attn_weight, dropout_p, training=False)
    return attn_weight @ value


def _sdpa_compatible_export(model, dummy_input, output_path, opset_version, dynamic_axes):
    """
    尝试导出，如果遇到 scaled_dot_product_attention 不支持，
    则用 monkey-patch 手动注意力后重试。
    """
    try:
        torch.onnx.export(
            model, dummy_input, output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes
        )
    except (torch.onnx.errors.UnsupportedOperatorError, Exception) as e:
        if 'scaled_dot_product_attention' in str(e):
            print("检测到 scaled_dot_product_attention，切换为手动注意力...")
            original = F.scaled_dot_product_attention
            F.scaled_dot_product_attention = _manual_scaled_dot_product_attention
            try:
                torch.onnx.export(
                    model, dummy_input, output_path,
                    export_params=True,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes=dynamic_axes
                )
            finally:
                F.scaled_dot_product_attention = original
        else:
            raise


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
    opset_version: int = 14,
    simplify: bool = True,
    dynamic_batch: bool = True
) -> str:
    """
    导出PyTorch模型为ONNX格式
    
    Args:
        model: PyTorch模型
        output_path: 输出路径
        input_shape: 输入形状
        opset_version: ONNX opset版本
        simplify: 是否简化模型
        dynamic_batch: 是否支持动态batch
    
    Returns:
        ONNX模型路径
    """
    model.eval()
    
    # 创建dummy输入
    dummy_input = torch.randn(input_shape)
    
    # 动态batch配置
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    
    # 导出ONNX（兼容 SDPA）
    _sdpa_compatible_export(
        model, dummy_input, output_path,
        opset_version=opset_version,
        dynamic_axes=dynamic_axes
    )
    
    print(f"模型已导出至: {output_path}")
    
    # 简化ONNX模型（可选）
    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify
            
            # 加载模型
            onnx_model = onnx.load(output_path)
            
            # 简化
            simplified_model, check = onnx_simplify(onnx_model)
            
            if check:
                onnx.save(simplified_model, output_path)
                print("模型已简化")
            
        except ImportError:
            print("提示: 安装onnxsim可简化模型 (pip install onnxsim)")
    
    # 验证模型
    verify_onnx_model(output_path, model, dummy_input)
    
    return output_path


def verify_onnx_model(
    onnx_path: str,
    pytorch_model: torch.nn.Module,
    test_input: torch.Tensor
) -> bool:
    """
    验证ONNX模型
    
    Args:
        onnx_path: ONNX模型路径
        pytorch_model: PyTorch模型
        test_input: 测试输入
    
    Returns:
        是否验证成功
    """
    import onnxruntime as ort
    
    # PyTorch推理
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input).numpy()
    
    # ONNX推理
    session = ort.InferenceSession(onnx_path)
    onnx_output = session.run(None, {'input': test_input.numpy()})[0]
    
    # 比较输出
    diff = np.abs(pytorch_output - onnx_output).max()
    
    if diff < 1e-5:
        print(f"验证成功! 最大差异: {diff:.6f}")
        return True
    else:
        print(f"验证失败! 最大差异: {diff:.6f}")
        return False


def export_checkpoint_to_onnx(
    checkpoint_path: str,
    output_path: str,
    input_size: int = 224,
    opset_version: int = 14
) -> str:
    """
    从检查点导出ONNX模型
    
    Args:
        checkpoint_path: 检查点路径
        output_path: 输出路径
        input_size: 输入尺寸
        opset_version: ONNX opset版本
    
    Returns:
        ONNX模型路径
    """
    # 加载模型
    model = load_model(checkpoint_path)
    
    # 导出
    return export_to_onnx(
        model,
        output_path,
        input_shape=(1, 3, input_size, input_size),
        opset_version=opset_version
    )


def export_all_models(
    checkpoint_dir: str,
    output_dir: str,
    input_size: int = 224
):
    """
    导出所有检查点为ONNX
    
    Args:
        checkpoint_dir: 检查点目录
        output_dir: 输出目录
        input_size: 输入尺寸
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有检查点
    checkpoints = [
        ('best_model.pth', 'fer_best.onnx'),
        ('vgg_best.pth', 'fer_vgg.onnx'),
        ('resnet_best.pth', 'fer_resnet.onnx'),
        ('mobilenet_best.pth', 'fer_mobilenet.onnx'),
        ('mobilevit_best.pth', 'fer_mobilevit.onnx'),
    ]
    
    for checkpoint_name, onnx_name in checkpoints:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        output_path = os.path.join(output_dir, onnx_name)
        
        if os.path.exists(checkpoint_path):
            print(f"\n导出: {checkpoint_name}")
            export_checkpoint_to_onnx(checkpoint_path, output_path, input_size)
        else:
            print(f"跳过: {checkpoint_name} (不存在)")