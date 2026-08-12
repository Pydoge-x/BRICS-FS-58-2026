"""
预训练模型下载脚本
将预训练权重下载到项目目录的 pretrained-models 文件夹中
"""

import os
import sys

def download_pretrained_models():
    """下载所有模型的预训练权重"""
    import torch
    from torchvision import models
    
    # 创建预训练模型目录
    pretrained_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pretrained-models')
    os.makedirs(pretrained_dir, exist_ok=True)
    
    print("=" * 60)
    print("下载预训练模型权重")
    print(f"保存目录: {pretrained_dir}")
    print("=" * 60)
    
    # VGG-16
    print("\n1. 下载 VGG-16...")
    try:
        vgg_model = models.vgg16(pretrained=True)
        vgg_path = os.path.join(pretrained_dir, 'vgg16.pth')
        torch.save(vgg_model.state_dict(), vgg_path)
        print(f"   ✓ VGG-16 已保存: {vgg_path}")
    except Exception as e:
        print(f"   ✗ VGG-16 下载失败: {e}")
    
    # ResNet-50
    print("\n2. 下载 ResNet-50...")
    try:
        resnet_model = models.resnet50(pretrained=True)
        resnet_path = os.path.join(pretrained_dir, 'resnet50.pth')
        torch.save(resnet_model.state_dict(), resnet_path)
        print(f"   ✓ ResNet-50 已保存: {resnet_path}")
    except Exception as e:
        print(f"   ✗ ResNet-50 下载失败: {e}")
    
    # MobileNetV3 Small
    print("\n3. 下载 MobileNetV3 Small...")
    try:
        mobilenet_model = models.mobilenet_v3_small(pretrained=True)
        mobilenet_path = os.path.join(pretrained_dir, 'mobilenetv3_small.pth')
        torch.save(mobilenet_model.state_dict(), mobilenet_path)
        print(f"   ✓ MobileNetV3 Small 已保存: {mobilenet_path}")
    except Exception as e:
        print(f"   ✗ MobileNetV3 Small 下载失败: {e}")
    
    # MobileViT 需要从 Hugging Face 下载
    print("\n4. 下载 MobileViT-XS...")
    try:
        from transformers import MobileViTForImageClassification
        
        mobilevit_model = MobileViTForImageClassification.from_pretrained("apple/mobilevit-xs")
        mobilevit_path = os.path.join(pretrained_dir, 'mobilevit_xs.pth')
        torch.save(mobilevit_model.state_dict(), mobilevit_path)
        print(f"   ✓ MobileViT-XS 已保存: {mobilevit_path}")
    except Exception as e:
        print(f"   ✗ MobileViT-XS 下载失败: {e}")
    
    print("\n" + "=" * 60)
    print("预训练模型下载完成！")
    print("=" * 60)


def setup_pretrained_path():
    """设置预训练模型路径环境变量"""
    pretrained_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pretrained-models')
    os.environ['FER_PRETRAINED_DIR'] = pretrained_dir
    print(f"已设置预训练模型目录: {pretrained_dir}")


if __name__ == '__main__':
    download_pretrained_models()
    setup_pretrained_path()
