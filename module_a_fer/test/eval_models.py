"""
评估所有模型在测试集上的表现
计算 Accuracy, F1 Score (Macro/Weighted), Confusion Matrix
"""

import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.factory import create_fer_model
from src.data.dataset import FERDataset, DatasetConfig

EXPRESSION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

MODEL_CONFIGS = [
    {
        'name': 'VGG-16',
        'type': 'vgg',
        'model_name': 'vgg16',
        'checkpoint': 'checkpoints/vgg_fer2013/20260613_2334/best_model.pth',
    },
    {
        'name': 'ResNet-50',
        'type': 'resnet',
        'model_name': 'resnet50',
        'checkpoint': 'checkpoints/resnet_fer2013/20260613_2334/best_model.pth',
    },
    {
        'name': 'MobileNet-V3',
        'type': 'mobilenet',
        'model_name': 'mobilenetv3_small',
        'checkpoint': 'checkpoints/mobilenet_fer2013/20260613_2334/best_model.pth',
    },
    {
        'name': 'MobileViT-XS',
        'type': 'mobilevit',
        'model_name': 'mobilevit_xs',
        'checkpoint': 'checkpoints/mobilevit_fer2013/20260613_2334/best_model.pth',
    },
]


def load_model(checkpoint_path, model_type, model_name, device):
    """加载 best_model.pth"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = create_fer_model(
        model_type=checkpoint.get('model_type', model_type),
        model_name=checkpoint.get('model_name', model_name),
        num_classes=7,
        pretrained=False
    )
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def create_test_loader():
    """创建测试集 DataLoader"""
    config = DatasetConfig(
        name='fer2013',
        root_dir='dataset_data/fer2013',
        split='test',
        image_size=224,
        use_augmentation=False,
        normalize=True,
    )
    dataset = FERDataset(config)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    print(f"测试集: {len(dataset)} 张图像, {len(loader)} 个 batch")
    return loader


def evaluate_model(model, loader, device):
    """评估单个模型，返回预测和标签"""
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    return all_preds, all_labels


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    print("=" * 60)

    loader = create_test_loader()

    all_results = {}

    for cfg in MODEL_CONFIGS:
        print(f"\n{'=' * 60}")
        print(f"评估: {cfg['name']}")
        print(f"{'=' * 60}")

        if not os.path.exists(cfg['checkpoint']):
            print(f"  SKIP: 检查点不存在 - {cfg['checkpoint']}")
            continue

        model = load_model(cfg['checkpoint'], cfg['type'], cfg['model_name'], device)

        # 参数量
        params = sum(p.numel() for p in model.parameters())
        print(f"  参数量: {params / 1e6:.2f}M")

        preds, labels = evaluate_model(model, loader, device)

        # 整体准确率
        acc = accuracy_score(labels, preds)
        print(f"  准确率 (Accuracy): {acc * 100:.2f}%")

        # F1 Score
        f1_macro = f1_score(labels, preds, average='macro')
        f1_weighted = f1_score(labels, preds, average='weighted')
        print(f"  F1 Macro:   {f1_macro:.4f}")
        print(f"  F1 Weighted: {f1_weighted:.4f}")

        # 混淆矩阵
        cm = confusion_matrix(labels, preds)
        print(f"\n  混淆矩阵 (行=真实, 列=预测):")
        header = "         " + " ".join(f"{l:>7}" for l in EXPRESSION_LABELS)
        print(header)
        for i, label in enumerate(EXPRESSION_LABELS):
            row = " ".join(f"{cm[i, j]:>7}" for j in range(7))
            print(f"  {label:>7}: {row}")

        # 各类别准确率
        print(f"\n  各类别准确率:")
        for i, label in enumerate(EXPRESSION_LABELS):
            class_total = cm[i].sum()
            class_correct = cm[i, i]
            class_acc = class_correct / class_total * 100 if class_total > 0 else 0
            print(f"    {label:>9}: {class_acc:5.1f}% ({class_correct}/{class_total})")

        # Classification Report
        print(f"\n  Classification Report:")
        print(classification_report(labels, preds, target_names=EXPRESSION_LABELS, digits=4))

        all_results[cfg['name']] = {
            'accuracy': round(acc * 100, 2),
            'f1_macro': round(f1_macro, 4),
            'f1_weighted': round(f1_weighted, 4),
            'confusion_matrix': cm.tolist(),
            'params_m': round(params / 1e6, 2),
            'per_class_acc': {EXPRESSION_LABELS[i]: round(cm[i, i] / cm[i].sum() * 100, 1) if cm[i].sum() > 0 else 0 for i in range(7)}
        }

    # 汇总
    print(f"\n{'=' * 60}")
    print("汇总对比")
    print(f"{'=' * 60}")
    print(f"{'模型':<16} {'Accuracy':>10} {'F1 Macro':>10} {'F1 Weighted':>12} {'参数(M)':>10}")
    print("-" * 58)
    for name, r in all_results.items():
        print(f"{name:<16} {r['accuracy']:>9.2f}% {r['f1_macro']:>9.4f} {r['f1_weighted']:>11.4f} {r['params_m']:>9.1f}")

    # 各类别对比
    print(f"\n各类别准确率对比 (%):")
    header = f"{'类别':<10}"
    for name in all_results:
        header += f" {name:>14}"
    print(header)
    print("-" * (10 + 15 * len(all_results)))
    for label in EXPRESSION_LABELS:
        row = f"{label:<10}"
        for name in all_results:
            row += f" {all_results[name]['per_class_acc'][label]:>13.1f}%"
        print(row)


if __name__ == '__main__':
    main()
