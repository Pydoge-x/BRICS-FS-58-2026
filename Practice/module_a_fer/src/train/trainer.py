"""
表情识别统一训练器
支持VGG、ResNet、MobileNet、MobileViT四种模型
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import time
import json
import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, asdict

from src.models.factory import create_fer_model, save_model, print_model_info, count_parameters


@dataclass
class TrainingConfig:
    """训练配置"""
    # 模型配置
    model_type: str = 'mobilevit'
    model_name: str = 'mobilevit_xs'
    num_classes: int = 7
    pretrained: bool = True
    dropout_rate: float = 0.2
    
    # 训练配置
    epochs: int = 100
    batch_size: int = 64
    num_workers: int = 4
    
    # 优化器配置
    optimizer_type: str = 'AdamW'
    lr: float = 3e-4
    weight_decay: float = 1e-5
    momentum: float = 0.9  # SGD使用
    
    # 学习率调度
    scheduler_type: str = 'CosineAnnealingWarmRestarts'
    warmup_epochs: int = 5
    T_0: int = 20  # CosineAnnealingWarmRestarts
    T_mult: int = 2
    eta_min: float = 1e-6
    
    # 损失函数
    label_smoothing: float = 0.1
    use_focal_loss: bool = False
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    
    # 正则化
    grad_clip: float = 1.0
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    
    # 早停
    early_stop_patience: int = 0  # 0=禁用, N=val_acc连续N个epoch不提升即停止
    
    # 其他
    seed: int = 42
    device: str = 'cuda'
    use_amp: bool = True  # 混合精度训练
    save_dir: str = 'checkpoints'
    log_interval: int = 10
    save_interval: int = 10  # 定期保存间隔，避免大模型频繁写盘
    keep_checkpoint_max: int = 5  # 最多保留定期checkpoint数量，超出自动删除旧的


class FERTrainer:
    """
    表情识别统一训练器
    """
    
    # 表情类别
    EXPRESSION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    def __init__(
        self,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None
    ):
        """
        初始化训练器
        
        Args:
            config: 训练配置
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器
        """
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # 设置随机种子
        self._set_seed(config.seed)
        
        # 设置设备
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 创建模型
        self.model = create_fer_model(
            model_type=config.model_type,
            model_name=config.model_name,
            num_classes=config.num_classes,
            pretrained=config.pretrained,
            dropout_rate=config.dropout_rate
        )
        self.model.to(self.device)
        
        # 打印模型信息
        print_model_info(self.model, config.model_type, config.model_name)
        
        # 创建损失函数
        self.criterion = self._create_criterion()
        
        # 创建优化器
        self.optimizer = self._create_optimizer()
        
        # 创建学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 混合精度训练
        self.scaler = GradScaler() if config.use_amp and self.device.type == 'cuda' else None
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        # 最佳模型
        self.best_acc = 0.0
        self.best_epoch = 0
        
        # 早停计数器
        self.early_stop_counter = 0
        
        # 创建保存目录
        os.makedirs(config.save_dir, exist_ok=True)
        
        # 保存配置
        config_path = os.path.join(config.save_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(asdict(config), f, indent=2)
    
    def _set_seed(self, seed: int):
        """设置随机种子"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
    
    def _create_criterion(self) -> nn.Module:
        """
        创建损失函数
        
        TODO: 考生需要实现此方法 (考点11/15: 损失函数创建)
        
        要求：
        1. 如果 self.config.use_focal_loss 为 True：
           from .losses import FocalLoss
           return FocalLoss(alpha=..., gamma=..., num_classes=...)
           参数从 self.config 获取
        2. 否则：
           return nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        """
        # TODO: 实现损失函数创建
        raise NotImplementedError("TODO: 请实现 FERTrainer._create_criterion() 方法")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """
        创建优化器
        
        TODO: 考生需要实现此方法 (考点12/15: 优化器创建)
        
        要求：根据 self.config.optimizer_type 创建对应优化器
        - 'Adam': optim.Adam(params=self.model.parameters(), lr=..., weight_decay=...)
        - 'AdamW': optim.AdamW(params=self.model.parameters(), lr=..., weight_decay=...)
        - 'SGD': optim.SGD(params=self.model.parameters(), lr=..., momentum=..., weight_decay=...)
        - 其他: raise ValueError
        所有参数从 self.config 获取
        """
        # TODO: 实现优化器创建
        raise NotImplementedError("TODO: 请实现 FERTrainer._create_optimizer() 方法")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """
        创建学习率调度器
        
        TODO: 考生需要实现此方法 (考点13/15: 学习率调度器创建)
        
        要求：根据 self.config.scheduler_type 创建对应调度器
        - 'StepLR': optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)
        - 'CosineAnnealingLR': optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.epochs, eta_min=self.config.eta_min)
        - 'CosineAnnealingWarmRestarts': optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=self.config.T_0, T_mult=self.config.T_mult, eta_min=self.config.eta_min)
        - 'LinearWarmup' 或 其他: return None
        """
        # TODO: 实现学习率调度器创建
        raise NotImplementedError("TODO: 请实现 FERTrainer._create_scheduler() 方法")
    
    def _warmup_scheduler(self, epoch: int):
        """Warmup学习率调整"""
        if epoch < self.config.warmup_epochs:
            warmup_lr = self.config.lr * (epoch + 1) / self.config.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = warmup_lr
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        训练一个epoch
        
        TODO: 考生需要实现此方法 (考点9/10: 训练循环)
        
        提示：
        1. 设置 self.model.train() 进入训练模式
        2. 遍历 self.train_loader 获取每个batch的数据
        3. 数据解析：train_loader 返回 (images, labels) 或 (images, labels, _)
        4. 将 images 和 labels 移动到 self.device
        5. 调用 self.optimizer.zero_grad() 清零梯度
        6. 调用 self.model(images) 进行前向传播得到 outputs
        7. 调用 self.criterion(outputs, labels) 计算损失 loss
        8. 调用 loss.backward() 反向传播
        9. 可选：梯度裁剪（self.config.grad_clip > 0 时）
        10. 调用 self.optimizer.step() 更新参数
        11. 统计累计损失和准确率
        12. 返回 {'loss': 平均损失, 'acc': 平均准确率(%)}

        参考框架：
        ```python
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        for data in self.train_loader:
            # 解析数据
            if isinstance(data, tuple) and len(data) == 3:
                images, labels, _ = data
            else:
                images, labels = data
            images, labels = images.to(self.device), labels.to(self.device)
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            # 反向传播
            loss.backward()
            self.optimizer.step()
            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        return {'loss': total_loss/len(self.train_loader), 'acc': 100.*correct/total}
        ```
        """
        # TODO: 实现训练一个epoch的逻辑
        raise NotImplementedError("TODO: 请实现 FERTrainer.train_epoch() 方法")
    
    def _mixup(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple:
        """Mixup数据增强"""
        alpha = self.config.mixup_alpha
        lam = np.random.beta(alpha, alpha)
        
        batch_size = images.size(0)
        index = torch.randperm(batch_size).to(self.device)
        
        mixed_images = lam * images + (1 - lam) * images[index]
        labels_a, labels_b = labels, labels[index]
        
        return mixed_images, (labels_a, labels_b, lam)
    
    def validate(self, loader: DataLoader, desc: str = 'Val') -> Dict[str, float]:
        """
        验证模型
        
        TODO: 考生需要实现此方法 (考点10/10: 验证逻辑)
        
        提示：
        1. 设置 self.model.eval() 进入评估模式
        2. 使用 torch.no_grad() 关闭梯度计算
        3. 遍历 loader 获取每个batch的数据
        4. 数据解析同 train_epoch
        5. 调用 self.model(images) 进行前向传播得到 outputs
        6. 调用 self.criterion(outputs, labels) 计算损失
        7. 统计累计损失和准确率
        8. 可选：统计各类别准确率（使用 class_correct, class_total）
        9. 返回 {'loss': 平均损失, 'acc': 平均准确率(%), 'class_acc': 各类别准确率数组}

        参考框架：
        ```python
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for data in loader:
                if isinstance(data, tuple) and len(data) == 3:
                    images, labels, _ = data
                else:
                    images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        return {'loss': total_loss/len(loader), 'acc': 100.*correct/total, 'class_acc': None}
        ```
        """
        # TODO: 实现验证逻辑
        raise NotImplementedError("TODO: 请实现 FERTrainer.validate() 方法")
    
    def train(self) -> float:
        """
        完整训练流程
        
        TODO: 考生需要实现此方法 (考点14/15: 完整训练流程编排)
        
        要求：编排完整的训练过程，每个 epoch 执行以下步骤：
        1. 打印训练开始信息
        2. for epoch in range(1, self.config.epochs + 1):
           a. Warmup: 如果 epoch <= self.config.warmup_epochs，调用 self._warmup_scheduler(epoch)
           b. 训练: train_metrics = self.train_epoch(epoch)
           c. 验证: 如果 self.val_loader 不为 None，val_metrics = self.validate(self.val_loader)
              否则 val_metrics = {'loss': 0, 'acc': 0, 'class_acc': np.zeros(...)}
           d. 更新学习率: 如果 scheduler 存在且超过 warmup 阶段，调用 self.scheduler.step()
           e. 记录历史到 self.history（train_loss, train_acc, val_loss, val_acc, lr）
           f. 打印当前 epoch 结果
           g. 保存最佳模型: 如果 val_acc > self.best_acc，更新 best_acc/best_epoch，
              调用 save_model(...) 保存到 'best_model.pth'（include_optimizer=False）
              否则 self.early_stop_counter += 1
           h. 早停检查: 如果 early_stop_counter >= patience > 0，break
           i. 定期保存: epoch % save_interval == 0 时保存 checkpoint，调用 self._cleanup_old_checkpoints()
        3. 打印训练完成信息
        4. 保存训练历史到 'history.json'
        5. 返回 self.best_acc

        关键提示：
        - save_model 签名: save_model(model, path, model_type, model_name, optimizer=None, epoch=0, best_acc=0.0, include_optimizer=True, history=None)
        - 最佳模型路径: os.path.join(self.config.save_dir, 'best_model.pth')
        - Checkpoint 路径: os.path.join(self.config.save_dir, f'checkpoint_epoch_{epoch}.pth')
        - epoch 从 1 开始
        """
        # TODO: 实现完整训练流程
        raise NotImplementedError("TODO: 请实现 FERTrainer.train() 方法")
    
    def _cleanup_old_checkpoints(self):
        """自动清理旧的定期checkpoint，只保留最近 keep_checkpoint_max 个"""
        if self.config.keep_checkpoint_max <= 0:
            return  # 0 或负数 = 不限制
        
        import glob
        pattern = os.path.join(self.config.save_dir, 'checkpoint_epoch_*.pth')
        files = glob.glob(pattern)
        
        if len(files) <= self.config.keep_checkpoint_max:
            return
        
        # 按 epoch 号排序（从文件名提取数字）
        def extract_epoch(filepath):
            import re
            match = re.search(r'checkpoint_epoch_(\d+)\.pth', filepath)
            return int(match.group(1)) if match else 0
        
        files.sort(key=extract_epoch)
        
        # 删除最旧的，保留最近 keep_checkpoint_max 个
        to_delete = files[:-self.config.keep_checkpoint_max]
        for f in to_delete:
            try:
                os.remove(f)
                print(f"  [CLEANUP] 已删除旧checkpoint: {os.path.basename(f)}")
            except OSError as e:
                print(f"  [WARN] 删除失败: {os.path.basename(f)} - {e}")
    
    def test(self) -> Dict[str, float]:
        """
        测试模型
        
        TODO: 考生需要实现此方法 (考点15/15: 测试流程)
        
        要求：
        1. 如果 self.test_loader 为 None，打印提示并返回 {}
        2. 从 os.path.join(self.config.save_dir, 'best_model.pth') 加载最佳模型：
           checkpoint = torch.load(best_path, map_location=self.device)
           self.model.load_state_dict(checkpoint['model_state_dict'])
        3. 调用 self.validate(self.test_loader, 'Test') 获取测试指标
        4. 打印测试结果（Loss, Accuracy, 各类别准确率）
           - 类别标签在 self.EXPRESSION_LABELS 中
        5. 返回 test_metrics
        """
        # TODO: 实现测试流程
        raise NotImplementedError("TODO: 请实现 FERTrainer.test() 方法")