"""
训练脚本
支持单数据集和多数据集训练
支持通过配置文件进行训练
"""

import os
import sys
import argparse
import torch

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import FERDataset, MultiDataset, DatasetConfig, create_dataloader
from src.data.fer2013 import FER2013Dataset
from src.data.rafdb import RAFDBDataset
from src.data.oulucasia import OuluCASIADataset
from src.train.trainer import FERTrainer, TrainingConfig
from src.models.factory import create_fer_model
from configs.config import get_config, Config


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='表情识别模型训练')
    
    # 配置文件参数（优先使用）
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径（.json文件）')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, default='mobilevit',
                        choices=['vgg', 'resnet', 'mobilenet', 'mobilevit'],
                        help='模型类型')
    parser.add_argument('--model_name', type=str, default='mobilevit_xs',
                        help='具体模型名称')
    parser.add_argument('--num_classes', type=int, default=7,
                        help='类别数')
    
    # 数据集参数
    parser.add_argument('--dataset', type=str, default='fer2013',
                        choices=['fer2013', 'rafdb', 'oulucasia', 'multi', 
                                'fer_affectnet', 'fer_ckplus_kdef', 'large_fer'],
                        help='数据集名称')
    parser.add_argument('--data_dir', type=str, default='dataset_data',
                        help='数据集目录')
    parser.add_argument('--multi_datasets', type=str, nargs='+',
                        default=['fer2013', 'rafdb'],
                        help='多数据集训练时的数据集列表')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批大小')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='学习率')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--optimizer', type=str, default='AdamW',
                        choices=['Adam', 'AdamW', 'SGD'],
                        help='优化器类型')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='权重衰减')
    
    # 其他参数
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='是否使用预训练权重')
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='是否使用混合精度训练')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='模型保存目录')
    
    return parser.parse_args()


def load_config(args):
    """加载配置文件"""
    if args.config and os.path.exists(args.config):
        print(f"加载配置文件: {args.config}")
        config = Config.load(args.config)
        
        # 用命令行参数覆盖配置文件（仅在显式指定时）
        # 我们需要区分用户显式指定的参数和默认值
        # 检查原始命令行中是否有对应的参数
        import sys
        provided_args = set()
        for i in range(1, len(sys.argv)):
            arg = sys.argv[i]
            if arg.startswith('--'):
                provided_args.add(arg[2:])
        
        if 'model_type' in provided_args:
            config.model.type = args.model_type
        if 'model_name' in provided_args:
            config.model.name = args.model_name
        if 'dataset' in provided_args:
            config.dataset.name = args.dataset
        if 'epochs' in provided_args:
            config.training.epochs = args.epochs
        if 'batch_size' in provided_args:
            config.training.batch_size = args.batch_size
        if 'lr' in provided_args:
            config.training.lr = args.lr
        if 'num_workers' in provided_args:
            config.training.num_workers = args.num_workers
        if 'save_dir' in provided_args:
            config.training.save_dir = args.save_dir
        
        return config
    else:
        # 使用命令行参数创建配置
        return Config(
            dataset=DatasetConfig(
                name=args.dataset,
                root_dir=os.path.join(args.data_dir, args.dataset),
                split='train',
                image_size=224,
                use_augmentation=True
            ),
            model=ModelConfig(
                type=args.model_type,
                name=args.model_name,
                num_classes=args.num_classes,
                pretrained=args.pretrained,
                dropout_rate=0.2
            ),
            training=TrainingConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                num_workers=args.num_workers,
                optimizer=args.optimizer,
                weight_decay=args.weight_decay,
                use_amp=args.use_amp,
                seed=args.seed,
                save_dir=args.save_dir
            )
        )


def create_dataset(config):
    """根据配置创建数据集"""
    dataset_name = config.dataset.name
    root_dir = config.dataset.root_dir
    image_size = config.dataset.image_size
    use_augmentation = config.dataset.use_augmentation
    
    # 数据集目录映射（Kaggle下载后的实际目录名）
    dataset_dir_map = {
        'fer2013': 'fer2013',
        'rafdb': 'RAF-DB',
        'oulucasia': 'Oulu-CASIA',
        'fer_affectnet': 'fer_affectnet',
        'fer_ckplus_kdef': 'fer_ckplus_kdef',
        'large_fer': 'large_fer',
    }
    
    # 解析root_dir，如果是相对路径则转换为绝对路径
    if not os.path.isabs(root_dir):
        root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), root_dir)
    
    if dataset_name == 'multi':
        # 多数据集训练
        dataset_configs = []
        for name in ['fer2013', 'rafdb']:  # 默认使用fer2013和rafdb
            dir_name = dataset_dir_map.get(name, name)
            ds_root_dir = os.path.join(os.path.dirname(root_dir), dir_name)
            if os.path.exists(ds_root_dir):
                ds_config = DatasetConfig(
                    name=name,
                    root_dir=ds_root_dir,
                    split='train',
                    image_size=image_size,
                    use_augmentation=use_augmentation
                )
                dataset_configs.append(ds_config)
            else:
                print(f"警告: 数据集目录不存在: {ds_root_dir}")
        
        if not dataset_configs:
            raise ValueError("没有找到任何数据集")
        
        train_dataset = MultiDataset(dataset_configs, balance=False)
        
        # 验证集（使用第一个数据集）
        val_configs = [DatasetConfig(
            name=dataset_configs[0].name,
            root_dir=dataset_configs[0].root_dir,
            split='val',
            image_size=image_size,
            use_augmentation=False
        )]
        val_dataset = MultiDataset(val_configs)
        
    elif dataset_name == 'fer2013':
        # FER-2013数据集 (Kaggle解压格式: train/angry/*.jpg, test/angry/*.jpg)
        train_dataset = FER2013Dataset(
            root_dir=root_dir,
            split='train',
            image_size=image_size,
            use_augmentation=use_augmentation
        )
        val_dataset = FER2013Dataset(
            root_dir=root_dir,
            split='test',
            image_size=image_size,
            use_augmentation=False
        )
        
    elif dataset_name == 'rafdb':
        # RAF-DB数据集 (Kaggle格式: DATASET/train/1/*.jpg)
        train_dataset = RAFDBDataset(
            root_dir=root_dir,
            split='train',
            image_size=image_size,
            use_augmentation=use_augmentation
        )
        val_dataset = RAFDBDataset(
            root_dir=root_dir,
            split='test',
            image_size=image_size,
            use_augmentation=False
        )
        
    elif dataset_name == 'oulucasia':
        # Oulu-CASIA数据集 (Kaggle格式: Oulu_CASIA_NIR_VIS/NI/Dark/P001/Anger/)
        train_dataset = OuluCASIADataset(
            root_dir=root_dir,
            split='train',
            image_size=image_size,
            use_augmentation=use_augmentation,
            modality='both',
            illumination='all'
        )
        val_dataset = OuluCASIADataset(
            root_dir=root_dir,
            split='test',
            image_size=image_size,
            use_augmentation=False,
            modality='both',
            illumination='all'
        )
    
    elif dataset_name in ['fer_affectnet', 'fer_ckplus_kdef', 'large_fer']:
        # 使用FER2013Dataset加载其他通用格式数据集
        train_dataset = FER2013Dataset(
            root_dir=root_dir,
            split='train',
            image_size=image_size,
            use_augmentation=use_augmentation
        )
        val_dataset = FER2013Dataset(
            root_dir=root_dir,
            split='test',
            image_size=image_size,
            use_augmentation=False
        )
    
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    return train_dataset, val_dataset


def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置
    config = load_config(args)
    
    print("=" * 60)
    print("表情识别模型训练")
    print("=" * 60)
    print(f"配置文件: {args.config if args.config else '命令行参数'}")
    print(f"模型: {config.model.type}/{config.model.name}")
    print(f"数据集: {config.dataset.name}")
    print(f"数据集目录: {config.dataset.root_dir}")
    print("-" * 60)
    print(f"训练轮数: {config.training.epochs}")
    print(f"批大小: {config.training.batch_size}")
    print(f"学习率: {config.training.lr}")
    print(f"优化器: {config.training.optimizer}")
    print(f"权重衰减: {config.training.weight_decay}")
    print(f"混合精度: {'开启' if config.training.use_amp else '关闭'}")
    print(f"预训练权重: {'使用' if config.model.pretrained else '不使用'}")
    print(f"模型保存目录: {config.training.save_dir}")
    print("=" * 60)
    
    # 创建数据集
    print("\n加载数据集...")
    train_dataset, val_dataset = create_dataset(config)
    
    # 创建数据加载器
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers
    )
    
    # 创建训练配置
    training_config = TrainingConfig(
        model_type=config.model.type,
        model_name=config.model.name,
        epochs=config.training.epochs,
        batch_size=config.training.batch_size,
        lr=config.training.lr,
        optimizer_type=config.training.optimizer,
        weight_decay=config.training.weight_decay,
        scheduler_type=config.training.scheduler,
        warmup_epochs=config.training.warmup_epochs,
        T_0=config.training.T_0,
        T_mult=config.training.T_mult,
        eta_min=config.training.eta_min,
        label_smoothing=config.training.label_smoothing,
        use_focal_loss=config.training.use_focal_loss,
        mixup_alpha=getattr(config.training, 'mixup_alpha', 0.0),
        early_stop_patience=getattr(config.training, 'early_stop_patience', 0),
        pretrained=config.model.pretrained,
        use_amp=config.training.use_amp,
        seed=config.training.seed,
        grad_clip=config.training.grad_clip,
        save_dir=config.training.save_dir,
        save_interval=config.training.save_interval,
        keep_checkpoint_max=getattr(config.training, 'keep_checkpoint_max', 5)
    )
    
    # 创建训练器
    trainer = FERTrainer(
        config=training_config,
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    # 开始训练
    best_acc = trainer.train()
    
    # 测试
    print("\n测试模型...")
    test_metrics = trainer.test()
    
    print("\n训练完成!")
    print(f"最佳准确率: {best_acc:.2f}%")
    
    return best_acc


if __name__ == '__main__':
    main()
