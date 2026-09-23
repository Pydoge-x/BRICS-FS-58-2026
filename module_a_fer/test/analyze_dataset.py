"""
数据集分析脚本
用于分析和检查下载的数据集结构
"""

import os
import sys
import argparse
from collections import defaultdict


def count_files_by_extension(directory):
    """统计目录中不同扩展名的文件数量"""
    ext_counts = defaultdict(int)
    total_files = 0
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            _, ext = os.path.splitext(file)
            ext_counts[ext.lower()] += 1
            total_files += 1
    
    return ext_counts, total_files


def analyze_directory_structure(directory, depth=2, prefix=""):
    """分析目录结构"""
    try:
        entries = sorted(os.listdir(directory))
        dirs = []
        files = []
        
        for entry in entries:
            full_path = os.path.join(directory, entry)
            if os.path.isdir(full_path):
                dirs.append(entry)
            else:
                files.append(entry)
        
        result = []
        
        # 显示当前目录下的文件（仅顶层）
        if prefix == "":
            for file in files[:5]:
                result.append(f"{prefix}├── {file}")
            if len(files) > 5:
                result.append(f"{prefix}└── ... ({len(files) - 5} more files)")
        
        # 递归显示子目录
        for i, dir_name in enumerate(dirs):
            full_path = os.path.join(directory, dir_name)
            is_last = i == len(dirs) - 1
            connector = "└── " if is_last else "├── "
            
            result.append(f"{prefix}{connector}{dir_name}/")
            
            if depth > 1:
                sub_prefix = prefix + ("    " if is_last else "│   ")
                sub_result = analyze_directory_structure(full_path, depth - 1, sub_prefix)
                result.extend(sub_result)
        
        return result
    
    except PermissionError:
        return [f"{prefix}└── [权限受限]"]


def analyze_fer_dataset(directory):
    """分析FER类数据集（按类别组织）"""
    categories = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    category_counts = {}
    
    for cat in categories:
        cat_dir = os.path.join(directory, cat)
        if os.path.exists(cat_dir) and os.path.isdir(cat_dir):
            count = len([f for f in os.listdir(cat_dir) if os.path.isfile(os.path.join(cat_dir, f))])
            category_counts[cat] = count
    
    return category_counts


def analyze_split_dataset(directory, splits=['train', 'test', 'val']):
    """分析按train/test/val划分的数据集"""
    split_info = {}
    
    for split in splits:
        split_dir = os.path.join(directory, split)
        if os.path.exists(split_dir) and os.path.isdir(split_dir):
            cat_counts = analyze_fer_dataset(split_dir)
            if cat_counts:
                split_info[split] = cat_counts
    
    return split_info


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='数据集分析工具')
    parser.add_argument('--data_dir', type=str, default='dataset_data',
                        help='数据集根目录')
    parser.add_argument('--dataset', type=str, default=None,
                        help='特定数据集名称（如fer2013, RAF-DB, Oulu-CASIA）')
    parser.add_argument('--depth', type=int, default=2,
                        help='目录结构显示深度')
    args = parser.parse_args()
    
    data_dir = args.data_dir
    
    if not os.path.exists(data_dir):
        print(f"错误: 目录不存在 - {data_dir}")
        sys.exit(1)
    
    if args.dataset:
        target_dir = os.path.join(data_dir, args.dataset)
        if not os.path.exists(target_dir):
            print(f"错误: 数据集目录不存在 - {target_dir}")
            sys.exit(1)
        datasets = [(args.dataset, target_dir)]
    else:
        # 获取所有子目录作为数据集
        datasets = []
        for entry in os.listdir(data_dir):
            entry_path = os.path.join(data_dir, entry)
            if os.path.isdir(entry_path):
                datasets.append((entry, entry_path))
        
        if not datasets:
            print(f"警告: {data_dir} 目录中没有子目录")
            return
    
    print("=" * 60)
    print("数据集分析报告")
    print("=" * 60)
    print()
    
    for dataset_name, dataset_dir in datasets:
        print(f"【数据集】{dataset_name}")
        print("-" * 40)
        
        # 目录结构
        print("目录结构:")
        structure = analyze_directory_structure(dataset_dir, depth=args.depth)
        for line in structure:
            print(line)
        print()
        
        # 文件统计
        ext_counts, total_files = count_files_by_extension(dataset_dir)
        print(f"文件统计:")
        print(f"  总文件数: {total_files}")
        print(f"  文件类型:")
        for ext, count in sorted(ext_counts.items(), key=lambda x: -x[1]):
            print(f"    {ext}: {count}")
        print()
        
        # 检查是否是按类别组织的数据集
        split_info = analyze_split_dataset(dataset_dir)
        if split_info:
            print("数据集划分:")
            for split, cat_counts in split_info.items():
                total = sum(cat_counts.values())
                print(f"  {split}: {total} 张图像")
                for cat, count in sorted(cat_counts.items()):
                    print(f"    {cat}: {count}")
        else:
            # 检查是否直接按类别组织
            cat_counts = analyze_fer_dataset(dataset_dir)
            if cat_counts:
                total = sum(cat_counts.values())
                print(f"类别分布:")
                print(f"  总图像数: {total}")
                for cat, count in sorted(cat_counts.items()):
                    print(f"    {cat}: {count}")
        print()
        
        # 检查特殊结构（如Oulu-CASIA）
        vis_dir = os.path.join(dataset_dir, 'VIS')
        nir_dir = os.path.join(dataset_dir, 'NIR')
        oulu_kaggle_dir = os.path.join(dataset_dir, 'Oulu_CASIA_NIR_VIS')
        
        if os.path.exists(vis_dir) or os.path.exists(nir_dir):
            print("检测到Oulu-CASIA标准格式:")
            if os.path.exists(vis_dir):
                vis_subdirs = [d for d in os.listdir(vis_dir) if os.path.isdir(os.path.join(vis_dir, d))]
                print(f"  VIS模态: {', '.join(vis_subdirs)}")
            if os.path.exists(nir_dir):
                print("  NIR模态: 存在")
        
        if os.path.exists(oulu_kaggle_dir):
            print("检测到Oulu-CASIA Kaggle格式")
        
        # 检查RAF-DB结构
        raf_data_dir = os.path.join(dataset_dir, 'DATASET')
        raf_image_dir = os.path.join(dataset_dir, 'Image')
        raf_label_dir = os.path.join(dataset_dir, 'label')
        
        if os.path.exists(raf_data_dir):
            print("检测到RAF-DB Kaggle格式")
            train_dir = os.path.join(raf_data_dir, 'train')
            test_dir = os.path.join(raf_data_dir, 'test')
            if os.path.exists(train_dir):
                train_classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
                print(f"  训练集类别数: {len(train_classes)}")
            if os.path.exists(test_dir):
                test_classes = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
                print(f"  测试集类别数: {len(test_classes)}")
        
        if os.path.exists(raf_image_dir) and os.path.exists(raf_label_dir):
            print("检测到RAF-DB标准格式")
        
        print()
        print("-" * 40)
        print()


if __name__ == '__main__':
    main()
