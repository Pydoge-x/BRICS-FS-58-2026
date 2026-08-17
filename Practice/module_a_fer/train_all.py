"""
批量训练脚本
依次训练四个模型：VGG、ResNet、MobileNet、MobileViT
"""

import os
import sys
import subprocess
import time
from datetime import datetime


def train_model(config_file, save_dir_suffix=None):
    """训练单个模型（实时输出进度）"""
    print(f"\n{'='*60}")
    print(f"开始训练: {config_file}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        python_executable = sys.executable
        cwd = os.path.dirname(os.path.abspath(__file__))
        
        # 构建命令（epochs 由配置文件控制）
        cmd = [python_executable, "train.py", "--config", f"configs/{config_file}"]
        if save_dir_suffix:
            cmd.extend(["--save_dir", save_dir_suffix])
        
        # 使用 Popen 实现实时输出
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='gbk',
            errors='replace'
        )
        
        # 实时读取并打印输出
        captured_output = []
        for line in process.stdout:
            print(line, end='')
            captured_output.append(line)
        
        process.wait()
        
        elapsed_time = time.time() - start_time
        print(f"\n训练耗时: {elapsed_time:.2f} 秒")
        
        return process.returncode == 0
    
    except Exception as e:
        print(f"训练失败: {e}")
        return False


def main():
    """主函数"""
    # 生成日期时间标识
    run_tag = datetime.now().strftime("%Y%m%d_%H%M")
    
    print("=" * 60)
    print("批量训练表情识别模型")
    print("=" * 60)
    print(f"运行标识: {run_tag}")
    print(f"每模型: 50 epochs")
    print("将依次训练以下模型：")
    print("1. VGG-16")
    print("2. ResNet-50")
    print("3. MobileNetV3")
    print("4. MobileViT-XS")
    print("=" * 60)
    
    # 模型配置文件列表 -> (config_file, save_dir)
    model_configs = [
        ('train_vgg_fer2013.json',       f'checkpoints/vgg_fer2013/{run_tag}'),
        ('train_resnet_fer2013.json',    f'checkpoints/resnet_fer2013/{run_tag}'),
        ('train_mobilenet_fer2013.json', f'checkpoints/mobilenet_fer2013/{run_tag}'),
        ('train_mobilevit_fer2013.json', f'checkpoints/mobilevit_fer2013/{run_tag}'),
    ]
    
    # 训练结果统计
    results = []
    
    for i, (config_file, save_dir) in enumerate(model_configs):
        config_path = os.path.join('configs', config_file)
        if not os.path.exists(config_path):
            print(f"警告: 配置文件不存在 - {config_path}")
            results.append((config_file, False))
            continue
        
        success = train_model(config_file, save_dir_suffix=save_dir)
        results.append((config_file, success))
        
        # 训练之间短暂休息
        if i < len(model_configs) - 1:
            print("\n等待5秒后开始下一个模型训练...")
            time.sleep(5)
    
    # 输出训练总结
    print("\n" + "=" * 60)
    print("训练总结")
    print("=" * 60)
    
    success_count = sum(1 for _, success in results if success)
    total_count = len(results)
    
    for config_file, success in results:
        status = "[OK] 成功" if success else "[FAIL] 失败"
        print(f"{config_file}: {status}")
    
    print(f"\n总计: {success_count}/{total_count} 个模型训练成功")
    
    if success_count == total_count:
        print("\n所有模型训练完成！")
    else:
        print(f"\n{total_count - success_count} 个模型训练失败，请检查错误信息")


if __name__ == '__main__':
    main()
