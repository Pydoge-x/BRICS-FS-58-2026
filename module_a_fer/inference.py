"""
推理脚本
使用ONNX模型进行表情识别
"""

import os
import sys
import argparse
import cv2
import numpy as np
from PIL import Image

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.inference.inferrer import FERInferrer, InferrerConfig
from src.inference.detector import FaceDetector
from src.inference.export import export_checkpoint_to_onnx


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='表情识别推理')
    
    parser.add_argument('--model', type=str, default='models/fer_mobilevit.onnx',
                        help='ONNX模型路径')
    parser.add_argument('--image', type=str, default=None,
                        help='输入图像路径')
    parser.add_argument('--output', type=str, default=None,
                        help='输出结果路径')
    parser.add_argument('--detect_face', action='store_true', default=False,
                        help='是否进行人脸检测')
    parser.add_argument('--benchmark', type=int, default=0,
                        help='性能测试次数（0表示不测试）')
    parser.add_argument('--export', type=str, default=None,
                        help='导出PyTorch检查点为ONNX')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 导出模型
    if args.export:
        print(f"导出模型: {args.export}")
        output_path = args.model if args.model else 'models/fer.onnx'
        export_checkpoint_to_onnx(args.export, output_path)
        return
    
    # 检查模型
    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在: {args.model}")
        return
    
    # 创建推理器
    config = InferrerConfig(model_path=args.model)
    inferrer = FERInferrer(config)
    
    # 性能测试
    if args.benchmark > 0:
        print(f"\n性能测试 ({args.benchmark} 次)...")
        results = inferrer.benchmark(args.benchmark)
        print(f"平均推理时间: {results['avg_time_ms']:.2f} ms")
        print(f"最小推理时间: {results['min_time_ms']:.2f} ms")
        print(f"最大推理时间: {results['max_time_ms']:.2f} ms")
        print(f"帧率: {results['fps']:.2f} FPS")
        return
    
    # 单图推理
    if args.image:
        print(f"\n推理图像: {args.image}")
        
        # 加载图像
        image = Image.open(args.image).convert('RGB')
        
        # 人脸检测
        if args.detect_face:
            detector = FaceDetector()
            image_np = np.array(image)
            faces = detector.detect(image_np)
            
            if len(faces) == 0:
                print("未检测到人脸")
                return
            
            # 提取最大人脸
            largest_face = max(faces, key=lambda f: f['bbox'][2] * f['bbox'][3])
            image = Image.fromarray(
                detector.extract_face(image_np, largest_face['bbox'], (224, 224))
            )
            
            print(f"检测到 {len(faces)} 个人脸")
        
        # 推理
        result = inferrer.predict(image)
        
        print(f"\n识别结果:")
        print(f"  表情: {result['expression']}")
        print(f"  置信度: {result['confidence']:.4f}")
        print(f"  推理时间: {result['process_time']*1000:.2f} ms")
        
        print(f"\n各类别概率:")
        for label, prob in result['probabilities'].items():
            print(f"  {label}: {prob:.4f}")
        
        # 保存结果
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n结果已保存至: {args.output}")
    
    else:
        print("\n提示: 请使用 --image 参数指定输入图像")
        print("示例: python inference.py --model models/fer.onnx --image test.jpg")


if __name__ == '__main__':
    main()