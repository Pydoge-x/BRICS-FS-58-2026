"""
ONNX推理器
支持VGG、ResNet、MobileNet、MobileViT四种模型
"""

import os
import numpy as np
import onnxruntime as ort
from PIL import Image
import cv2
from typing import Dict, List, Union, Optional, Tuple
from dataclasses import dataclass
import time


@dataclass
class InferrerConfig:
    """推理配置"""
    model_path: str
    model_type: str = 'auto'  # auto/vgg/resnet/mobilenet/mobilevit
    num_threads: int = 4
    use_gpu: bool = True
    input_size: int = 224
    
    # 预处理参数
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)


class FERInferrer:
    """
    表情识别ONNX推理器
    支持四种模型的统一推理
    """
    
    # 表情类别
    EXPRESSION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    def __init__(self, config: InferrerConfig):
        """
        初始化推理器
        
        Args:
            config: 推理配置
        """
        self.config = config
        self.model_path = config.model_path
        
        # 配置推理会话
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = config.num_threads
        sess_options.inter_op_num_threads = 2
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # 启用内存优化
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True
        
        # 推理提供者
        providers = []
        if config.use_gpu and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        
        # 创建推理会话
        self.session = ort.InferenceSession(config.model_path, sess_options, providers=providers)
        
        # 获取输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        
        # 动态batch size
        self.input_size = (input_shape[2], input_shape[3]) if len(input_shape) == 4 else (config.input_size, config.input_size)
        
        # 自动检测模型类型
        if config.model_type == 'auto':
            self.model_type = self._detect_model_type()
        else:
            self.model_type = config.model_type
        
        print(f"加载模型: {config.model_path}")
        print(f"模型类型: {self.model_type}")
        print(f"输入尺寸: {self.input_size}")
        print(f"推理设备: {providers[0]}")
        
        # 预热
        self._warmup()
    
    def _detect_model_type(self) -> str:
        """自动检测模型类型"""
        model_name = self.model_path.lower()
        if 'vgg' in model_name:
            return 'vgg'
        elif 'resnet' in model_name:
            return 'resnet'
        elif 'mobilenet' in model_name:
            return 'mobilenet'
        elif 'mobilevit' in model_name:
            return 'mobilevit'
        else:
            return 'unknown'
    
    def _warmup(self, iterations: int = 5):
        """预热推理"""
        dummy_input = np.random.randn(1, 3, self.input_size[0], self.input_size[1]).astype(np.float32)
        for _ in range(iterations):
            self.session.run([self.output_name], {self.input_name: dummy_input})
    
    def preprocess(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        图像预处理
        
        Args:
            image: 输入图像
        
        Returns:
            预处理后的numpy数组
        """
        # 转换为numpy数组
        if isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        
        # 确保RGB格式
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # 调整大小
        image = cv2.resize(image, self.input_size)
        
        # 归一化
        image = image.astype(np.float32) / 255.0
        
        # 标准化
        mean = np.array(self.config.mean, dtype=np.float32)
        std = np.array(self.config.std, dtype=np.float32)
        image = (image - mean) / std
        
        # CHW格式
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict(self, image: Union[np.ndarray, Image.Image]) -> Dict:
        """
        单图推理
        
        Args:
            image: 输入图像
        
        Returns:
            推理结果字典
        """
        start_time = time.time()
        
        # 预处理
        input_data = self.preprocess(image)
        
        # 推理
        outputs = self.session.run([self.output_name], {self.input_name: input_data})
        
        # 后处理
        probabilities = self._softmax(outputs[0][0])
        predicted_class = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class])
        
        result = {
            'model_type': self.model_type,
            'expression': self.EXPRESSION_LABELS[predicted_class],
            'label': predicted_class,
            'confidence': confidence,
            'probabilities': {
                label: float(prob)
                for label, prob in zip(self.EXPRESSION_LABELS, probabilities)
            },
            'process_time': time.time() - start_time
        }
        
        return result
    
    def batch_predict(self, images: List[Union[np.ndarray, Image.Image]]) -> List[Dict]:
        """
        批量推理
        
        Args:
            images: 图像列表
        
        Returns:
            结果列表
        """
        start_time = time.time()
        
        # 批量预处理
        batch_inputs = np.concatenate([self.preprocess(img) for img in images], axis=0)
        
        # 批量推理
        outputs = self.session.run([self.output_name], {self.input_name: batch_inputs})
        
        # 后处理
        results = []
        for probs in outputs[0]:
            probabilities = self._softmax(probs)
            predicted_class = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_class])
            
            results.append({
                'model_type': self.model_type,
                'expression': self.EXPRESSION_LABELS[predicted_class],
                'label': predicted_class,
                'confidence': confidence,
                'probabilities': {
                    label: float(prob)
                    for label, prob in zip(self.EXPRESSION_LABELS, probabilities)
                }
            })
        
        total_time = time.time() - start_time
        for result in results:
            result['batch_time'] = total_time
            result['avg_time'] = total_time / len(images)
        
        return results
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax归一化"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def benchmark(self, num_iterations: int = 100) -> Dict:
        """
        性能基准测试
        
        Args:
            num_iterations: 测试次数
        
        Returns:
            性能统计
        """
        dummy_input = np.random.randn(1, 3, self.input_size[0], self.input_size[1]).astype(np.float32)
        
        # 预热
        for _ in range(10):
            self.session.run([self.output_name], {self.input_name: dummy_input})
        
        # 计时
        times = []
        for _ in range(num_iterations):
            start = time.time()
            self.session.run([self.output_name], {self.input_name: dummy_input})
            times.append(time.time() - start)
        
        return {
            'model_type': self.model_type,
            'avg_time_ms': np.mean(times) * 1000,
            'min_time_ms': np.min(times) * 1000,
            'max_time_ms': np.max(times) * 1000,
            'std_time_ms': np.std(times) * 1000,
            'fps': 1.0 / np.mean(times)
        }
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            'model_path': self.model_path,
            'model_type': self.model_type,
            'input_size': self.input_size,
            'input_name': self.input_name,
            'output_name': self.output_name,
            'providers': self.session.get_providers()
        }