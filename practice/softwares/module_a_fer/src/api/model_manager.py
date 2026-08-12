"""
模型管理器
管理多个模型的加载和切换
支持PyTorch格式存储，服务启动时自动转换为ONNX
自动扫描 checkpoints/ 目录发现可用模型
"""

import os
import glob
import torch
from typing import Dict, Optional
from src.inference.inferrer import FERInferrer, InferrerConfig
from src.models.factory import load_model
from src.inference.export import export_to_onnx


class ModelManager:
    """
    模型管理器
    支持多模型加载和切换
    优先加载PyTorch模型(.pth)，自动转换为ONNX格式用于推理
    自动扫描 checkpoints/{model}_*/best_model.pth 格式的文件
    """
    
    def __init__(self, model_dir: str = 'checkpoints', onnx_dir: str = 'models'):
        """
        初始化模型管理器
        
        Args:
            model_dir: PyTorch模型目录
            onnx_dir: ONNX模型目录（自动生成）
        """
        self.model_dir = model_dir
        self.onnx_dir = onnx_dir
        self.current_model: Optional[str] = None
        self.current_inferrer: Optional[FERInferrer] = None
        self.loaded_models: Dict[str, FERInferrer] = {}
        
        # 确保ONNX目录存在
        os.makedirs(self.onnx_dir, exist_ok=True)
        
        # 自动扫描可用模型
        self.available_models = self._scan_models()
    
    def _scan_models(self) -> Dict[str, str]:
        """
        自动扫描 checkpoints/ 目录，发现所有可用模型
        
        支持的文件结构：
        - checkpoints/{model}_*/best_model.pth  (推荐)
        - checkpoints/{model}_best.pth  (兼容旧格式)
        
        Returns:
            {model_name: pth_file_path} 字典
        """
        models = {}
        
        # 扫描子目录中的 best_model.pth
        for subdir in os.listdir(self.model_dir):
            subdir_path = os.path.join(self.model_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
            
            best_path = os.path.join(subdir_path, 'best_model.pth')
            if os.path.exists(best_path):
                # 从目录名提取模型名：vgg_fer2013 -> vgg
                model_name = subdir.split('_')[0]
                models[model_name] = best_path
                print(f"发现模型: {model_name} -> {best_path}")
        
        # 兼容旧格式: checkpoints/{model}_best.pth
        for fname in os.listdir(self.model_dir):
            if fname.endswith('_best.pth'):
                model_name = fname.replace('_best.pth', '')
                if model_name not in models:
                    pth_path = os.path.join(self.model_dir, fname)
                    models[model_name] = pth_path
                    print(f"发现模型(旧格式): {model_name} -> {pth_path}")
        
        if not models:
            print(f"警告: 在 {self.model_dir}/ 下未发现任何模型文件")
        
        return models
    
    def _get_onnx_path(self, model_name: str) -> str:
        """获取ONNX模型路径"""
        return os.path.join(self.onnx_dir, f'fer_{model_name}.onnx')
    
    def _pth_to_onnx(self, model_name: str, pth_path: str) -> str:
        """
        将PyTorch模型转换为ONNX格式
        
        Args:
            model_name: 模型名称
            pth_path: PyTorch模型路径
        
        Returns:
            ONNX模型路径
        """
        onnx_path = self._get_onnx_path(model_name)
        
        # 如果ONNX已存在且比pth新，则跳过转换
        if os.path.exists(onnx_path):
            pth_mtime = os.path.getmtime(pth_path)
            onnx_mtime = os.path.getmtime(onnx_path)
            if onnx_mtime > pth_mtime:
                print(f"ONNX模型已存在且更新，跳过转换: {onnx_path}")
                return onnx_path
        
        # 加载PyTorch模型
        print(f"加载PyTorch模型: {pth_path}")
        model = load_model(pth_path)
        model.eval()
        
        # 转换为ONNX
        print(f"转换为ONNX: {onnx_path}")
        try:
            export_to_onnx(model, onnx_path, input_shape=(1, 3, 224, 224))
        except Exception as e:
            print(f"ONNX导出失败: {e}")
            # 如果简化验证失败但文件已生成，仍可使用
            if os.path.exists(onnx_path):
                print(f"ONNX文件已生成，跳过验证错误继续使用")
            else:
                raise
        
        return onnx_path
    
    def pre_convert_all(self):
        """
        预转换所有可用模型为ONNX格式
        在服务启动时调用，避免首次切换模型时等待过久
        """
        print("\n开始预转换所有模型为ONNX格式...")
        for model_name, pth_path in self.available_models.items():
            try:
                print(f"\n--- 转换 {model_name} ---")
                self._pth_to_onnx(model_name, pth_path)
            except Exception as e:
                print(f"警告: 模型 {model_name} 的ONNX转换失败: {e}")
        print("\n预转换完成。")

    def load_model(self, model_name: str, num_threads: int = 4) -> FERInferrer:
        """
        加载模型
        
        Args:
            model_name: 模型名称
            num_threads: 推理线程数
        
        Returns:
            推理器
        """
        # 检查模型是否已加载
        if model_name in self.loaded_models:
            self.current_model = model_name
            self.current_inferrer = self.loaded_models[model_name]
            return self.current_inferrer
        
        # 检查模型是否支持
        if model_name not in self.available_models:
            raise ValueError(
                f"不支持的模型: {model_name}。可用模型: {list(self.available_models.keys())}"
            )
        
        # 获取PyTorch模型路径
        pth_path = self.available_models[model_name]
        
        if not os.path.exists(pth_path):
            raise FileNotFoundError(f"PyTorch模型文件不存在: {pth_path}")
        
        # 转换为ONNX（如果需要）
        onnx_path = self._pth_to_onnx(model_name, pth_path)
        
        # 创建推理器
        config = InferrerConfig(
            model_path=onnx_path,
            model_type=model_name,
            num_threads=num_threads
        )
        
        inferrer = FERInferrer(config)
        
        # 缓存模型
        self.loaded_models[model_name] = inferrer
        self.current_model = model_name
        self.current_inferrer = inferrer
        
        print(f"已加载模型: {model_name}")
        
        return inferrer
    
    def switch_model(self, model_name: str) -> str:
        """
        切换模型
        
        Args:
            model_name: 模型名称
        
        Returns:
            当前模型名称
        """
        if model_name not in self.available_models:
            raise ValueError(
                f"不支持的模型: {model_name}。可用模型: {list(self.available_models.keys())}"
            )
        
        self.load_model(model_name)
        
        return self.current_model
    
    def get_available_models(self) -> list:
        """获取可用模型列表"""
        available = []
        for name, pth_path in self.available_models.items():
            onnx_path = self._get_onnx_path(name)
            available.append({
                'name': name,
                'pth_path': pth_path,
                'pth_exists': os.path.exists(pth_path),
                'onnx_exists': os.path.exists(onnx_path),
                'loaded': name in self.loaded_models,
                'pth_size_mb': round(os.path.getsize(pth_path) / (1024*1024), 1) if os.path.exists(pth_path) else 0
            })
        return available
    
    def get_current_model(self) -> Optional[str]:
        """获取当前模型名称"""
        return self.current_model
    
    def get_inferrer(self) -> Optional[FERInferrer]:
        """获取当前推理器"""
        return self.current_inferrer
    
    def unload_model(self, model_name: str):
        """卸载模型"""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            print(f"已卸载模型: {model_name}")
    
    def unload_all(self):
        """卸载所有模型"""
        self.loaded_models.clear()
        self.current_model = None
        self.current_inferrer = None
        print("已卸载所有模型")