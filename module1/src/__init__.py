"""
人脸表情识别项目
支持VGG、ResNet、MobileNet、MobileViT四种模型
支持FER-2013、RAF-DB、Oulu-CASIA等多种数据集
"""

__version__ = '1.0.0'


def __getattr__(name):
    """延迟导入，避免 API 启动时加载不必要的 torchvision"""
    _imports = {
        'FERDataset': '.data',
        'MultiDataset': '.data',
        'DatasetConfig': '.data',
        'create_fer_model': '.models',
        'get_model_config': '.models',
        'FERTrainer': '.train',
        'TrainingConfig': '.train',
        'FERInferrer': '.inference',
        'InferrerConfig': '.inference',
        'FaceDetector': '.inference',
        'create_app': '.api',
        'run_app': '.api',
    }
    if name in _imports:
        import importlib
        mod = importlib.import_module(_imports[name], __package__)
        attr = getattr(mod, name)
        # Cache for next time
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    'FERDataset', 'MultiDataset', 'DatasetConfig',
    'create_fer_model', 'get_model_config',
    'FERTrainer', 'TrainingConfig',
    'FERInferrer', 'InferrerConfig', 'FaceDetector',
    'create_app', 'run_app',
]