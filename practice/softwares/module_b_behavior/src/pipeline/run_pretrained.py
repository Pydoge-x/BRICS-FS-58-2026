"""兼容旧入口，转发至 inference 模块。"""
from src.pipeline.inference import PretrainedPipeline, run_pretrained

__all__ = ["PretrainedPipeline", "run_pretrained"]
