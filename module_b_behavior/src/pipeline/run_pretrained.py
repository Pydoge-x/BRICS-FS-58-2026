"""兼容旧入口，转发至 inference 模块。"""
from module_b_behavior.src.pipeline.inference import PretrainedPipeline, run_pretrained

__all__ = ["PretrainedPipeline", "run_pretrained"]
