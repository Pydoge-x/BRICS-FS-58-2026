"""
表情识别数据集模块
支持多个数据集：FER-2013, RAF-DB, Oulu-CASIA
"""

from .dataset import FERDataset, MultiDataset, DatasetConfig
from .fer2013 import FER2013Dataset
from .rafdb import RAFDBDataset
from .oulucasia import OuluCASIADataset

__all__ = [
    'FERDataset',
    'MultiDataset',
    'DatasetConfig',
    'FER2013Dataset',
    'RAFDBDataset',
    'OuluCASIADataset'
]