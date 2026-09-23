"""规则行为推断单元测试。"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from module_b_behavior.src.behavior.rule_engine import infer_behavior_from_pose


def _base_kpts() -> np.ndarray:
    k = np.zeros((17, 3))
    k[5] = [100, 100, 0.9]   # left shoulder
    k[6] = [200, 100, 0.9]   # right shoulder
    k[11] = [110, 200, 0.9]  # left hip
    k[12] = [190, 200, 0.9]  # right hip
    k[0] = [150, 80, 0.9]    # nose
    return k


def test_raise_hand():
    k = _base_kpts()
    k[9] = [80, 50, 0.9]  # left wrist up
    behavior, conf = infer_behavior_from_pose(k)
    assert behavior == "raise_hand"
    assert conf > 0.5


def test_sit_listen():
    k = _base_kpts()
    k[11] = [110, 150, 0.9]
    k[12] = [190, 150, 0.9]
    behavior, _ = infer_behavior_from_pose(k)
    assert behavior == "sit_listen"


def test_unknown():
    k = np.zeros((17, 3))
    behavior, conf = infer_behavior_from_pose(k)
    assert behavior == "unknown"
    assert conf <= 0.5
