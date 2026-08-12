"""规则行为推断验证（普通 Python 脚本，无需 pytest）。"""
import sys
from pathlib import Path

import numpy as np

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.behavior.rule_engine import infer_behavior_from_pose


def _base_kpts() -> np.ndarray:
    """基础坐姿关键点：肩部、臀部、鼻子可见。"""
    k = np.zeros((17, 3))
    k[5] = [100, 100, 0.9]   # left shoulder
    k[6] = [200, 100, 0.9]   # right shoulder
    k[11] = [110, 200, 0.9]  # left hip
    k[12] = [190, 200, 0.9]  # right hip
    k[0] = [150, 80, 0.9]    # nose
    return k


passed = 0
failed = 0


def check(name: str, ok: bool, detail: str = ""):
    global passed, failed
    if ok:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        print(f"  ❌ {name} — {detail}")


# ===== 测试 1：举手 =====
print("测试举手:")
k = _base_kpts()
k[9] = [80, 50, 0.9]  # 左手腕高于左肩
behavior, conf = infer_behavior_from_pose(k)
check("举手 → 标签为 raise_hand", behavior == "raise_hand",
      f"得到 {behavior}")
check("举手 → 置信度 > 0.5", conf > 0.5,
      f"得到 {conf:.2f}")


# ===== 测试 2：听讲 =====
print("测试听讲:")
k = _base_kpts()
k[11] = [110, 150, 0.9]  # 臀部距离肩部较近（坐姿时躯干较短）
k[12] = [190, 150, 0.9]
behavior, _ = infer_behavior_from_pose(k)
check("听讲 → 标签为 sit_listen", behavior == "sit_listen",
      f"得到 {behavior}")


# ===== 测试 3：未知 =====
print("测试未知（无关键点）:")
k = np.zeros((17, 3))
behavior, conf = infer_behavior_from_pose(k)
check("未知 → 标签为 unknown", behavior == "unknown",
      f"得到 {behavior}")
check("未知 → 置信度 <= 0.5", conf <= 0.5,
      f"得到 {conf:.2f}")


# ===== 测试 4：站立 =====
print("测试站立:")
k = _base_kpts()
k[11] = [110, 300, 0.9]  # 拉长躯干（站立姿态）
k[12] = [190, 300, 0.9]
behavior, _ = infer_behavior_from_pose(k)
check("站立 → 标签为 stand", behavior == "stand",
      f"得到 {behavior}")


# ===== 汇总 =====
print(f"\n{'='*30}")
print(f"通过: {passed}  |  失败: {failed}  |  总计: {passed + failed}")
if failed == 0:
    print("🎉 全部通过!")
else:
    print(f"❌ 有 {failed} 个测试未通过")
    sys.exit(1)
