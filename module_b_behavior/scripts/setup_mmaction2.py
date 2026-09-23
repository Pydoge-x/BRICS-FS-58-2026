"""安装并验证 MMAction2 环境（数据与权重已随仓库提供）。

建议：
  1. pip install -r requirements.txt
  2. python scripts/setup_mmaction2.py --skip-pip

步骤:
  1. pip 安装 mmengine / mmcv-lite / mmaction2 / decord（或已用 requirements.txt）
  2. 修补 Windows 下 pip 包缺失的 DRN 模块
  3. 确认 models/mmaction2/ 与 data/demo/extracurricular/ 已就绪
  4. 推理冒烟测试
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
MODELS_DIR = ROOT / "models" / "mmaction2"
DEMO_VIDEO_DIR = ROOT / "data" / "demo" / "extracurricular"


def log(msg: str) -> None:
    print(msg, flush=True)


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    log(f"  $ {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=cwd or ROOT)


def patch_mmaction_localizers() -> None:
    import mmaction  # noqa: WPS433

    init_py = Path(mmaction.__file__).resolve().parent / "models" / "localizers" / "__init__.py"
    text = init_py.read_text(encoding="utf-8")
    if "except ModuleNotFoundError" in text:
        log(f"  已修补: {init_py}")
        return
    patched = """# Copyright (c) OpenMMLab. All rights reserved.
from .bmn import BMN
from .bsn import PEM, TEM
try:
    from .drn.drn import DRN
except ModuleNotFoundError:
    DRN = None
from .tcanet import TCANet

__all__ = ['TEM', 'PEM', 'BMN', 'TCANet', 'DRN']
"""
    init_py.write_text(patched, encoding="utf-8")
    log(f"  已修补 DRN 导入: {init_py}")


def install_pip_deps() -> None:
    log("[1/3] 安装 pip 依赖")
    run([str(PYTHON), "-m", "pip", "install", "-U", "openmim"])
    run([str(PYTHON), "-m", "mim", "install", "mmengine"])
    run([str(PYTHON), "-m", "mim", "install", "mmcv-lite"])
    run([
        str(PYTHON), "-m", "pip", "install",
        "mmaction2>=1.2.0", "decord>=0.6.0", "einops", "importlib_metadata",
    ])
    patch_mmaction_localizers()


def check_local_assets() -> None:
    log("[2/3] 确认本地模型与 demo 数据")
    required = [
        MODELS_DIR / "label_map_k400.txt",
        DEMO_VIDEO_DIR / "mmaction_official_demo.mp4",
    ]
    pth_files = list(MODELS_DIR.glob("*.pth"))
    if not pth_files:
        log("  警告: models/mmaction2/ 下未找到 .pth 权重")
    for path in required:
        if path.exists():
            log(f"  OK: {path.relative_to(ROOT)}")
        else:
            log(f"  缺失: {path.relative_to(ROOT)}")


def verify() -> None:
    log("[3/3] MMAction2 推理冒烟测试")
    sys.path.insert(0, str(ROOT))
    from module_b_behavior.src.extracurricular.mmaction import MMActionRecognizer  # noqa: WPS433

    import yaml
    with open(ROOT / "configs" / "extracurricular.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    rec = MMActionRecognizer(cfg)
    status = rec.status()
    log(f"  MMAction 状态: {status}")
    if not rec.available:
        raise RuntimeError(f"MMAction2 未就绪: {status['note']}")

    demo = DEMO_VIDEO_DIR / "mmaction_official_demo.mp4"
    if not demo.exists():
        demo = ROOT / "data" / "demo" / "coco_person_demo.mp4"
    if demo.exists():
        label, score = rec.predict_video(demo)
        log(f"  试跑 {demo.name}: {label} ({score:.3f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="安装 MMAction2 并验证本地资源")
    parser.add_argument("--skip-pip", action="store_true", help="跳过 pip 安装，仅修补与验证")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    if not args.skip_pip:
        install_pip_deps()
    else:
        patch_mmaction_localizers()

    check_local_assets()

    if not args.skip_verify:
        verify()

    log("完成。课外推理: python scripts/run_extracurricular_demo.py")


if __name__ == "__main__":
    main()
