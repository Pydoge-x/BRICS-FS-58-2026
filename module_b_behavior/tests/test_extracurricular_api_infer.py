"""课外视频 API 端到端推理测试（需 MMAction2，较慢）。"""
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from module_b_behavior.src.api.app import app  # noqa: E402
from module_b_behavior.src.extracurricular.mmaction import MMActionRecognizer  # noqa: E402
from module_b_behavior.src.common.config import load_config  # noqa: E402

client = TestClient(app)
VIDEO = ROOT / "data" / "demo" / "extracurricular" / "mmaction_skeleton_demo.mp4"


@pytest.fixture(scope="module")
def mmaction_available() -> bool:
    cfg = load_config(ROOT / "configs" / "extracurricular.yaml")
    rec = MMActionRecognizer(cfg)
    return rec.available


def test_mmaction_direct_on_skeleton_demo(mmaction_available):
    if not mmaction_available:
        pytest.skip("MMAction2 未安装")
    cfg = load_config(ROOT / "configs" / "extracurricular.yaml")
    rec = MMActionRecognizer(cfg)
    detail = rec.predict_detailed(VIDEO)
    assert detail["behavior"] != "unknown", detail
    assert detail.get("raw_label"), detail
    assert detail["label_mode"] == "campus"
    assert detail.get("behavior_zh"), detail


def test_api_extracurricular_skeleton_demo(mmaction_available):
    if not mmaction_available:
        pytest.skip("MMAction2 未安装")
    assert VIDEO.exists()

    with open(VIDEO, "rb") as f:
        r = client.post(
            "/api/v1/behavior/analyze",
            files={"media": (VIDEO.name, f, "video/mp4")},
            data={"scene": "extracurricular", "max_frames": "72"},
        )
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]

    for _ in range(120):
        status = client.get(f"/api/v1/jobs/{job_id}").json()
        if status["status"] in ("completed", "failed"):
            break
        time.sleep(2)

    assert status["status"] == "completed", status
    result = client.get(f"/api/v1/jobs/{job_id}/result").json()
    scene_beh = result.get("meta", {}).get("scene_behavior", {})
    assert scene_beh.get("behavior") != "unknown", result
    assert scene_beh.get("label_mode") == "campus", scene_beh

    tracks = result.get("tracks", [])
    assert tracks, "应有 track 数据"
    all_behaviors = [seg["behavior"] for t in tracks for seg in t.get("segments", [])]
    assert any(b != "unknown" for b in all_behaviors), all_behaviors
    # 单片段有效归类（如 roughhouse）即通过；多片段时再要求 unknown 占比不过半
    if len(all_behaviors) > 1:
        assert all_behaviors.count("unknown") < len(all_behaviors) // 2, all_behaviors
