"""REST API 基础测试（不加载 YOLO 模型）。"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from module_b_behavior.src.api.app import app  # noqa: E402

client = TestClient(app)


def test_health():
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_behaviors_classroom():
    r = client.get("/api/v1/behaviors", params={"scene": "classroom"})
    assert r.status_code == 200
    data = r.json()
    assert data["scene"] == "classroom"
    assert "sit_listen" in data["labels"]
    assert "raise_hand" in data["labels"]


def test_behaviors_extracurricular():
    r = client.get("/api/v1/behaviors", params={"scene": "extracurricular"})
    assert r.status_code == 200
    data = r.json()
    assert data["scene"] == "extracurricular"
    assert "abseiling" in data["labels"] or "cheerleading" in data["labels"]


def test_behaviors_extracurricular_kinetics_mode():
    r = client.get(
        "/api/v1/behaviors",
        params={"scene": "extracurricular", "label_mode": "kinetics"},
    )
    assert r.status_code == 200
    data = r.json()
    assert "kinetics" in data["description"].lower() or "abseiling" in data["labels"]


def test_analyze_rejects_empty_video():
    r = client.post(
        "/api/v1/behavior/analyze",
        files={"video": ("empty.mp4", b"x", "video/mp4")},
    )
    assert r.status_code == 400


def test_analyze_rejects_unsupported_format():
    r = client.post(
        "/api/v1/behavior/analyze",
        files={"media": ("test.txt", b"hello world", "text/plain")},
    )
    assert r.status_code == 400


def test_health_supports_image_video():
    r = client.get("/api/v1/health")
    data = r.json()
    assert "image" in data["supports"]
    assert "video" in data["supports"]
    assert "classroom" in data["scenes"]
    assert "extracurricular" in data["scenes"]


def test_job_not_found():
    r = client.get("/api/v1/jobs/nonexistent")
    assert r.status_code == 404


def test_index_page():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")
