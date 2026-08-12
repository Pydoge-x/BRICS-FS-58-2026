"""MMAction 标签映射测试。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.extracurricular.labels import (  # noqa: E402
    campus_label_zh,
    map_kinetics_scores_to_campus,
    map_kinetics_to_campus,
    map_pretrain_label,
    resolve_display_label,
)


def test_campus_basketball():
    assert map_kinetics_to_campus("playing basketball") == "basketball"
    assert campus_label_zh("basketball") == "打篮球"


def test_campus_soccer():
    assert map_kinetics_to_campus("kicking soccer ball") == "soccer"


def test_campus_volleyball():
    assert map_kinetics_to_campus("playing volleyball") == "volleyball"


def test_campus_gymnastics():
    assert map_kinetics_to_campus("gymnastics tumbling") == "gymnastics"


def test_campus_roughhouse():
    assert map_kinetics_to_campus("wrestling") == "roughhouse"
    assert map_kinetics_to_campus("punching person (boxing)") == "roughhouse"


def test_campus_fall():
    assert map_kinetics_to_campus("faceplanting") == "fall"


def test_campus_walk_run():
    assert map_kinetics_to_campus("walking the dog") == "walk"
    assert map_kinetics_to_campus("jogging") == "run"


def test_campus_sit_not_in_kinetics():
    assert map_kinetics_to_campus("reading book") == "unknown"


def test_campus_table_tennis_from_kinetics_tennis():
    assert map_kinetics_to_campus("playing tennis") == "table_tennis"
    assert campus_label_zh("table_tennis") == "打乒乓球"


def test_campus_table_tennis_from_ucf():
    assert map_kinetics_to_campus("TableTennisShot") == "table_tennis"


def test_campus_table_tennis_badminton_proxy():
    assert map_kinetics_to_campus("playing badminton") == "table_tennis"


def test_topk_fallback_soccer_from_scores():
    labels = ["catching or throwing frisbee", "kicking soccer ball", "texting"]
    scores = np.array([0.77, 0.36, 0.1])
    campus, score, raw = map_kinetics_scores_to_campus(scores, labels)
    assert campus == "soccer"
    assert raw == "kicking soccer ball"
    assert score == 0.36


def test_kinetics_basketball_maps_sport():
    assert map_pretrain_label("playing basketball") == "sport"


def test_kinetics_walking_dog_maps_walk():
    assert map_pretrain_label("walking the dog") == "walk"


def test_kinetics_texting_maps_chat():
    assert map_pretrain_label("texting") == "chat"


def test_ucf_basketball_maps_sport():
    assert map_pretrain_label("Basketball") == "sport"


def test_kinetics_mode_returns_raw():
    assert resolve_display_label("throwing ball", "kinetics") == "throwing ball"


def test_campus_mode_returns_id():
    assert resolve_display_label("throwing ball", "campus") == "unknown"
    assert resolve_display_label("dribbling basketball", "campus") == "basketball"


def test_frisbee_maps_play_school_mode():
    assert map_pretrain_label("catching or throwing frisbee") == "play"


def test_unknown_label():
    assert map_pretrain_label("ironing") == "unknown"
    assert map_kinetics_to_campus("ironing") == "unknown"
