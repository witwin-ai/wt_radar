import importlib.util
from pathlib import Path

import witwin.studio as ws

PROJECT = Path(__file__).resolve().parents[1]


def load_script():
    spec = importlib.util.spec_from_file_location("prepare_radar", PROJECT / "scripts" / "prepare_radar.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scene_has_a_radar_aimed_at_a_target():
    scene = ws.open_project(PROJECT).scene()
    radar = load_script().find_radar(scene)
    target = radar.components["Radar"]["snapshot_target_id"]
    assert target in scene.objects
    assert scene.objects[target].type == "builtin.mesh"


def test_prepare_script_reports_ready(capsys):
    assert load_script().main([]) == 0
    assert "Ready to solve" in capsys.readouterr().out
