"""Synthesize, train and evaluate on a copy of this Project, without a GPU.

The dataset Study runs through the real Study process
(``python -m witwin.studio.studies``) with ``tests/standin_export.py`` in
place of the GPU export and fewer samples. Training and evaluation are the
real scripts.
"""
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import witwin.studio as ws
from witwin.studio import datasets

PROJECT = Path(__file__).resolve().parents[1]
SKIP = shutil.ignore_patterns(".venv", ".witwin", "datasets", "checkpoints", "reports",
                              "__pycache__", ".pytest_cache", "uv.lock")


def load(path: Path):
    spec = importlib.util.spec_from_file_location(f"_{path.parent.parent.name}_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_synthesize_train_evaluate(tmp_path, monkeypatch, capsys):
    root = tmp_path / "project"
    shutil.copytree(PROJECT, root, ignore=SKIP)
    project = ws.open_project(root)
    project.studies.update("radar-returns", method={"samples": 150},
                           export={"kind": "script", "path": "tests/standin_export.py", "function": "export"})

    out = root / ".witwin" / "runs" / "test"
    done = subprocess.run([sys.executable, "-m", "witwin.studio.studies", "--project", str(root),
                           "--study", "radar-returns", "--out", str(out)], capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["dataset"]["samples"] == 150

    data = datasets.open(root / "datasets" / "radar-returns")
    first = data[0]
    assert first["profile"].shape == (256,)
    assert 3.0 < first["range_m"] < 11.0

    monkeypatch.chdir(root)  # tracking records the headless Run in the Project it finds from here
    assert load(root / "scripts" / "train.py").main(["--epochs", "60", "--lr", "0.003"]) == 0
    assert list((root / "checkpoints").glob("*.npz"))

    assert load(root / "scripts" / "evaluate.py").main(["--split", "test"]) == 0
    report = json.loads((root / "reports" / "evaluation.json").read_text(encoding="utf-8"))
    assert report["samples"] > 0
    assert report["mae"] < report["baseline_mae"]
