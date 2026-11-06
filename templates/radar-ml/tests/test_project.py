"""The Project files fit together: Scene parameters, Studies and the Workflow."""
from pathlib import Path

import witwin.studio as ws
from witwin.studio.workflows import find_workflow

PROJECT = Path(__file__).resolve().parents[1]


def test_target_position_follows_the_parameters():
    scene = ws.open_project(PROJECT).scene()
    assert scene.parameters["target_range"] == 6.0
    scene.parameters.set("target_range", "8.5")   # in memory only; nothing is saved
    scene.parameters.set("target_offset", "-1.0")
    car = scene.objects["radar_target"].transform.position
    cabin = scene.objects["target_cabin"].transform.position
    assert (car[0], car[2]) == (-1.0, -8.5)
    assert (cabin[0], cabin[2]) == (-1.0, -8.8)


def test_studies_load_and_vary_scene_parameters():
    project = ws.open_project(PROJECT)
    assert project.studies.problems() == []
    dataset = project.studies.get("radar-returns")
    assert str(dataset.kind) == "dataset" and dataset.method.kind == "monte_carlo"
    assert set(dataset.variables) <= set(project.scene(dataset.scene).parameters)
    assert (PROJECT / dataset.export.path).is_file()
    train = project.studies.get("train-range")
    assert str(train.kind) == "train" and (PROJECT / train.script).is_file()
    assert train.inputs["dataset"] == dataset.dataset.dir


def test_workflow_chains_the_studies():
    workflow = find_workflow(PROJECT, "radar-pipeline")
    steps = {step.id: step for step in workflow.steps}
    assert [step.id for step in workflow.steps] == ["synthesize", "train", "evaluate"]
    assert steps["synthesize"].study == "radar-returns"
    assert steps["train"].study == "train-range" and len(steps["train"].matrix["lr"]) == 3
    assert (PROJECT / steps["evaluate"].entry).is_file()
