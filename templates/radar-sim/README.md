# Radar Sim

A WiTwin Studio Project. It is also an ordinary Python project: you can open,
edit and run it without the editor.

## Set up

```bash
uv sync                 # or: python -m venv .venv, activate it, pip install -e .
```

## Use from Python

```python
import witwin.studio as ws

project = ws.open_project(".")
scene = project.scene()          # the main scene
print(list(scene.objects))
```

## Commands

- `witwin check` validates the Project files.
- `witwin run scripts/<name>.py` runs a Script with the Project on its path.
- `witwin open` opens the Project in WiTwin Studio.

## Layout

- `witwin.toml`: the Studio manifest (Project id, main scene).
- `pyproject.toml`: the Python dependencies, including the Plugins this Project uses.
- `scenes/`: the Scene files (`.wtscene`, TOML). Studio and code edit the same
  files; saving from either side rewrites the file, so comments in it are not kept.
- `.witwin/`: local runtime state (caches, Agent Sessions, Result history). It
  is not shared and git ignores it.

## What is in the Scene

`scenes/main.wtscene` is the Radar plugin's library demo: a `Radar` entity
holding the 77 GHz FMCW sensor configuration, and `Radar Target`, a metal box
the sensor's static snapshot is aimed at (`snapshot_target_id`).

## Solve

Open the Project in WiTwin Studio, select **Radar** and press **Run Static
Snapshot** or **Simulate & Prepare Replay**. Solving needs the Radar solver
runtime and a CUDA GPU.

## Run the example

```bash
uv sync
uv run python scripts/prepare_radar.py
uv run pytest
```

`scripts/prepare_radar.py` opens this Project with `witwin.studio`, checks the setup
the solver needs and prints it. Pass an option to change the setup; the script
saves `scenes/main.wtscene`, the same file Studio edits.
