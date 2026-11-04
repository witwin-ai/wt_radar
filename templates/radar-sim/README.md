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

`scenes/main.wtscene` is a short road seen by a 77 GHz FMCW sensor. `Radar`
holds the sensor configuration and sits 1 m above the `Road`,
looking down -z; it sees out to about 11 m. `Radar Target` is the car 6 m
ahead that the static snapshot is aimed at (`snapshot_target_id`), with its
`Radar Target Cabin` on top. `Parked Car` and `Sign Post` are more metal in
the field of view for the multi-target simulation.

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
