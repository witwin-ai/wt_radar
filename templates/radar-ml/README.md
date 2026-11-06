# Radar ML

A WiTwin Studio Project that synthesizes radar data, trains a model on it
and evaluates the model. It is also an ordinary Python project: you can open,
edit and run it without the editor.

The task: from one radar range profile, estimate how far away the car is. The
Scene is a 77 GHz FMCW radar on a road with a car ahead (`Radar Target`), a
parked car and a sign post. Two Scene parameters place the car:
`target_range` (metres ahead of the radar) and `target_offset` (metres to the
side).

## Set up

```bash
uv sync
```

This installs the Radar Plugin with its solver stack (`witwin-radar`, CUDA
torch). Synthesizing data needs a CUDA GPU; training, evaluating and the tests
run on the CPU.

## The pipeline

| Step | File | What it does |
| --- | --- | --- |
| Synthesize | `studies/radar-returns.toml` (dataset Study) | Monte Carlo draws 600 car positions. For each, `scripts/synthesize.py` simulates the radar and stores the range profile (dB per range bin) and the true range. Writes `datasets/radar-returns/`, split 80/10/10 into train, val and test. |
| Train | `studies/train-range.toml` (train Study) | `scripts/train.py` fits a small neural network (`scripts/model.py`, numpy) and logs `train/loss`, `val/loss` and `val/mae_m`. Saves the best epoch to `checkpoints/<run id>.npz`. |
| Evaluate | `scripts/evaluate.py` | Scores the checkpoint with the lowest validation error on the test split against a predict-the-mean baseline. Writes `reports/evaluation.json`. |

`tasks/radar-pipeline.toml` is a Workflow that runs the three steps in order.
Its `train` step trains three learning rates (`matrix`), and `evaluate` picks
the best of them.

## Run it

In Studio: open **Tasks**, pick **Synthesize, train, evaluate** and run it.
Or run each Study from the **Studies** page; the train Run shows its curves
there.

Without Studio:

```bash
uv run witwin workflow run radar-pipeline
```

or step by step:

```bash
uv run python -m witwin.studio.studies --project . --study radar-returns --out .witwin/runs/synthesize
uv run python scripts/train.py --dataset datasets/radar-returns
uv run python scripts/evaluate.py --dataset datasets/radar-returns --split test
uv run pytest
```

## Change it

- More or different data: edit the variables and `samples` in
  `studies/radar-returns.toml`, or bind more Scene fields to parameters and
  vary those.
- Another label: return it from `export` in `scripts/synthesize.py` and set
  `TARGET` in `scripts/model.py`.
- Another model: replace `scripts/model.py`. For torch,
  `witwin.studio.datasets.open("datasets/radar-returns").split("train").torch()`
  returns the samples as a torch `Dataset`.

## Notes

- `scripts/synthesize.py` calls `witwin.radar` directly, the library the Radar
  Plugin's solver uses, with the radar settings of the Scene's `Radar`
  component. Each object is one point scatterer; room reflections are left
  out. The first sample compiles CUDA kernels and takes about 15 s.
- Dataset shards are stored with Git LFS (`datasets/**/shards/**`). If this
  folder is a git repository, install git-lfs before running the dataset Study.
- `.witwin/` is local runtime state (caches, Runs, Agent Sessions); git ignores
  it, and `checkpoints/`.
