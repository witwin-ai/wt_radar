"""Score the best trained range regressor on one split of the radar dataset.

    python scripts/evaluate.py --dataset datasets/radar-returns --split test
    python scripts/evaluate.py --model checkpoints/<run id>.npz

Without ``--model`` it takes, of the ``checkpoints/*.npz`` trained on the
same dataset, the one with the lowest validation MAE; after the Workflow's
``train`` matrix that is the best learning rate. It prints the MAE next to
a baseline that always predicts the training mean, and writes
``reports/evaluation.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from witwin.studio import datasets

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import TARGET, Model, arrays  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]


def best_checkpoint(dataset: str) -> Path | None:
    """The checkpoint of ``dataset`` with the lowest saved ``val_mae_m``, or None."""
    found = []
    for path in sorted((PROJECT / "checkpoints").glob("*.npz")):
        _, meta = Model.load(path)
        if Path(meta.get("dataset", "")).as_posix() == Path(dataset).as_posix():
            found.append((float(meta["val_mae_m"]), path))
    return min(found)[1] if found else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="datasets/radar-returns", help="dataset folder, relative to the Project")
    parser.add_argument("--split", default="test")
    parser.add_argument("--model", help="checkpoint file; default: the best one of the dataset")
    args = parser.parse_args(argv)

    path = PROJECT / args.model if args.model else best_checkpoint(args.dataset)
    if path is None or not path.is_file():
        print(f"no checkpoint of {args.dataset} in checkpoints/; run the train Study first", file=sys.stderr)
        return 1
    model, meta = Model.load(path)
    data = datasets.open(PROJECT / args.dataset)
    x, y = arrays(data.split(args.split))
    if len(y) == 0:
        print(f"split {args.split!r} of {args.dataset} has no samples", file=sys.stderr)
        return 1
    _, y_train = arrays(data.split("train"))

    error = model.predict(x) - y
    baseline = float(np.mean(np.abs(y_train.mean() - y))) if len(y_train) else float("nan")
    report = {
        "model": path.relative_to(PROJECT).as_posix(),
        "run": meta.get("run"),
        "lr": meta.get("lr"),
        "dataset": args.dataset,
        "split": args.split,
        "samples": int(len(y)),
        "target": TARGET,
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "max_abs_error": float(np.max(np.abs(error))),
        "baseline_mae": baseline,
    }
    out = PROJECT / "reports" / "evaluation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{report['model']} on {args.split} ({len(y)} samples): MAE {report['mae']:.3f} m, "
          f"RMSE {report['rmse']:.3f} m; predicting the train mean: MAE {baseline:.3f} m")
    print(f"wrote {out.relative_to(PROJECT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
