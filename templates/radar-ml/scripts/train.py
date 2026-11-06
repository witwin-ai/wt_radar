"""Train the range regressor on the radar dataset and log its curves.

    python scripts/train.py --dataset datasets/radar-returns --lr 0.003 --epochs 300

The train Study ``studies/train-range.toml`` starts this script with its
``args`` and ``inputs`` as these options. ``witwin.studio.tracking`` sends
each logged value to the Run's metric curves in Studio; run by hand, the
same calls write ``.witwin/runs/<run id>/`` in this Project.

The model with the lowest validation MAE is saved to
``checkpoints/<run id>.npz``; ``evaluate.py`` picks the best of them.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import witwin.studio as ws
from witwin.studio import datasets

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import FEATURE, TARGET, Model, arrays  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="datasets/radar-returns", help="dataset folder, relative to the Project")
    parser.add_argument("--lr", type=float, default=3e-3, help="Adam learning rate")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden", type=int, nargs="+", default=[64], help="width of each hidden layer")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    ws.config(total_steps=args.epochs, lr=args.lr, hidden=args.hidden, batch_size=args.batch_size)
    data = datasets.open(PROJECT / args.dataset)
    x_train, y_train = arrays(data.split("train"))
    x_val, y_val = arrays(data.split("val"))
    if len(y_train) < 2 or len(y_val) < 1:
        raise SystemExit(f"{args.dataset} needs at least 2 train and 1 val samples; "
                         f"it has {len(y_train)} and {len(y_val)}. Run the dataset Study first.")

    rng = np.random.default_rng(args.seed)
    model = Model.new(x_train, y_train, args.hidden, rng)
    # Studio sets WITWIN_RUN_ID; by hand the time and process id keep parallel runs apart.
    run = os.environ.get("WITWIN_RUN_ID") or f"local-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    path = PROJECT / "checkpoints" / f"{run}.npz"
    best_mae, best_epoch = float("inf"), -1
    for epoch in range(args.epochs):
        order = rng.permutation(len(y_train))
        losses = [model.step(x_train[i], y_train[i], args.lr)
                  for i in np.array_split(order, max(1, len(order) // args.batch_size))]
        error = model.predict(x_val) - y_val
        val_mae = float(np.mean(np.abs(error)))
        ws.log({"train/loss": float(np.mean(losses)), "val/loss": float(np.mean(error ** 2)),
                "val/mae_m": val_mae}, step=epoch, epoch=epoch)
        if val_mae < best_mae:
            best_mae, best_epoch = val_mae, epoch
            model.save(path, {"run": run, "dataset": args.dataset, "feature": FEATURE, "target": TARGET,
                              "epoch": epoch, "val_mae_m": val_mae, "lr": args.lr, "hidden": args.hidden})

    ws.checkpoint(path.relative_to(PROJECT).as_posix(), metrics={"val/mae_m": best_mae})
    ws.summary(best_val_mae_m=best_mae, best_epoch=best_epoch, train_samples=len(y_train))
    print(f"best val MAE {best_mae:.3f} m at epoch {best_epoch}; saved {path.relative_to(PROJECT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
