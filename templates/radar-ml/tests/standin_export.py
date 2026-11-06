"""A GPU-free stand-in for ``scripts/synthesize.py``'s export, used by the tests.

It reads the same setup from the Scene and returns the same sample, but the
range profile comes from a formula instead of ``witwin.radar``: each
scatterer is a peak one range bin wide at its distance, with power
RCS / distance^4 (the radar equation without constants).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import synthesize  # noqa: E402


def power_profile(setup: synthesize.Setup) -> np.ndarray:
    bins = np.arange(int(setup.config["adc_samples"]), dtype=float)
    distances = np.linalg.norm(setup.points - setup.position, axis=1)
    centres = distances / setup.range_resolution_m
    return sum(rcs / d ** 4 * np.exp(-0.5 * (bins - c) ** 2)
               for rcs, d, c in zip(setup.rcs, distances, centres))


def export(ctx):
    setup = synthesize.read_setup(ctx)
    return synthesize.sample(setup, power_profile(setup), ctx.rng)
