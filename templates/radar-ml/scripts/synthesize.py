"""Export function of the dataset Study ``studies/radar-returns.toml``.

One sample is the range profile the Scene's Radar sees with the car at the
Study's sampled ``target_range`` and ``target_offset``, plus the true range
as the label.

The Radar Plugin's solver runs inside Studio. This script calls the same
library (``witwin.radar`` 0.4, CUDA only) directly, so a dataset Run needs
no editor. It follows the Plugin's Run Static Snapshot: ``Radar.from_dict``
on the Radar component's fields, the sensor pose from its Transform, and
the car as one point scatterer with the component's ``snapshot_rcs_m2``.
The meshes in ``CLUTTER_IDS`` are point scatterers too, so the profile has
returns the model must learn to ignore. Room reflections are left out.

``read_setup`` and ``sample`` need only numpy; ``simulate`` is the GPU
part. ``tests/standin_export.py`` swaps ``simulate`` for a formula so the
tests run without a GPU.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

# The Radar component fields Radar.from_dict reads (wt_radar adapter/radar04.py).
CONFIG_KEYS = ("num_tx", "num_rx", "fc", "slope", "power", "adc_samples", "adc_start_time",
               "sample_rate", "idle_time", "ramp_end_time", "chirp_per_frame", "tx_loc", "rx_loc")
INT_KEYS = frozenset({"num_tx", "num_rx", "adc_samples", "chirp_per_frame"})

# Other objects that scatter, each as one point at its centre. Missing ids are skipped.
CLUTTER_IDS = ("parked_car", "sign_post")
# RCS of each clutter point in m^2. Uncalibrated, like the component's snapshot RCS.
CLUTTER_RCS_M2 = 0.05
# Receiver noise floor, in dB below the strongest return of the sample.
SNR_DB = 35.0
# Profile values are dB relative to the peak, cut off here.
FLOOR_DB = -60.0
C = 299_792_458.0

# The Radar instrument of this process, rebuilt only when its config or pose
# changes: the first simulate compiles CUDA kernels (about 15 s).
_RADAR: dict[str, Any] = {}


@dataclass(frozen=True)
class Setup:
    """What one sample simulates. Positions are world metres."""
    config: dict[str, Any]       # Radar.from_dict input, in the component's units (Hz, MHz/us, ksps, us)
    position: np.ndarray         # sensor position
    forward: np.ndarray          # unit look direction
    up: np.ndarray               # unit up direction
    polarization: tuple[float, float, float]
    points: np.ndarray           # (n, 3) scatterers; row 0 is the target
    rcs: np.ndarray              # (n,) m^2

    @property
    def target_range_m(self) -> float:
        return float(np.linalg.norm(self.points[0] - self.position))

    @property
    def range_resolution_m(self) -> float:
        """c / 2B with B the swept bandwidth over the ADC window."""
        slope_hz_per_s = float(self.config["slope"]) * 1e12
        window_s = int(self.config["adc_samples"]) / (float(self.config["sample_rate"]) * 1e3)
        return C / (2.0 * slope_hz_per_s * window_s)


def export(ctx) -> dict[str, Any]:
    """One dataset sample: ``profile`` (dB per range bin), ``range_m`` and ``offset_m``."""
    setup = read_setup(ctx)
    return sample(setup, simulate(setup), ctx.rng)


def read_setup(ctx) -> Setup:
    """The radar and scatterers of this sample, read from ``ctx.scene`` and the sampled values.

    ``ctx.scene`` holds the saved Scene; a sampled point changes only the
    fields bound to formulas, which ``ctx.values.bindings`` holds. So every
    position goes through ``_position``. Bound component fields (a formula
    on ``Radar.fc``, say) are not read here.
    """
    scene = ctx.scene
    sensor = next((e for e in scene.objects.values() if "Radar" in e.components), None)
    if sensor is None:
        raise RuntimeError("the Scene has no Radar")
    radar = sensor.components["Radar"]
    config = {key: int(radar[key]) if key in INT_KEYS else radar[key] for key in CONFIG_KEYS}
    target = scene.objects.get(str(radar.get("snapshot_target_id") or ""))
    if target is None:
        raise RuntimeError("the Radar's Snapshot Target ID names no object in the Scene")
    local = np.asarray(radar.get("snapshot_local_point") or (0.0, 0.0, 0.0), dtype=float)
    target_point = _position(ctx, target) + _rotate(target.transform.rotation,
                                                    local * np.asarray(target.transform.scale))
    clutter = [_position(ctx, scene.objects[i]) for i in CLUTTER_IDS if i in scene.objects]
    rotation = sensor.transform.rotation
    return Setup(
        config=config,
        position=_position(ctx, sensor),
        forward=_rotate(rotation, (0.0, 0.0, -1.0)),
        up=_rotate(rotation, (0.0, 1.0, 0.0)),
        polarization=tuple(float(v) for v in radar.get("snapshot_polarization") or (0.0, 1.0, 0.0)),
        points=np.stack([target_point, *clutter]),
        rcs=np.asarray([float(radar["snapshot_rcs_m2"])] + [CLUTTER_RCS_M2] * len(clutter)),
    )


def simulate(setup: Setup) -> np.ndarray:
    """Power per range bin (linear, any unit): the FMCW cube averaged over Tx, Rx and chirps."""
    import torch
    import witwin.core as core
    from witwin.radar import Motion, PointTargets

    if not torch.cuda.is_available():
        raise RuntimeError("witwin.radar simulates on a CUDA GPU; this machine has none")
    radar, world = _radar(setup), core.Scene()
    # One simulate per scatterer, summed: PointTargets takes one isotropic RCS
    # for all its points, and the received signal is linear in the scatterers.
    cube = None
    for point, rcs in zip(setup.points, setup.rcs):
        targets = PointTargets(positions=torch.tensor([point], dtype=torch.float32, device=radar.device),
                               rcs=float(rcs))
        result = radar.simulate(world, targets, times=(0.0,), los=True, reflections=0,
                                motion=Motion.static())
        part = result.frame(0).processing_cube().data  # (tx, rx, chirp, range bin), complex
        cube = part if cube is None else cube + part
    power = cube.abs().square().mean(dim=tuple(range(cube.ndim - 1)))
    return power.double().cpu().numpy()


def sample(setup: Setup, power: np.ndarray, rng: np.random.Generator) -> dict[str, Any] | None:
    """Add receiver noise to ``power`` and return the dataset sample (None drops it)."""
    power = np.asarray(power, dtype=float)
    peak = float(power.max())
    if not peak > 0.0:
        return None  # nothing reached the radar: drop the sample
    # Exponential noise power per bin: the power of complex Gaussian noise.
    noisy = power + rng.exponential(peak * 10.0 ** (-SNR_DB / 10.0), size=power.shape)
    profile = 10.0 * np.log10(noisy / noisy.max())
    offset = float(np.dot(setup.points[0] - setup.position, np.cross(setup.forward, setup.up)))
    return {
        "profile": np.maximum(profile, FLOOR_DB).astype(np.float32),
        "range_m": setup.target_range_m,
        "offset_m": offset,
    }


# A position with its bound components replaced by the sample's values.
def _position(ctx, entity) -> np.ndarray:
    position = np.asarray(entity.transform.position, dtype=float)
    for key, value in ctx.values.bindings.items():
        if key.entity_id == entity.id and key.target == "Transform.position" and key.index is not None:
            position[key.index] = float(value)
    return position


# Rotate v by the unit quaternion q = (x, y, z, w), the Scene file's order.
def _rotate(q, v) -> np.ndarray:
    x, y, z, w = (float(c) for c in q)
    u, v = np.array([x, y, z]), np.asarray(v, dtype=float)
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def _radar(setup: Setup):
    from witwin.radar import Radar

    key = json.dumps([setup.config, setup.position.tolist(), setup.forward.tolist(),
                      setup.up.tolist(), setup.polarization])
    if _RADAR.get("key") != key:
        look_at = setup.position + setup.forward
        _RADAR["radar"] = Radar.from_dict(
            setup.config, position=tuple(setup.position), look_at=tuple(look_at),
            up=tuple(setup.up), polarization=setup.polarization, device="cuda")
        _RADAR["key"] = key
    return _RADAR["radar"]
