"""Shared round-trip assertion helpers for the wt-radar test suite."""
from typing import Any, Optional

from witwin_server.platform_bridge import GeometryMap

# Combined abs+rel tolerance: carrier/slope values are large (e.g. fc=77e9) and may pass
# through float32 component storage, so a pure absolute tolerance is too strict for them.
ATOL = 1e-6
RTOL = 1e-6


def approx(a: Any, b: Any) -> bool:
    """Float closeness within combined absolute + relative tolerance."""
    a, b = float(a), float(b)
    return abs(a - b) <= ATOL + RTOL * max(abs(a), abs(b))


# --- geometry / material (mirror the base round-trip via the frozen GeometryMap) ---

def _params_equal(a: dict, b: dict) -> bool:
    if a.keys() != b.keys():
        return False
    for key, va in a.items():
        vb = b[key]
        if isinstance(va, list):
            if len(va) != len(vb) or not all(approx(x, y) for x, y in zip(va, vb)):
                return False
        elif isinstance(va, (int, float)):
            if not approx(va, vb):
                return False
        elif va != vb:
            return False
    return True


def assert_geometry_equal(g1: Any, g2: Any) -> None:
    """Assert two platform geometries match (kind + params + position + rotation)."""
    a, b = GeometryMap.to_studio(g1), GeometryMap.to_studio(g2)
    assert a.kind == b.kind, f"geometry kind {a.kind} != {b.kind}"
    assert _params_equal(a.params, b.params), f"geometry params {a.params} != {b.params}"
    assert all(approx(x, y) for x, y in zip(a.position, b.position)), "geometry position"
    assert all(approx(x, y) for x, y in zip(a.rotation_quat, b.rotation_quat)), "geometry rotation"


def assert_material_scalar_equal(m1: Any, m2: Any) -> None:
    """Assert the scalar part of two materials matches (eps_r/mu_r/sigma_e/name)."""
    assert approx(complex(m1.eps_r).real, complex(m2.eps_r).real), "eps_r real"
    assert approx(complex(m1.eps_r).imag, complex(m2.eps_r).imag), "eps_r imag"
    assert approx(m1.mu_r, m2.mu_r), "mu_r"
    assert approx(m1.sigma_e, m2.sigma_e), "sigma_e"
    assert (m1.name or None) == (m2.name or None), "material name"


def assert_structures_equal(list1: Any, list2: Any) -> None:
    """Assert two radar structure lists match (geometry + material + meta + radar metadata)."""
    s1 = {s.name: s for s in list1}
    s2 = {s.name: s for s in list2}
    assert s1.keys() == s2.keys(), f"structure names {s1.keys()} != {s2.keys()}"
    for name, a in s1.items():
        b = s2[name]
        assert_geometry_equal(a.geometry, b.geometry)
        assert_material_scalar_equal(a.material, b.material)
        assert int(a.priority) == int(b.priority), f"{name} priority"
        assert bool(a.enabled) == bool(b.enabled), f"{name} enabled"
        assert tuple(a.tags) == tuple(b.tags), f"{name} tags"
        assert dict(a.metadata) == dict(b.metadata), f"{name} metadata (incl bsdf/dynamic)"


# --- RadarConfig (the (Scene, RadarConfig) pair's config half) ---

_RADAR_SCALAR_FIELDS = (
    "fc", "slope", "power", "adc_samples", "adc_start_time", "sample_rate",
    "idle_time", "ramp_end_time", "chirp_per_frame", "frame_per_second",
    "num_doppler_bins", "num_range_bins", "num_angle_bins", "num_tx", "num_rx",
)


def _locs_equal(a: Any, b: Any) -> bool:
    a, b = tuple(a), tuple(b)
    if len(a) != len(b):
        return False
    return all(all(approx(x, y) for x, y in zip(pa, pb)) for pa, pb in zip(a, b))


def assert_radar_config_equal(c1: Any, c2: Any) -> None:
    """Assert two RadarConfig dataclasses match across every field + the 4 sub-configs."""
    for field in _RADAR_SCALAR_FIELDS:
        assert approx(getattr(c1, field), getattr(c2, field)), f"RadarConfig.{field}"
    assert _locs_equal(c1.tx_loc, c2.tx_loc), "RadarConfig.tx_loc"
    assert _locs_equal(c1.rx_loc, c2.rx_loc), "RadarConfig.rx_loc"
    assert _subconfig_equal(c1.antenna_pattern, c2.antenna_pattern), "antenna_pattern"
    assert _subconfig_equal(c1.noise_model, c2.noise_model), "noise_model"
    assert _subconfig_equal(c1.polarization, c2.polarization), "polarization"
    assert _subconfig_equal(c1.receiver_chain, c2.receiver_chain), "receiver_chain"


def _subconfig_equal(a: Optional[dict], b: Optional[dict]) -> bool:
    # Sub-configs are validated dicts (or None); compare numerically (nested lists/floats).
    if a is None or b is None:
        return a is None and b is None
    if a.keys() != b.keys():
        return False
    return all(_value_equal(a[k], b[k]) for k in a)


def _value_equal(a: Any, b: Any) -> bool:
    if isinstance(a, dict):
        return isinstance(b, dict) and _subconfig_equal(a, b)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_value_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, (int, float)):
        return approx(a, b)
    return a == b
