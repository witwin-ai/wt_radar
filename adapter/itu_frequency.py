"""ITU-R P.2040 building materials evaluated at the radar carrier.

Studio's Material component stores a named ITU material as its nominal 1 GHz
snapshot (``eps_r == a``, ``sigma_e == c``; see Studio's ``itu_materials``).
P.2040 models the real permittivity as ``a * f^b`` and the conductivity as
``c * f^d`` with ``f`` in GHz, so at 77 GHz plasterboard is ~22x and floorboard
~350x more lossy than the stored values. The scene keeps the 1 GHz values;
the radar export evaluates them at its own carrier.

Coefficients are P.2040-1 Table 3, the edition Studio's stored values come
from. A material is converted only when its name is in the table and its
stored values are still that entry's nominal ones: a hand-edited value has no
coefficients to scale. The ground models are valid only from 1 to 10 GHz and
are left as stored outside it.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

#: name -> (a, b, c, d, low GHz, high GHz, range is a hard limit)
P2040_1 = {
    "Vacuum": (1.0, 0.0, 0.0, 0.0, 0.001, 100.0, False),
    "Concrete": (5.31, 0.0, 0.0326, 0.8095, 1.0, 100.0, False),
    "Brick": (3.75, 0.0, 0.038, 0.0, 1.0, 10.0, False),
    "Plasterboard": (2.94, 0.0, 0.0116, 0.7076, 1.0, 100.0, False),
    "Wood": (1.99, 0.0, 0.0047, 1.0718, 0.001, 100.0, False),
    "Glass": (6.27, 0.0, 0.0043, 1.1925, 0.1, 100.0, False),
    "Ceiling board": (1.50, 0.0, 0.0005, 1.1634, 1.0, 100.0, False),
    "Chipboard": (2.58, 0.0, 0.0217, 0.7800, 1.0, 100.0, False),
    "Floorboard": (3.66, 0.0, 0.0044, 1.3515, 50.0, 100.0, False),
    "Metal": (1.0, 0.0, 1.0e7, 0.0, 1.0, 100.0, False),
    "Very dry ground": (3.0, 0.0, 0.00015, 2.52, 1.0, 10.0, True),
    "Medium dry ground": (15.0, -0.1, 0.035, 1.63, 1.0, 10.0, True),
    "Wet ground": (30.0, -0.4, 0.15, 1.30, 1.0, 10.0, True),
}

#: Studio stores float32; a nominal value round-trips within this.
NOMINAL_TOLERANCE = 1e-5


def _nominal(stored: float, expected: float) -> bool:
    return math.isclose(float(stored), expected, rel_tol=NOMINAL_TOLERANCE,
                        abs_tol=NOMINAL_TOLERANCE * 1e-3)


def itu_material_at(name: Optional[str], eps_r: complex, sigma_e: float,
                    frequency_hz: float) -> Optional[Tuple[float, float]]:
    """``(eps_r, sigma_e)`` of a nominal ITU material at ``frequency_hz``.

    ``None`` leaves the stored values in place: not an ITU name, a complex or
    edited value, or a ground model outside its valid band.
    """
    entry = P2040_1.get(str(name or ""))
    if entry is None or complex(eps_r).imag != 0.0:
        return None
    a, b, c, d, low, high, strict = entry
    if not (_nominal(complex(eps_r).real, a) and _nominal(sigma_e, c)):
        return None
    f_ghz = float(frequency_hz) / 1e9
    if not math.isfinite(f_ghz) or f_ghz <= 0.0 or (strict and not low <= f_ghz <= high):
        return None
    return a * f_ghz ** b, c * f_ghz ** d
