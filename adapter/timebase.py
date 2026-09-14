"""Shared Radar animation timebase validation.

Studio component floats can make a JSON value such as ``16.9`` round-trip as
the nearest IEEE-754 float32 value. Frame-aligned intervals must therefore be
checked against float32 storage precision, not a near-zero absolute epsilon.
"""
import math


_FLOAT32_EPSILON = 1.1920928955078125e-7


def aligned_frame_count(duration_s: float, fps: float, *, minimum: int = 2) -> int | None:
    """Return the nearest valid frame count, or ``None`` when not frame-aligned."""
    product = float(duration_s) * float(fps)
    if not math.isfinite(product):
        return None
    count = int(round(product))
    # Four float32 ULPs cover component serialization and multiplication while
    # remaining far below a meaningful fraction of one Radar frame.
    tolerance = max(1e-7, 4.0 * _FLOAT32_EPSILON * max(abs(product), 1.0))
    if count < minimum or not math.isclose(product, count, rel_tol=0.0, abs_tol=tolerance):
        return None
    return count
