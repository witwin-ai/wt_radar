"""Motion-fidelity provenance and offline Radar before/after evidence.

This module does not alter synthesis, DSP, or the online authoring gate.  It
compares already processed whole-recording products.  A before/after pair can
describe change; it can claim improvement only when a separately identified
trusted reference is supplied.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping

import numpy as np


SCHEMA = "witwin.radar.motion-fidelity-ab.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FIDELITY_STATUSES = frozenset({
    "passed", "needs_review", "failed", "not_evaluated",
})


def normalize_fidelity_provenance(
    value: Mapping[str, Any], *, radar_motion_fingerprint: str | None = None,
) -> dict[str, str]:
    """Validate and reduce a full fidelity report to immutable provenance."""

    report_id = str(value.get("report_id") or value.get("fidelity_report_id") or "")
    snapshot = str(
        value.get("snapshot_fingerprint")
        or value.get("motion_snapshot_fingerprint") or "")
    status = str(value.get("status") or value.get("fidelity_status") or "")
    motion = str(
        radar_motion_fingerprint
        or value.get("radar_motion_fingerprint") or "")
    for label, item in (
        ("fidelity report id", report_id),
        ("motion snapshot fingerprint", snapshot),
        ("Radar motion fingerprint", motion),
    ):
        if not _SHA256.fullmatch(item):
            raise ValueError(f"Invalid {label}; expected a lowercase SHA-256 value.")
    if status not in _FIDELITY_STATUSES:
        raise ValueError("Invalid motion fidelity status.")
    return {
        "report_id": report_id,
        "snapshot_fingerprint": snapshot,
        "status": status,
        "radar_motion_fingerprint": motion,
    }


def bind_fidelity_provenance(component: Any, report: Mapping[str, Any]) -> dict[str, str]:
    """Bind one report to the component's current baked Timeline identity."""

    if getattr(component, "scene", None) is None:
        raise ValueError("Radar component must belong to a scene before binding fidelity.")
    from .replay import motion_fingerprint
    provenance = normalize_fidelity_provenance(
        report,
        radar_motion_fingerprint=motion_fingerprint(component.scene),
    )
    component._motion_fidelity_provenance = provenance
    return dict(provenance)


def attach_fidelity_provenance(
    metadata: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    authored_motion_fingerprint: str,
) -> dict[str, Any]:
    """Copy verified provenance into producer metadata or leave it absent."""

    result = dict(metadata)
    source = request.get("motion_fidelity")
    if source is None:
        return result
    provenance = normalize_fidelity_provenance(source)
    if provenance["radar_motion_fingerprint"] != authored_motion_fingerprint:
        raise ValueError(
            "Bound motion fidelity belongs to a different baked Timeline; "
            "rerun fidelity before Radar synthesis."
        )
    result["motion_fidelity"] = provenance
    return result


def _require_compatible(before: Any, after: Any) -> None:
    for name in ("times_s", "velocity_mps", "range_m"):
        left = np.asarray(getattr(before, name), dtype=np.float64)
        right = np.asarray(getattr(after, name), dtype=np.float64)
        if left.shape != right.shape or not np.allclose(left, right, rtol=0.0, atol=1e-9):
            raise ValueError(
                f"Radar A/B requires identical {name}; use the same interval and configuration.")
    for name in ("micro_doppler_db", "range_time_db", "motion_energy"):
        left = np.asarray(getattr(before, name), dtype=np.float64)
        right = np.asarray(getattr(after, name), dtype=np.float64)
        if left.shape != right.shape:
            raise ValueError(f"Radar A/B product shape differs for {name}.")


def _distribution(db_values: Any) -> np.ndarray:
    amplitude = np.power(10.0, np.asarray(db_values, dtype=np.float64) / 20.0)
    distribution = amplitude.sum(axis=0)
    total = float(distribution.sum())
    return distribution / total if total > 0.0 else np.zeros_like(distribution)


def _recording_metrics(overview: Any) -> dict[str, float]:
    velocity = np.asarray(overview.velocity_mps, dtype=np.float64)
    distribution = _distribution(overview.micro_doppler_db)
    energy = np.asarray(overview.motion_energy, dtype=np.float64)
    signed_mean = float(np.sum(distribution * velocity))
    return {
        "mean_abs_doppler_velocity_m_s": float(
            np.sum(distribution * np.abs(velocity))),
        "doppler_velocity_centroid_m_s": signed_mean,
        "doppler_velocity_rms_m_s": float(
            math.sqrt(max(0.0, np.sum(distribution * velocity * velocity)))),
        "mean_normalized_motion_energy": float(energy.mean()),
        "motion_energy_temporal_std": float(energy.std()),
        "motion_energy_delta_rms": float(
            math.sqrt(np.mean(np.diff(energy) ** 2))) if len(energy) > 1 else 0.0,
    }


def _distance(left: Any, right: Any) -> dict[str, float]:
    micro_l1 = float(np.abs(
        _distribution(left.micro_doppler_db)
        - _distribution(right.micro_doppler_db)).sum() / 2.0)
    left_energy = np.asarray(left.motion_energy, dtype=np.float64)
    right_energy = np.asarray(right.motion_energy, dtype=np.float64)
    energy_rmse = float(math.sqrt(np.mean((left_energy - right_energy) ** 2)))
    return {
        "micro_doppler_distribution_l1": micro_l1,
        "motion_energy_rmse": energy_rmse,
        "combined_distance": (micro_l1 + energy_rmse) / 2.0,
    }


def _stable_id(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compare_recording_overviews(
    before: Any,
    after: Any,
    *,
    before_provenance: Mapping[str, Any],
    after_provenance: Mapping[str, Any],
    reference: Any | None = None,
    reference_id: str | None = None,
) -> dict[str, Any]:
    """Build auditable Radar A/B evidence without promoting it to a gate."""

    _require_compatible(before, after)
    before_identity = normalize_fidelity_provenance(before_provenance)
    after_identity = normalize_fidelity_provenance(after_provenance)
    before_metrics = _recording_metrics(before)
    after_metrics = _recording_metrics(after)
    delta = {
        name: after_metrics[name] - before_metrics[name]
        for name in before_metrics
    }
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "before": before_identity,
        "after": after_identity,
        "sample_count": int(len(before.times_s)),
        "interval_s": [float(before.times_s[0]), float(before.times_s[-1])],
        "metrics": {
            "before": before_metrics,
            "after": after_metrics,
            "after_minus_before": delta,
            "ab_distance": _distance(before, after),
        },
        "assessment": {
            "status": "comparison_only",
            "can_claim_radar_improvement": False,
            "reason": "no trusted Radar reference was supplied",
        },
        "online_gate_effect": "none",
        "limitations": [
            "Relative amplitude is not calibrated radar cross section.",
            "A before/after delta alone does not establish biological realism.",
            "Visual/anatomical fidelity remains independently authoritative.",
        ],
    }
    if reference is not None:
        if not reference_id:
            raise ValueError("A trusted reference_id is required with a Radar reference.")
        _require_compatible(before, reference)
        before_distance = _distance(before, reference)
        after_distance = _distance(after, reference)
        improved = (
            after_distance["combined_distance"]
            < before_distance["combined_distance"] - 1e-9
        )
        result["reference"] = {
            "reference_id": str(reference_id),
            "before_distance": before_distance,
            "after_distance": after_distance,
        }
        result["assessment"] = {
            "status": "closer_to_reference" if improved else "not_closer_to_reference",
            "can_claim_radar_improvement": improved,
            "reason": (
                "after is closer on the declared combined metric"
                if improved else
                "after did not reduce the declared combined reference distance"
            ),
        }
    identity_payload = {
        "schema_version": result["schema_version"],
        "before": result["before"],
        "after": result["after"],
        "reference": result.get("reference"),
        "metrics": result["metrics"],
    }
    result["comparison_id"] = _stable_id(identity_payload)
    return result


__all__ = [
    "SCHEMA",
    "attach_fidelity_provenance",
    "bind_fidelity_provenance",
    "compare_recording_overviews",
    "normalize_fidelity_provenance",
]
