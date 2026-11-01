from types import SimpleNamespace

import numpy as np
import pytest

from wt_radar.adapter.fidelity_ab import (
    attach_fidelity_provenance,
    compare_recording_overviews,
    normalize_fidelity_provenance,
)


def _provenance(seed, status="passed"):
    return {
        "report_id": seed * 64,
        "snapshot_fingerprint": chr(ord(seed) + 1) * 64,
        "status": status,
        "radar_motion_fingerprint": chr(ord(seed) + 2) * 64,
    }


def _overview(center=2, energy=(0.1, 0.5, 1.0)):
    values = np.full((3, 5), -40.0)
    values[:, center] = 0.0
    return SimpleNamespace(
        times_s=np.array([0.0, 0.1, 0.2]),
        velocity_mps=np.array([-2.0, -1.0, 0.0, 1.0, 2.0]),
        range_m=np.array([0.0, 1.0]),
        micro_doppler_db=values,
        range_time_db=np.zeros((3, 2)),
        motion_energy=np.asarray(energy, dtype=np.float64),
    )


def test_provenance_is_attached_only_to_the_same_authored_timeline():
    provenance = _provenance("a")
    metadata = attach_fidelity_provenance(
        {"model": "fixture"}, {"motion_fidelity": provenance},
        authored_motion_fingerprint="c" * 64,
    )

    assert metadata["motion_fidelity"] == provenance
    with pytest.raises(ValueError, match="different baked Timeline"):
        attach_fidelity_provenance(
            {}, {"motion_fidelity": provenance},
            authored_motion_fingerprint="d" * 64,
        )


def test_ab_without_reference_describes_change_but_cannot_claim_improvement():
    report = compare_recording_overviews(
        _overview(center=2), _overview(center=3),
        before_provenance=_provenance("a"),
        after_provenance=_provenance("d"),
    )

    assert report["assessment"]["status"] == "comparison_only"
    assert report["assessment"]["can_claim_radar_improvement"] is False
    assert report["online_gate_effect"] == "none"
    assert len(report["comparison_id"]) == 64
    assert report["metrics"]["after_minus_before"][
        "mean_abs_doppler_velocity_m_s"] > 0.0


def test_ab_claims_improvement_only_when_after_is_closer_to_reference():
    reference = _overview(center=3, energy=(0.2, 0.6, 1.0))
    before = _overview(center=1, energy=(1.0, 0.2, 0.1))
    after = _overview(center=3, energy=(0.2, 0.6, 1.0))

    report = compare_recording_overviews(
        before, after,
        before_provenance=_provenance("a"),
        after_provenance=_provenance("d"),
        reference=reference,
        reference_id="trusted-cat-session-17",
    )

    assert report["assessment"]["status"] == "closer_to_reference"
    assert report["assessment"]["can_claim_radar_improvement"] is True
    assert report["reference"]["after_distance"]["combined_distance"] == 0.0


def test_ab_rejects_different_radar_axes_or_intervals():
    after = _overview()
    after.times_s = np.array([0.0, 0.2, 0.4])

    with pytest.raises(ValueError, match="identical times_s"):
        compare_recording_overviews(
            _overview(), after,
            before_provenance=_provenance("a"),
            after_provenance=_provenance("d"),
        )


def test_provenance_refuses_unverifiable_identifiers():
    with pytest.raises(ValueError, match="report id"):
        normalize_fidelity_provenance({
            **_provenance("a"), "report_id": "not-a-hash",
        })
