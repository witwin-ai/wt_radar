import pytest


def test_throughput_summary_groups_channels_by_frame_timestamp():
    from wt_radar.tools.live_stream_throughput import PublishRecord, summarize_records

    records = [
        PublishRecord(1.00, 10.00, "radar.test.signal", "raw", 8, (2,), 1000),
        PublishRecord(1.01, 10.01, "radar.test.signal", "rd", 16, (2, 2), 1000),
        PublishRecord(1.10, 10.10, "radar.test.signal", "raw", 8, (2,), 2000),
        PublishRecord(1.11, 10.11, "radar.test.signal", "rd", 16, (2, 2), 2000),
        PublishRecord(1.20, 10.20, "radar.test.signal", "raw", 8, (2,), 3000),
        PublishRecord(1.21, 10.21, "radar.test.signal", "rd", 16, (2, 2), 3000),
    ]

    summary = summarize_records(
        records,
        solve_ms=[120.0, 30.0, 31.0],
        post_ms={"rd": [3.0, 4.0, 5.0]},
        warmup_frames=1,
    )

    assert summary["frames_total"] == 3
    assert summary["frames_measured"] == 2
    assert summary["fps"] == pytest.approx(10.0)
    assert summary["payload_bytes_per_frame_mean"] == pytest.approx(24.0)
    assert summary["payload_bytes_per_second"] == pytest.approx(240.0)
    assert summary["solve_ms"]["mean"] == pytest.approx(30.5)
    assert summary["post_ms"]["rd"]["mean"] == pytest.approx(4.5)
