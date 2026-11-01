"""Compare two provenance-bound Radar recordings for motion-fidelity evidence."""

import argparse
import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from wt_radar.adapter.fidelity_ab import compare_recording_overviews
from wt_radar.adapter.recording_overview import build_recording_overview
from wt_radar.adapter.saved_result import SavedRadarResult


def _load(path, tx, rx):
    result = SavedRadarResult.load(path)
    provenance = result.producer.get("motion_fidelity")
    if not isinstance(provenance, dict):
        raise ValueError(
            f"{path} has no embedded motion_fidelity provenance; rerun Radar "
            "with final_motion_fidelity supplied to submit_animation_measurement."
        )
    return build_recording_overview(result, tx, rx), provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--reference-id")
    parser.add_argument("--tx", type=int, default=0)
    parser.add_argument("--rx", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to replace an existing A/B report.")
    if bool(args.reference) != bool(args.reference_id):
        raise ValueError("--reference and --reference-id must be supplied together.")

    before, before_provenance = _load(args.before, args.tx, args.rx)
    after, after_provenance = _load(args.after, args.tx, args.rx)
    reference = None
    if args.reference:
        saved_reference = SavedRadarResult.load(args.reference)
        reference = build_recording_overview(saved_reference, args.tx, args.rx)
    report = compare_recording_overviews(
        before,
        after,
        before_provenance=before_provenance,
        after_provenance=after_provenance,
        reference=reference,
        reference_id=args.reference_id,
    )
    report["sources"] = {
        "before": str(args.before.resolve()),
        "after": str(args.after.resolve()),
        "reference": str(args.reference.resolve()) if args.reference else None,
        "tx": args.tx,
        "rx": args.rx,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {report['assessment']['status']} A/B evidence "
        f"{report['comparison_id']} to {args.output}"
    )


if __name__ == "__main__":
    main()
