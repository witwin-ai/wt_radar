"""Export the official Range Profile selection as NPZ and CSV, without DSP changes."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from wt_radar.adapter.saved_result import SavedRadarResult


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tx", type=int, default=0)
    parser.add_argument("--rx", type=int, default=0)
    parser.add_argument("--chirp", type=int, default=0)
    args = parser.parse_args()
    csv_path = args.output.with_suffix(".csv")
    if args.output.exists() or csv_path.exists():
        raise FileExistsError("Refusing to replace an existing NPZ/CSV export.")
    result = SavedRadarResult.load(args.input)
    profiles = np.stack([
        result.selected_profile(i, args.tx, args.rx, args.chirp)[1].numpy()
        for i in range(len(result.times_s))
    ])
    ranges = result.axes.range_m.numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, range_profile_complex=profiles, amplitude=np.abs(profiles),
        range_m=ranges, times_s=result.times_s,
        tx=np.asarray(args.tx), rx=np.asarray(args.rx), chirp=np.asarray(args.chirp),
        source_file=np.asarray(str(result.path)),
        processing=np.asarray("witwin.radar.processing.range_profile(window='rectangular', remove_dc=False)"),
        producer_metadata_json=np.asarray(json.dumps(result.producer)),
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame", "time_s", "tx", "rx", "chirp", "range_m", "real", "imag", "amplitude"])
        for index, (time_s, profile) in enumerate(zip(result.times_s, profiles)):
            for range_m, value in zip(ranges, profile):
                writer.writerow([index, time_s, args.tx, args.rx, args.chirp, range_m, value.real, value.imag, abs(value)])
    print(f"Exported {profiles.shape}: {args.output} and {csv_path}")


if __name__ == "__main__":
    main()
