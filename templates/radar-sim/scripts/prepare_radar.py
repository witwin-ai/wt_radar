"""Check and print the Radar setup of scenes/main.wtscene.

    python scripts/prepare_radar.py                  # print the setup
    python scripts/prepare_radar.py --fc-ghz 60      # change the carrier, save
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import witwin.studio as ws

PROJECT = Path(__file__).resolve().parents[1]


def find_radar(scene: ws.Scene) -> ws.Entity:
    for entity in scene.objects.values():
        if "Radar" in entity.components:
            return entity
    raise SystemExit("no entity with a Radar component in the main scene")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fc-ghz", type=float, help="new carrier frequency in GHz")
    args = parser.parse_args(argv)

    project = ws.open_project(PROJECT)
    scene = project.scene()
    radar = find_radar(scene)
    cfg = radar.components["Radar"]

    if args.fc_ghz is not None:
        cfg["fc"] = args.fc_ghz * 1e9
        scene.save()
        print(f"saved: fc = {args.fc_ghz} GHz")

    print(f"Project {project.name}, scene {scene.id}: radar {radar.name!r}")
    print(f"  carrier       {cfg['fc'] / 1e9:.2f} GHz")
    print(f"  chirp slope   {cfg['slope']:.3f} MHz/us")
    print(f"  ADC           {int(cfg['adc_samples'])} samples at {cfg['sample_rate']} ksps")
    print(f"  frame         {int(cfg['chirp_per_frame'])} chirps, {cfg['frame_per_second']} frames/s")
    print(f"  antennas      {int(cfg['num_tx'])} Tx x {int(cfg['num_rx'])} Rx")

    target_id = cfg.get("snapshot_target_id")
    target = scene.objects.get(target_id) if target_id else None
    if target is None:
        print(f"  snapshot target {target_id!r} is not in the scene", file=sys.stderr)
        return 1
    distance = math.dist(radar.transform.position, target.transform.position)
    print(f"  target        {target.name!r} at {distance:.2f} m from the sensor")
    print("Ready to solve in Studio (Run Static Snapshot).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
