"""Run the supported Radar 0.4 animation path against a saved Studio scene."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


class Context:
    def __init__(self):
        self._last_message = ""

    def log(self, message):
        print(message, flush=True)

    def progress(self, fraction, message):
        if message != self._last_message:
            print(f"[{float(fraction):.0%}] {message}", flush=True)
            self._last_message = message

    def throw_if_cancelled(self):
        return None

    def result_ref(self, payload, **_kwargs):
        return bytes(payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("project", type=Path)
    parser.add_argument("scene_id")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    project = args.project.resolve()
    os.environ["WITWIN_PROJECT_DIR"] = str(project)

    import wt_radar  # noqa: F401 - register component schemas before scene load
    from witwin_server.persistence import PersistenceManager
    from witwin_server.features.platform.bridge.offline_scene import hydrate_project_meshes
    from witwin_server.features.assets.imported.gltf_import_handler import GLTFImportHandler
    from witwin_server.features.assets.imported.skinned_mesh_loader import (
        hydrate_skinned_mesh,
        needs_skin_hydration,
    )
    from wt_radar.adapter.animation import animation_request, export_animation, solve_animation

    scene = PersistenceManager(object()).load_scene_by_id(args.scene_id)
    if scene is None:
        raise FileNotFoundError(f"Scene not found: {args.scene_id}")
    hydration = hydrate_project_meshes(scene, project)
    print(
        f"Hydrated {hydration.hydrated_meshes}/{hydration.project_meshes} project meshes",
        flush=True,
    )
    gltf_reader = object.__new__(GLTFImportHandler)
    for obj in scene.objects.values():
        skin = obj.get_component("SkinnedMesh")
        if skin is not None and needs_skin_hydration(skin):
            hydrate_skinned_mesh(skin, obj.mesh_path, project, gltf_reader)
    radar_objects = [obj for obj in scene.objects.values() if obj.get_component("Radar") is not None]
    if len(radar_objects) != 1:
        raise RuntimeError(f"Expected exactly one Radar object, found {len(radar_objects)}")
    component = radar_objects[0].get_component("Radar")
    request = animation_request(component)

    started = time.monotonic()
    ctx = Context()
    result = solve_animation(ctx, scene, request)
    payload = export_animation(ctx, result)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(payload)

    cube = result.cube.numpy()
    frame_nonzero = np.any(np.abs(cube) > 0, axis=tuple(range(1, cube.ndim)))
    environment = dict(result.metadata.get("environment_reflection") or {})
    target = dict(result.metadata.get("target_propagation") or {})
    topology = dict(result.metadata.get("topology_preflight") or {})
    summary = {
        "scene_id": scene.scene_id,
        "output": str(output),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "elapsed_s": round(time.monotonic() - started, 3),
        "cube_shape": list(cube.shape),
        "all_finite": bool(np.isfinite(cube).all()),
        "nonzero_frames": int(np.count_nonzero(frame_nonzero)),
        "frame_count": int(cube.shape[0]),
        "active_site_count": int(topology.get("active_site_count", 0)),
        "target_multipath_enabled": bool(target.get("multipath_enabled")),
        "target_max_depth": int(target.get("max_depth", -1)),
        "target_motion_sampling": target.get("motion_sampling"),
        "reflected_path_count": int(environment.get("reflected_path_count", 0)),
        "material_slot_count": int(environment.get("material_slot_count", 0)),
        "environment_model": environment.get("model"),
        "max_depth": environment.get("max_depth"),
    }
    print(json.dumps(summary, indent=2), flush=True)
    accepted = (
        summary["all_finite"]
        and summary["nonzero_frames"] == summary["frame_count"]
        and summary["active_site_count"] > 0
        and not summary["target_multipath_enabled"]
        and summary["target_max_depth"] == 0
        and summary["target_motion_sampling"] == "chirp"
        and summary["reflected_path_count"] > 0
        and summary["material_slot_count"] > 0
        and summary["max_depth"] == 1
    )
    return 0 if accepted else 2


if __name__ == "__main__":
    sys.exit(main())
