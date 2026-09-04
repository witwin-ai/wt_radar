"""Create an isolated Studio review project; never overwrite an existing scene."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from witwin_server import Scene, SceneObject
from witwin_server.core.wtscene import dumps
from witwin_server.projects import ensure_witwin_project
from wt_radar.adapter.saved_result import SavedRadarResult
from wt_radar.components.radar import RadarComponent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    result = SavedRadarResult.load(args.result)
    scene_path = args.project / "scenes" / "main.wtscene"
    if scene_path.exists():
        raise FileExistsError(f"Refusing to replace {scene_path}")
    ensure_witwin_project(args.project)
    scene = Scene(scene_id="main", name="Saved Radar Result Review")
    obj = SceneObject(id="radar-result-review", name="bedroom_004 GPU Result", mesh_type="Empty")
    component = obj.add_component(RadarComponent())
    component.saved_result_path = str(result.path)
    component.view = "range_profile"
    scene.add_object(obj)
    scene_path.parent.mkdir(parents=True, exist_ok=True)
    scene_path.write_text(dumps(scene.to_dict()), encoding="utf-8")
    print(f"Created {scene_path}; select bedroom_004 GPU Result and click Load Saved Result.")


if __name__ == "__main__":
    main()
