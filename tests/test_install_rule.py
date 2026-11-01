"""Radar installation rule (design 7.1-7.4): room geometry only, no actors chased."""
import math

import numpy as np
import pytest

from witwin_server import Scene, SceneObject

from wt_radar import install_rule
from wt_radar.install_rule import InstallRule, InstallRuleError
from wt_radar.los import StaticOccluders


def add_box(scene, object_id, lo, hi, *, name=None, parent=None):
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    obj = SceneObject(id=object_id, name=name or object_id, mesh_type="Cube")
    obj.get_component("Transform").position = ((lo + hi) / 2).tolist()
    obj.get_component("Transform").scale = (hi - lo).tolist()
    scene.add_object(obj)
    if parent is not None:
        scene.set_parent(object_id, parent, keep_local_transform=False)
    return obj


def box_room(width=5.0, depth=4.0, *, height=2.5, scene_id="install-room"):
    """A closed room: floor top at y=0, 0.1 m walls centred on the room bounds."""
    scene = Scene(scene_id=scene_id, name=scene_id)
    add_box(scene, "floor", [0, -.1, 0], [width, 0, depth])
    add_box(scene, "wall_south", [0, 0, -.05], [width, height, .05])
    add_box(scene, "wall_north", [0, 0, depth - .05], [width, height, depth + .05])
    add_box(scene, "wall_west", [-.05, 0, 0], [.05, height, depth])
    add_box(scene, "wall_east", [width - .05, 0, 0], [width + .05, height, depth])
    return scene


def local_angles(mount, point):
    """Azimuth/elevation of a world point in the mount's antenna axes."""
    forward = np.asarray(mount["forward"], dtype=float)
    right = np.cross(forward, [0., 1., 0.])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    v = np.asarray(point, dtype=float) - np.asarray(mount["position_m"], dtype=float)
    z = v @ forward
    return math.degrees(math.atan2(v @ right, z)), math.degrees(math.atan2(v @ up, z))


def test_mount_looking_straight_across_the_room_is_rejected_for_the_opposite_wall():
    geometry = install_rule.room_geometry_from_scene(box_room())
    rule = InstallRule()
    straight = install_rule.make_mount(
        position=[2.5, 2.0, .1], heading_xz=[0., 1.], tilt_deg=15., style="wall", wall="south")
    verdict = install_rule.planar_normal_check(geometry, straight, rule)
    assert not verdict["ok"]
    assert verdict["surface"]["object_id"] == "wall_north"
    assert abs(verdict["worst_normal_angle_deg"]) < 35.

    yawed = install_rule.make_mount(
        position=[.6, 2.0, .1], heading_xz=[math.sin(math.radians(40)), math.cos(math.radians(40))],
        tilt_deg=15., style="wall", wall="south")
    assert install_rule.planar_normal_check(geometry, yawed, rule)["ok"]


def test_floor_passes_below_55_degrees_tilt_and_fails_above():
    geometry = install_rule.room_geometry_from_scene(box_room())
    rule = InstallRule()
    heading = [math.sin(math.radians(40)), math.cos(math.radians(40))]
    shallow = install_rule.make_mount(position=[.6, 2., .1], heading_xz=heading, tilt_deg=50.)
    steep = install_rule.make_mount(position=[.6, 2., .1], heading_xz=heading, tilt_deg=60.)
    assert install_rule.planar_normal_check(geometry, shallow, rule)["ok"]
    verdict = install_rule.planar_normal_check(geometry, steep, rule)
    assert not verdict["ok"] and verdict["surface"]["object_id"] == "floor"


def test_installation_is_a_wall_or_corner_mount_at_two_metres_tilted_down_with_no_normal_in_beam():
    scene = box_room()
    result = install_rule.plan_installation(scene)
    mount = result["mount"]
    assert mount["style"] in {"wall", "corner"}
    assert mount["position_m"][1] == pytest.approx(2.0)
    assert 5. < mount["tilt_deg"] < 55.
    # The mount sits on the inner wall face plus the stand-off.
    x, _, z = mount["position_m"]
    assert min(x - .05, 4.95 - x, z - .05, 3.95 - z) == pytest.approx(.05, abs=1e-6)
    assert abs(mount["worst_normal_angle_deg"]) >= 35.
    # The planner's own rejected list proves the straight-across mounts were tried.
    reasons = {item["reason"] for item in result["rejected_mounts"]}
    assert "planar_normal_in_beam" in reasons
    # Proposed ensure_sensor pose: boresight crosses aim height 0.5 m at the aim point.
    assert result["aim_point_m"][1] == pytest.approx(.5, abs=1e-6)
    assert result["rule"]["height_m"] == 2.0


def test_installation_is_deterministic():
    first = install_rule.plan_installation(box_room())
    second = install_rule.plan_installation(box_room())
    assert first["mount"] == second["mount"]
    assert first["zones"]["person"]["area_m2"] == second["zones"]["person"]["area_m2"]


def test_large_furniture_face_in_the_beam_rejects_the_mount():
    scene = box_room()
    # A wardrobe front (1.6 x 2.2 m) facing straight back at a mount on the west wall.
    add_box(scene, "wardrobe", [4.3, 0, 1.2], [4.9, 2.2, 2.8], name="wardrobe")
    geometry = install_rule.room_geometry_from_scene(scene)
    mount = install_rule.make_mount(position=[.1, 2., 2.], heading_xz=[1., 0.], tilt_deg=20.)
    verdict = install_rule.planar_normal_check(geometry, mount, InstallRule())
    # The east wall is hidden behind the wardrobe, so the wardrobe front is the offender.
    assert not verdict["ok"]
    assert verdict["surface"]["object_id"] == "wardrobe"


def test_sensing_zone_is_line_of_sight_floor_inside_the_narrowed_beam_and_range():
    scene = box_room()
    add_box(scene, "cabinet", [2.2, 0, 1.8], [2.8, 1.6, 2.2], name="cabinet")
    geometry = install_rule.room_geometry_from_scene(scene)
    rule = InstallRule()
    mount = install_rule.make_mount(
        position=[.1, 2., .1], heading_xz=[math.sqrt(.5), math.sqrt(.5)], tilt_deg=25., style="corner")
    zones = install_rule.sensing_zones(geometry, mount, rule)
    person = zones["person"]
    # Inside the cabinet footprint: never in the zone.
    assert not person.contains(2.5, 2.0)
    # Directly behind the cabinet from the corner, at cat height: occluded.
    assert not zones["cat"].contains(3.2, 2.6)
    # Within 1 m of the mount: out of range.
    assert not person.contains(.5, .5)
    # Open floor on the boresight about 3 m out: in both zones.
    assert person.contains(2.4, 1.5) and zones["cat"].contains(2.4, 1.5)
    # Outside the 25 degree narrowed beam on azimuth.
    assert not person.contains(4.5, .4)
    assert 0. < person.area_m2 < 20.
    payload = person.to_dict()
    assert payload["cell_m"] == pytest.approx(.1)
    assert payload["polygon_xz"]


def test_point_in_zone_uses_the_support_height_for_events_on_furniture():
    scene = box_room()
    add_box(scene, "bed", [3.0, 0, 2.4], [4.95, .45, 3.95], name="bed")
    geometry = install_rule.room_geometry_from_scene(scene)
    mount = install_rule.make_mount(
        position=[.1, 2., .1], heading_xz=[math.sqrt(.5), math.sqrt(.5)], tilt_deg=25., style="corner")
    on_bed = {"id": "cat_on_bed", "actor_kind": "cat", "position_m": [3.6, .45, 3.0]}
    check = install_rule.event_in_view(geometry, mount, on_bed, InstallRule())
    assert check["in_zone"], check
    assert check["test_point_m"][1] == pytest.approx(.45 + .25)


def test_room_with_most_key_events_is_chosen_and_missed_event_is_reported():
    scene = box_room(width=8.0)
    # Interior wall splits the apartment into x<4 and x>4, with a doorway at z in [1.5, 2.5].
    add_box(scene, "wall_interior_1a", [3.95, 0, 0], [4.05, 2.5, 1.5])
    add_box(scene, "wall_interior_1b", [3.95, 0, 2.5], [4.05, 2.5, 4.0])
    geometry = install_rule.room_geometry_from_scene(scene, rooms=[
        {"index": 0, "type": "living_room", "bounds": [0, 0, 4, 4]},
        {"index": 1, "type": "bedroom", "bounds": [4, 0, 8, 4]},
    ], doorways=[{"position_xz": [4, 2], "width_m": 1.0, "axis": "z"}])
    events = [
        {"id": "person_stop", "actor_kind": "person", "position_m": [6.0, 0, 2.0]},
        {"id": "cat_land", "actor_kind": "cat", "position_m": [6.5, 0, 1.0]},
        {"id": "person_start", "actor_kind": "person", "position_m": [1.0, 0, 1.0]},
    ]
    result = install_rule.plan_installation(scene, geometry=geometry, key_events=events, require_all_events=False)
    assert result["room"]["index"] == 1
    assert 4. <= result["mount"]["position_m"][0] <= 8.
    by_id = {item["id"]: item for item in result["key_events"]}
    assert by_id["person_stop"]["in_zone"] and by_id["cat_land"]["in_zone"]
    # The living-room event cannot be seen through the wall from the bedroom.
    assert not by_id["person_start"]["in_zone"]
    with pytest.raises(InstallRuleError) as raised:
        install_rule.plan_installation(scene, geometry=geometry, key_events=events)
    assert raised.value.code == "RADAR_ZONE_MISSES_EVENT"
    detail = raised.value.detail
    assert detail["missed_events"][0]["id"] == "person_start"
    assert detail["best_mount"]["position_m"]
    assert detail["missed_events"][0]["miss_distance_m"] > 0


def test_candidates_avoid_openings_corners_and_tall_furniture_at_the_wall():
    scene = box_room()
    add_box(scene, "bookshelf", [1.5, 0, 3.55], [2.7, 2.1, 3.95], name="bookshelf")
    geometry = install_rule.room_geometry_from_scene(scene, doorways=[
        {"position_xz": [3.5, 0.0], "width_m": .9, "axis": "x"}])
    candidates, rejected = install_rule.mount_positions(geometry, geometry.rooms[0], InstallRule())
    walls = [item for item in candidates if item["style"] == "wall"]
    assert walls
    for item in walls:
        x, _, z = item["position_m"]
        assert min(x, 5 - x, z, 4 - z) >= .05 - 1e-9
        along = x if item["wall"] in {"south", "north"} else z
        span = 5. if item["wall"] in {"south", "north"} else 4.
        assert .3 + .05 - 1e-9 <= along <= span - .3 - .05 + 1e-9
        if item["wall"] == "south":
            assert abs(x - 3.5) >= .45 + .3 - 1e-9
        if item["wall"] == "north":
            assert not 1.5 - .3 < x < 2.7 + .3
    reasons = {item["reason"] for item in rejected}
    assert {"opening", "tall_furniture"} <= reasons
    assert len([item for item in candidates if item["style"] == "corner"]) == 4


def test_actor_path_under_a_low_mount_is_rejected():
    scene = box_room()
    geometry = install_rule.room_geometry_from_scene(scene)
    rule = InstallRule(height_m=1.6)
    path = [{"object_id": "person", "points_xz": [[x, .3] for x in np.linspace(.5, 4.5, 20)], "height_m": 1.75}]
    candidates, rejected = install_rule.mount_positions(geometry, geometry.rooms[0], rule, actor_paths=path)
    assert not any(item["wall"] == "south" for item in candidates if item["style"] == "wall")
    assert "actor_path" in {item["reason"] for item in rejected}


def test_no_site_raises_radar_no_install_site():
    # A pattern this wide sees some wall at normal incidence from every mount.
    with pytest.raises(InstallRuleError) as raised:
        install_rule.plan_installation(box_room(), rule=InstallRule(beam_half_deg=80.))
    assert raised.value.code == "RADAR_NO_INSTALL_SITE"
    assert raised.value.detail["walls"]


def test_occluders_ignore_hits_at_the_segment_end():
    scene = box_room()
    add_box(scene, "bed", [3.0, 0, 2.4], [4.95, .45, 3.95], name="bed")
    occluders = StaticOccluders.from_scene(scene)
    origin = np.array([.1, 2., .1])
    points = np.array([[3.5, .45, 3.0], [3.5, .2, 3.0], [2.0, .5, 2.0]])
    blocked = occluders.segments_blocked(origin, points)
    assert blocked.tolist() == [False, True, False]
