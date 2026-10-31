"""Radar installation rule (design 7.1-7.4): mount from room geometry only.

A wall or corner mount at a fixed height, tilted down, chosen the way an
installer would: no wall, floor or large furniture face may meet the beam at
normal incidence (the static bright spot of meeting sample 03), and the mount
must not sit in an opening, inside furniture or in an actor's path. The mount
then defines a sensing zone, the floor area the Radar sees by static line of
sight, which the choreography uses to place key events. Actors never move the
Radar.

Geometry and line of sight are a Studio-side approximation (``studio_los_v1``);
the native simulation keeps its own path discovery.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Iterable

import numpy as np

from .los import LOS_METHOD, StaticMesh, StaticOccluders, static_meshes


INSTALL_STYLES = ("auto", "wall", "corner")
WALL_STEP_M = .25
CORNER_CLEARANCE_M = .3
OPENING_CLEARANCE_M = .3
FURNITURE_CLEARANCE_M = .3
ACTOR_PATH_CLEARANCE_M = .3
FURNITURE_SHELL_M = .05
WALL_YAW_SPAN_DEG, WALL_YAW_STEP_DEG = 40., 5.
CORNER_YAW_SPAN_DEG = 20.
ZONE_CELL_M = .1
ZONE_BEAM_MARGIN_DEG = 5.
# Static line-of-sight heights of the two sensing zones: cat body and person torso.
ZONE_HEIGHTS_M = {"cat": .25, "person": 1.}
MIN_PLANE_AREA_M2 = .5
FLOOR_COVER_HEIGHT_M = .05


class InstallRuleError(Exception):
    """A blocked installation with a scenario error code and its evidence."""

    def __init__(self, code, message, *, next_step, detail=None):
        super().__init__(message)
        self.code = code
        self.next_step = next_step
        self.detail = detail or {}

    def to_dict(self):
        return {"code": self.code, "message": str(self), "next_step": self.next_step,
                "detail": self.detail}


@dataclass(frozen=True)
class InstallRule:
    """``radar.rule`` of ``witwin.scenario/v1`` (design 3.6)."""

    style: str = "auto"
    height_m: float = 2.0
    aim_height_m: float = .5
    beam_half_deg: float = 30.
    normal_margin_deg: float = 5.
    range_m: tuple = (1., 6.)
    wall_standoff_m: float = .05

    def __post_init__(self):
        if self.style not in INSTALL_STYLES:
            raise ValueError(f"install style must be one of {INSTALL_STYLES}, got {self.style!r}")
        low, high = (float(value) for value in self.range_m)
        object.__setattr__(self, "range_m", (low, high))
        checks = {
            "height_m": .2 <= self.height_m <= 3.,
            "aim_height_m": 0. <= self.aim_height_m < self.height_m,
            "beam_half_deg": 1. <= self.beam_half_deg <= 89.,
            "normal_margin_deg": 0. <= self.normal_margin_deg <= 30.,
            "range_m": 0. <= low < high <= 30.,
            "wall_standoff_m": 0. <= self.wall_standoff_m <= .5,
        }
        bad = [name for name, ok in checks.items() if not ok]
        if bad:
            raise ValueError("invalid installation rule fields: " + ", ".join(bad))

    @classmethod
    def from_dict(cls, value=None):
        value = dict(value or {})
        unknown = set(value) - {item for item in cls.__dataclass_fields__}
        if unknown:
            raise ValueError("unknown installation rule fields: " + ", ".join(sorted(unknown)))
        if "range_m" in value:
            value["range_m"] = tuple(value["range_m"])
        return cls(**value)

    @property
    def exclusion_deg(self):
        return self.beam_half_deg + self.normal_margin_deg

    @property
    def zone_half_deg(self):
        return self.beam_half_deg - ZONE_BEAM_MARGIN_DEG

    def to_dict(self):
        payload = asdict(self)
        payload["range_m"] = list(self.range_m)
        return payload


# ----------------------------------------------------------------- geometry


@dataclass
class Room:
    index: int
    type: str
    bounds: tuple  # x0, z0, x1, z1 (wall centre lines)


@dataclass
class Opening:
    """A door or doorway: centre on a wall, width along the wall."""

    position_xz: tuple
    width_m: float
    axis: str  # "x": the wall runs along x (constant z); "z": along z (constant x)


@dataclass
class Plane:
    object_id: str
    name: str
    kind: str
    normal: np.ndarray
    offset: float
    triangles: np.ndarray
    area_m2: float


@dataclass
class RoomGeometry:
    rooms: list
    openings: list
    meshes: list
    kinds: dict
    owners: dict
    floor_y: float
    occluders: StaticOccluders
    planes: list
    _faces: dict = field(default_factory=dict, repr=False)
    _room_occluders: dict = field(default_factory=dict, repr=False)

    def room(self, index):
        for room in self.rooms:
            if room.index == index:
                return room
        raise KeyError(index)

    def room_of_point(self, x, z):
        best = None
        for room in self.rooms:
            x0, z0, x1, z1 = room.bounds
            outside = max(x0 - x, 0., x - x1) + max(z0 - z, 0., z - z1)
            if best is None or outside < best[0]:
                best = (outside, room)
        return best[1]

    def mesh_kind(self, object_id):
        return self.kinds.get(str(object_id), "furniture")

    def furniture(self):
        return [mesh for mesh in self.meshes if self.kinds[mesh.object_id] == "furniture"]

    def walls(self):
        return [mesh for mesh in self.meshes if self.kinds[mesh.object_id] == "wall"]

    def inner_faces(self, room):
        """Inner wall face coordinates of a room: {side: coordinate}."""
        if room.index not in self._faces:
            self._faces[room.index] = _inner_faces(self, room)
        return self._faces[room.index]

    def room_occluders(self, room):
        """Meshes that can cut a segment kept inside this room's walls."""
        if room.index not in self._room_occluders:
            x0, z0, x1, z1 = room.bounds
            kept = [mesh for mesh in self.meshes
                    if mesh.hi[0] >= x0 - .2 and mesh.lo[0] <= x1 + .2
                    and mesh.hi[2] >= z0 - .2 and mesh.lo[2] <= z1 + .2]
            self._room_occluders[room.index] = StaticOccluders(kept)
        return self._room_occluders[room.index]


def _owner_names(scene):
    """Display name of each object's furniture owner (the StructureMeta ancestor)."""
    owners = {}
    for object_id, obj in scene.objects.items():
        current, name = obj, str(obj.name)
        seen = set()
        while current is not None and str(current.id) not in seen:
            seen.add(str(current.id))
            if current.get_component("StructureMeta") is not None:
                name = str(current.name)
                break
            parent_id = getattr(current, "parent_id", None)
            current = scene.get_object(parent_id) if parent_id and parent_id != "root" else None
        owners[str(object_id)] = name
    return owners


def _classify(mesh, name, floor_top):
    lowered = name.lower()
    size = mesh.hi - mesh.lo
    if lowered.startswith("floor"):
        return "floor"
    if lowered.startswith("ceiling"):
        return "ceiling"
    if lowered.startswith("wall"):
        return "wall"
    thin_horizontal = size[1] <= .25 and size[0] * size[2] >= 4.
    if thin_horizontal and mesh.hi[1] <= floor_top + .1:
        return "floor"
    if size[1] >= 1.5 and min(size[0], size[2]) <= .3 and max(size[0], size[2]) >= 1.:
        return "wall"
    if mesh.hi[1] <= floor_top + FLOOR_COVER_HEIGHT_M:
        return "floor_cover"
    return "furniture"


def _planes(meshes, kinds, owners):
    planes = []
    for mesh in meshes:
        triangles = mesh.triangles()
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        doubled = np.linalg.norm(cross, axis=1)
        keep = doubled > 1e-10
        if not keep.any():
            continue
        triangles, cross, doubled = triangles[keep], cross[keep], doubled[keep]
        normals = cross / doubled[:, None]
        offsets = np.sum(normals * triangles[:, 0], axis=1)
        keys = np.column_stack((np.round(normals * 50.), np.round(offsets * 50.)))
        _, inverse = np.unique(keys, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        for group in range(int(inverse.max()) + 1):
            rows = np.flatnonzero(inverse == group)
            area = float(doubled[rows].sum() / 2.)
            if area < MIN_PLANE_AREA_M2:
                continue
            normal = normals[rows].mean(axis=0)
            normal /= np.linalg.norm(normal)
            planes.append(Plane(
                mesh.object_id, owners.get(mesh.object_id, mesh.name), kinds[mesh.object_id],
                normal, float(np.mean(np.sum(triangles[rows, 0] * normal, axis=1))),
                triangles[rows], area))
    return planes


def _room_generation(scene):
    for obj in scene.objects.values():
        component = obj.get_component("RoomGeneration")
        if component is not None:
            return obj, component
    return None, None


def _component_value(component, name, default=None):
    value = getattr(component, name, None)
    if value is None and isinstance(getattr(component, "properties", None), dict):
        value = component.properties.get(name)
    return default if value is None else value


def _normalize_openings(doorways):
    openings = []
    for item in doorways or ():
        if isinstance(item, Opening):
            openings.append(item)
        elif isinstance(item, dict):
            openings.append(Opening(tuple(float(v) for v in item["position_xz"]),
                                    float(item["width_m"]), str(item["axis"])))
        else:
            # Roomgen doorway rows: [x, z, width, orientation]; orientation 0 is
            # an interior wall at constant x, 1 a wall at constant z.
            x, z, width, orientation = (float(value) for value in list(item)[:4])
            openings.append(Opening((x, z), width, "z" if orientation < .5 else "x"))
    return openings


def room_geometry_from_scene(scene, *, rooms=None, doorways=None, excluded_ids=()):
    """Static room geometry, rooms and openings; actors and Radar objects are skipped."""
    excluded = {str(value) for value in excluded_ids}
    for object_id, obj in scene.objects.items():
        if obj.get_component("Radar") is not None:
            excluded.add(str(object_id))
    meshes = static_meshes(scene, excluded_ids=excluded)
    owners = _owner_names(scene)
    floor_top = 0.
    flats = [mesh for mesh in meshes if (mesh.hi - mesh.lo)[1] <= .25
             and (mesh.hi - mesh.lo)[0] * (mesh.hi - mesh.lo)[2] >= 1.]
    named_floor = [mesh for mesh in meshes if mesh.name.lower().startswith("floor")]
    if named_floor or flats:
        floor = max(named_floor or flats, key=lambda m: (m.hi - m.lo)[0] * (m.hi - m.lo)[2])
        floor_top = float(floor.hi[1])
    kinds = {mesh.object_id: _classify(mesh, mesh.name, floor_top) for mesh in meshes}

    owner, component = _room_generation(scene)
    if rooms is None and component is not None:
        bounds = _component_value(component, "room_bounds", []) or []
        types = list(_component_value(component, "room_types", []) or [])
        rooms = [{"index": index, "type": types[index] if index < len(types) else "room",
                  "bounds": [float(v) for v in item]} for index, item in enumerate(bounds)]
        if owner is not None and rooms:
            rooms = _transform_rooms(scene, owner, rooms)
    if doorways is None and component is not None:
        doorways = list(_component_value(component, "doorways", []) or [])
        door = _component_value(component, "front_door")
        wall = str(_component_value(component, "front_door_wall", "") or "")
        if door is not None and len(door) >= 2:
            axis = "x" if wall in {"south", "north"} else "z"
            doorways.append({"position_xz": [float(door[0]), float(door[1])],
                             "width_m": float(_component_value(component, "front_door_width_m", .9)),
                             "axis": axis})
    if not rooms:
        floors = [mesh for mesh in meshes if kinds[mesh.object_id] == "floor"]
        if not floors:
            raise InstallRuleError(
                "RADAR_NO_INSTALL_SITE", "The scene has no floor or room bounds to install a Radar in.",
                next_step="Generate or open a room before installing the Radar.")
        floor = max(floors, key=lambda m: (m.hi - m.lo)[0] * (m.hi - m.lo)[2])
        rooms = [{"index": 0, "type": "room",
                  "bounds": [float(floor.lo[0]), float(floor.lo[2]), float(floor.hi[0]), float(floor.hi[2])]}]
    room_list = [Room(int(item["index"]), str(item.get("type", "room")),
                      tuple(float(v) for v in item["bounds"])) for item in rooms]
    return RoomGeometry(
        rooms=room_list, openings=_normalize_openings(doorways), meshes=meshes, kinds=kinds,
        owners=owners, floor_y=floor_top, occluders=StaticOccluders(meshes),
        planes=_planes(meshes, kinds, owners))


def _transform_rooms(scene, owner, rooms):
    from .adapter.snapshot import _hierarchy

    matrices, _ = _hierarchy(scene)
    matrix = np.asarray(matrices[str(owner.id)], dtype=np.float64)
    if np.allclose(matrix, np.eye(4)):
        return rooms
    out = []
    for item in rooms:
        x0, z0, x1, z1 = item["bounds"]
        corners = np.array([[x0, 0, z0, 1], [x1, 0, z0, 1], [x0, 0, z1, 1], [x1, 0, z1, 1]]) @ matrix.T
        out.append({**item, "bounds": [float(corners[:, 0].min()), float(corners[:, 2].min()),
                                       float(corners[:, 0].max()), float(corners[:, 2].max())]})
    return out


# Each side: fixed axis, the coordinate index along the wall, inward normal.
_SIDES = {
    "south": (2, 0, (0., 1.)),
    "north": (2, 0, (0., -1.)),
    "west": (0, 2, (1., 0.)),
    "east": (0, 2, (-1., 0.)),
}


def _side_line(room, side):
    x0, z0, x1, z1 = room.bounds
    return {"south": z0, "north": z1, "west": x0, "east": x1}[side]


def _side_walls(geometry, room, side):
    fixed_axis, along_axis, _ = _SIDES[side]
    line = _side_line(room, side)
    lo_along, hi_along = ((room.bounds[0], room.bounds[2]) if along_axis == 0
                          else (room.bounds[1], room.bounds[3]))
    # A wall slab on this side is thin across the side line and long along it.
    return [mesh for mesh in geometry.walls()
            if mesh.lo[fixed_axis] - .05 <= line <= mesh.hi[fixed_axis] + .05
            and (mesh.hi - mesh.lo)[fixed_axis] < (mesh.hi - mesh.lo)[along_axis]
            and mesh.hi[along_axis] > lo_along and mesh.lo[along_axis] < hi_along]


def _inner_faces(geometry, room):
    faces = {}
    for side, (fixed_axis, _, inward) in _SIDES.items():
        line = _side_line(room, side)
        walls = _side_walls(geometry, room, side)
        sign = inward[0] + inward[1]
        if not walls:
            faces[side] = line
        elif sign > 0:
            faces[side] = float(max(mesh.hi[fixed_axis] for mesh in walls))
        else:
            faces[side] = float(min(mesh.lo[fixed_axis] for mesh in walls))
    return faces


def _wall_covers(walls, along_axis, along, height):
    return any(mesh.lo[along_axis] - 1e-6 <= along <= mesh.hi[along_axis] + 1e-6
               and mesh.lo[1] <= height <= mesh.hi[1] for mesh in walls)


def _opening_near(geometry, room, side, along):
    fixed_axis, along_axis, _ = _SIDES[side]
    line = _side_line(room, side)
    axis = "x" if along_axis == 0 else "z"
    for opening in geometry.openings:
        if opening.axis != axis:
            continue
        fixed = opening.position_xz[1] if axis == "x" else opening.position_xz[0]
        centre = opening.position_xz[0] if axis == "x" else opening.position_xz[1]
        if abs(fixed - line) <= .2 and abs(along - centre) < opening.width_m / 2 + OPENING_CLEARANCE_M:
            return opening
    return None


def _position_blockers(geometry, position, actor_paths):
    point = np.asarray(position, dtype=np.float64)
    for mesh in geometry.furniture():
        if np.all(point >= mesh.lo - FURNITURE_SHELL_M) and np.all(point <= mesh.hi + FURNITURE_SHELL_M):
            return "inside_furniture", {"object_id": mesh.object_id,
                                        "object_name": geometry.owners.get(mesh.object_id, mesh.name)}
    for mesh in geometry.furniture():
        if mesh.hi[1] < point[1]:
            continue
        dx = max(mesh.lo[0] - point[0], 0., point[0] - mesh.hi[0])
        dz = max(mesh.lo[2] - point[2], 0., point[2] - mesh.hi[2])
        if math.hypot(dx, dz) < FURNITURE_CLEARANCE_M:
            return "tall_furniture", {"object_id": mesh.object_id,
                                      "object_name": geometry.owners.get(mesh.object_id, mesh.name),
                                      "top_m": float(mesh.hi[1])}
    for path in actor_paths or ():
        top = float(path.get("height_m", 1.8)) + geometry.floor_y
        if top < point[1] - .1:
            continue
        points = np.asarray(path["points_xz"], dtype=np.float64).reshape(-1, 2)
        if not len(points):
            continue
        distance = float(np.min(np.hypot(points[:, 0] - point[0], points[:, 1] - point[2])))
        if distance < ACTOR_PATH_CLEARANCE_M:
            return "actor_path", {"object_id": str(path.get("object_id", "")), "distance_m": distance}
    return None, None


def mount_positions(geometry, room, rule, *, actor_paths=None):
    """Wall samples every 0.25 m and the four inner corners (design 7.1)."""
    faces = geometry.inner_faces(room)
    height = geometry.floor_y + rule.height_m
    standoff = rule.wall_standoff_m
    candidates, rejected = [], []

    def accept(item):
        reason, detail = _position_blockers(geometry, item["position_m"], actor_paths)
        if reason is None:
            item["index"] = len(candidates)
            candidates.append(item)
        else:
            rejected.append({**item, "reason": reason, "detail": detail})

    if rule.style in {"auto", "wall"}:
        for side, (fixed_axis, along_axis, inward) in _SIDES.items():
            walls = _side_walls(geometry, room, side)
            if along_axis == 0:
                low, high = faces["west"], faces["east"]
            else:
                low, high = faces["south"], faces["north"]
            for along in np.arange(low + CORNER_CLEARANCE_M, high - CORNER_CLEARANCE_M + 1e-9, WALL_STEP_M):
                along = float(along)
                position = [0., height, 0.]
                position[fixed_axis] = faces[side] + (inward[0] + inward[1]) * standoff
                position[along_axis] = along
                item = {"style": "wall", "wall": side, "position_m": position, "inward_xz": list(inward)}
                covered = walls and all(
                    _wall_covers(walls, along_axis, along + offset, height)
                    for offset in (-OPENING_CLEARANCE_M, 0., OPENING_CLEARANCE_M))
                opening = _opening_near(geometry, room, side, along)
                if not covered or opening is not None:
                    detail = {"opening": asdict(opening)} if opening is not None else {"wall_missing": True}
                    rejected.append({**item, "reason": "opening", "detail": detail})
                    continue
                accept(item)
    if rule.style in {"auto", "corner"}:
        for name, (x_side, z_side) in {"south_west": ("west", "south"), "south_east": ("east", "south"),
                                       "north_west": ("west", "north"),
                                       "north_east": ("east", "north")}.items():
            nx, nz = _SIDES[x_side][2][0], _SIDES[z_side][2][1]
            position = [faces[x_side] + nx * standoff, height, faces[z_side] + nz * standoff]
            bisector = [nx / math.sqrt(2.), nz / math.sqrt(2.)]
            item = {"style": "corner", "wall": name, "position_m": position, "inward_xz": bisector}
            near_x = _opening_near(geometry, room, z_side, faces[x_side])
            near_z = _opening_near(geometry, room, x_side, faces[z_side])
            if near_x is not None or near_z is not None:
                rejected.append({**item, "reason": "opening",
                                 "detail": {"opening": asdict(near_x or near_z)}})
                continue
            accept(item)
    return candidates, rejected


# ------------------------------------------------------------------ mounts


def _axes(forward):
    forward = np.asarray(forward, dtype=np.float64)
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, [0., 1., 0.])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return forward, right, up


def antenna_angles(position, forward, points):
    """Azimuth/elevation in degrees, as the separable pattern evaluates them.

    Local x is right and local -z the boresight, so ``az = atan2(v_x, -v_z)``
    and ``el = atan2(v_y, -v_z)`` (witwin.radar ``SeparablePattern``).
    """
    forward, right, up = _axes(forward)
    v = np.asarray(points, dtype=np.float64).reshape(-1, 3) - np.asarray(position, dtype=np.float64)
    along = v @ forward
    return np.degrees(np.arctan2(v @ right, along)), np.degrees(np.arctan2(v @ up, along))


def make_mount(position, heading_xz, tilt_deg, *, style="wall", wall=None, aim_height_m=None, index=None):
    heading = np.asarray(heading_xz, dtype=np.float64)
    heading /= np.linalg.norm(heading)
    tilt = math.radians(float(tilt_deg))
    forward = np.array([math.cos(tilt) * heading[0], -math.sin(tilt), math.cos(tilt) * heading[1]])
    position = [float(value) for value in position]
    if aim_height_m is not None and tilt > 1e-6:
        distance = (position[1] - float(aim_height_m)) / math.sin(tilt)
    else:
        distance = 3.
    aim = np.asarray(position) + forward * distance
    return {
        "style": style,
        "wall": wall,
        "index": index,
        "position_m": position,
        "heading_xz": heading.tolist(),
        "yaw_deg": float(math.degrees(math.atan2(heading[0], heading[1]))),
        "tilt_deg": float(tilt_deg),
        "forward": forward.tolist(),
        "aim_point_m": aim.tolist(),
    }


def _plane_feet(geometry, position):
    """Foot points of every large plane that lie on the surface and are in sight."""
    point = np.asarray(position, dtype=np.float64)
    feet = []
    for plane in geometry.planes:
        signed = float(plane.normal @ point - plane.offset)
        if abs(signed) < 1e-6:
            continue
        foot = point - signed * plane.normal
        if _on_triangles(foot, plane.triangles):
            feet.append((plane, foot))
    if not feet:
        return []
    visible = geometry.occluders.visible(point, np.array([foot for _, foot in feet]))
    return [item for item, ok in zip(feet, visible) if ok]


def _on_triangles(point, triangles, tolerance=1e-4):
    v0 = triangles[:, 0]
    e1, e2 = triangles[:, 1] - v0, triangles[:, 2] - v0
    w = point[None, :] - v0
    d00, d01, d11 = np.sum(e1 * e1, 1), np.sum(e1 * e2, 1), np.sum(e2 * e2, 1)
    d20, d21 = np.sum(w * e1, 1), np.sum(w * e2, 1)
    denom = d00 * d11 - d01 * d01
    ok = np.abs(denom) > 1e-14
    denom = np.where(ok, denom, 1.)
    v = (d11 * d20 - d01 * d21) / denom
    w2 = (d00 * d21 - d01 * d20) / denom
    return bool(np.any(ok & (v >= -tolerance) & (w2 >= -tolerance) & (v + w2 <= 1. + tolerance)))


def planar_normal_check(geometry, mount, rule, *, feet=None):
    """Design 7.2: reject when a large plane's foot point is within the beam on both axes."""
    if feet is None:
        feet = _plane_feet(geometry, mount["position_m"])
    limit = rule.exclusion_deg
    worst = None
    for plane, foot in feet:
        azimuth, elevation = antenna_angles(mount["position_m"], mount["forward"], foot)
        azimuth, elevation = float(azimuth[0]), float(elevation[0])
        angle = max(abs(azimuth), abs(elevation))
        if worst is None or angle < worst[0]:
            worst = (angle, plane, foot, azimuth, elevation)
    if worst is None:
        return {"ok": True, "worst_normal_angle_deg": None, "surface": None, "limit_deg": limit}
    angle, plane, foot, azimuth, elevation = worst
    return {
        "ok": bool(angle > limit),
        "worst_normal_angle_deg": float(angle),
        "limit_deg": float(limit),
        "surface": {
            "object_id": plane.object_id,
            "object_name": plane.name,
            "kind": plane.kind,
            "area_m2": float(plane.area_m2),
            "foot_point_m": [float(v) for v in foot],
            "azimuth_deg": azimuth,
            "elevation_deg": elevation,
        },
    }


# ------------------------------------------------------------------- zones


class SensingZone:
    """Floor cells seen from a mount at one static line-of-sight height."""

    def __init__(self, *, kind, height_m, origin_xz, cell_m, mask, open_mask):
        self.kind = kind
        self.height_m = float(height_m)
        self.origin_xz = (float(origin_xz[0]), float(origin_xz[1]))
        self.cell_m = float(cell_m)
        self.mask = np.asarray(mask, dtype=bool)
        self.open_mask = np.asarray(open_mask, dtype=bool)

    def _cell(self, x, z):
        col = int(math.floor((float(x) - self.origin_xz[0]) / self.cell_m))
        row = int(math.floor((float(z) - self.origin_xz[1]) / self.cell_m))
        return row, col

    def contains(self, x, z):
        row, col = self._cell(x, z)
        return bool(0 <= row < self.mask.shape[0] and 0 <= col < self.mask.shape[1] and self.mask[row, col])

    def cell_centres(self, mask=None):
        rows, cols = np.nonzero(self.mask if mask is None else mask)
        return np.column_stack((self.origin_xz[0] + (cols + .5) * self.cell_m,
                                self.origin_xz[1] + (rows + .5) * self.cell_m))

    @property
    def area_m2(self):
        return float(self.mask.sum() * self.cell_m ** 2)

    @property
    def open_floor_m2(self):
        return float(self.open_mask.sum() * self.cell_m ** 2)

    @property
    def centroid_xz(self):
        centres = self.cell_centres()
        return centres.mean(axis=0).tolist() if len(centres) else None

    def distance_xz(self, x, z):
        """Horizontal distance to the nearest zone cell centre (0 inside the zone)."""
        if self.contains(x, z):
            return 0.
        centres = self.cell_centres()
        if not len(centres):
            return None
        return float(np.min(np.hypot(centres[:, 0] - float(x), centres[:, 1] - float(z))))

    def polygon_xz(self):
        """Outer contours of the zone, simplified to about one cell."""
        if not self.mask.any():
            return []
        try:
            import cv2
        except ImportError:  # pragma: no cover - cv2 ships with the Studio env
            centres = self.cell_centres()
            lo, hi = centres.min(axis=0), centres.max(axis=0)
            return [[[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]]]
        image = np.pad(self.mask.astype(np.uint8), 1)
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = []
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            approx = cv2.approxPolyDP(contour, 1., True).reshape(-1, 2)
            # Contour points are cell indices of the 1-cell padded image.
            polygons.append([[round(float(self.origin_xz[0] + (col - .5) * self.cell_m), 3),
                              round(float(self.origin_xz[1] + (row - .5) * self.cell_m), 3)]
                             for col, row in approx])
        return polygons

    def to_dict(self):
        runs = []
        for row in range(self.mask.shape[0]):
            values = self.mask[row]
            col = 0
            while col < len(values):
                if values[col]:
                    end = col
                    while end + 1 < len(values) and values[end + 1]:
                        end += 1
                    runs.append([row, col, end])
                    col = end + 1
                else:
                    col += 1
        polygons = self.polygon_xz()
        return {
            "kind": self.kind,
            "height_m": self.height_m,
            "cell_m": self.cell_m,
            "origin_xz": list(self.origin_xz),
            "shape": list(self.mask.shape),
            "runs": runs,
            "area_m2": self.area_m2,
            "open_floor_m2": self.open_floor_m2,
            "fraction_of_open_floor": self.area_m2 / self.open_floor_m2 if self.open_floor_m2 else 0.,
            "centroid_xz": self.centroid_xz,
            "polygon_xz": polygons[0] if polygons else [],
            "polygons_xz": polygons,
        }

    @classmethod
    def from_dict(cls, payload):
        rows, cols = payload["shape"]
        mask = np.zeros((rows, cols), dtype=bool)
        for row, start, end in payload["runs"]:
            mask[row, start:end + 1] = True
        return cls(kind=payload["kind"], height_m=payload["height_m"], origin_xz=payload["origin_xz"],
                   cell_m=payload["cell_m"], mask=mask, open_mask=mask)


@dataclass
class _Raster:
    origin_xz: tuple
    shape: tuple
    centres: np.ndarray  # (rows*cols, 2)
    open_mask: np.ndarray  # (rows, cols)


def _raster(geometry, room):
    faces = geometry.inner_faces(room)
    x0, x1 = faces["west"], faces["east"]
    z0, z1 = faces["south"], faces["north"]
    cols = max(0, int(math.floor((x1 - x0) / ZONE_CELL_M + 1e-9)))
    rows = max(0, int(math.floor((z1 - z0) / ZONE_CELL_M + 1e-9)))
    xs = x0 + (np.arange(cols) + .5) * ZONE_CELL_M
    zs = z0 + (np.arange(rows) + .5) * ZONE_CELL_M
    grid_x, grid_z = np.meshgrid(xs, zs)
    centres = np.column_stack((grid_x.ravel(), grid_z.ravel()))
    open_mask = np.ones(len(centres), dtype=bool)
    for mesh in geometry.furniture():
        inside = ((centres[:, 0] >= mesh.lo[0]) & (centres[:, 0] <= mesh.hi[0])
                  & (centres[:, 1] >= mesh.lo[2]) & (centres[:, 1] <= mesh.hi[2]))
        open_mask &= ~inside
    return _Raster((x0, z0), (rows, cols), centres, open_mask.reshape(rows, cols))


def _zone_visibility(geometry, room, raster, position):
    """Static line of sight from a mount position to every open cell, per zone height."""
    occluders = geometry.room_occluders(room)
    flat_open = raster.open_mask.ravel()
    out = {}
    for kind, height in ZONE_HEIGHTS_M.items():
        visible = np.zeros(len(raster.centres), dtype=bool)
        rows = np.flatnonzero(flat_open)
        points = np.column_stack((raster.centres[rows, 0], np.full(len(rows), geometry.floor_y + height),
                                  raster.centres[rows, 1]))
        visible[rows] = occluders.visible(position, points)
        out[kind] = visible
    return out


def _zone_masks(geometry, raster, position, forward, rule, visibility):
    position = np.asarray(position, dtype=np.float64)
    zones = {}
    for kind, height in ZONE_HEIGHTS_M.items():
        points = np.column_stack((raster.centres[:, 0], np.full(len(raster.centres), geometry.floor_y + height),
                                  raster.centres[:, 1]))
        distance = np.linalg.norm(points - position, axis=1)
        azimuth, elevation = antenna_angles(position, forward, points)
        half = rule.zone_half_deg
        mask = (visibility[kind] & (distance >= rule.range_m[0]) & (distance <= rule.range_m[1])
                & (np.abs(azimuth) <= half) & (np.abs(elevation) <= half))
        zones[kind] = SensingZone(kind=kind, height_m=height, origin_xz=raster.origin_xz, cell_m=ZONE_CELL_M,
                                  mask=mask.reshape(raster.shape), open_mask=raster.open_mask)
    return zones


def sensing_zones(geometry, mount, rule=None, *, room=None):
    """Design 7.3: ``{"person": zone, "cat": zone}`` for one mount."""
    rule = rule or InstallRule()
    if room is None:
        room = geometry.room_of_point(mount["position_m"][0], mount["position_m"][2])
    raster = _raster(geometry, room)
    visibility = _zone_visibility(geometry, room, raster, mount["position_m"])
    return _zone_masks(geometry, raster, mount["position_m"], mount["forward"], rule, visibility)


def event_test_point(event, floor_y=0.):
    """The point that must be seen for a key event: support height plus zone height."""
    position = [float(value) for value in event["position_m"]]
    kind = str(event.get("actor_kind", "person"))
    height = event.get("test_height_m")
    if height is None:
        height = ZONE_HEIGHTS_M.get(kind, ZONE_HEIGHTS_M["person"])
    support = position[1] if len(position) == 3 else floor_y
    return [position[0], support + float(height), position[2]]


def event_in_view(geometry, mount, event, rule=None, *, zones=None):
    """Whether one key event's test point is in the narrowed beam, range and static LOS."""
    rule = rule or InstallRule()
    point = np.asarray(event_test_point(event, geometry.floor_y), dtype=np.float64)
    position = np.asarray(mount["position_m"], dtype=np.float64)
    distance = float(np.linalg.norm(point - position))
    azimuth, elevation = antenna_angles(position, mount["forward"], point)
    azimuth, elevation = float(azimuth[0]), float(elevation[0])
    reason = None
    if not rule.range_m[0] <= distance <= rule.range_m[1]:
        reason = "out_of_range"
    elif max(abs(azimuth), abs(elevation)) > rule.zone_half_deg:
        reason = "out_of_beam"
    elif geometry.occluders.segments_blocked(position, point[None])[0]:
        reason = "occluded"
    result = {
        "id": str(event.get("id", "")),
        "actor_kind": str(event.get("actor_kind", "person")),
        "in_zone": reason is None,
        "reason": reason,
        "test_point_m": point.tolist(),
        "range_m": distance,
        "azimuth_deg": azimuth,
        "elevation_deg": elevation,
    }
    if reason is not None and zones is not None:
        zone = zones.get(result["actor_kind"]) or zones["person"]
        result["miss_distance_m"] = zone.distance_xz(point[0], point[2])
    return result


# ---------------------------------------------------------------- planning


def _headings(candidate):
    base = math.atan2(candidate["inward_xz"][0], candidate["inward_xz"][1])
    span, step = ((WALL_YAW_SPAN_DEG, WALL_YAW_STEP_DEG) if candidate["style"] == "wall"
                  else (CORNER_YAW_SPAN_DEG, WALL_YAW_STEP_DEG))
    for offset in np.arange(-span, span + 1e-9, step):
        angle = base + math.radians(float(offset))
        yield float(offset), [math.sin(angle), math.cos(angle)]


def _provisional_tilt(geometry, raster, position, heading, rule, visibility):
    """Tilt so the boresight crosses the aim height at the zone centroid (design 7.1)."""
    centres = raster.centres
    dx, dz = centres[:, 0] - position[0], centres[:, 1] - position[2]
    horizontal = np.hypot(dx, dz)
    off_axis = np.degrees(np.abs(np.arctan2(dx * heading[1] - dz * heading[0], dx * heading[0] + dz * heading[1])))
    height = geometry.floor_y + ZONE_HEIGHTS_M["person"]
    slant = np.hypot(horizontal, position[1] - height)
    mask = (visibility["person"] & (off_axis <= rule.zone_half_deg)
            & (slant >= rule.range_m[0]) & (slant <= rule.range_m[1]))
    if not mask.any():
        return None, None
    centroid = centres[mask].mean(axis=0)
    distance = float(math.hypot(centroid[0] - position[0], centroid[1] - position[2]))
    drop = position[1] - (geometry.floor_y + rule.aim_height_m)
    return float(math.degrees(math.atan2(drop, max(distance, 1e-6)))), distance


def _room_doors(geometry, room):
    x0, z0, x1, z1 = room.bounds
    doors = []
    for opening in geometry.openings:
        x, z = opening.position_xz
        if x0 - .2 <= x <= x1 + .2 and z0 - .2 <= z <= z1 + .2 and (
                min(abs(x - x0), abs(x - x1)) <= .2 or min(abs(z - z0), abs(z - z1)) <= .2):
            doors.append((x, z))
    return doors


def choose_room(geometry, key_events, room_index=None):
    if room_index is not None:
        return geometry.room(int(room_index)), "explicit"
    counts = {room.index: 0 for room in geometry.rooms}
    for event in key_events:
        x, _, z = event["position_m"]
        counts[geometry.room_of_point(x, z).index] += 1

    def area(room):
        x0, z0, x1, z1 = room.bounds
        return (x1 - x0) * (z1 - z0)

    best = max(geometry.rooms, key=lambda room: (counts[room.index], area(room), -room.index))
    return best, ("most_key_events" if key_events else "largest_room")


def _summarize_rejections(rejected, per_reason=3):
    counts, examples = {}, []
    seen = {}
    for item in rejected:
        reason = item["reason"]
        counts[reason] = counts.get(reason, 0) + 1
        key = (reason, item.get("wall"))
        if seen.get(key, 0) < per_reason:
            seen[key] = seen.get(key, 0) + 1
            examples.append(item)
    return counts, examples


def plan_installation(scene=None, *, rule=None, key_events=(), actor_paths=None, geometry=None,
                      require_all_events=True, room_index=None):
    """Choose the installation mount; read-only. Raises :class:`InstallRuleError`."""
    rule = rule or InstallRule()
    if geometry is None:
        geometry = room_geometry_from_scene(scene)
    key_events = [dict(event) for event in key_events or ()]
    for number, event in enumerate(key_events):
        event.setdefault("id", f"event_{number}")
        if len(event["position_m"]) == 2:
            event["position_m"] = [event["position_m"][0], geometry.floor_y, event["position_m"][1]]
    room, room_reason = choose_room(geometry, key_events, room_index)
    positions, rejected = mount_positions(geometry, room, rule, actor_paths=actor_paths)
    walls = sorted({item["wall"] for item in positions + rejected if item.get("wall")})
    if not positions:
        counts, examples = _summarize_rejections(rejected)
        raise InstallRuleError(
            "RADAR_NO_INSTALL_SITE", "No wall or corner position is free of openings, furniture and actor paths.",
            next_step="Allow corner mounts, change the mount height, or use an explicit mount.",
            detail={"room": asdict(room), "walls": walls, "rejected_counts": counts,
                    "rejected_mounts": examples})
    raster = _raster(geometry, room)
    doors = _room_doors(geometry, room)
    valid = []
    for candidate in positions:
        position = candidate["position_m"]
        visibility = _zone_visibility(geometry, room, raster, position)
        feet = _plane_feet(geometry, position)
        for offset, heading in _headings(candidate):
            tilt, _ = _provisional_tilt(geometry, raster, position, heading, rule, visibility)
            base = {"style": candidate["style"], "wall": candidate["wall"], "position_m": position,
                    "yaw_offset_deg": offset}
            if tilt is None:
                rejected.append({**base, "reason": "empty_zone", "detail": {}})
                continue
            mount = make_mount(position, heading, tilt, style=candidate["style"], wall=candidate["wall"],
                               aim_height_m=geometry.floor_y + rule.aim_height_m)
            mount["yaw_offset_deg"] = offset
            normal = planar_normal_check(geometry, mount, rule, feet=feet)
            if not normal["ok"]:
                rejected.append({**base, "tilt_deg": tilt, "reason": "planar_normal_in_beam",
                                 "detail": {"surface": normal["surface"],
                                            "normal_angle_deg": normal["worst_normal_angle_deg"]}})
                continue
            zones = _zone_masks(geometry, raster, position, mount["forward"], rule, visibility)
            events = [event_in_view(geometry, mount, event, rule, zones=zones) for event in key_events]
            open_floor = zones["person"].open_floor_m2
            fraction = ((zones["person"].area_m2 + zones["cat"].area_m2) / (2. * open_floor)
                        if open_floor else 0.)
            centroid = zones["person"].centroid_xz
            door_distance = (min(math.hypot(centroid[0] - x, centroid[1] - z) for x, z in doors)
                             if doors and centroid else 0.)
            mount.update(worst_normal_angle_deg=normal["worst_normal_angle_deg"],
                         worst_normal_surface=normal["surface"])
            mount["index"] = len(valid)
            score = (sum(item["in_zone"] for item in events), round(fraction, 2),
                     round(door_distance, 1), -mount["index"])
            valid.append({"mount": mount, "zones": zones, "events": events, "score": score,
                          "zone_fraction": fraction, "door_distance_m": door_distance})
    counts, examples = _summarize_rejections(rejected)
    usable = [item for item in valid if item["zones"]["person"].area_m2 > 0.]
    if not usable:
        raise InstallRuleError(
            "RADAR_NO_INSTALL_SITE",
            "Every mount sees a wall, floor or large furniture face at normal incidence, or sees no floor.",
            next_step="Allow corner mounts or another height; the walls and reasons are listed.",
            detail={"room": asdict(room), "walls": walls, "rejected_counts": counts,
                    "rejected_mounts": examples})
    best = max(usable, key=lambda item: item["score"])
    missed = [item for item in best["events"] if not item["in_zone"]]
    if missed and require_all_events:
        raise InstallRuleError(
            "RADAR_ZONE_MISSES_EVENT",
            "No installation mount keeps every key event inside its sensing zone.",
            next_step="Move the event (another role or start), or use an explicit mount.",
            detail={"room": asdict(room), "missed_events": missed,
                    "best_mount": best["mount"], "key_events": best["events"]})
    mount = best["mount"]
    return {
        "ok": True,
        "label_method": LOS_METHOD,
        "placement_objective": "installation",
        "rule": rule.to_dict(),
        "room": {**asdict(room), "bounds": list(room.bounds), "selected_by": room_reason},
        "mount": mount,
        "position_m": mount["position_m"],
        "aim_point_m": mount["aim_point_m"],
        "zones": {kind: zone.to_dict() for kind, zone in best["zones"].items()},
        "zone_fraction_of_open_floor": best["zone_fraction"],
        "key_events": best["events"],
        "key_events_in_zone": not missed,
        "candidate_count": len(positions),
        "valid_mount_count": len(usable),
        "rejected_counts": counts,
        "rejected_mounts": examples,
        "score": list(best["score"]),
    }


__all__ = [
    "InstallRule",
    "InstallRuleError",
    "RoomGeometry",
    "SensingZone",
    "ZONE_HEIGHTS_M",
    "antenna_angles",
    "event_in_view",
    "event_test_point",
    "make_mount",
    "mount_positions",
    "plan_installation",
    "planar_normal_check",
    "room_geometry_from_scene",
    "sensing_zones",
]
