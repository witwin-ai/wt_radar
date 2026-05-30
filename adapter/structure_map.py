"""Radar structure overlay on top of the frozen base ``StructureMap``.

The live editor path is mesh-first: any visible, non-empty Studio ``Mesh`` can be
exported as a radar structure without radar-specific sidecar components. Radar-only
metadata from imported platform scenes is intentionally discarded.
"""
from typing import Any, Optional

import numpy as np

from witwin_server import SceneObject
from witwin_server.features.platform.bridge import MaterialMap, StructureMap, TransformMap
from witwin_server.utils.mitsuba_utils import get_world_transform


class RadarStructureMap:
    """Map radar structures while keeping the common editor path component-light."""

    @staticmethod
    def to_studio(structure: Any) -> SceneObject:
        """Build a normal mesh-backed object."""
        obj = RadarStructureMap._native_to_studio(structure)
        if obj is None:
            obj = StructureMap.to_studio(structure)
            RadarStructureMap._strip_platform_components_if_mesh_backed(obj)
        return obj

    @staticmethod
    def to_platform(obj: SceneObject) -> Optional[Any]:
        """Rebuild a ``core.Structure`` from ordinary Studio mesh/material fields."""
        base = (
            StructureMap.to_platform(obj)
            if obj.get_component("PlatformGeometry") is not None
            else RadarStructureMap._mesh_to_platform(obj)
        )
        if base is None:
            return None  # cameras / lights / empties (e.g. the Radar Settings object)

        import witwin.core as wc

        metadata = dict(base.metadata)

        return wc.Structure(
            base.geometry,
            base.material,
            name=base.name,
            priority=int(base.priority),
            enabled=bool(base.enabled),
            tags=tuple(base.tags),
            metadata=metadata,
        )

    @staticmethod
    def _native_to_studio(structure: Any) -> Optional[SceneObject]:
        geometry = structure.geometry
        kind = str(geometry.kind)
        if kind == "box":
            obj = SceneObject(name=structure.name or "Structure", mesh_type="Cube")
            obj.get_component("Transform").scale = RadarStructureMap._list(geometry.size)
        elif kind == "sphere":
            radius = RadarStructureMap._num(geometry.radius)
            obj = SceneObject(name=structure.name or "Structure", mesh_type="Sphere")
            obj.get_component("Transform").scale = [radius * 2.0] * 3
        elif kind in {"cylinder", "cone"} and str(geometry.axis).lower() == "z":
            radius = RadarStructureMap._num(geometry.radius)
            height = RadarStructureMap._num(geometry.height)
            mesh_type = "Cylinder" if kind == "cylinder" else "Cone"
            obj = SceneObject(name=structure.name or "Structure", mesh_type=mesh_type)
            obj.get_component("Transform").scale = [radius * 2.0, radius * 2.0, height]
        else:
            return None

        transform = obj.get_component("Transform")
        transform.position = RadarStructureMap._list(geometry.position)
        transform.rotation = TransformMap.quat_to_euler(RadarStructureMap._list(geometry.rotation))
        obj.visible = bool(structure.enabled)
        material = obj.get_component("Material")
        if material is not None:
            MaterialMap.fill_component(material, MaterialMap.to_studio(structure.material))
        return obj

    @staticmethod
    def _mesh_to_platform(obj: SceneObject) -> Optional[Any]:
        import witwin.core as wc

        if not obj.visible:
            return None
        mesh = obj.get_component("Mesh")
        if mesh is None or mesh.is_empty:
            return None

        vertices = np.asarray(mesh.get_vertices_numpy(), dtype=np.float32).reshape(-1, 3)
        faces = np.asarray(mesh.get_faces_numpy(), dtype=np.int64).reshape(-1, 3)
        ones = np.ones((vertices.shape[0], 1), dtype=np.float32)
        world = (get_world_transform(obj) @ np.concatenate([vertices, ones], axis=1).T).T[:, :3]
        geometry = wc.Mesh(world, faces, position=(0.0, 0.0, 0.0), rotation=None,
                           scale=1.0, recenter=False)

        material = obj.get_component("Material")
        return wc.Structure(
            geometry,
            MaterialMap.to_platform(material) if material is not None else wc.Material(),
            name=obj.name,
        )

    @staticmethod
    def _strip_platform_components_if_mesh_backed(obj: SceneObject) -> None:
        mesh = obj.get_component("Mesh")
        if mesh is None or mesh.is_empty:
            return
        obj.remove_component("PlatformGeometry")
        obj.remove_component("StructureMeta")

    @staticmethod
    def _num(value: Any) -> float:
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)

    @staticmethod
    def _list(value: Any) -> list:
        if hasattr(value, "tolist"):
            return value.tolist()
        return list(value)
