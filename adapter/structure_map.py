"""Radar structure overlay on top of the frozen base ``StructureMap``.

The base ``StructureMap`` already round-trips geometry + scalar material + transform +
the full opaque ``metadata`` dict (via ``StructureMeta``). Radar adds nothing to that
except presenting two metadata keys nicely: ``dynamic`` and ``bsdf`` move into a
dedicated ``RadarStructureMeta`` component on ``to_studio``, and merge back into the
``core.Structure`` metadata on ``to_platform``. Everything geometric/material is
delegated to the frozen base map — no reimplementation.
"""
from typing import Any, Optional

from witwin_server import SceneObject
from witwin_server.platform_bridge import StructureMap


class RadarStructureMap:
    """Overlay radar ``dynamic`` / ``bsdf`` metadata on the base structure round trip."""

    @staticmethod
    def to_studio(structure: Any) -> SceneObject:
        """Base structure object + a ``RadarStructureMeta`` carrying dynamic/bsdf."""
        from ..components.structure_meta import RadarStructureMetaComponent

        obj = StructureMap.to_studio(structure)
        meta = obj.get_component("StructureMeta")
        metadata = dict(meta.metadata) if meta is not None else {}

        radar_meta = obj.add_component(RadarStructureMetaComponent())
        radar_meta.dynamic = bool(metadata.get("dynamic", False))
        bsdf = metadata.get("bsdf")
        radar_meta.bsdf = bsdf if isinstance(bsdf, dict) else None

        # The two keys now live on RadarStructureMeta; keep StructureMeta opaque-only.
        if meta is not None:
            metadata.pop("dynamic", None)
            metadata.pop("bsdf", None)
            meta.metadata = metadata
        return obj

    @staticmethod
    def to_platform(obj: SceneObject) -> Optional[Any]:
        """Rebuild a ``core.Structure``, merging dynamic/bsdf back into its metadata."""
        base = StructureMap.to_platform(obj)
        if base is None:
            return None  # cameras / lights / empties (e.g. the Radar Settings object)

        import witwin.core as wc

        metadata = dict(base.metadata)
        radar_meta = obj.get_component("RadarStructureMeta")
        if radar_meta is not None:
            # Match the platform convention: store the keys only when meaningful, so a
            # static structure round-trips to the same (empty) metadata it came from.
            if bool(radar_meta.dynamic):
                metadata["dynamic"] = True
            bsdf = radar_meta.bsdf
            if bsdf is not None:
                metadata["bsdf"] = bsdf

        return wc.Structure(
            base.geometry,
            base.material,
            name=base.name,
            priority=int(base.priority),
            enabled=bool(base.enabled),
            tags=tuple(base.tags),
            metadata=metadata,
        )
