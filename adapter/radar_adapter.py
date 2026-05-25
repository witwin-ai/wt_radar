"""RadarAdapter — ``witwin.radar`` ``(Scene, RadarConfig)`` <-> studio ``Scene``.

Implements the frozen ``SceneAdapter`` protocol (detect / to_studio / to_platform)
and is registered on the core ``PlatformSceneHandler`` by the plugin's ``__init__``.

Radar is the one domain whose sensor lives **outside** the scene, so its load/export
payload is a ``(radar.Scene, RadarConfig)`` pair (master plan §6.4) rather than a bare
scene. ``to_studio`` accepts either the pair or a bare ``radar.Scene`` (defaulting the
config); ``to_platform`` always emits the pair. The frozen ``PlatformSceneHandler`` is
tuple-safe: it keeps the pair in ``_last_export`` for the solver hand-off and never
crashes on the tuple return.

Geometry / scalar material / transform / structure metadata are delegated to the frozen
base maps via :class:`RadarStructureMap`; this adapter owns only the radar-specific
parts (the ``RadarConfig`` settings object and, from R2, the motion graph).
"""
from typing import Any, Optional, Tuple

from witwin_server import Scene, SceneObject

from .config_map import MARKER_COMPONENT, ConfigMap
from .motion_map import MotionMap
from .structure_map import RadarStructureMap


class RadarAdapter:
    """Bidirectional adapter between a radar ``(Scene, RadarConfig)`` pair and a studio ``Scene``."""

    def detect(self, scene: Any) -> bool:
        """Claim a radar ``Scene`` or a ``(Scene, RadarConfig)`` pair (graceful if radar absent)."""
        try:
            import witwin.radar as wr
        except Exception:  # noqa: BLE001 - external boundary: radar/mitsuba/CUDA may be absent
            return False
        if isinstance(scene, tuple):
            return len(scene) == 2 and isinstance(scene[0], wr.Scene)
        return isinstance(scene, wr.Scene)

    # --- platform -> studio --------------------------------------------------

    def to_studio(self, obj: Any) -> Scene:
        """Translate a radar ``(Scene, RadarConfig)`` pair into a studio ``Scene`` (batched)."""
        platform_scene, config = self._split(obj)
        studio = Scene(kind="local")
        studio.begin_batch()
        studio.add_object(ConfigMap.settings_to_studio(config))
        for structure in platform_scene.structures:
            sobj = RadarStructureMap.to_studio(structure)
            motion = platform_scene.get_structure_motion(structure.name)
            if motion is not None:
                from ..components.motion import RadarMotionComponent
                MotionMap.to_studio(sobj.add_component(RadarMotionComponent()), motion)
            studio.add_object(sobj)
        studio.end_batch()
        return studio

    # --- studio -> platform --------------------------------------------------

    def to_platform(self, studio_scene: Scene) -> Tuple[Any, Any]:
        """Rebuild the ``(radar.Scene, RadarConfig)`` pair from the studio scene."""
        import witwin.radar as wr

        settings = self._find_settings(studio_scene)
        if settings is None:
            raise ValueError(
                "radar export requires a 'Radar Settings' object with a RadarConfig component.")
        config = ConfigMap.build_config(settings)
        scene = wr.Scene(device="cpu")
        # Add all structures first, then motions: add_structure_motion validates that
        # the parent structure exists and that the motion graph stays acyclic.
        movers = []
        for obj in studio_scene.objects.values():
            if obj is settings:
                continue
            if obj.get_component("PlatformGeometry") is None:
                continue
            structure = RadarStructureMap.to_platform(obj)
            if structure is not None:
                scene.add_structure(structure)
                movers.append(obj)
        for obj in movers:
            radar_motion = obj.get_component("RadarMotion")
            if radar_motion is None:
                continue
            motion = MotionMap.build(radar_motion)
            if motion is not None:
                scene.add_structure_motion(obj.name, motion)
        return scene, config

    # --- helpers -------------------------------------------------------------

    @staticmethod
    def _split(obj: Any) -> Tuple[Any, Optional[Any]]:
        # Unpack the (Scene, RadarConfig) pair; a bare Scene defaults the config.
        if isinstance(obj, tuple):
            return obj[0], obj[1]
        return obj, None

    @staticmethod
    def _find_settings(studio_scene: Scene) -> Optional[SceneObject]:
        # The Radar Settings object is the one carrying a RadarConfig component.
        for obj in studio_scene.objects.values():
            if obj.get_component(MARKER_COMPONENT) is not None:
                return obj
        return None
