"""WiTwin Radar 0.3 ``CoreScene``/``DynamicScene`` <-> Studio scene adapter."""
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from witwin_server import Scene, SceneObject
from witwin_server.utils.mitsuba_utils import get_world_transform

from .config_map import MARKER_COMPONENT, ConfigMap
from .motion_map import MotionMap
from .structure_map import RadarStructureMap


@dataclass(frozen=True)
class RadarExport:
    """A Core world plus the explicit scatter-site declarations authored by Studio."""

    scene: Any
    config: Any
    site_positions_m: tuple[tuple[float, float, float], ...]
    site_structure_ids: tuple[int, ...]

    def positions_at(self, time_s: float, *, device: Any) -> Any:
        """Return explicit scatter sites at one frame instant on the Radar device."""
        import torch

        if not self.site_positions_m:
            raise ValueError(
                "radar solve requires at least one enabled RadarScatterer with an explicit surface site"
            )
        dynamic = self.scene if hasattr(self.scene, "structure_trajectories") else None
        trajectories = {} if dynamic is None else dynamic.structure_trajectories
        rows = []
        for base, structure_id in zip(self.site_positions_m, self.site_structure_ids, strict=True):
            position = np.asarray(base, dtype=np.float64)
            trajectory = trajectories.get(structure_id)
            if trajectory is not None:
                motion = trajectory.at(float(time_s))
                if motion.rotation is not None:
                    value = np.asarray(motion.rotation.detach().cpu(), dtype=np.float64)
                    rotation = (
                        Rotation.from_quat((value[1], value[2], value[3], value[0]))
                        if value.shape == (4,)
                        else Rotation.from_euler("zyx", value)
                    )
                    position = rotation.apply(position)
                if motion.translation is not None:
                    position = position + np.asarray(motion.translation.detach().cpu(), dtype=np.float64)
            rows.append(tuple(float(value) for value in position))
        return torch.tensor(rows, dtype=torch.float32, device=device).contiguous()


class RadarAdapter:
    """Bidirectional adapter for ``(witwin.core Scene|DynamicScene, RadarConfig)``."""

    def detect(self, scene: Any) -> bool:
        try:
            import witwin.core as wc
        except Exception:  # noqa: BLE001 - optional platform boundary
            return False
        candidate = scene[0] if isinstance(scene, tuple) and len(scene) == 2 else scene
        return isinstance(candidate, (wc.Scene, wc.DynamicScene))

    def to_studio(self, obj: Any) -> Scene:
        platform_scene, config = self._split(obj)
        dynamic = platform_scene if hasattr(platform_scene, "structure_trajectories") else None
        scene = platform_scene.scene if dynamic is not None else platform_scene
        trajectories = {} if dynamic is None else dynamic.structure_trajectories

        studio = Scene()
        studio.begin_batch()
        studio.add_object(ConfigMap.settings_to_studio(config))
        for structure in scene.structures:
            sobj = RadarStructureMap.to_studio(structure)
            trajectory = trajectories.get(structure.structure_id)
            if trajectory is not None:
                from ..components.motion import RadarMotionComponent

                MotionMap.to_studio(sobj.add_component(RadarMotionComponent()), trajectory)
            studio.add_object(sobj)
        studio.end_batch()
        return studio

    def to_platform(self, studio_scene: Scene, *, device: str | None = None) -> Tuple[Any, Any]:
        """Rebuild the current immutable Core world and its Radar configuration."""
        exported = self.build_export(studio_scene)
        return exported.scene, exported.config

    def build_export(self, studio_scene: Scene) -> RadarExport:
        """Build the Core world and explicitly authored RadarScatterer sites."""
        import witwin.core as wc

        settings = self._find_settings(studio_scene)
        if settings is None:
            raise ValueError("radar export requires a 'Radar Settings' object with a Radar component")

        structures = []
        positions = []
        structure_ids = []
        trajectories = {}
        for obj in studio_scene.objects.values():
            if obj is settings:
                continue
            identity = len(structures) + 1
            structure = RadarStructureMap.to_platform(obj, identity=identity)
            if structure is None:
                continue
            structures.append(structure)
            scatterer = obj.get_component("RadarScatterer")
            if scatterer is not None and bool(scatterer.enabled):
                matrix = np.asarray(get_world_transform(obj), dtype=np.float64)
                local_site = np.asarray((*scatterer.site_offset, 1.0), dtype=np.float64)
                world_site = matrix @ local_site
                positions.append(tuple(float(value) for value in world_site[:3]))
                structure_ids.append(identity)
            motion_component = obj.get_component("RadarMotion")
            if motion_component is not None:
                trajectory = MotionMap.build(motion_component)
                if trajectory is not None:
                    trajectories[identity] = trajectory

        scene = wc.Scene(structures=tuple(structures))
        world = wc.DynamicScene(scene, structure_trajectories=trajectories) if trajectories else scene
        return RadarExport(
            scene=world,
            config=ConfigMap.build_config(settings),
            site_positions_m=tuple(positions),
            site_structure_ids=tuple(structure_ids),
        )

    @staticmethod
    def _split(obj: Any) -> Tuple[Any, Optional[Any]]:
        if isinstance(obj, tuple):
            if len(obj) != 2:
                raise ValueError("a radar platform payload must be (CoreScene, RadarConfig)")
            return obj[0], obj[1]
        return obj, None

    @staticmethod
    def _find_settings(studio_scene: Scene) -> Optional[SceneObject]:
        for obj in studio_scene.objects.values():
            if obj.get_component(MARKER_COMPONENT) is not None:
                return obj
        return None
