"""wt-radar — witwin.radar solver-scene round-trip + FMCW solve + in-component signal viz.

Importing this package (eager: no ``activationEvents``):
  1. registers the radar scene components (config / sensor / sub-configs / motion /
     structure-meta / tracer / result) via their ``@component`` decorators,
  2. registers the radar library prefabs, and
  3. registers :class:`RadarAdapter` under the ``"radar"`` domain on the core
     :class:`PlatformSceneHandler` so ``load_platform_scene`` /
     ``export_platform_scene`` dispatch radar scenes here.

No ``witwin.radar`` import happens at module load: the adapter maps lazy-import it
inside their methods (its ``__init__`` eagerly initializes the mitsuba CUDA variant),
so the plugin still registers its schema-only components on a machine without the
radar solver installed. The shared base layer (``witwin_server.features.platform.bridge`` +
the unified ``Material``/``PlatformGeometry``/``StructureMeta`` components) is in
core and frozen; this plugin only reads/writes it.

Radar is unlike maxwell/channel in two ways the round trip must respect:
  * the sensor (``Radar``/``RadarConfig``) lives **outside** the scene, so the
    load/export contract carries a **(Scene, RadarConfig) pair** (master plan §6.4);
  * dynamics are a **separate acyclic motion graph** (``scene._structure_motions``),
    never the studio ``Transform`` parent tree.
"""
from witwin_server.features.platform.scene_handlers import PlatformSceneHandler

from . import components  # noqa: F401  (side-effect: registers radar components)
from . import library_items  # noqa: F401  (side-effect: registers radar library prefabs)
from .adapter import RadarAdapter

PlatformSceneHandler.register_adapter("radar", RadarAdapter())
