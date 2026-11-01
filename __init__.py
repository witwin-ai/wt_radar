"""Studio components and tools for the official Radar 0.4 simulation path.

The supported solver inputs are immutable Studio snapshots produced by the
explicit-point snapshot and baked-animation adapters.  The historical
``(radar.Scene, RadarConfig)`` platform round trip targeted Radar 0.3 and is no
longer registered as a Studio platform-scene adapter.

Importing the package only registers component schemas, library items, and agent
tools.  Simulation packages remain lazy imports in the solver process.
"""

from . import components  # noqa: F401  (side-effect: registers radar components)
from . import library_items  # noqa: F401  (side-effect: registers radar library prefabs)
