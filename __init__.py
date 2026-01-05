"""
Radar Simulation Extension - Radar system simulation and visualization.

This extension provides:
1. Radar component for simulating radar systems
2. SimpleRadar signal processing class
3. Radar library item in the RF category

Usage:
    import extensions.radar_simulation
"""
from witwin.utils.logging import get_logger

logger = get_logger("Radar Simulation")

# Import modules to trigger auto-registration
from . import components
from . import library_items

logger.info("Loaded!")
