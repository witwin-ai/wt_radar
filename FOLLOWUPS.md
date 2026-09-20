# WT Radar follow-ups

The current supported boundary is official Radar 0.4 / Channel 0.5 with one
specular environment bounce assembled by WT Radar.

Possible future work, each requiring an explicit physical-model decision:

- Diffuse environment clutter or material roughness.
- More than one environment bounce.
- Calibrated animal/skin scattering instead of the explicit scalar RCS policy.
- Time-varying room geometry or a moving sensor during a measurement.
- A Radar 0.4-native realtime streaming contract.

These are not silently approximated by the current plugin. Maxwell remains out
of the room-scale Radar workflow unless a separate full-wave use case is defined.
