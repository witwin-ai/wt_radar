"""WT Radar bridge for direct Channel environment legs.

Official Radar 0.4 composes two propagation legs through explicit targets.  A
static room return is instead one direct Channel leg from a transmitter to a
receiver.  This module performs only the missing bookkeeping conversion: it
orders those rows by Radar's sink-major sensor-pair convention and publishes an
official ``RadarPathBatch``.  Channel remains the owner of propagation and
material response; Radar remains the owner of waveform synthesis.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


_NO_SITE = -1


def _stable_ids(values, name: str) -> list[int]:
    if isinstance(values, torch.Tensor):
        if values.ndim != 1:
            raise ValueError(f"{name} must be a 1-D sequence of stable IDs")
        listed = [int(value) for value in values.tolist()]
    else:
        listed = [int(value) for value in values]
    if not listed:
        raise ValueError(f"{name} must not be empty")
    if len(set(listed)) != len(listed):
        raise ValueError(f"{name} must not repeat a stable ID, got {listed}")
    return listed


def _leg_identity(leg) -> tuple[list[int], list[int], list[tuple]]:
    source = [int(value) for value in leg.source_id.tolist()]
    sink = [int(value) for value in leg.sink_id.tolist()]
    component = [int(value) for value in leg.component_id.tolist()]
    depth = [int(value) for value in leg.depth.tolist()]
    primitive = [tuple(int(value) for value in row) for row in leg.primitive_sequence.tolist()]
    material = [tuple(int(value) for value in row) for row in leg.material_sequence.tolist()]
    rows = len(source)
    for label, column in (
        ("sink_id", sink),
        ("component_id", component),
        ("depth", depth),
        ("primitive_sequence", primitive),
        ("material_sequence", material),
    ):
        if len(column) != rows:
            raise ValueError(f"direct leg {label} has {len(column)} rows, expected {rows}")
    keys = [
        (component[row], depth[row], primitive[row], material[row])
        for row in range(rows)
    ]
    seen: dict[tuple[int, int], dict[tuple, int]] = {}
    for row, endpoints in enumerate(zip(source, sink, strict=True)):
        claimed = seen.setdefault(endpoints, {})
        if keys[row] in claimed:
            raise ValueError(
                f"direct leg rows {claimed[keys[row]]} and {row} share the "
                f"identity key {keys[row]} within endpoint pair {endpoints}"
            )
        claimed[keys[row]] = row
    return source, sink, keys


def _pair_offsets(pair_of_row: list[int], pair_count: int) -> list[int]:
    counts = [0] * pair_count
    for row, pair in enumerate(pair_of_row):
        if not 0 <= pair < pair_count:
            raise ValueError(
                f"direct row {row} claims sensor pair {pair}, outside {pair_count} pairs"
            )
        counts[pair] += 1
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    return offsets


@dataclass(frozen=True, slots=True, eq=False)
class DirectEnvironmentComposer:
    """Frozen direct Channel rows in Radar's canonical sensor-pair order."""

    row_index: torch.Tensor
    topology: object
    sensor_pair_index: torch.Tensor
    pair_offsets: torch.Tensor
    sensor_pair_count: int
    reference_frequency_hz: float

    @classmethod
    def freeze(
        cls, leg, *, radar_source_ids, radar_sink_ids, reference_frequency_hz: float,
    ) -> "DirectEnvironmentComposer":
        from witwin.radar.paths import RadarPathTopology

        device = leg.source_id.device
        sources = _stable_ids(radar_source_ids, "radar_source_ids")
        sinks = _stable_ids(radar_sink_ids, "radar_sink_ids")
        source, sink, keys = _leg_identity(leg)
        source_rank = {value: rank for rank, value in enumerate(sources)}
        sink_rank = {value: rank for rank, value in enumerate(sinks)}
        stray_sources = sorted(set(source) - set(sources))
        stray_sinks = sorted(set(sink) - set(sinks))
        if stray_sources:
            raise ValueError(f"direct leg carries unknown radar source IDs {stray_sources}")
        if stray_sinks:
            raise ValueError(f"direct leg carries unknown radar sink IDs {stray_sinks}")

        def pair_rank(source_id: int, sink_id: int) -> int:
            return sink_rank[sink_id] * len(sources) + source_rank[source_id]

        rows = [
            (pair_rank(source[row], sink[row]), source[row], sink[row], row, keys[row])
            for row in range(len(source))
        ]
        rows.sort(key=lambda item: (item[0], item[4]))

        def column(index: int) -> torch.Tensor:
            return torch.tensor([row[index] for row in rows], dtype=torch.int64, device=device)

        def constant(value: int) -> torch.Tensor:
            return torch.full((len(rows),), value, dtype=torch.int64, device=device)

        pair_count = len(sources) * len(sinks)
        pair_index = column(0)
        pair_list = [int(row[0]) for row in rows]
        return cls(
            row_index=column(3),
            topology=RadarPathTopology(
                radar_source_id=column(1),
                site_id=constant(_NO_SITE),
                radar_sink_id=column(2),
                inbound_row=column(3),
                outbound_row=constant(-1),
            ),
            sensor_pair_index=pair_index,
            pair_offsets=torch.tensor(
                _pair_offsets(pair_list, pair_count), dtype=torch.int64, device=device,
            ),
            sensor_pair_count=pair_count,
            reference_frequency_hz=float(reference_frequency_hz),
        )

    @property
    def path_count(self) -> int:
        return int(self.row_index.shape[0])

    @property
    def site_count(self) -> int:
        return 0

    def compose(self, leg):
        from witwin.radar.paths import RadarPathBatch

        if int(leg.delay_s.shape[0]) != self.path_count:
            raise ValueError(
                f"direct leg carries {int(leg.delay_s.shape[0])} rows but the frozen "
                f"topology expects {self.path_count}"
            )
        rows = self.row_index
        return RadarPathBatch(
            sensor_pair_count=self.sensor_pair_count,
            path_count=self.path_count,
            sensor_pair_index=self.sensor_pair_index,
            pair_offsets=self.pair_offsets,
            total_delay_s=leg.delay_s.index_select(0, rows),
            delay_rate=None,
            complex_transfer_ref=leg.coefficient.index_select(0, rows),
            reference_frequency_hz=self.reference_frequency_hz,
            row_valid=None if leg.row_valid is None else leg.row_valid.index_select(0, rows),
            topology=self.topology,
        )


__all__ = ["DirectEnvironmentComposer"]
