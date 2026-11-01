"""Canonical legacy-compatible target contract for animated Radar measurements."""
from __future__ import annotations

import json
from typing import Any

from .studio_motion import MAX_TARGET_SITE_BUDGET, resolved_target_site_budget, target_kind


def normalize_target_specs(
    scene: Any,
    values: Any,
    *,
    budget_overrides: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Validate, deduplicate, and resolve one or more skinned target declarations."""
    if not isinstance(values, list) or not values:
        raise ValueError("Radar animation requires at least one target.")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            raise ValueError(f"Radar target {index} must be an object.")
        object_id = str(value.get("object_id") or "").strip()
        if not object_id:
            raise ValueError(f"Radar target {index} requires object_id.")
        if object_id in seen:
            raise ValueError(f"Radar target is duplicated: {object_id}")
        target = scene.get_object(object_id) if scene is not None else None
        if target is None:
            raise ValueError(f"Radar target does not exist in scene: {object_id}")
        if target.get_component("SkinnedMesh") is None:
            raise ValueError(f"Radar target must have a SkinnedMesh component: {object_id}")
        kind = target_kind(scene, object_id)
        persisted_source = str(value.get("budget_source") or "")
        explicit = "site_budget" in value and persisted_source not in {
            "component_default", "type_default",
        }
        if explicit:
            authored_budget = value["site_budget"]
            budget_source = "explicit"
        elif budget_overrides is not None and kind in budget_overrides:
            authored_budget = budget_overrides[kind]
            budget_source = "component_default"
        else:
            authored_budget = None
            budget_source = "type_default"
        resolved_budget = resolved_target_site_budget(scene, object_id, authored_budget)
        if (
            isinstance(resolved_budget, bool)
            or int(resolved_budget) != resolved_budget
            or not 1 <= int(resolved_budget) <= MAX_TARGET_SITE_BUDGET
        ):
            raise ValueError(
                f"Radar target site_budget must be an integer from 1 to "
                f"{MAX_TARGET_SITE_BUDGET}: {object_id}"
            )
        normalized.append({
            "object_id": object_id,
            "target_kind": kind,
            "site_budget": int(resolved_budget),
            "budget_source": budget_source,
        })
        seen.add(object_id)
    return normalized


def component_target_specs(component: Any) -> list[dict[str, Any]]:
    """Read the new list contract, falling back to the legacy single target fields."""
    scene = getattr(component, "scene", None)
    encoded = str(getattr(component, "animation_targets_json", "") or "").strip()
    if encoded:
        try:
            values = json.loads(encoded)
        except json.JSONDecodeError as exc:
            raise ValueError("Radar animation target list is not valid JSON.") from exc
        return normalize_target_specs(scene, values, budget_overrides={
            "human": int(getattr(component, "human_target_site_budget", 720)),
            "animal": int(getattr(component, "animal_target_site_budget", 360)),
            "skinned": int(getattr(component, "target_site_budget", 24)),
        })
    object_id = str(getattr(component, "snapshot_target_id", "") or "").strip()
    if not object_id:
        raise ValueError("Radar animation requires at least one target.")
    return normalize_target_specs(scene, [{"object_id": object_id}], budget_overrides={
        "human": int(getattr(component, "human_target_site_budget", 720)),
        "animal": int(getattr(component, "animal_target_site_budget", 360)),
        "skinned": int(getattr(component, "target_site_budget", 24)),
    })


def encode_target_specs(specs: list[dict[str, Any]]) -> str:
    """Persist a stable compact representation in the Studio component."""
    return json.dumps(specs, sort_keys=True, separators=(",", ":"), allow_nan=False)
