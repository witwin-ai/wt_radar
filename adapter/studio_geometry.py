"""Imported Studio meshes to Core geometry for the explicit radar models.

This production adapter has no dependency on local experiment scripts. Meshes
are baked with the full parent affine matrix, preserving scale and shear.
"""
from types import SimpleNamespace

import numpy as np
import torch


def export_static_mesh_scene(scene, *, excluded_object_ids=()):
    import witwin.core as wc
    from witwin_server.features.platform.bridge.material_map import MaterialMap
    from .snapshot import _hierarchy

    matrices, visible = _hierarchy(scene)
    excluded = set(excluded_object_ids)
    structures, mapping = [], {}
    faces_count = 0
    for oid, obj in scene.objects.items():
        oid = str(oid)
        if oid in excluded or not visible[oid]:
            continue
        if obj.get_component("SkinnedMesh") is not None:
            raise ValueError(f"Undeclared skinned target: {oid}")
        mesh = obj.get_component("Mesh")
        if mesh is None:
            continue
        material = obj.get_component("Material")
        if mesh.is_empty or material is None:
            raise ValueError(f"Room mesh {oid} requires hydrated geometry and Material.")
        vertices = np.asarray(mesh.get_vertices_numpy(), dtype=np.float32)
        faces = np.asarray(mesh.get_faces_numpy(), dtype=np.int64).reshape(-1, 3)
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
                or not len(faces) or faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError(f"Invalid room geometry: {oid}")
        vertices = (np.column_stack((vertices, np.ones(len(vertices)))) @ matrices[oid].T)[:, :3]
        sid = 10_000 + len(structures)
        geometry = wc.Mesh(torch.tensor(vertices, dtype=torch.float32), torch.from_numpy(faces),
                           recenter=False, fill_mode="surface", topology_diagnostics=False)
        structures.append(wc.Structure(
            geometry, MaterialMap.to_platform(material, material_id=sid + 100_000), name=str(obj.name),
            structure_id=sid, material_id=sid + 100_000, assignment_id=sid + 200_000,
            surface_id=sid + 300_000, metadata={"studio_object_id": oid}))
        mapping[oid] = sid
        faces_count += len(faces)
    if not structures:
        raise ValueError("Studio scene contains no exportable static room meshes.")
    return SimpleNamespace(scene=wc.Scene(structures=tuple(structures)), mesh_count=len(structures),
                           face_count=faces_count, object_to_structure_id=mapping)
