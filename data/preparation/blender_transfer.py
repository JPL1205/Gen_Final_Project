"""
Blender render script — produces 40-view RGBA renders + camera JSONs per OBJ.

Run inside Blender:
    blender --background --python data/preparation/blender_transfer.py

Output staging dir:  data/chibi_renders/{obj_stem}/{00000-00039}.{png,json}
Then run:            python data/preparation/pack_to_parquet.py
"""

import bpy
import math
import json
import os
import sys
from mathutils import Vector
from pathlib import Path

# ── Paths (Linux absolute paths for WSL) ──────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
OBJ_DIR      = PROJECT_ROOT / "data" / "chibi_train_obj_data"
RENDER_DIR   = PROJECT_ROOT / "data" / "chibi_renders"

# ── Render settings ───────────────────────────────────────────────────────
RESOLUTION       = 512
CAMERA_DISTANCE  = 2.2     # matches GObjaverse distance
CYCLES_SAMPLES   = 64      # lower for faster test; raise to 128+ for quality

# 40 views: 5 elevation rings × 8 azimuths
# Keep elevation low so camera stays near eye-level for standing characters
ELEVATIONS = [-5, 5, 15, 25, 35]           # degrees above XY plane (Blender Z-up)
AZIMUTHS   = [0, 45, 90, 135, 180, 225, 270, 315]


# ── Scene helpers ─────────────────────────────────────────────────────────

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for block in list(bpy.data.meshes) + list(bpy.data.materials) + list(bpy.data.images):
        bpy.data.meshes.remove(block) if hasattr(bpy.data.meshes, 'remove') and block in bpy.data.meshes.values() else None


def setup_render():
    scene = bpy.context.scene
    scene.render.engine                       = 'CYCLES'
    scene.render.resolution_x                 = RESOLUTION
    scene.render.resolution_y                 = RESOLUTION
    scene.render.image_settings.file_format   = 'PNG'
    scene.render.image_settings.color_mode    = 'RGBA'
    scene.render.film_transparent             = True
    scene.cycles.samples                      = CYCLES_SAMPLES
    scene.cycles.use_denoising               = False
    # Black world background (transparent via film_transparent)
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (0, 0, 0, 1)


def import_obj(filepath: str):
    bpy.ops.wm.obj_import(filepath=filepath)
    meshes = [o for o in bpy.context.selected_objects if o.type == 'MESH']
    if not meshes:
        raise RuntimeError(f"No mesh found in {filepath}")

    # If multiple mesh objects, join them
    bpy.ops.object.select_all(action='DESELECT')
    for m in meshes:
        m.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.active_object

    # Center at origin, normalize to unit bounding box
    bbox = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    lo = Vector(map(min, zip(*bbox)))
    hi = Vector(map(max, zip(*bbox)))
    center = (lo + hi) / 2
    size   = max(hi - lo)
    obj.location -= center
    obj.scale     = Vector((1.0 / size,) * 3)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # Default grey material if no material
    if not obj.data.materials:
        mat = bpy.data.materials.new("DefaultMat")
        mat.use_nodes = True
        mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.8, 0.8, 0.8, 1)
        obj.data.materials.append(mat)

    return obj


def setup_camera():
    bpy.ops.object.camera_add()
    cam = bpy.context.active_object
    cam.name = "RenderCam"
    bpy.context.scene.camera = cam
    return cam


def setup_lights():
    bpy.ops.object.light_add(type='SUN', location=(4, 4, 6))
    bpy.context.active_object.data.energy = 3.0
    bpy.ops.object.light_add(type='AREA', location=(-3, -2, 3))
    bpy.context.active_object.data.energy = 1.0
    bpy.context.active_object.data.size   = 3


def camera_json(cam) -> dict:
    """
    Export camera matrix columns as x/y/z/origin.
    Stored in Blender world + OpenCV cam convention,
    matching the GObjaverse format.  The dataset loader
    converts to OpenGL convention internally.
    """
    m = cam.matrix_world
    return {
        "x":      [m[0][0], m[1][0], m[2][0]],   # col 0 — camera right
        "y":      [m[0][1], m[1][1], m[2][1]],   # col 1 — camera up
        "z":      [m[0][2], m[1][2], m[2][2]],   # col 2 — camera backward
        "origin": [m[0][3], m[1][3], m[2][3]],   # col 3 — camera position
    }


def render_object(obj_path: Path):
    print(f"\n=== Rendering {obj_path.name} ===")
    out_dir = RENDER_DIR / obj_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Scene reset
    bpy.ops.wm.read_factory_settings(use_empty=True)
    setup_render()

    import_obj(str(obj_path))
    cam = setup_camera()
    setup_lights()

    view_idx = 0
    for elev in ELEVATIONS:
        for azim in AZIMUTHS:
            er = math.radians(elev)
            ar = math.radians(azim)
            # Spherical → Cartesian (Blender: Z-up)
            x = CAMERA_DISTANCE * math.cos(er) * math.cos(ar)
            y = CAMERA_DISTANCE * math.cos(er) * math.sin(ar)
            z = CAMERA_DISTANCE * math.sin(er)
            cam.location = Vector((x, y, z))
            # Point camera at origin
            direction = Vector((0, 0, 0)) - cam.location
            cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
            bpy.context.view_layer.update()

            png_path = out_dir / f"{view_idx:05d}.png"
            json_path = out_dir / f"{view_idx:05d}.json"

            bpy.context.scene.render.filepath = str(png_path)
            bpy.ops.render.render(write_still=True)

            with open(json_path, "w") as f:
                json.dump(camera_json(cam), f)

            print(f"  view {view_idx:05d}  elev={elev}  azim={azim}")
            view_idx += 1

    print(f"Done: {view_idx} views → {out_dir}")


def main():
    obj_files = sorted(OBJ_DIR.glob("*.obj"))
    if not obj_files:
        print(f"No .obj files found in {OBJ_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(obj_files)} OBJ file(s): {[f.name for f in obj_files]}")
    RENDER_DIR.mkdir(parents=True, exist_ok=True)

    for obj_path in obj_files:
        render_object(obj_path)

    print(f"\nAll renders saved to: {RENDER_DIR}")
    print("Next step: python data/preparation/pack_to_parquet.py")


main()
