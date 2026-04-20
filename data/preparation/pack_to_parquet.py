"""
Pack Blender renders into the parquet format expected by GObjaverseParquetDataset.

Usage (after running blender_transfer.py):
    cd ~/Gen_Final_Project
    PYTHONPATH=. python data/preparation/pack_to_parquet.py

Input:   data/chibi_renders/{obj_stem}/{00000-00039}.{png,json}
Output:  data/chibi_train/
            train.parquet, train.pkl
            val.parquet,   val.pkl
            prompt_embeds_sd15/null.npy
            prompt_embeds_sd15/{uid}.npy
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import io

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent
RENDER_DIR   = PROJECT_ROOT / "data" / "chibi_renders"
OUT_DIR      = PROJECT_ROOT / "data" / "chibi_train"
EMBED_DIR    = OUT_DIR / "prompt_embeds_sd15"

NUM_VIEWS = 40


# ── Helpers ───────────────────────────────────────────────────────────────

def png_to_bytes(png_path: Path) -> bytes:
    """Read PNG and re-encode to ensure standard format."""
    with open(png_path, "rb") as f:
        return f.read()


def build_row(obj_dir: Path, uid: str, caption: str) -> dict:
    row = {"uid": uid, "caption": caption}
    for i in range(NUM_VIEWS):
        png_path  = obj_dir / f"{i:05d}.png"
        json_path = obj_dir / f"{i:05d}.json"
        if not png_path.exists() or not json_path.exists():
            print(f"  WARNING: missing view {i:05d} for {uid}, skipping view")
            row[f"{i:05d}.png"]  = None
            row[f"{i:05d}.json"] = None
            continue
        row[f"{i:05d}.png"]  = png_to_bytes(png_path)
        row[f"{i:05d}.json"] = json_path.read_bytes()
    return row


def generate_caption(stem: str) -> str:
    """Simple caption from filename: 'lina' → 'a 3D character named lina'."""
    return f"a 3D character named {stem.replace('_', ' ')}"


def make_embed(caption: str) -> np.ndarray:
    """
    Generate SD1.5 CLIP text embedding (77, 768).
    Falls back to zeros if transformers is unavailable.
    """
    try:
        from transformers import CLIPTokenizer, CLIPTextModel
        import torch
        model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        text_enc  = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
        text_enc.eval()
        with torch.no_grad():
            tokens = tokenizer(
                [caption], padding="max_length", max_length=77,
                truncation=True, return_tensors="pt"
            )
            embed = text_enc(**tokens).last_hidden_state[0]  # (77, 768)
        return embed.float().numpy()
    except Exception as e:
        print(f"  [embed] falling back to zeros ({e})")
        return np.zeros((77, 768), dtype=np.float32)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    obj_dirs = sorted([d for d in RENDER_DIR.iterdir() if d.is_dir()])
    if not obj_dirs:
        print(f"No render directories found in {RENDER_DIR}")
        sys.exit(1)

    print(f"Found {len(obj_dirs)} object(s): {[d.name for d in obj_dirs]}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EMBED_DIR.mkdir(parents=True, exist_ok=True)

    # Split: last object → val, rest → train (for ≥2 objects)
    if len(obj_dirs) >= 2:
        train_dirs = obj_dirs[:-1]
        val_dirs   = obj_dirs[-1:]
    else:
        train_dirs = obj_dirs
        val_dirs   = obj_dirs   # single object: use same for both

    null_embed = np.zeros((77, 768), dtype=np.float32)
    np.save(EMBED_DIR / "null.npy", null_embed)
    print("Saved null.npy")

    for split_name, dirs in [("train", train_dirs), ("val", val_dirs)]:
        rows = []
        for i, obj_dir in enumerate(dirs):
            stem    = obj_dir.name
            uid     = f"chibi_{split_name}_{i:03d}"
            caption = generate_caption(stem)
            print(f"  [{split_name}] {uid}  caption='{caption}'  source={stem}")

            row = build_row(obj_dir, uid, caption)
            rows.append(row)

            # Prompt embed
            embed = make_embed(caption)
            np.save(EMBED_DIR / f"{uid}.npy", embed)

        df = pd.DataFrame(rows)
        df.to_parquet(OUT_DIR / f"{split_name}.parquet", index=False)
        with open(OUT_DIR / f"{split_name}.pkl", "wb") as f:
            pickle.dump(df, f)
        print(f"  Saved {split_name}.parquet + .pkl  ({len(df)} rows)")

    print(f"\nDone. Output: {OUT_DIR}")
    print("You can now train with:")
    print(f"  DIFFSPLAT_DATA_DIR={OUT_DIR} DIFFSPLAT_VAL_DIR={OUT_DIR} \\")
    print(f"  python src/train_gsdiff_sd.py --config_file configs/gsdiff_sd15.yaml \\")
    print(f"    opt.dataset_size={len(train_dirs)} opt.file_name_train=train opt.file_name_test=val \\")
    print(f"    opt.prompt_embed_dir={EMBED_DIR} opt.load_normal=false opt.load_coord=false")


if __name__ == "__main__":
    main()
