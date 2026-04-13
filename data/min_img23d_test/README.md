# Minimal SD1.5 image-to-3D test dataset

This package is a **synthetic smoke-test dataset** for checking whether your repo's
dataset loader / finetune pipeline can start.

## Included files
- `train.parquet` / `val.parquet` (or `.pkl` fallback if parquet engine is unavailable)
- `prompt_embeds_sd15/sample_train_000.npy`
- `prompt_embeds_sd15/sample_val_000.npy`
- `prompt_embeds_sd15/null.npy`

## Row schema
Each row contains:
- `uid`
- `caption`
- `00000.png` ... `00039.png` (PNG bytes, RGBA)
- `00000.json` ... `00039.json` (camera JSON strings with keys `x`, `y`, `z`, `origin`)

## Important
- This dataset does **not** include `*_nd.png`, `*_albedo.png`, or `*_mr.png`.
- Use:
  - `opt.load_normal=false`
  - `opt.load_coord=false`
- Prompt embeddings here are **dummy arrays** with shape `(77, 768)`.
  They are useful for smoke-testing data loading / training startup, but not for meaningful training quality.
- Full finetuning still requires your repo's pretrained checkpoints and all runtime dependencies.

## Notes
Parquet engine unavailable in this environment, so pickle files were written instead.
