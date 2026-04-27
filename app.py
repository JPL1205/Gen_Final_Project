"""
Gradio UI
Launch:
    conda activate diffsplat_sm120
    cd ~/Gen_Final_Project
    PYTHONPATH=. python app.py
"""

import os
import re
import sys
import time
import glob
import threading
import subprocess
from pathlib import Path
from typing import Optional

import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────── Constants ──────────────────────────────
PROJECT_ROOT   = Path(__file__).parent
OUT_DIR        = PROJECT_ROOT / "out"
DATA_DIR       = PROJECT_ROOT / "data"
CONDA_ENV      = "diffsplat_sm120"
MODEL_TEXT     = "gsdiff_gobj83k_sd15__render"
MODEL_IMAGE    = "gsdiff_gobj83k_sd15_image__render"
CHIBI_PHASE2_CONFIGS = {
    "text": "configs/gsdiff_sd15_chibi_text.yaml",
    "image": "configs/gsdiff_sd15_chibi_image.yaml",
}
DEFAULT_CHIBI_VARIANT = "image"
DEFAULT_CHIBI_CONFIG = CHIBI_PHASE2_CONFIGS[DEFAULT_CHIBI_VARIANT]

# ─────────────────────────── i18n strings ───────────────────────────
STRINGS = {
    "en": {
        # Header
        "header_html": """
        <div style="padding:14px 0 6px">
          <h1 style="margin:0;font-size:1.45rem;font-weight:800;color:#1e293b">
            🚀 GS-Diff Finetune GenModel &amp; Inference Lab
          </h1>
          <p style="margin:4px 0 0;font-size:0.75rem;color:#94a3b8;font-family:monospace">
            Base: gsdiff_gobj83k_sd15__render &nbsp;|&nbsp; Env: diffsplat_sm120
          </p>
        </div>""",
        # Tab 1
        "sec_config":       "### ⚙️ Setup",
        "lbl_config_file":  "Config File",
        "lbl_tag":          "Experiment Tag",
        "lbl_train_dir":    "Training Data Dir",
        "lbl_val_dir":      "Validation Data Dir",
        "lbl_ds_size":      "dataset_size",
        "lbl_fname_tr":     "file_name_train",
        "lbl_fname_val":    "file_name_val",
        "lbl_embed_dir":    "Prompt Embed Dir (optional)",
        "lbl_steps":        "Max Steps",
        "lbl_lr":           "Learning Rate",
        "lbl_batch":        "Batch / GPU",
        "lbl_accum":        "Grad Accum",
        "lbl_train_variant":"Training Type",
        "lbl_load_model":   "Load Pretrained UNet (empty = from backbone)",
        "lbl_lora_acc":     "LoRA Settings",
        "lbl_use_lora":     "Enable LoRA Finetuning",
        "lbl_lora_r":       "Rank r",
        "lbl_lora_a":       "Alpha",
        "btn_start":        "▶  Start Training",
        "btn_stop":         "■  Stop",
        "lbl_status":       "Status",
        "lbl_chart":        "Training Metrics",
        "btn_refresh":      "🔄 Refresh",
        "sec_terminal":     "**📟 Terminal Output**",
        "chart_titles":     ("Train Loss", "Learning Rate", "Val PSNR ↑", "Val LPIPS ↓"),
        # Tab 2
        "sec_filter":       "### 🔍 Filter Results",
        "lbl_val_exp":      "Experiment",
        "btn_refresh_exp":  "🔄 Refresh List",
        "sec_metrics":      "### 📊 Latest Validation Metrics",
        "lbl_psnr":         "PSNR ↑",
        "lbl_lpips":        "LPIPS ↓",
        "lbl_ssim":         "SSIM ↑",
        "lbl_gallery":      "Validation Renders (GT vs Pred, chronological)",
        "tip_val":          "_Tip: Images are multi-view composites saved by WandB offline._",
        # Tab 3
        "sec_input":        "### 🎨 Input",
        "tab_text":         "✏️ Text-to-3D",
        "tab_image":        "🖼 Image-to-3D",
        "lbl_prompt":       "Prompt",
        "ph_prompt":        "A shiny ceramic vase, studio lighting, white background",
        "lbl_neg_prompt":   "Negative Prompt (optional)",
        "lbl_image_in":     "Upload object image (white background preferred)",
        "sec_params":       "### ⚙️ Parameters",
        "lbl_cfg":          "Guidance Scale",
        "lbl_seed":         "Seed",
        "lbl_steps_inf":    "Diffusion Steps",
        "lbl_elevation":    "Image Elevation (deg)",
        "lbl_use_elevest":  "Estimate elevation automatically",
        "sec_ckpt":         "### 📂 Checkpoint",
        "lbl_infer_exp":    "Experiment",
        "lbl_infer_ckpt":   "Checkpoint Step",
        "sec_output_opt":   "### 💾 Output Options",
        "lbl_save_ply":     "Save .PLY file",
        "lbl_vid_type":     "Video format",
        "btn_gen":          "✨ Generate 3D",
        "lbl_infer_status": "Status",
        "sec_preview":      "### 🔭 Render Preview",
        "lbl_out_img":      "Multi-view PNG",
        "lbl_out_vid":      "360° Rotation (GIF / MP4)",
        "tip_infer":        "_Outputs are also saved to `out/{tag}/infer/`._",
        # Status messages
        "status_idle":      "⚪ Idle",
        "status_running":   "🟢 Training",
        "status_done_ok":   "✅ Finished",
        "status_done_err":  "❌ Failed",
        # Tab 0 — Data Prep
        "sec_data_prep":         "### 🗂️ Data Preparation Pipeline",
        "lbl_obj_dir":           "OBJ Directory (input)",
        "lbl_render_dir":        "Render Output Directory",
        "lbl_out_dir_parquet":   "Parquet Output Directory",
        "lbl_blender_exe":       "Blender Executable",
        "btn_run_blender":       "▶  Run Blender Render",
        "btn_pack_parquet":      "▶  Pack to Parquet",
        "btn_stop_data":         "■  Stop",
        "btn_refresh_data":      "🔄 Refresh",
        "lbl_data_status":       "Status",
        "sec_data_terminal":     "**📟 Terminal Output**",
        "dp_status_idle":        "⚪ Idle",
        "dp_status_running":     "🟢 Running",
        "dp_status_done_ok":     "✅ Done",
        "dp_status_done_err":    "❌ Failed",
    },
    "zh": {
        # Header
        "header_html": """
        <div style="padding:14px 0 6px">
          <h1 style="margin:0;font-size:1.45rem;font-weight:800;color:#1e293b">
            🚀 CS5788 GenModel Finetune & Inference /Diffsplat
          </h1>
          <p style="margin:4px 0 0;font-size:0.75rem;color:#94a3b8;font-family:monospace">
            底座: gsdiff_gobj83k_sd15__render &nbsp;|&nbsp; 环境: diffsplat_sm120
          </p>
        </div>""",
        # Tab 1
        "sec_config":       "### ⚙️ 配置",
        "lbl_config_file":  "配置文件",
        "lbl_tag":          "实验名称 (Tag)",
        "lbl_train_dir":    "训练数据目录",
        "lbl_val_dir":      "验证数据目录",
        "lbl_ds_size":      "dataset_size（样本数）",
        "lbl_fname_tr":     "file_name_train",
        "lbl_fname_val":    "file_name_val",
        "lbl_embed_dir":    "Prompt Embed 目录（可选，留空跳过）",
        "lbl_steps":        "最大步数 (Max Steps)",
        "lbl_lr":           "学习率 (LR)",
        "lbl_batch":        "每卡批量 (Batch)",
        "lbl_accum":        "梯度累积 (Grad Accum)",
        "lbl_train_variant":"训练类型",
        "lbl_load_model":   "加载预训练 UNet（空 = 从 backbone 初始化）",
        "lbl_lora_acc":     "LoRA 设置",
        "lbl_use_lora":     "启用 LoRA 微调",
        "lbl_lora_r":       "Rank r",
        "lbl_lora_a":       "Alpha",
        "btn_start":        "▶  启动训练",
        "btn_stop":         "■  停止",
        "lbl_status":       "状态",
        "lbl_chart":        "训练指标",
        "btn_refresh":      "🔄 刷新",
        "sec_terminal":     "**📟 终端输出**",
        "chart_titles":     ("训练 Loss", "学习率", "验证 PSNR ↑", "验证 LPIPS ↓"),
        # Tab 2
        "sec_filter":       "### 🔍 结果筛选",
        "lbl_val_exp":      "实验版本",
        "btn_refresh_exp":  "🔄 刷新列表",
        "sec_metrics":      "### 📊 关键指标（最新 Validation）",
        "lbl_psnr":         "PSNR ↑",
        "lbl_lpips":        "LPIPS ↓",
        "lbl_ssim":         "SSIM ↑",
        "lbl_gallery":      "Validation 渲染结果（GT vs Pred，按时间顺序）",
        "tip_val":          "_提示：图像为 WandB 离线保存的多视角拼图。_",
        # Tab 3
        "sec_input":        "### 🎨 输入",
        "tab_text":         "✏️ 文本生成",
        "tab_image":        "🖼 图像生成",
        "lbl_prompt":       "Prompt",
        "ph_prompt":        "A shiny ceramic vase, studio lighting, white background",
        "lbl_neg_prompt":   "负向提示词（可选）",
        "lbl_image_in":     "上传物体图片（建议白底）",
        "sec_params":       "### ⚙️ 参数",
        "lbl_cfg":          "引导强度 (CFG Scale)",
        "lbl_seed":         "随机种子",
        "lbl_steps_inf":    "扩散步数",
        "lbl_elevation":    "图像仰角（度）",
        "lbl_use_elevest":  "自动估计仰角",
        "sec_ckpt":         "### 📂 Checkpoint",
        "lbl_infer_exp":    "实验版本",
        "lbl_infer_ckpt":   "Checkpoint 步数",
        "sec_output_opt":   "### 💾 输出选项",
        "lbl_save_ply":     "保存 .PLY 文件",
        "lbl_vid_type":     "视频格式",
        "btn_gen":          "✨ 立即生成 3D",
        "lbl_infer_status": "状态",
        "sec_preview":      "### 🔭 渲染预览",
        "lbl_out_img":      "多视角拼图（PNG）",
        "lbl_out_vid":      "360° 旋转（GIF / MP4）",
        "tip_infer":        "_推理输出同时保存至 `out/{tag}/infer/` 目录。_",
        # Status messages
        "status_idle":      "⚪ 未运行",
        "status_running":   "🟢 训练中",
        "status_done_ok":   "✅ 已完成",
        "status_done_err":  "❌ 已失败",
        # Tab 0 — Data Prep
        "sec_data_prep":         "### 🗂️ 数据准备流水线",
        "lbl_obj_dir":           "OBJ 输入目录",
        "lbl_render_dir":        "渲染输出目录",
        "lbl_out_dir_parquet":   "Parquet 输出目录",
        "lbl_blender_exe":       "Blender 可执行文件",
        "btn_run_blender":       "▶  运行 Blender 渲染",
        "btn_pack_parquet":      "▶  打包为 Parquet",
        "btn_stop_data":         "■  停止",
        "btn_refresh_data":      "🔄 刷新",
        "lbl_data_status":       "状态",
        "sec_data_terminal":     "**📟 终端输出**",
        "dp_status_idle":        "⚪ 未运行",
        "dp_status_running":     "🟢 运行中",
        "dp_status_done_ok":     "✅ 已完成",
        "dp_status_done_err":    "❌ 已失败",
    },
}

# ─────────────────────────── Global state ──────────────────────────
class _State:
    proc:      Optional[subprocess.Popen] = None
    log_lines: list = []
    metrics:   dict = dict(
        step=[], loss=[], lr=[],
        val_step=[], val_psnr=[], val_lpips=[], val_ssim=[]
    )
    lock: threading.Lock = threading.Lock()
    lang: str = "en"

S = _State()


class _DataPrepState:
    proc:      Optional[subprocess.Popen] = None
    log_lines: list = []
    lock: threading.Lock = threading.Lock()

DS = _DataPrepState()


# ══════════════════════════════════════════════════════════════
#  Utilities
# ══════════════════════════════════════════════════════════════

def _list_configs() -> list:
    cfg_dir = PROJECT_ROOT / "configs"
    if not cfg_dir.exists():
        return []
    return sorted(
        str(p.relative_to(PROJECT_ROOT))
        for p in cfg_dir.iterdir()
        if p.is_file() and p.name.endswith(".yaml") and ":Zone.Identifier" not in p.name
    )


def _infer_training_variant(config_file: str) -> str:
    stem = Path(config_file).stem.lower()
    if stem.endswith("_image"):
        return "image"
    if stem.endswith("_text"):
        return "text"
    return "image" if "image" in stem else "text"


def _resolve_training_config(variant: str) -> str:
    return CHIBI_PHASE2_CONFIGS.get(variant, DEFAULT_CHIBI_CONFIG)


def _list_experiments() -> list:
    if not OUT_DIR.exists():
        return []
    result = []
    for d in OUT_DIR.iterdir():
        ckpt_dir = d / "checkpoints"
        if d.is_dir() and ckpt_dir.exists() and any(ckpt_dir.iterdir()):
            result.append(d.name)
    return sorted(result)


def _list_checkpoints(tag: str) -> list:
    ckpt_dir = OUT_DIR / tag / "checkpoints"
    if not ckpt_dir.exists():
        return []
    return sorted([d.name for d in ckpt_dir.iterdir() if d.is_dir()])


def _get_unet_in_channels(tag: str, ckpt_step) -> Optional[int]:
    import json
    # Normalize: "013020" or 13020 → "013020"
    ckpt_folder = f"{int(ckpt_step):06d}"
    for subfolder in ("unet_ema", "unet"):
        config_path = OUT_DIR / tag / "checkpoints" / ckpt_folder / subfolder / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    return json.load(f).get("in_channels")
            except Exception:
                return None
    return None


def _get_val_images(tag: str) -> list:
    pattern = str(OUT_DIR / tag / "wandb" / "offline-run-*"
                  / "files" / "media" / "images" / "images" / "*.png")
    return sorted(glob.glob(pattern))


def _parse_train_line(line: str):
    m = re.search(
        r'\|\s*(\d+)/\d+\s*\[.*?loss=([\d.e+\-]+|nan).*?lr=([\d.e+\-]+)',
        line
    )
    if m:
        step  = int(m.group(1))
        loss_s = m.group(2)
        loss  = None if loss_s == "nan" else float(loss_s)
        lr    = float(m.group(3))
        return step, loss, lr
    return None, None, None


def _parse_val_line(line: str):
    psnr  = re.search(r'psnr=([\d.e+\-]+)',  line)
    lpips = re.search(r'lpips=([\d.e+\-]+)', line)
    ssim  = re.search(r'ssim=([\d.e+\-]+)',  line)
    if psnr and lpips:
        return (float(psnr.group(1)),
                float(lpips.group(1)),
                float(ssim.group(1)) if ssim else None)
    return None, None, None


def _stream_proc(proc: subprocess.Popen):
    for raw in iter(proc.stdout.readline, b""):
        line = raw.decode("utf-8", errors="replace").rstrip()
        with S.lock:
            S.log_lines.append(line)
            if len(S.log_lines) > 600:
                S.log_lines = S.log_lines[-600:]
        step, loss, lr = _parse_train_line(line)
        if step is not None and loss is not None:
            with S.lock:
                S.metrics["step"].append(step)
                S.metrics["loss"].append(loss)
                S.metrics["lr"].append(lr)
        p, l, s = _parse_val_line(line)
        if p is not None:
            with S.lock:
                vs = S.metrics["step"][-1] if S.metrics["step"] else 0
                S.metrics["val_step"].append(vs)
                S.metrics["val_psnr"].append(p)
                S.metrics["val_lpips"].append(l)
                if s is not None:
                    S.metrics["val_ssim"].append(s)
    proc.stdout.close()


def _ensure_wandb_token():
    token_path = PROJECT_ROOT / "wandb" / "token"
    token_path.parent.mkdir(exist_ok=True)
    if not token_path.exists():
        token_path.write_text("offline")


# ══════════════════════════════════════════════════════════════
#  Tab 0 — Data Preparation
# ══════════════════════════════════════════════════════════════

def _stream_data_prep(proc: subprocess.Popen):
    for raw in iter(proc.stdout.readline, b""):
        line = raw.decode("utf-8", errors="replace").rstrip()
        with DS.lock:
            DS.log_lines.append(line)
            if len(DS.log_lines) > 600:
                DS.log_lines = DS.log_lines[-600:]
    proc.stdout.close()


def _data_prep_busy() -> bool:
    with DS.lock:
        return DS.proc is not None and DS.proc.poll() is None


def start_blender_render(obj_dir: str, render_dir: str, blender_exe: str):
    if _data_prep_busy():
        return "⚠️ A process is already running. Stop it first."
    with DS.lock:
        DS.log_lines.clear()
    script = str(PROJECT_ROOT / "data" / "preparation" / "blender_transfer.py")
    cmd = (
        f"source ~/miniconda3/etc/profile.d/conda.sh && "
        f"conda activate {CONDA_ENV} && "
        f"cd {PROJECT_ROOT} && "
        f"{blender_exe} --background --python {script} "
        f"-- --obj_dir {obj_dir} --render_dir {render_dir}"
    )
    proc = subprocess.Popen(
        ["bash", "-c", cmd],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
    )
    with DS.lock:
        DS.proc = proc
    threading.Thread(target=_stream_data_prep, args=(proc,), daemon=True).start()
    return f"🟢 Blender render started  PID {proc.pid}"


def start_pack_parquet(render_dir: str, out_dir_parquet: str):
    if _data_prep_busy():
        return "⚠️ A process is already running. Stop it first."
    with DS.lock:
        DS.log_lines.clear()
    script = str(PROJECT_ROOT / "data" / "preparation" / "pack_to_parquet.py")
    cmd = (
        f"source ~/miniconda3/etc/profile.d/conda.sh && "
        f"conda activate {CONDA_ENV} && "
        f"cd {PROJECT_ROOT} && "
        f"PYTHONPATH=. python {script} "
        f"--render_dir {render_dir} --out_dir {out_dir_parquet}"
    )
    proc = subprocess.Popen(
        ["bash", "-c", cmd],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
    )
    with DS.lock:
        DS.proc = proc
    threading.Thread(target=_stream_data_prep, args=(proc,), daemon=True).start()
    return f"🟢 Pack-to-parquet started  PID {proc.pid}"


def stop_data_prep():
    with DS.lock:
        if DS.proc is not None and DS.proc.poll() is None:
            DS.proc.terminate()
            return "🛑 Stop signal sent."
    return "ℹ️ No process running."


def get_data_prep_logs():
    with DS.lock:
        return "\n".join(DS.log_lines[-100:])


def get_data_prep_status():
    with DS.lock:
        if DS.proc is None:
            return STRINGS[S.lang]["dp_status_idle"]
        if DS.proc.poll() is None:
            return STRINGS[S.lang]["dp_status_running"]
        rc  = DS.proc.returncode
        key = "dp_status_done_ok" if rc == 0 else "dp_status_done_err"
        return f"{STRINGS[S.lang][key]}  (returncode={rc})"


# ══════════════════════════════════════════════════════════════
#  Tab 1 — Training
# ══════════════════════════════════════════════════════════════

def auto_fill_from_config(config_file: str):
    """Read a config yaml and return sensible UI defaults for all training fields."""
    import yaml

    tag = "my_finetune"
    train_dir = str(DATA_DIR / "min_img23d_test")
    val_dir = str(DATA_DIR / "min_img23d_test")
    ds_size = 1
    fname_tr = "train"
    fname_vl = "val"
    embed_dir = ""
    steps = 5000
    lr = "1e-4"
    batch = 1
    accum = 1
    load_model = MODEL_TEXT

    if not config_file:
        return tag, train_dir, val_dir, ds_size, fname_tr, fname_vl, embed_dir, steps, lr, batch, accum, load_model

    cfg_path = PROJECT_ROOT / config_file.strip()
    if not cfg_path.exists():
        return tag, train_dir, val_dir, ds_size, fname_tr, fname_vl, embed_dir, steps, lr, batch, accum, load_model

    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        tag = Path(config_file).stem + "_run"

        if "optimizer" in cfg and "lr" in cfg["optimizer"]:
            lr = str(cfg["optimizer"]["lr"])
        if "train" in cfg:
            tr = cfg["train"]
            if "batch_size_per_gpu" in tr:
                batch = int(tr["batch_size_per_gpu"])
        ui_cfg = cfg.get("ui", {}) or {}
        if ui_cfg.get("load_pretrained_model"):
            load_model = str(ui_cfg["load_pretrained_model"])
        else:
            opt_cfg = cfg.get("opt", {}) or {}
            if opt_cfg.get("view_concat_condition") or opt_cfg.get("input_concat_binary_mask"):
                load_model = MODEL_IMAGE

        opt_type = cfg.get("opt_type", "")
        if opt_type:
            try:
                from src.options import opt_dict
                if opt_type in opt_dict:
                    opt = opt_dict[opt_type]
                    if opt.file_dir_train:
                        train_dir = opt.file_dir_train
                    if opt.file_dir_test:
                        val_dir = opt.file_dir_test
                    if opt.file_name_train:
                        fname_tr = opt.file_name_train
                    if opt.file_name_test:
                        fname_vl = opt.file_name_test
                    if opt.prompt_embed_dir:
                        embed_dir = opt.prompt_embed_dir
                    if opt.dataset_size:
                        ds_size = opt.dataset_size
            except Exception:
                pass

        if "train" in cfg and "epochs" in cfg["train"]:
            epochs = int(cfg["train"]["epochs"])
            steps = min(50000, max(epochs, epochs * ds_size // max(1, batch)))

    except Exception:
        pass

    return tag, train_dir, val_dir, ds_size, fname_tr, fname_vl, embed_dir, steps, lr, batch, accum, load_model


def apply_training_config(config_file: str):
    return (_infer_training_variant(config_file), *auto_fill_from_config(config_file))


def apply_training_variant(variant: str):
    config_file = _resolve_training_config(variant)
    return (config_file, *auto_fill_from_config(config_file))


def start_training(tag, config_file, train_dir, val_dir, dataset_size,
                   fname_train, fname_val, embed_dir,
                   steps, lr, batch_size, grad_accum,
                   use_lora, lora_r, lora_alpha, load_model):
    with S.lock:
        if S.proc is not None and S.proc.poll() is None:
            t = S.lang
            return "⚠️ Training already running. Stop it first." if t == "en" \
                   else "⚠️ 训练已在运行中，请先点击「停止」。"
        S.log_lines.clear()
        for k in S.metrics:
            S.metrics[k].clear()

    _ensure_wandb_token()

    cfg = config_file.strip() if config_file and config_file.strip() else "configs/gsdiff_sd15.yaml"
    env_vars = (
        f"PYTHONPATH=. "
        f"HF_HOME=~/.cache/huggingface "
        f"TORCH_HOME=~/.cache/torch "
    )
    cmd_parts = [
        "accelerate launch --num_processes 1",
        "src/train_gsdiff_sd.py",
        f"--config_file {cfg}",
        f"--tag {tag}",
        f"--max_train_steps {int(steps)}",
        "--num_workers 4 --allow_tf32 --offline_wandb",
        f"--gradient_accumulation_steps {int(grad_accum)}",
    ]
    if load_model and load_model.strip():
        cmd_parts.append(f"--load_pretrained_model {load_model.strip()}")

    overrides = [
        f"optimizer.lr={lr}",
        f"train.batch_size_per_gpu={int(batch_size)}",
        f"opt.dataset_size={int(dataset_size)}",
        f"opt.file_dir_train={train_dir}",
        f"opt.file_dir_test={val_dir}",
        f"opt.file_name_train={fname_train}",
        f"opt.file_name_test={fname_val}",
        "opt.load_normal=false",
        "opt.load_coord=false",
    ]
    if embed_dir and embed_dir.strip():
        overrides.append(f"opt.prompt_embed_dir={embed_dir.strip()}")
    if use_lora:
        overrides += [
            "opt.use_peft=true",
            f"opt.peft_r={int(lora_r)}",
            f"opt.peft_alpha={int(lora_alpha)}",
        ]

    full_cmd = (
        f"source ~/miniconda3/etc/profile.d/conda.sh && "
        f"conda activate {CONDA_ENV} && "
        f"cd {PROJECT_ROOT} && "
        f"{env_vars} {' '.join(cmd_parts + overrides)}"
    )
    proc = subprocess.Popen(
        ["bash", "-c", full_cmd],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
    )
    with S.lock:
        S.proc = proc
    threading.Thread(target=_stream_proc, args=(proc,), daemon=True).start()
    return f"🟢 Started  PID {proc.pid}  tag={tag}"


def stop_training():
    with S.lock:
        if S.proc is not None and S.proc.poll() is None:
            S.proc.terminate()
            return "🛑 Stop signal sent." if S.lang == "en" else "🛑 已发送停止信号。"
    return "ℹ️ No training running." if S.lang == "en" else "ℹ️ 当前没有运行中的训练。"


def get_train_status():
    with S.lock:
        if S.proc is None:
            return STRINGS[S.lang]["status_idle"]
        if S.proc.poll() is None:
            step = S.metrics["step"][-1] if S.metrics["step"] else 0
            loss = f'{S.metrics["loss"][-1]:.4f}' if S.metrics["loss"] else "—"
            lr   = f'{S.metrics["lr"][-1]:.2e}'   if S.metrics["lr"]   else "—"
            base = STRINGS[S.lang]["status_running"]
            return f"{base}  │  Step {step}  │  Loss {loss}  │  LR {lr}"
        rc   = S.proc.returncode
        key  = "status_done_ok" if rc == 0 else "status_done_err"
        return f"{STRINGS[S.lang][key]}  (returncode={rc})"


def get_logs():
    with S.lock:
        return "\n".join(S.log_lines[-100:])


def get_charts(lang="en"):
    with S.lock:
        steps     = list(S.metrics["step"])
        losses    = list(S.metrics["loss"])
        lrs       = list(S.metrics["lr"])
        val_steps = list(S.metrics["val_step"])
        val_psnr  = list(S.metrics["val_psnr"])
        val_lpips = list(S.metrics["val_lpips"])

    titles = STRINGS[lang]["chart_titles"]
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=titles,
        vertical_spacing=0.18, horizontal_spacing=0.12,
    )
    ORANGE, BLUE, GREEN = "#ea580c", "#3b82f6", "#10b981"

    if steps:
        fig.add_trace(go.Scatter(x=steps, y=losses, mode="lines",
            line=dict(color=ORANGE, width=2), name="Loss"), row=1, col=1)
        fig.add_trace(go.Scatter(x=steps, y=lrs, mode="lines",
            line=dict(color=BLUE, width=2), name="LR"), row=1, col=2)
    if val_psnr:
        fig.add_trace(go.Scatter(x=val_steps, y=val_psnr, mode="lines+markers",
            line=dict(color=GREEN, width=2), marker=dict(size=6), name="PSNR"), row=2, col=1)
        fig.add_trace(go.Scatter(x=val_steps, y=val_lpips, mode="lines+markers",
            line=dict(color=ORANGE, width=2), marker=dict(size=6), name="LPIPS"), row=2, col=2)

    fig.update_layout(
        height=420,
        margin=dict(l=50, r=20, t=45, b=30),
        paper_bgcolor="white", plot_bgcolor="#f8fafc",
        showlegend=False,
        font=dict(size=11, family="monospace"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e2e8f0", linecolor="#cbd5e1")
    fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0", linecolor="#cbd5e1")
    return fig


def get_charts_auto():
    return get_charts(S.lang)


# ══════════════════════════════════════════════════════════════
#  Tab 2 — Validation viewer
# ══════════════════════════════════════════════════════════════

def refresh_experiments():
    exps = _list_experiments()
    return gr.Dropdown(choices=exps, value=exps[0] if exps else None)


def load_val_view(tag: str):
    if not tag:
        return [], "—", "—", "—"
    images = _get_val_images(tag)
    log_path = OUT_DIR / tag / "log.txt"
    psnr_s = lpips_s = ssim_s = "—"
    if log_path.exists():
        text = log_path.read_text(errors="replace")
        psnrs  = re.findall(r'psnr=([\d.]+)', text)
        lpipss = re.findall(r'lpips=([\d.]+)', text)
        ssims  = re.findall(r'ssim=([\d.]+)', text)
        if psnrs:  psnr_s  = f"{float(psnrs[-1]):.3f}"
        if lpipss: lpips_s = f"{float(lpipss[-1]):.3f}"
        if ssims:  ssim_s  = f"{float(ssims[-1]):.3f}"
    return images, psnr_s, lpips_s, ssim_s


# ══════════════════════════════════════════════════════════════
#  Tab 3 — Inference
# ══════════════════════════════════════════════════════════════

def update_ckpt_list(tag: str):
    ckpts = _list_checkpoints(tag) if tag else []
    return gr.Dropdown(choices=ckpts, value=ckpts[-1] if ckpts else None)


def _progress_desc(stage: str, value: float) -> str:
    value = max(0.0, min(1.0, float(value)))
    return f"{stage} - {value * 100:.1f}%"


def _update_progress(progress, current: float, target: float, stage: str):
    value = max(current, target)
    progress(value, desc=_progress_desc(stage, value))
    return value


def _collect_recent_outputs(infer_dir: Path, start_time: float):
    files = []
    for ext in ("*.png", "*.gif", "*.mp4"):
        files.extend(p for p in infer_dir.glob(ext) if p.is_file())
    recent = [p for p in files if p.stat().st_mtime >= start_time - 1.0]
    recent.sort(key=lambda p: (p.stat().st_mtime, p.name))
    return recent


def _pick_output_pair(infer_dir: Path, start_time: float):
    recent = _collect_recent_outputs(infer_dir, start_time)
    image_candidates = [p for p in recent if p.name.endswith("_gs.png")]
    video_candidates = [p for p in recent if p.suffix.lower() in {".gif", ".mp4"}]

    out_image = str(image_candidates[-1]) if image_candidates else None
    out_video = None

    if out_image:
        image_stem = Path(out_image).name[:-7]  # trim "_gs.png"
        matching_videos = [p for p in video_candidates if p.stem == image_stem]
        if matching_videos:
            matching_videos.sort(key=lambda p: (p.stat().st_mtime, p.name))
            out_video = str(matching_videos[-1])

    if out_video is None and video_candidates:
        out_video = str(video_candidates[-1])

    if out_image is None:
        fallback_pngs = sorted(
            [p for p in infer_dir.glob("*_gs.png") if p.is_file()],
            key=lambda p: (p.stat().st_mtime, p.name),
        )
        out_image = str(fallback_pngs[-1]) if fallback_pngs else None

    if out_video is None:
        fallback_videos = sorted(
            [p for p in infer_dir.glob("*.gif") if p.is_file()] +
            [p for p in infer_dir.glob("*.mp4") if p.is_file()],
            key=lambda p: (p.stat().st_mtime, p.name),
        )
        out_video = str(fallback_videos[-1]) if fallback_videos else None

    return out_image, out_video


def _load_experiment_params(tag: str) -> dict:
    import yaml

    params_path = OUT_DIR / tag / "params.yaml"
    if not params_path.exists():
        return {}
    try:
        with open(params_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _get_unet_in_channels(tag: str, ckpt_step, _visited=None) -> Optional[int]:
    import json

    if _visited is None:
        _visited = set()

    try:
        ckpt_int = int(ckpt_step)
    except Exception:
        return None

    visit_key = (tag, ckpt_int)
    if visit_key in _visited:
        return None
    _visited.add(visit_key)

    ckpt_root = OUT_DIR / tag / "checkpoints" / f"{ckpt_int:06d}"
    for subfolder in ("unet_ema", "unet"):
        config_path = ckpt_root / subfolder / "config.json"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    return json.load(f).get("in_channels")
            except Exception:
                return None

    # LoRA checkpoints only save adapter weights, so resolve the base
    # checkpoint recorded in params.yaml and inspect that UNet config instead.
    if (ckpt_root / "unet_lora").exists():
        params = _load_experiment_params(tag)
        base_tag = params.get("load_pretrained_model")
        base_ckpt = params.get("load_pretrained_model_ckpt")
        if base_tag and base_ckpt not in (None, "", "None"):
            return _get_unet_in_channels(str(base_tag), base_ckpt, _visited)

    return None


def run_inference(prompt, image, cfg_scale, seed, num_steps, elevation, use_elevest,
                  exp_tag, ckpt_step, save_ply, output_video_type,
                  progress=gr.Progress()):
    if not exp_tag:
        return None, None, "⚠️ Select an experiment first."
    if not ckpt_step:
        return None, None, "⚠️ Select a checkpoint."
    if not prompt and image is None:
        return None, None, "⚠️ Enter a prompt or upload an image."

    in_ch = _get_unet_in_channels(exp_tag, ckpt_step)
    if image is not None:
        if in_ch is None:
            return None, None, (
                f"⚠️ Cannot read checkpoint config for [{exp_tag} / {ckpt_step}].\n"
                "Image conditioning cannot be verified. Please use a text prompt only."
            )
        if in_ch < 11:
            return None, None, (
                f"⚠️ Checkpoint [{exp_tag}] has {in_ch} input channels — text-only model.\n"
                "Image conditioning requires an image-conditioned checkpoint (11ch).\n"
                "Please:\n"
                "  1) Use a text prompt only with this checkpoint, OR\n"
                "  2) Download: python download_ckpt.py --model_type sd15 --image_cond"
            )

    image_tmp_path = None
    if image is not None:
        from PIL import Image as PILImage
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir="/tmp")
        PILImage.fromarray(image.astype("uint8")).save(tmp.name)
        image_tmp_path = tmp.name

    cmd = [
        sys.executable, "src/infer_gsdiff_sd.py",
        "--config_file", "configs/gsdiff_sd15.yaml",
        "--tag", exp_tag,
        "--infer_from_iter", str(int(ckpt_step)),
        "--output_dir", str(OUT_DIR),
        "--guidance_scale", str(cfg_scale),
        "--seed", str(int(seed)),
        "--num_inference_steps", str(int(num_steps)),
        "--half_precision", "--allow_tf32",
        "--prompt", prompt,
    ]
    if image_tmp_path:
        cmd += ["--image_path", image_tmp_path]
        if use_elevest:
            cmd.append("--use_elevest")
        else:
            cmd += ["--elevation", str(float(elevation))]
    if output_video_type != "none":
        cmd += ["--output_video_type", output_video_type]
    if save_ply:
        cmd.append("--save_ply")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    infer_dir = OUT_DIR / exp_tag / "inference"
    start_time = time.time()
    progress_value = 0.0
    progress_value = _update_progress(progress, progress_value, 0.0, "Starting inference...")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        cwd=str(PROJECT_ROOT), env=env,
    )
    log_buf = []
    stage_patterns = [
        ("Load image", 0.08),
        ("No image condition", 0.08),
        ("Load checkpoint", 0.22),
        ("Load GSVAE checkpoint", 0.32),
        ("Load GSRecon checkpoint", 0.42),
        ("Load ElevEst checkpoint", 0.48),
        ("Elevation estimation", 0.55),
        ("Rendering", 0.70),
        ("Inference finished!", 0.98),
    ]
    for raw in iter(proc.stdout.readline, b""):
        line = raw.decode("utf-8", errors="replace").rstrip()
        log_buf.append(line)
        for marker, target in stage_patterns:
            if marker in line:
                progress_value = _update_progress(progress, progress_value, target, marker)
                break

        render_match = re.search(r"Rendering:\s+(\d+)%", line)
        if render_match:
            render_pct = int(render_match.group(1)) / 100.0
            progress_value = _update_progress(
                progress,
                progress_value,
                0.70 + 0.28 * render_pct,
                "Rendering",
            )
    proc.stdout.close()
    proc.wait()

    if proc.returncode != 0:
        return None, None, "❌ Inference failed:\n" + "\n".join(log_buf[-20:])

    if not infer_dir.exists():
        candidates = sorted((OUT_DIR / exp_tag).glob("infer*"))
        infer_dir  = candidates[-1] if candidates else OUT_DIR / exp_tag

    out_image, out_video = _pick_output_pair(infer_dir, start_time)
    progress(1.0, desc=_progress_desc("Done", 1.0))
    return out_image, out_video, f"✅ Done. Output dir: {infer_dir}"


# ══════════════════════════════════════════════════════════════
#  Language switch
# ══════════════════════════════════════════════════════════════

def switch_lang(lang_label: str):
    """Return gr.update() for every translatable component."""
    lang = "zh" if lang_label == "中文" else "en"
    with S.lock:
        S.lang = lang
    t = STRINGS[lang]

    return (
        gr.update(value=t["header_html"]),
        # Tab 0 — Data Prep
        gr.update(value=t["sec_data_prep"]),
        gr.update(label=t["lbl_obj_dir"]),
        gr.update(label=t["lbl_render_dir"]),
        gr.update(label=t["lbl_out_dir_parquet"]),
        gr.update(label=t["lbl_blender_exe"]),
        gr.update(value=t["btn_run_blender"]),
        gr.update(value=t["btn_pack_parquet"]),
        gr.update(value=t["btn_stop_data"]),
        gr.update(label=t["lbl_data_status"]),
        gr.update(value=t["sec_data_terminal"]),
        gr.update(value=t["btn_refresh_data"]),
        # Tab 1 — Training
        gr.update(value=t["sec_config"]),
        gr.update(label=t["lbl_config_file"]),
        gr.update(label=t["lbl_tag"]),
        gr.update(label=t["lbl_train_dir"]),
        gr.update(label=t["lbl_val_dir"]),
        gr.update(label=t["lbl_ds_size"]),
        gr.update(label=t["lbl_fname_tr"]),
        gr.update(label=t["lbl_fname_val"]),
        gr.update(label=t["lbl_embed_dir"]),
        gr.update(label=t["lbl_steps"]),
        gr.update(label=t["lbl_lr"]),
        gr.update(label=t["lbl_batch"]),
        gr.update(label=t["lbl_accum"]),
        gr.update(label=t["lbl_train_variant"]),
        gr.update(label=t["lbl_load_model"]),
        gr.update(label=t["lbl_lora_acc"]),
        gr.update(label=t["lbl_use_lora"]),
        gr.update(label=t["lbl_lora_r"]),
        gr.update(label=t["lbl_lora_a"]),
        gr.update(value=t["btn_start"]),
        gr.update(value=t["btn_stop"]),
        gr.update(label=t["lbl_status"]),
        gr.update(label=t["lbl_chart"], value=get_charts(lang)),
        gr.update(value=t["btn_refresh"]),
        gr.update(value=t["sec_terminal"]),
        gr.update(value=t["sec_filter"]),
        gr.update(label=t["lbl_val_exp"]),
        gr.update(value=t["btn_refresh_exp"]),
        gr.update(value=t["sec_metrics"]),
        gr.update(label=t["lbl_psnr"]),
        gr.update(label=t["lbl_lpips"]),
        gr.update(label=t["lbl_ssim"]),
        gr.update(label=t["lbl_gallery"]),
        gr.update(value=t["tip_val"]),
        gr.update(value=t["sec_input"]),
        gr.update(label=t["lbl_prompt"]),
        gr.update(label=t["lbl_neg_prompt"]),
        gr.update(label=t["lbl_image_in"]),
        gr.update(value=t["sec_params"]),
        gr.update(label=t["lbl_cfg"]),
        gr.update(label=t["lbl_seed"]),
        gr.update(label=t["lbl_steps_inf"]),
        gr.update(label=t["lbl_elevation"]),
        gr.update(label=t["lbl_use_elevest"]),
        gr.update(value=t["sec_ckpt"]),
        gr.update(label=t["lbl_infer_exp"]),
        gr.update(label=t["lbl_infer_ckpt"]),
        gr.update(value=t["sec_output_opt"]),
        gr.update(label=t["lbl_save_ply"]),
        gr.update(label=t["lbl_vid_type"]),
        gr.update(value=t["btn_gen"]),
        gr.update(label=t["lbl_infer_status"]),
        gr.update(value=t["sec_preview"]),
        gr.update(label=t["lbl_out_img"]),
        gr.update(label=t["lbl_out_vid"]),
        gr.update(value=t["tip_infer"]),
    )


# ══════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════

CSS = """
body, .gradio-container { font-family: 'Inter', 'Helvetica Neue', sans-serif !important; }
.tab-nav button.selected {
    border-bottom: 2px solid #ea580c !important;
    color: #ea580c !important; font-weight: 700 !important;
}
.btn-orange { background: #ea580c !important; border-color: #ea580c !important; }
.btn-orange:hover { background: #c2410c !important; }
.btn-stop { background: #475569 !important; border-color: #475569 !important; }
.btn-stop:hover { background: #334155 !important; }
.terminal textarea {
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    font-size: 11.5px !important;
    background: #0f172a !important; color: #94a3b8 !important;
    border: none !important; padding: 12px !important;
    border-radius: 12px !important; line-height: 1.6 !important;
}
.metric-card textarea {
    font-size: 1.3rem !important; font-weight: 900; color: #1e293b !important;
}
.lang-radio label { font-size: 0.78rem !important; }
.lang-radio .wrap { gap: 4px !important; }
"""


# ══════════════════════════════════════════════════════════════
#  Build UI
# ══════════════════════════════════════════════════════════════

def build_ui():
    L = STRINGS["en"]   # initial language = English

    with gr.Blocks(title="GS-Diff Lab", css=CSS) as demo:

        # ─── Header ───
        with gr.Row(equal_height=True):
            with gr.Column(scale=10):
                header_html = gr.HTML(value=L["header_html"])
            with gr.Column(scale=2, min_width=130):
                lang_radio = gr.Radio(
                    choices=["English", "中文"],
                    value="English",
                    label="",
                    container=False,
                    elem_classes="lang-radio",
                )

        with gr.Tabs(elem_classes="tab-nav"):

            # ══════════ TAB 0 ══════════
            with gr.Tab("0 · Data Prep"):
                with gr.Row(equal_height=False):

                    # Left panel — path inputs + controls
                    with gr.Column(scale=4, min_width=320):
                        sec_data_prep = gr.Markdown(L["sec_data_prep"])
                        dp_obj_dir = gr.Textbox(
                            label=L["lbl_obj_dir"],
                            value=str(DATA_DIR / "chibi_train_obj_data"),
                        )
                        dp_render_dir = gr.Textbox(
                            label=L["lbl_render_dir"],
                            value=str(DATA_DIR / "chibi_renders"),
                        )
                        dp_out_dir = gr.Textbox(
                            label=L["lbl_out_dir_parquet"],
                            value=str(DATA_DIR / "chibi_train"),
                        )
                        dp_blender_exe = gr.Textbox(
                            label=L["lbl_blender_exe"],
                            value="blender",
                        )
                        gr.Markdown("---")
                        with gr.Row():
                            dp_btn_blender  = gr.Button(L["btn_run_blender"],
                                                        variant="primary",
                                                        elem_classes="btn-orange", scale=2)
                            dp_btn_parquet  = gr.Button(L["btn_pack_parquet"],
                                                        variant="primary",
                                                        elem_classes="btn-orange", scale=2)
                            dp_btn_stop     = gr.Button(L["btn_stop_data"],
                                                        elem_classes="btn-stop", scale=1)
                        dp_status_bar = gr.Textbox(
                            label=L["lbl_data_status"],
                            value=L["dp_status_idle"],
                            interactive=False, max_lines=1,
                        )

                    # Right panel — terminal
                    with gr.Column(scale=8):
                        with gr.Row():
                            dp_sec_terminal = gr.Markdown(L["sec_data_terminal"])
                            dp_btn_refresh  = gr.Button(L["btn_refresh_data"], size="sm", scale=0)
                        dp_log_box = gr.Textbox(
                            value="", lines=25, max_lines=25,
                            interactive=False, show_label=False,
                            elem_classes="terminal",
                        )

                dp_timer = gr.Timer(value=3)
                dp_timer.tick(fn=get_data_prep_logs,   outputs=[dp_log_box])
                dp_timer.tick(fn=get_data_prep_status, outputs=[dp_status_bar])

                dp_btn_blender.click(
                    fn=start_blender_render,
                    inputs=[dp_obj_dir, dp_render_dir, dp_blender_exe],
                    outputs=[dp_status_bar],
                )
                dp_btn_parquet.click(
                    fn=start_pack_parquet,
                    inputs=[dp_render_dir, dp_out_dir],
                    outputs=[dp_status_bar],
                )
                dp_btn_stop.click(fn=stop_data_prep, outputs=[dp_status_bar])
                dp_btn_refresh.click(fn=get_data_prep_logs,   outputs=[dp_log_box])
                dp_btn_refresh.click(fn=get_data_prep_status, outputs=[dp_status_bar])

            # ══════════ TAB 1 ══════════
            with gr.Tab("1 · Training Monitor"):
                with gr.Row(equal_height=False):

                    # Left panel
                    with gr.Column(scale=4, min_width=320):
                        sec_config   = gr.Markdown(L["sec_config"])
                        train_variant = gr.Radio(
                            label=L["lbl_train_variant"],
                            choices=["text", "image"],
                            value=DEFAULT_CHIBI_VARIANT,
                        )
                        config_dd = gr.Dropdown(
                            label=L["lbl_config_file"],
                            choices=_list_configs(),
                            value=DEFAULT_CHIBI_CONFIG,
                            allow_custom_value=True,
                        )
                        tag_in       = gr.Textbox(label=L["lbl_tag"], value=Path(DEFAULT_CHIBI_CONFIG).stem + "_run")
                        train_dir    = gr.Textbox(label=L["lbl_train_dir"],
                                                  value=str(DATA_DIR / "chibi_train"))
                        val_dir      = gr.Textbox(label=L["lbl_val_dir"],
                                                  value=str(DATA_DIR / "chibi_train"))
                        with gr.Row():
                            ds_size  = gr.Number(label=L["lbl_ds_size"],  value=1,   precision=0, min_width=90)
                            fname_tr = gr.Textbox(label=L["lbl_fname_tr"], value="train", min_width=100)
                            fname_vl = gr.Textbox(label=L["lbl_fname_val"], value="val",  min_width=100)
                        embed_dir = gr.Textbox(
                            label=L["lbl_embed_dir"],
                            value=str(DATA_DIR / "chibi_train" / "prompt_embeds_sd15"),
                        )
                        with gr.Row():
                            steps_in = gr.Number(label=L["lbl_steps"], value=500, precision=0)
                            lr_in    = gr.Textbox(label=L["lbl_lr"],   value="1e-5")
                        with gr.Row():
                            batch_in = gr.Number(label=L["lbl_batch"], value=1, precision=0)
                            accum_in = gr.Number(label=L["lbl_accum"], value=1, precision=0)
                        load_model_dd = gr.Dropdown(
                            label=L["lbl_load_model"],
                            choices=[""] + _list_experiments(),
                            value=MODEL_IMAGE,
                            allow_custom_value=True,
                        )
                        with gr.Accordion(L["lbl_lora_acc"], open=True) as lora_acc:
                            use_lora = gr.Checkbox(label=L["lbl_use_lora"], value=True)
                            with gr.Row():
                                lora_r = gr.Number(label=L["lbl_lora_r"], value=16, precision=0)
                                lora_a = gr.Number(label=L["lbl_lora_a"], value=16, precision=0)
                        gr.Markdown("---")
                        with gr.Row():
                            start_btn = gr.Button(L["btn_start"], variant="primary",
                                                  elem_classes="btn-orange", scale=2)
                            stop_btn  = gr.Button(L["btn_stop"],  elem_classes="btn-stop", scale=1)
                        status_bar = gr.Textbox(label=L["lbl_status"],
                                                value=L["status_idle"],
                                                interactive=False, max_lines=1)

                    # Right panel
                    with gr.Column(scale=8):
                        chart = gr.Plot(label=L["lbl_chart"], value=get_charts("en"))
                        with gr.Row():
                            sec_terminal = gr.Markdown(L["sec_terminal"])
                            refresh_btn  = gr.Button(L["btn_refresh"], size="sm", scale=0)
                        log_box = gr.Textbox(
                            value="", lines=20, max_lines=20,
                            interactive=False, show_label=False,
                            elem_classes="terminal",
                        )

                # Auto-refresh
                timer = gr.Timer(value=3)
                timer.tick(fn=get_logs,         outputs=[log_box])
                timer.tick(fn=get_charts_auto,  outputs=[chart])
                timer.tick(fn=get_train_status, outputs=[status_bar])

                start_btn.click(
                    fn=start_training,
                    inputs=[tag_in, config_dd, train_dir, val_dir, ds_size, fname_tr, fname_vl,
                            embed_dir, steps_in, lr_in, batch_in, accum_in,
                            use_lora, lora_r, lora_a, load_model_dd],
                    outputs=[status_bar],
                )

                config_dd.change(
                    fn=apply_training_config,
                    inputs=[config_dd],
                    outputs=[train_variant, tag_in, train_dir, val_dir, ds_size,
                             fname_tr, fname_vl, embed_dir, steps_in, lr_in,
                             batch_in, accum_in, load_model_dd],
                )
                train_variant.change(
                    fn=apply_training_variant,
                    inputs=[train_variant],
                    outputs=[config_dd, tag_in, train_dir, val_dir, ds_size,
                             fname_tr, fname_vl, embed_dir, steps_in, lr_in,
                             batch_in, accum_in, load_model_dd],
                )
                stop_btn.click(fn=stop_training, outputs=[status_bar])
                refresh_btn.click(fn=get_logs,        outputs=[log_box])
                refresh_btn.click(fn=get_charts_auto, outputs=[chart])
                refresh_btn.click(fn=get_train_status, outputs=[status_bar])

            # ══════════ TAB 2 ══════════
            with gr.Tab("2 · Validation Viewer"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3, min_width=260):
                        sec_filter      = gr.Markdown(L["sec_filter"])
                        val_exp_dd      = gr.Dropdown(label=L["lbl_val_exp"],
                                                      choices=_list_experiments(),
                                                      allow_custom_value=True)
                        btn_refresh_exp = gr.Button(L["btn_refresh_exp"], size="sm")
                        gr.Markdown("---")
                        sec_metrics     = gr.Markdown(L["sec_metrics"])
                        with gr.Row():
                            psnr_m  = gr.Textbox(label=L["lbl_psnr"],  value="—",
                                                 interactive=False, max_lines=1,
                                                 elem_classes="metric-card")
                            lpips_m = gr.Textbox(label=L["lbl_lpips"], value="—",
                                                 interactive=False, max_lines=1,
                                                 elem_classes="metric-card")
                            ssim_m  = gr.Textbox(label=L["lbl_ssim"],  value="—",
                                                 interactive=False, max_lines=1,
                                                 elem_classes="metric-card")
                        tip_val = gr.Markdown(L["tip_val"])

                    with gr.Column(scale=9):
                        val_gallery = gr.Gallery(label=L["lbl_gallery"],
                                                 columns=2, height=520,
                                                 object_fit="contain")

                val_exp_dd.change(fn=load_val_view, inputs=[val_exp_dd],
                                  outputs=[val_gallery, psnr_m, lpips_m, ssim_m])
                btn_refresh_exp.click(fn=refresh_experiments, outputs=[val_exp_dd])

            # ══════════ TAB 3 ══════════
            with gr.Tab("3 · Inference"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, min_width=360):
                        sec_input = gr.Markdown(L["sec_input"])
                        with gr.Tabs():
                            with gr.Tab("✏️ Text") as tab_text:
                                prompt_in  = gr.Textbox(label=L["lbl_prompt"],
                                                        placeholder=L["ph_prompt"],
                                                        lines=4)
                                neg_prompt = gr.Textbox(label=L["lbl_neg_prompt"],
                                                        value="", lines=2)
                            with gr.Tab("🖼 Image") as tab_image:
                                image_in = gr.Image(label=L["lbl_image_in"],
                                                    type="numpy", height=240)
                        gr.Markdown("---")
                        sec_params  = gr.Markdown(L["sec_params"])
                        with gr.Row():
                            cfg_in     = gr.Slider(label=L["lbl_cfg"],
                                                   minimum=1.0, maximum=10.0, value=7.5, step=0.5)
                            seed_in    = gr.Number(label=L["lbl_seed"], value=42, precision=0)
                        steps_inf = gr.Slider(label=L["lbl_steps_inf"],
                                              minimum=10, maximum=50, value=20, step=5)
                        with gr.Row():
                            elevation_in = gr.Number(label=L["lbl_elevation"], value=20, precision=1)
                            use_elevest_cb = gr.Checkbox(label=L["lbl_use_elevest"], value=False)
                        gr.Markdown("---")
                        sec_ckpt       = gr.Markdown(L["sec_ckpt"])
                        _init_exps  = _list_experiments()
                        _init_exp   = MODEL_TEXT if MODEL_TEXT in _init_exps else (_init_exps[0] if _init_exps else None)
                        _init_ckpts = _list_checkpoints(_init_exp) if _init_exp else []
                        _init_ckpt  = _init_ckpts[-1] if _init_ckpts else None
                        infer_exp_dd   = gr.Dropdown(label=L["lbl_infer_exp"],
                                                     choices=_init_exps,
                                                     value=_init_exp,
                                                     allow_custom_value=True)
                        infer_ckpt_dd  = gr.Dropdown(label=L["lbl_infer_ckpt"],
                                                     choices=_init_ckpts,
                                                     value=_init_ckpt)
                        gr.Markdown("---")
                        sec_output_opt = gr.Markdown(L["sec_output_opt"])
                        with gr.Row():
                            save_ply_cb = gr.Checkbox(label=L["lbl_save_ply"], value=False)
                            vid_type    = gr.Dropdown(label=L["lbl_vid_type"],
                                                      choices=["none", "gif", "mp4", "fancy_gif"],
                                                      value="gif")
                        gen_btn      = gr.Button(L["btn_gen"], variant="primary",
                                                 elem_classes="btn-orange")
                        infer_status = gr.Textbox(label=L["lbl_infer_status"],
                                                  interactive=False, max_lines=3)

                    with gr.Column(scale=7):
                        sec_preview = gr.Markdown(L["sec_preview"])
                        output_img  = gr.Image(label=L["lbl_out_img"], height=280)
                        output_vid  = gr.Video(label=L["lbl_out_vid"], height=280)
                        tip_infer   = gr.Markdown(L["tip_infer"])

                infer_exp_dd.change(fn=update_ckpt_list,
                                    inputs=[infer_exp_dd], outputs=[infer_ckpt_dd])

                def _switch_to_image_model():
                    exps   = _list_experiments()
                    exp    = MODEL_IMAGE if MODEL_IMAGE in exps else None
                    ckpts  = _list_checkpoints(exp) if exp else []
                    ckpt   = ckpts[-1] if ckpts else None
                    return (
                        gr.Dropdown(choices=exps, value=exp),
                        gr.Dropdown(choices=ckpts, value=ckpt),
                        gr.Slider(value=2.0),   # recommended CFG for image-cond
                    )

                def _switch_to_text_model():
                    exps   = _list_experiments()
                    exp    = MODEL_TEXT if MODEL_TEXT in exps else None
                    ckpts  = _list_checkpoints(exp) if exp else []
                    ckpt   = ckpts[-1] if ckpts else None
                    return (
                        gr.Dropdown(choices=exps, value=exp),
                        gr.Dropdown(choices=ckpts, value=ckpt),
                        gr.Slider(value=7.5),   # recommended CFG for text-cond
                    )

                tab_image.select(fn=_switch_to_image_model,
                                 outputs=[infer_exp_dd, infer_ckpt_dd, cfg_in])
                tab_text.select(fn=_switch_to_text_model,
                                outputs=[infer_exp_dd, infer_ckpt_dd, cfg_in])

                gen_btn.click(
                    fn=run_inference,
                    inputs=[prompt_in, image_in, cfg_in, seed_in, steps_inf, elevation_in, use_elevest_cb,
                            infer_exp_dd, infer_ckpt_dd, save_ply_cb, vid_type],
                    outputs=[output_img, output_vid, infer_status],
                )

        # ─── Language switch: wire ALL translatable components ───
        lang_outputs = [
            header_html,
            # Tab 0
            sec_data_prep, dp_obj_dir, dp_render_dir, dp_out_dir, dp_blender_exe,
            dp_btn_blender, dp_btn_parquet, dp_btn_stop,
            dp_status_bar, dp_sec_terminal, dp_btn_refresh,
            # Tab 1
            sec_config, train_variant, config_dd, tag_in, train_dir, val_dir,
            ds_size, fname_tr, fname_vl, embed_dir,
            steps_in, lr_in, batch_in, accum_in, load_model_dd,
            lora_acc, use_lora, lora_r, lora_a,
            start_btn, stop_btn, status_bar, chart, refresh_btn,
            sec_terminal,
            sec_filter, val_exp_dd, btn_refresh_exp, sec_metrics,
            psnr_m, lpips_m, ssim_m, val_gallery, tip_val,
            sec_input, prompt_in, neg_prompt, image_in,
            sec_params, cfg_in, seed_in, steps_inf, elevation_in, use_elevest_cb,
            sec_ckpt, infer_exp_dd, infer_ckpt_dd,
            sec_output_opt, save_ply_cb, vid_type, gen_btn,
            infer_status, sec_preview, output_img, output_vid, tip_infer,
        ]

        lang_radio.change(
            fn=switch_lang,
            inputs=[lang_radio],
            outputs=lang_outputs,
        )

    return demo


# ═══════════════════════ Entry point ═══════════════════════
if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=5788,
        share=False,
        show_error=True,
    )
