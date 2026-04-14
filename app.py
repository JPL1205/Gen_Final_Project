"""
GS-Diff Lab — Gradio UI
Launch:
    conda activate diffsplat_sm120
    cd ~/Gen_Final_Project
    PYTHONPATH=. python app.py
"""

import os
import re
import glob
import threading
import subprocess
from pathlib import Path
from typing import Optional

import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────── Constants ──────────────────────────────
PROJECT_ROOT = Path(__file__).parent
OUT_DIR      = PROJECT_ROOT / "out"
DATA_DIR     = PROJECT_ROOT / "data"
CONDA_ENV    = "diffsplat_sm120"

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


# ══════════════════════════════════════════════════════════════
#  Utilities
# ══════════════════════════════════════════════════════════════

def _list_experiments() -> list:
    if not OUT_DIR.exists():
        return []
    return sorted([
        d.name for d in OUT_DIR.iterdir()
        if d.is_dir() and (d / "checkpoints").exists()
    ])


def _list_checkpoints(tag: str) -> list:
    ckpt_dir = OUT_DIR / tag / "checkpoints"
    if not ckpt_dir.exists():
        return []
    return sorted([d.name for d in ckpt_dir.iterdir() if d.is_dir()])


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
#  Tab 1 — Training
# ══════════════════════════════════════════════════════════════

def start_training(tag, train_dir, val_dir, dataset_size,
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

    env_vars = (
        f"PYTHONPATH=. "
        f"DIFFSPLAT_DATA_DIR={train_dir} "
        f"DIFFSPLAT_VAL_DIR={val_dir} "
        f"HF_HOME=~/.cache/huggingface "
        f"TORCH_HOME=~/.cache/torch "
    )
    cmd_parts = [
        "accelerate launch --num_processes 1",
        "src/train_gsdiff_sd.py",
        "--config_file configs/gsdiff_sd15.yaml",
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


def run_inference(prompt, image, cfg_scale, seed, num_steps,
                  exp_tag, ckpt_step, save_ply, output_video_type,
                  progress=gr.Progress()):
    if not exp_tag:
        return None, None, "⚠️ Select an experiment first."
    if not ckpt_step:
        return None, None, "⚠️ Select a checkpoint."
    if not prompt and image is None:
        return None, None, "⚠️ Enter a prompt or upload an image."

    image_arg = ""
    if image is not None:
        from PIL import Image as PILImage
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir="/tmp")
        PILImage.fromarray(image.astype("uint8")).save(tmp.name)
        image_arg = f"--image_path {tmp.name}"

    video_arg = f"--output_video_type {output_video_type}" if output_video_type != "none" else ""
    ply_arg   = "--save_ply" if save_ply else ""

    cmd_parts = [
        "python src/infer_gsdiff_sd.py",
        "--config_file configs/gsdiff_sd15.yaml",
        f"--tag {exp_tag}",
        f"--infer_from_iter {int(ckpt_step)}",
        f"--output_dir {OUT_DIR}",
        f"--guidance_scale {cfg_scale}",
        f"--seed {int(seed)}",
        f"--num_inference_steps {int(num_steps)}",
        "--half_precision --allow_tf32",
        f'--prompt "{prompt}"',
        image_arg, video_arg, ply_arg,
    ]
    full_cmd = (
        f"source ~/miniconda3/etc/profile.d/conda.sh && "
        f"conda activate {CONDA_ENV} && "
        f"cd {PROJECT_ROOT} && "
        f"PYTHONPATH=. {' '.join(p for p in cmd_parts if p)}"
    )
    progress(0, desc="Starting inference...")
    proc = subprocess.Popen(["bash", "-c", full_cmd],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            cwd=str(PROJECT_ROOT))
    log_buf = []
    for raw in iter(proc.stdout.readline, b""):
        line = raw.decode("utf-8", errors="replace").rstrip()
        log_buf.append(line)
        if "Rendering" in line:
            progress(0.5, desc="Rendering...")
    proc.stdout.close()
    proc.wait()

    if proc.returncode != 0:
        return None, None, "❌ Inference failed:\n" + "\n".join(log_buf[-20:])

    infer_dir = OUT_DIR / exp_tag / "infer"
    if not infer_dir.exists():
        candidates = list((OUT_DIR / exp_tag).glob("infer*"))
        infer_dir  = candidates[-1] if candidates else OUT_DIR / exp_tag

    png_files = sorted(glob.glob(str(infer_dir / "*.png")))
    gif_files = sorted(glob.glob(str(infer_dir / "*.gif")))
    mp4_files = sorted(glob.glob(str(infer_dir / "*.mp4")))

    out_image = png_files[-1] if png_files else None
    out_video = gif_files[-1] if gif_files else (mp4_files[-1] if mp4_files else None)
    progress(1.0, desc="Done")
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
        gr.update(value=t["sec_config"]),
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

            # ══════════ TAB 1 ══════════
            with gr.Tab("1 · Training Monitor"):
                with gr.Row(equal_height=False):

                    # Left panel
                    with gr.Column(scale=4, min_width=320):
                        sec_config   = gr.Markdown(L["sec_config"])
                        tag_in       = gr.Textbox(label=L["lbl_tag"], value="my_finetune")
                        train_dir    = gr.Textbox(label=L["lbl_train_dir"],
                                                  value=str(DATA_DIR / "min_img23d_test"))
                        val_dir      = gr.Textbox(label=L["lbl_val_dir"],
                                                  value=str(DATA_DIR / "min_img23d_test"))
                        with gr.Row():
                            ds_size  = gr.Number(label=L["lbl_ds_size"],  value=1,   precision=0, min_width=90)
                            fname_tr = gr.Textbox(label=L["lbl_fname_tr"], value="train", min_width=100)
                            fname_vl = gr.Textbox(label=L["lbl_fname_val"], value="val",  min_width=100)
                        embed_dir = gr.Textbox(
                            label=L["lbl_embed_dir"],
                            value=str(DATA_DIR / "min_img23d_test" / "prompt_embeds_sd15"),
                        )
                        with gr.Row():
                            steps_in = gr.Number(label=L["lbl_steps"], value=5000, precision=0)
                            lr_in    = gr.Textbox(label=L["lbl_lr"],   value="5e-5")
                        with gr.Row():
                            batch_in = gr.Number(label=L["lbl_batch"], value=1, precision=0)
                            accum_in = gr.Number(label=L["lbl_accum"], value=1, precision=0)
                        load_model_dd = gr.Dropdown(
                            label=L["lbl_load_model"],
                            choices=[""] + _list_experiments(),
                            value="gsdiff_gobj83k_sd15__render",
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
                    inputs=[tag_in, train_dir, val_dir, ds_size, fname_tr, fname_vl,
                            embed_dir, steps_in, lr_in, batch_in, accum_in,
                            use_lora, lora_r, lora_a, load_model_dd],
                    outputs=[status_bar],
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
                            with gr.Tab("✏️ Text"):
                                prompt_in  = gr.Textbox(label=L["lbl_prompt"],
                                                        placeholder=L["ph_prompt"],
                                                        lines=4)
                                neg_prompt = gr.Textbox(label=L["lbl_neg_prompt"],
                                                        value="", lines=2)
                            with gr.Tab("🖼 Image"):
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
                        gr.Markdown("---")
                        sec_ckpt       = gr.Markdown(L["sec_ckpt"])
                        infer_exp_dd   = gr.Dropdown(label=L["lbl_infer_exp"],
                                                     choices=_list_experiments(),
                                                     allow_custom_value=True)
                        infer_ckpt_dd  = gr.Dropdown(label=L["lbl_infer_ckpt"], choices=[])
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
                gen_btn.click(
                    fn=run_inference,
                    inputs=[prompt_in, image_in, cfg_in, seed_in, steps_inf,
                            infer_exp_dd, infer_ckpt_dd, save_ply_cb, vid_type],
                    outputs=[output_img, output_vid, infer_status],
                )

        # ─── Language switch: wire ALL translatable components ───
        lang_outputs = [
            header_html,
            sec_config, tag_in, train_dir, val_dir,
            ds_size, fname_tr, fname_vl, embed_dir,
            steps_in, lr_in, batch_in, accum_in, load_model_dd,
            lora_acc, use_lora, lora_r, lora_a,
            start_btn, stop_btn, status_bar, chart, refresh_btn,
            sec_terminal,
            sec_filter, val_exp_dd, btn_refresh_exp, sec_metrics,
            psnr_m, lpips_m, ssim_m, val_gallery, tip_val,
            sec_input, prompt_in, neg_prompt, image_in,
            sec_params, cfg_in, seed_in, steps_inf,
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
        server_port=7860,
        share=False,
        show_error=True,
    )
