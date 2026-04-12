from typing import *

import os
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import pyarrow.parquet as pq
from tqdm import tqdm
from transformers import (
    CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer,
    T5EncoderModel, T5Tokenizer, T5TokenizerFast,
)


MODEL_TO_HF = {
    "sd15": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "sd21": "stabilityai/stable-diffusion-2-1",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "paa": "PixArt-alpha/PixArt-XL-2-512x512",
    "pas": "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
    "sd3m": "stabilityai/stable-diffusion-3-medium-diffusers",
    "sd35m": "stabilityai/stable-diffusion-3.5-medium",
    "sd35l": "stabilityai/stable-diffusion-3.5-large",
}


def discover_parquet_files(file_dir: str, file_name: Optional[str]) -> List[str]:
    if os.path.isfile(file_dir):
        return [file_dir]

    if not os.path.isdir(file_dir):
        raise FileNotFoundError(f"Data directory does not exist: {file_dir}")

    files = []
    for fn in sorted(os.listdir(file_dir)):
        path = os.path.join(file_dir, fn)
        if not os.path.isfile(path):
            continue

        if file_name is None:
            if fn.endswith(".parquet"):
                files.append(path)
        else:
            if fn == file_name or (fn.endswith(".parquet") and fn.startswith(file_name)):
                files.append(path)

    if len(files) == 0:
        raise FileNotFoundError(f"No parquet files found in [{file_dir}] with prefix [{file_name}]")
    return files


def to_str(x: Any) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def collect_uid_captions(
    files: List[str],
    uid_key: str,
    caption_key: str,
) -> List[Tuple[str, str]]:
    pairs = []
    for path in tqdm(files, desc="Reading parquet captions", ncols=120):
        table = pq.read_table(path, columns=[uid_key, caption_key])
        rows = table.to_pylist()
        for row in rows:
            uid = to_str(row[uid_key]).split("/")[-1].split(".")[0]
            caption = to_str(row[caption_key]).strip()
            if caption == "":
                caption = "3d object"
            pairs.append((uid, caption))
    return pairs


@torch.no_grad()
def encode_batch(
    model_name: str,
    captions: List[str],
    tokenizer,
    text_encoder,
    tokenizer_2=None,
    text_encoder_2=None,
    tokenizer_3=None,
    text_encoder_3=None,
    device: str = "cuda",
):
    if model_name in ["sd15", "sd21"]:
        inputs = tokenizer(
            captions,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        ids = inputs.input_ids.to(device)
        prompt = text_encoder(ids)[0]
        return prompt, None, None

    if model_name in ["sdxl"]:
        inputs = tokenizer(
            captions,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        ids = inputs.input_ids.to(device)
        p1 = text_encoder(ids, output_hidden_states=True).hidden_states[-2]

        inputs2 = tokenizer_2(
            captions,
            padding="max_length",
            max_length=tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        ids2 = inputs2.input_ids.to(device)
        out2 = text_encoder_2(ids2, output_hidden_states=True)
        pooled = out2.text_embeds
        p2 = out2.hidden_states[-2]
        prompt = torch.cat([p1, p2], dim=-1)
        return prompt, pooled, None

    if model_name in ["paa", "pas"]:
        max_length = {"paa": 120, "pas": 300}[model_name]
        captions = [x.lower().strip() for x in captions]
        inputs = tokenizer(
            captions,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        ids = inputs.input_ids.to(device)
        mask = inputs.attention_mask.to(device)
        prompt = text_encoder(ids, attention_mask=mask)[0]
        return prompt, None, mask

    if model_name in ["sd3m", "sd35m", "sd35l"]:
        in1 = tokenizer(
            captions,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        ids1 = in1.input_ids.to(device)
        out1 = text_encoder(ids1, output_hidden_states=True)
        pooled1 = out1.text_embeds
        p1 = out1.hidden_states[-2]

        in2 = tokenizer_2(
            captions,
            padding="max_length",
            max_length=tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        ids2 = in2.input_ids.to(device)
        out2 = text_encoder_2(ids2, output_hidden_states=True)
        pooled2 = out2.text_embeds
        p2 = out2.hidden_states[-2]
        pooled = torch.cat([pooled1, pooled2], dim=-1)
        clip_prompt = torch.cat([p1, p2], dim=-1)

        in3 = tokenizer_3(
            captions,
            padding="max_length",
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )
        ids3 = in3.input_ids.to(device)
        p3 = text_encoder_3(ids3)[0]
        clip_prompt = F.pad(clip_prompt, (0, p3.shape[-1] - clip_prompt.shape[-1]))
        prompt = torch.cat([clip_prompt, p3], dim=-2)
        return prompt, pooled, None

    raise NotImplementedError(f"Unsupported model_name: {model_name}")


def build_text_encoders(model_name: str, pretrained_model_name_or_path: str):
    variant = "fp16" if model_name not in ["pas", "sd3m", "sd35m", "sd35l"] else None

    if model_name in ["sd15", "sdxl"]:
        tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder", variant=variant)
    elif model_name in ["paa", "pas"]:
        tokenizer = T5Tokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
        text_encoder = T5EncoderModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder", variant=variant)
    elif model_name in ["sd3m", "sd35m", "sd35l"]:
        tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
        text_encoder = CLIPTextModelWithProjection.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder", variant=variant)
    else:
        raise NotImplementedError(f"Unsupported model_name: {model_name}")

    if model_name in ["sdxl", "sd3m", "sd35m", "sd35l"]:
        tokenizer_2 = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer_2")
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder_2", variant=variant)
    else:
        tokenizer_2 = None
        text_encoder_2 = None

    if model_name in ["sd3m", "sd35m", "sd35l"]:
        tokenizer_3 = T5TokenizerFast.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer_3")
        text_encoder_3 = T5EncoderModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder_3", variant=variant)
    else:
        tokenizer_3 = None
        text_encoder_3 = None

    return tokenizer, text_encoder, tokenizer_2, text_encoder_2, tokenizer_3, text_encoder_3


def main():
    parser = argparse.ArgumentParser("Generic prompt embedding encoder for parquet datasets")
    parser.add_argument("--model_name", required=True, choices=list(MODEL_TO_HF.keys()))
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None)
    parser.add_argument("--file_dir", required=True, type=str)
    parser.add_argument("--file_name", default=None, type=str)
    parser.add_argument("--uid_key", default="uid", type=str)
    parser.add_argument("--caption_key", default="caption", type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    args = parser.parse_args()

    pretrained_model_name_or_path = args.pretrained_model_name_or_path or MODEL_TO_HF[args.model_name]
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer, text_encoder, tokenizer_2, text_encoder_2, tokenizer_3, text_encoder_3 = build_text_encoders(
        args.model_name,
        pretrained_model_name_or_path,
    )
    text_encoder = text_encoder.to(args.device)
    if text_encoder_2 is not None:
        text_encoder_2 = text_encoder_2.to(args.device)
    if text_encoder_3 is not None:
        text_encoder_3 = text_encoder_3.to(args.device)

    files = discover_parquet_files(args.file_dir, args.file_name)
    pairs = collect_uid_captions(files, args.uid_key, args.caption_key)

    for i in tqdm(range(0, len(pairs), args.batch_size), desc=f"Encoding [{args.model_name}]", ncols=120):
        chunk = pairs[i: i + args.batch_size]
        uids = [u for u, _ in chunk]
        captions = [c for _, c in chunk]
        prompt, pooled, mask = encode_batch(
            args.model_name, captions,
            tokenizer, text_encoder,
            tokenizer_2, text_encoder_2,
            tokenizer_3, text_encoder_3,
            args.device,
        )
        for j, uid in enumerate(uids):
            np.save(os.path.join(args.output_dir, f"{uid}.npy"), prompt[j].float().cpu().numpy())
            if pooled is not None:
                np.save(os.path.join(args.output_dir, f"{uid}_pooled.npy"), pooled[j].float().cpu().numpy())
            if mask is not None:
                np.save(os.path.join(args.output_dir, f"{uid}_attention_mask.npy"), mask[j].float().cpu().numpy())

    # Save unconditional embeddings
    prompt, pooled, mask = encode_batch(
        args.model_name, [""],
        tokenizer, text_encoder,
        tokenizer_2, text_encoder_2,
        tokenizer_3, text_encoder_3,
        args.device,
    )
    np.save(os.path.join(args.output_dir, "null.npy"), prompt[0].float().cpu().numpy())
    if pooled is not None:
        np.save(os.path.join(args.output_dir, "null_pooled.npy"), pooled[0].float().cpu().numpy())
    if mask is not None:
        np.save(os.path.join(args.output_dir, "null_attention_mask.npy"), mask[0].float().cpu().numpy())


if __name__ == "__main__":
    main()
