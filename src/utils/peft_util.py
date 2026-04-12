from typing import *

import os

import torch

from src.options import Options


def _import_peft():
    try:
        from peft import LoraConfig, PeftModel, get_peft_model
        from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
    except ImportError as e:
        raise ImportError(
            "PEFT is required for LoRA finetuning. Please install `peft` in your environment."
        ) from e

    return LoraConfig, PeftModel, get_peft_model, load_peft_weights, set_peft_model_state_dict


def _parse_target_modules(target_modules: Optional[str], default_target_modules: Sequence[str]) -> List[str]:
    if target_modules is None:
        return list(default_target_modules)

    modules = [m.strip() for m in target_modules.split(",") if m.strip()]
    if len(modules) == 0:
        raise ValueError("`peft_target_modules` is empty after parsing.")
    return modules


def is_peft_model(model: torch.nn.Module) -> bool:
    return hasattr(model, "peft_config")


def inject_lora_adapters(
    model: torch.nn.Module,
    opt: Options,
    default_target_modules: Sequence[str],
):
    if not opt.use_peft:
        return model, []
    if opt.peft_type != "lora":
        raise NotImplementedError(f"Unsupported peft_type: {opt.peft_type}")

    LoraConfig, _, get_peft_model, _, _ = _import_peft()
    target_modules = _parse_target_modules(opt.peft_target_modules, default_target_modules)

    peft_config = LoraConfig(
        r=opt.peft_r,
        lora_alpha=opt.peft_alpha,
        target_modules=target_modules,
        lora_dropout=opt.peft_dropout,
        bias=opt.peft_bias,
        task_type=None,
    )
    model = get_peft_model(model, peft_config, adapter_name=opt.peft_adapter_name)
    if len(get_trainable_params(model)) == 0:
        raise ValueError(
            f"No trainable LoRA parameters were created. Please check `peft_target_modules`: {target_modules}"
        )
    return model, target_modules


def get_trainable_params(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def get_trainable_param_names(model: torch.nn.Module) -> List[str]:
    return [n for n, p in model.named_parameters() if p.requires_grad]


def adapter_checkpoint_exists(adapter_dir: str) -> bool:
    return os.path.exists(os.path.join(adapter_dir, "adapter_config.json"))


def resolve_adapter_dir(path: str, subfolder: str) -> str:
    if adapter_checkpoint_exists(path):
        return path

    candidate = os.path.join(path, subfolder)
    if adapter_checkpoint_exists(candidate):
        return candidate
    return candidate


def save_lora_adapter(
    model: torch.nn.Module,
    adapter_dir: str,
    safe_serialization: bool = True,
):
    if not is_peft_model(model):
        return
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir, safe_serialization=safe_serialization)


def load_lora_adapter_weights(
    model: torch.nn.Module,
    adapter_dir: str,
    adapter_name: str,
) -> bool:
    if not is_peft_model(model):
        return False
    if not adapter_checkpoint_exists(adapter_dir):
        return False

    _, _, _, load_peft_weights, set_peft_model_state_dict = _import_peft()
    peft_state = load_peft_weights(adapter_dir, device="cpu")
    set_peft_model_state_dict(model, peft_state, adapter_name=adapter_name)
    if hasattr(model, "set_adapter"):
        model.set_adapter(adapter_name)
    return True


def build_peft_model_for_inference(
    model: torch.nn.Module,
    adapter_dir: str,
    adapter_name: str,
):
    if not adapter_checkpoint_exists(adapter_dir):
        return model, False

    _, PeftModel, _, _, _ = _import_peft()
    model = PeftModel.from_pretrained(model, adapter_dir, adapter_name=adapter_name, is_trainable=False)
    if hasattr(model, "set_adapter"):
        model.set_adapter(adapter_name)
    return model, True
