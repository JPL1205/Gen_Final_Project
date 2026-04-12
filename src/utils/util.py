from typing import *
from argparse import Namespace
from omegaconf import DictConfig
from torch.nn import Module
from accelerate import Accelerator
from src.options import Options

import os
import shutil
from omegaconf import OmegaConf
from accelerate import load_checkpoint_and_dispatch


def load_ckpt(
    ckpt_dir: str, ckpt_iter: int,
    hdfs_dir: Optional[str] = None,
    model: Optional[Module] = None,
    accelerator: Optional[Accelerator] = None,
    strict: bool = True,
    use_ema: bool = False,
) -> Module:
    # Find the latest checkpoint
    if ckpt_iter < 0:
        if hdfs_dir is not None:
            ckpt_iter = int(sorted(get_hdfs_files(hdfs_dir))[-1].split(".")[0])
        else:
            ckpt_iter = int(sorted(os.listdir(ckpt_dir))[-1])

    # Download checkpoint
    ckpt_path = f"{ckpt_dir}/{ckpt_iter:06d}" + ("/ema_states.pth" if use_ema else "")
    if not os.path.exists(ckpt_path):
        assert hdfs_dir is not None
        if accelerator is not None:
            if accelerator.is_main_process:
                ensure_sysrun(f"mkdir -p {ckpt_dir}")
                ensure_sysrun(f"hdfs dfs -get {hdfs_dir}/{ckpt_iter:06d}.tar {ckpt_dir}")
                ensure_sysrun(f"tar -xvf {ckpt_dir}/{ckpt_iter:06d}.tar -C {ckpt_dir}")
                ensure_sysrun(f"rm {ckpt_dir}/{ckpt_iter:06d}.tar")
            accelerator.wait_for_everyone()  # wait before preparing checkpoints by the main process
        else:
            ensure_sysrun(f"mkdir -p {ckpt_dir}")
            ensure_sysrun(f"hdfs dfs -get {hdfs_dir}/{ckpt_iter:06d}.tar {ckpt_dir}")
            ensure_sysrun(f"tar -xvf {ckpt_dir}/{ckpt_iter:06d}.tar -C {ckpt_dir}")
            ensure_sysrun(f"rm {ckpt_dir}/{ckpt_iter:06d}.tar")

    if model is None:
        return ckpt_iter

    # Load checkpoint
    else:
        ckpt_dir = f"{ckpt_dir}/{ckpt_iter:06d}"
        if not os.path.exists(f"{ckpt_dir}/zero_to_fp32.py"):
            load_checkpoint_and_dispatch(model, ckpt_path, strict=strict)
        else:  # from DeepSpeed
            if accelerator is not None:
                if accelerator.is_main_process:
                    ensure_sysrun(f"python3 {ckpt_dir}/zero_to_fp32.py {ckpt_dir} {ckpt_dir} --safe_serialization")
                accelerator.wait_for_everyone()  # wait before preparing checkpoints by the main process
            else:
                ensure_sysrun(f"python3 {ckpt_dir}/zero_to_fp32.py {ckpt_dir} {ckpt_dir} --safe_serialization")
            load_checkpoint_and_dispatch(model, ckpt_path, strict=strict)

        return model


def save_ckpt(ckpt_dir: str, ckpt_iter: int, hdfs_dir: Optional[str] = None):
    if hdfs_dir is not None:
        ensure_sysrun(f"tar -cf {ckpt_dir}/{ckpt_iter:06d}.tar -C {ckpt_dir} {ckpt_iter:06d}")
        ensure_sysrun(f"hdfs dfs -put -f {ckpt_dir}/{ckpt_iter:06d}.tar {hdfs_dir}")
        ensure_sysrun(f"rm -rf {ckpt_dir}/{ckpt_iter:06d}.tar {ckpt_dir}/{ckpt_iter:06d}")


def get_configs(yaml_path: str, cli_configs: List[str]=[], **kwargs) -> DictConfig:
    yaml_configs = OmegaConf.load(yaml_path)
    cli_configs = OmegaConf.from_cli(cli_configs)

    configs = OmegaConf.merge(yaml_configs, cli_configs, kwargs)
    OmegaConf.resolve(configs)  # resolve ${...} placeholders
    return configs


def save_experiment_params(args: Namespace, configs: DictConfig, opt: Options, save_dir: str) -> Dict[str, Any]:
    os.makedirs(save_dir, exist_ok=True)

    params = OmegaConf.merge(configs, {k: str(v) for k, v in vars(args).items()})
    params = OmegaConf.merge(params, OmegaConf.create(vars(opt)))
    OmegaConf.save(params, os.path.join(save_dir, "params.yaml"))
    return dict(params)


def save_model_architecture(model: Module, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)

    num_buffers = sum(b.numel() for b in model.buffers())
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    message = f"Number of buffers: {num_buffers}\n" +\
        f"Number of trainable / all parameters: {num_trainable_params} / {num_params}\n\n" +\
        f"Model architecture:\n{model}"

    with open(os.path.join(save_dir, "model.txt"), "w") as f:
        f.write(message)


def ensure_sysrun(cmd: str):
    while True:
        result = os.system(cmd)
        if result == 0:
            break
        else:
            print(f"Retry running {cmd}")


def prepare_hdfs_output_dir(requested_hdfs_dir: Optional[str], tag: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Prepare remote checkpoint dir if HDFS is available.

    Returns:
        (hdfs_dir_for_current_run, project_hdfs_root_dir)
    """
    if requested_hdfs_dir is None:
        return None, None

    project_hdfs_dir = requested_hdfs_dir
    if shutil.which("hdfs") is None:
        print(
            f"[Warning] HDFS dir was provided [{requested_hdfs_dir}] but `hdfs` command is not available. "
            "Remote checkpoint sync is disabled."
        )
        return None, project_hdfs_dir

    hdfs_dir = os.path.join(requested_hdfs_dir, tag)
    ensure_sysrun(f"hdfs dfs -mkdir -p {hdfs_dir}")
    return hdfs_dir, project_hdfs_dir


def maybe_run_dataset_setup(file_dir_test: Optional[str], dataset_setup_script: Optional[str]):
    """
    Run optional dataset setup only when test/val directory is missing.
    """
    if file_dir_test is None or os.path.exists(file_dir_test):
        return

    if dataset_setup_script is None or str(dataset_setup_script).strip() == "":
        print(
            f"[Warning] Validation dataset directory [{file_dir_test}] does not exist and no "
            "`dataset_setup_script` is configured."
        )
        return

    if "hdfs" in dataset_setup_script and shutil.which("hdfs") is None:
        print(
            f"[Warning] Validation dataset directory [{file_dir_test}] does not exist and setup script "
            "requires `hdfs`, but `hdfs` command is unavailable."
        )
        return

    result = os.system(dataset_setup_script)
    if result != 0:
        print(
            f"[Warning] dataset_setup_script exited with non-zero status ({result}). "
            f"Script: {dataset_setup_script}"
        )


def get_hdfs_files(hdfs_path: str) -> List[str]:
    lines = get_hdfs_lines(hdfs_path)
    if len(lines) == 0:
        raise ValueError(f"No files found in {hdfs_path}")

    return [line.split()[-1].split("/")[-1] for line in lines]


def get_hdfs_size(hdfs_path: str, unit: str="B") -> int:
    lines = get_hdfs_lines(hdfs_path)
    if len(lines) == 0:
        raise ValueError(f"No files found in {hdfs_path}")

    byte_size = sum(int(line.split()[4]) for line in lines)
    if unit == "B":
        return byte_size
    elif unit == "KB":
        return byte_size / 1024
    elif unit == "MB":
        return byte_size / 1024 / 1024
    elif unit == "GB":
        return byte_size / 1024 / 1024 / 1024
    elif unit == "TB":
        return byte_size / 1024 / 1024 / 1024 / 1024
    else:
        raise ValueError(f"Invalid unit: {unit}")


def get_hdfs_lines(hdfs_path: str) -> List[str]:
    return [line for line in os.popen(f"hdfs dfs -ls {hdfs_path}").read().strip().split("\n")[1:]]
