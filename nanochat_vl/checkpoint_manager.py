"""
Utils for loading and saving checkpoints
"""

import json
import os
import re

import torch

from nanochat_vl.common import get_base_dir, log0, logger, print0, setup_default_logging
from nanochat_vl.gpt import GPT, GPTConfig
from nanochat_vl.tokenizer import get_tokenizer


def save_checkpoint(
    checkpoint_dir, step, model_data, optimizer_data, meta_data, rank=0
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    if rank == 0:
        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        torch.save(model_data, model_path)
        log0(f"Saved model checkpoint to {model_path}")
        # save the metadata dict
        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        with open(meta_path, "w") as f:
            json.dump(meta_data, f, indent=2)
        log0(f"Saved metadata to {meta_path}")

    # Since optimizer state is sharded across ranks, all ranks need to save it
    if optimizer_data is not None:
        optimizer_path = os.path.join(
            checkpoint_dir, f"optimizer_{step:06d}_{rank:d}.pt"
        )
        torch.save(optimizer_data, optimizer_path)
        logger.info(f"Saved optimizer checkpoint to {optimizer_path}")


def load_checkpoint(checkpoint_dir, step, device, load_optimizer=False, rank=0):
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location=device)
    optimizer_data = None
    if load_optimizer:
        optimizer_path = os.path.join(
            checkpoint_dir, f"optimizer_{step:06d}_{rank:d}.pt"
        )
        optimizer_data = torch.load(optimizer_path, map_location=device)
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r") as f:
        meta_data = json.load(f)

    return model_data, optimizer_data, meta_data


# Utils to load trained checkpoints for eval and inference.


def get_last_step(checkpoint_dir):
    # find the last step
    pattern = re.compile(r"model_(\d+)\.pt$")

    steps = []
    for filename in os.listdir(checkpoint_dir):
        match = pattern.match(filename)
        if match:
            steps.append(int(match.group(1)))

    if not steps:
        return -1

    return max(steps)


def build_model(checkpoint_dir, step, device, phase):
    assert phase in ["train", "eval"]
    model_data, _, meta_data = load_checkpoint(checkpoint_dir, step, device)
    if device.type in ("mps", "cpu"):
        # convert bfloat16 tensors to float32 for nonGPU inference
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v
            for k, v in model_data.items()
        }

    # Fix torch compile issue which
    # model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    assert all(not k.startswith("_orig_mod.") for k in model_data.keys())
    model_config_kwargs = meta_data["model_config"]
    log0(f"Building model with config: {model_config_kwargs}")
    model_config = GPTConfig(**model_config_kwargs)
    with torch.device("meta"):
        model = GPT(model_config)

    model.to_empty(device=device)
    # need to call this to reinit the pos embeddings as they are not saved in the checkpoint
    model.init_weights()
    model.load_state_dict(model_data, strict=False, assign=True)
    if phase == "train":
        model.train()
    else:
        model.eval()

    tokenizer = get_tokenizer()
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"]
    return model, tokenizer, meta_data


def load_model_from_dir(checkpoint_dir, device, phase, model_tag, step=None):
    assert model_tag is not None
    checkpoint_dir = os.path.join(checkpoint_dir, model_tag)
    if step is None:
        step = get_last_step(checkpoint_dir)

    assert step >= 0, f"Invalid step {step} in {checkpoint_dir}"
    log0(f"Loading model from {checkpoint_dir} at step {step}")
    model, tokenizer, meta_data = build_model(checkpoint_dir, step, device, phase)
    return model, tokenizer, meta_data


def load_model(source, *args, **kwargs):
    model_dir = {
        "base": "base_checkpoints",
        "mid": "mid_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }[source]
    base_dir = get_base_dir()
    checkpoint_dir = os.path.join(base_dir, model_dir)
    return load_model_from_dir(checkpoint_dir, *args, **kwargs)
