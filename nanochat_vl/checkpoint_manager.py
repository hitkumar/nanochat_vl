"""
Utils for loading and saving checkpoints
"""

import json
import os

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
    if load_optimizer:
        optimizer_path = os.path.join(
            checkpoint_dir, f"optimizer_{step:06d}_{rank:d}.pt"
        )
        optimizer_data = torch.load(optimizer_path, map_location=device)
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    with open(meta_path, "r") as f:
        meta_data = json.load(f)

    return model_data, optimizer_data, meta_data
