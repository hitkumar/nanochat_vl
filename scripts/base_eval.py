"""
Evaluate the CORE metric for a given model
"""

import argparse
import csv
import json
import os
import random
import shutil
import tempfile
import time
import zipfile
from contextlib import nullcontext

import torch
import yaml
from nanochat_vl.checkpoint_manager import load_model

from nanochat_vl.common import (
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    download_file_with_lock,
    get_base_dir,
    print0,
)
from nanochat_vl.tokenizer import HuggingFaceTokenizer


# Utils for loading HF models for evaluation
class ModelWrapper:
    """
    Lightweight wrapper for a HuggingFace model
    Hugggingface models return a complex output structure, this wrapper simplified it to return logits only.
    """

    def __init__(self, model, max_seq_len=None):
        self.model = model
        self.max_seq_len = max_seq_len

    def __call__(self, input_ids):
        outputs = self.model(input_ids)
        logits = outputs.logits
        return logits


def load_hf_model(hf_path: str, device):
    print0(f"Loading model from: {hf_path}")
    # Load the model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(hf_path)
    model.to(device)
    model.eval()
    max_seq_len = 1024 if "openai-community/gpt2" in hf_path else None
    model = ModelWrapper(model, max_seq_len)
    tokenizer = HuggingFaceTokenizer.from_pretrained(hf_path)
    return model, tokenizer


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf-path", type=str, default=None, help="HuggingFace model path to evaluate"
    )
    parser.add_argument(
        "--model_tag", type=str, default="base", help="Nanochat Model tag to evaluate"
    )
    args = parser.parse_args()
    assert args.hf_path is not None or args.model_tag is not None

    # distributed training setup
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    autocast_ctx = (
        torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
        if device_type == "cuda"
        else nullcontext()
    )
    if args.hf_path is not None:
        hf_path = args.hf_path
        model, tokenizer = load_hf_model(hf_path, device)
        raw_inputs = torch.ones(2, 2, dtype=torch.long, device=device)
        logits = model(raw_inputs)
        print0(f"model logit shape is {logits.shape}")
        model_name = hf_path
    else:
        model, tokenizer, meta_data = load_model("base", device, "eval", "d34_full")
        raw_inputs = torch.ones(2, 2, dtype=torch.long, device=device)
        logits = model(raw_inputs)
        print0(f"model logit shape is {logits.shape}")


if __name__ == "__main__":
    main()
