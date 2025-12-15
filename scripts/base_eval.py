"""
Evaluate the CORE metric for a given model

Test commands
HF models

torchrun --standalone --nproc_per_node=8 -m scripts.base_eval --hf-path openai-community/gpt2-xl

nanochat models

torchrun --standalone --nproc_per_node=8 -m scripts.base_eval --model-tag d34_full
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
from nanochat_vl.core_eval import evaluate_task
from nanochat_vl.tokenizer import HuggingFaceTokenizer

# ~162MB of data needed to evaluate the CORE metric
EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"


def place_eval_bundle(file_path):
    """File path is the local path to eval bundle zip file"""
    eval_bundle_dir = os.path.join(get_base_dir(), "eval_bundle")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)
        extracted_bundle_dir = os.path.join(tmpdir, "eval_bundle")
        shutil.move(extracted_bundle_dir, eval_bundle_dir)
    print0(f"Placed eval bundle in: {eval_bundle_dir}")


def evaluate_model(model, tokenizer, device, max_per_task=-1):
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    if not os.path.exists(eval_bundle_dir):
        download_file_with_lock(
            EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=place_eval_bundle
        )

    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # In context learning tasks
    tasks = config["icl_tasks"]
    print0(f"Running In Context Learning tasks: {len(tasks)}, example task: {tasks[0]}")

    # Load random baselines values from eval metadata
    random_baselines = {}
    with open(eval_meta_data, "r", encoding="utf-8") as f:
        # uses first row of the csv as keys
        reader = csv.DictReader(f)
        for row in reader:
            task_name = row["Eval Task"]
            random_baseline = row["Random baseline"]
            random_baselines[task_name] = float(random_baseline)

    # Evaluate each task
    results = {}
    centered_results = {}
    for task in tasks:
        start_time = time.time()
        label = task["label"]
        continuation_delimiter = task.get("continuation_delimiter", " ")
        task_meta = {
            "task_type": task["icl_task_type"],
            "dataset_uri": task["dataset_uri"],
            "num_fewshot": task["num_fewshot"][0],
            "continuation_delimiter": continuation_delimiter,
        }

        print0(
            f"Evaluating: {label} ({task_meta['num_fewshot']}-shot, continuation_delimiter: {continuation_delimiter}, type: {task_meta['task_type']})... ",
            end="",
        )

        # Load data for this task
        data_path = os.path.join(data_base_path, task["dataset_uri"])
        with open(data_path, "r", encoding="utf-8") as f:
            data = [json.loads(line.strip()) for line in f]

        # shuffle the data
        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        accuracy = evaluate_task(model, tokenizer, data, device, task_meta)
        results[label] = accuracy
        random_baseline = random_baselines[label]
        # random baseline in the eval set is a %age, so we need * with 0.01 here
        centered_result = (accuracy - 0.01 * random_baseline) / (
            1.0 - 0.01 * random_baseline
        )
        centered_results[label] = centered_result

        end_time = time.time()
        print0(
            f"accuracy: {accuracy:.4f} | centered: {centered_result:.4f} | random_baseline: {random_baseline*0.01:.4f} time: {end_time - start_time:.2f}s"
        )

    core_metric = sum(centered_results.values()) / len(centered_results)
    out = {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric,
    }
    return out


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
        "--model-tag", type=str, default=None, help="Nanochat Model tag to evaluate"
    )
    parser.add_argument(
        "--max-per-task",
        type=int,
        default=-1,
        help="Max examples per task to evaluate (-1 = disable)",
    )
    args = parser.parse_args()
    assert (
        args.hf_path is not None or args.model_tag is not None
    ), "Either hf-path or model_tag must be specified"

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
        model_slug = hf_path.replace("/", "_")
    else:
        model, tokenizer, meta_data = load_model("base", device, "eval", args.model_tag)
        raw_inputs = torch.ones(2, 2, dtype=torch.long, device=device)
        with autocast_ctx:
            logits = model(raw_inputs)
        print0(f"model logit shape is {logits.shape}")
        model_name = f"base_model_{args.model_tag} (step {meta_data['step']:06d})"
        model_slug = f"base_model_{args.model_tag}_{meta_data['step']:06d}"

    with autocast_ctx:
        out = evaluate_model(model, tokenizer, device, args.max_per_task)

    # print(f"Results are {out}")
    if ddp_rank == 0:
        base_dir = get_base_dir()
        output_csv_file = os.path.join(base_dir, "base_eval", f"{model_slug}.csv")
        os.makedirs(os.path.dirname(output_csv_file), exist_ok=True)
        print0(f"writing eval results to {output_csv_file}")
        results, centered_results, core_metric = (
            out["results"],
            out["centered_results"],
            out["core_metric"],
        )
        with open(output_csv_file, "w", encoding="utf-8", newline="") as f:
            f.write(f"{'Task':<35}, {'Accuracy': <10}, {'Centered': <10}\n")
            for label in results:
                f.write(
                    f"{label:<35}, {results[label]:10.6f}, {centered_results[label]:10.6f}\n"
                )

            f.write(f"{'CORE':<35}, {'':<10}, {core_metric:10.6f}\n")

    compute_cleanup()


if __name__ == "__main__":
    main()
