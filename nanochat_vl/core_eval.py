"""
Functions for evaluating the CORE metric, as described in the DCLM paper.
https://arxiv.org/abs/2406.11794
"""

import random

import torch
import torch.distributed as dist
import torch.nn.functional as F
from jinja2 import Template
from nanochat_vl.common import get_dist_info
from pydantic._internal._utils import ValueItems


# Prompt rendering utils
def render_prompt_mc(item, continuation_delimiter, fewshot_examples=None):
    """
    Render prompt for multiple choice evaluation.
    Returns a list of prompts, one for each choice in the item.
    We check which one the model is most likely to choose to determine if the model is correct.
    """
    template_str = """{%- for example in fewshot_examples -%}
{{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}
{% endfor -%}
{{ item.query }}{{ continuation_delimiter }}{{ choice }}"""
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        "item": item,
        "continuation_delimiter": continuation_delimiter,
        "fewshot_examples": fewshot_examples,
    }
    prompts = [template.render(choice=choice, **context) for choice in item["choices"]]
    return prompts


def render_prompts_schema(item, continuation_delimiter, fewshot_examples=None):
    """Render complete prompts for schema questions
    Each items has these fields
    {"context_options": ["Sarah was a much better surgeon than Maria so Sarah", "Sarah was a much better surgeon than Maria so Maria"],
     "continuation": "always got the easier cases.", "gold": 1}
    """
    template_str = """
    {%- for example in fewshot_examples -%}
    {{ example.context_options[example.gold]}} {{ continuation_delimiter }} {{ example.continuation }}

    {% endfor -%}
    {{ context }} {{ continuation_delimiter }} {{ item.continuation }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        "item": item,
        "continuation_delimiter": continuation_delimiter,
        "fewshot_examples": fewshot_examples,
    }
    prompts = [
        template.render(context=context_option, **context)
        for context_option in item["context_options"]
    ]
    return prompts


def render_prompts_lm(item, continuation_delimiter, fewshot_examples=None):
    """
    Render prompts for language modeling tasks
    We manually trim the context in the template as some datasets have trailing whitespaces
    Example is
    {"context": "Given two strings, determine the length of the longest common subsequence.\n\nStrings: RHULBSLGMH EPRVXZZITZ\nLength of longest common subsequence:", "continuation": "1"}
    """
    template_str = """
    {%- for example in fewshot_examples -%}
    {{ example.context | trim }}{{ continuation_delimiter }}{{ example.continuation }}

    {% endfor -%}
    {{ item.context | trim }}{{ continuation_delimiter }}{% if include_continuation %}{{ item.continuation }}{% endif %}"""

    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        "item": item,
        "continuation_delimiter": continuation_delimiter,
        "fewshot_examples": fewshot_examples,
    }
    # Return two prompts: with and without the continuation, this is so that we can exactly determine the continuation tokens in the tokenizer space
    prompt_without = template.render(include_continuation=False, **context)
    prompt_with = template.render(include_continuation=True, **context)
    prompt_without = prompt_without.strip()
    return [prompt_without, prompt_with]


def find_common_length(token_sequences, direction="left"):
    """
    Find the length of common prefix or suffix of all token sequences.
    direction: 'left' for prefix, 'right' for suffix
    """
    min_len = min(len(seq) for seq in token_sequences)
    indices = {
        "left": range(min_len),
        "right": range(-1, -min_len - 1, -1),
    }[direction]
    for i, idx in enumerate(indices):
        token = token_sequences[0][idx]
        if not all(seq[idx] == token for seq in token_sequences):
            return i
    return min_len


def stack_sequences(tokens, pad_token_id):
    bsz, seq_len = len(tokens), max(len(seq) for seq in tokens)
    input_ids = torch.full((bsz, seq_len), pad_token_id, dtype=torch.long)
    for i, seq in enumerate(tokens):
        input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return input_ids


def batch_sequences_mc(prompts, tokenizer):
    # print(f"prompts are {prompts}, length is {len(prompts)}")
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    # print(f"tokens shape is {tokens.shape}")
    answer_start_idx = find_common_length(tokens, direction="left")
    start_indices = [answer_start_idx] * len(tokens)
    end_indices = [len(t) for t in tokens]
    return tokens, start_indices, end_indices


def batch_sequences_schema(prompts, tokenizer):
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    suffix_length = find_common_length(tokens, direction="right")
    end_indices = [len(t) for t in tokens]
    start_indices = [e - suffix_length for e in end_indices]
    return tokens, start_indices, end_indices


def batch_sequences_lm(prompts, tokenizer):
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    tokens_without, tokens_with = tokens
    start_idx, end_idx = len(tokens_without), len(tokens_with)
    assert start_idx < end_idx
    return [tokens_with], [start_idx], [end_idx]


@torch.no_grad()
def forward_model(model, input_ids):
    """
    Takes input_ids of shape (B, T) and returns losses and predictions of shape (B, T)
    """
    B, T = input_ids.size()
    outputs = model(input_ids)
    target_ids = input_ids[:, 1:]  # (B, T-1)
    output_logits = outputs[:, :-1, :]  # (B, T-1, C)

    # (losses for valid tokens)
    losses = F.cross_entropy(
        output_logits.reshape(-1, output_logits.size(-1)),
        target_ids.reshape(-1),
        reduction="none",
    ).view(B, T - 1)
    losses = torch.cat(
        [losses, torch.full((B, 1), float("nan"), device=input_ids.device)], dim=1
    )
    predictions = outputs.argmax(dim=-1)
    return losses, predictions


@torch.no_grad()
def evaluate_example(idx, model, tokenizer, data, device, task_meta):
    """
    Evaluates a single example, returns true if model predicts correctly False otherwise
    """
    item = data[idx]
    task_type, num_fewshot, continuation_delimiter = (
        task_meta["task_type"],
        task_meta["num_fewshot"],
        task_meta["continuation_delimiter"],
    )
    fewshot_examples = []
    if num_fewshot > 0:
        rng = random.Random(1234 + idx)
        available_indices = [i for i in range(len(data)) if i != idx]
        selected_indices = rng.sample(available_indices, num_fewshot)
        fewshot_examples = [data[i] for i in selected_indices]

    if task_type == "multiple_choice":
        prompts = render_prompt_mc(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_mc(prompts, tokenizer)
    elif task_type == "schema":
        prompts = render_prompts_schema(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_schema(prompts, tokenizer)
    elif task_type == "language_modeling":
        prompts = render_prompts_lm(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_lm(prompts, tokenizer)
    else:
        raise ValueError(f"Invalid task type: {task_type}")

    if hasattr(model, "max_seq_len") and model.max_seq_len is not None:
        max_tokens = model.max_seq_len
        new_tokens, new_start_idxs, new_end_idxs = [], [], []
        for t, s_idx, e_idx in zip(tokens, start_idxs, end_idxs):
            if len(t) > max_tokens:
                num_to_crop = len(t) - max_tokens
                new_tokens.append(t[-max_tokens:])
                assert s_idx - num_to_crop >= 0
                assert e_idx - num_to_crop >= 0
                new_start_idxs.append(s_idx - num_to_crop)
                new_end_idxs.append(e_idx - num_to_crop)
            else:
                new_tokens.append(t)
                new_start_idxs.append(s_idx)
                new_end_idxs.append(e_idx)

        tokens, start_idxs, end_idxs = new_tokens, new_start_idxs, new_end_idxs

    pad_token_id = tokenizer.get_bos_token_id()
    input_ids = stack_sequences(tokens, pad_token_id)
    input_ids = input_ids.to(device)

    losses, predictions = forward_model(model, input_ids)
    if task_type in ["multiple_choice", "schema"]:
        mean_losses = [
            losses[i, si - 1 : ei - 1].mean()
            for i, (si, ei) in enumerate(zip(start_idxs, end_idxs))
        ]
        pred_idx = torch.argmin(torch.tensor(mean_losses)).item()
        is_correct = pred_idx == item["gold"]
    elif task_type == "language_modeling":
        # batch size 1 for this
        si = start_idxs[0]
        ei = end_idxs[0]
        # prediction at i-1 predicts token at pos i
        predicted_tokens = predictions[0, si - 1 : ei - 1]
        actual_tokens = input_ids[0, si:ei]
        is_correct = torch.all(predicted_tokens == actual_tokens).item()
    else:
        raise ValueError(f"Invalid task_type: {task_type}")
    return is_correct


def evaluate_task(model, tokenizer, data, device, task_meta):
    """
    Evaluates a task across many examples in a distributed fashion if run using torchrun
    """
    _, rank, _, world_size = get_dist_info()
    correct = torch.zeros(len(data), dtype=torch.float32, device=device)
    for idx in range(rank, len(data), world_size):
        is_correct = evaluate_example(idx, model, tokenizer, data, device, task_meta)
        correct[idx] = float(is_correct)
    if world_size > 1:
        dist.barrier()
        # this is basically gathering results from all ranks as all ranks work on different examples
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)

    mean_correct = correct.mean().item()
    return mean_correct
