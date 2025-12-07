"""
Training script for model training

python base_train.py

distributed training use torchrun
torchrun --nproc_per_node=8 base_train.py

"""

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
from contextlib import nullcontext

import torch
from nanochat_vl.checkpoint_manager import load_checkpoint, save_checkpoint
from nanochat_vl.common import (
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    DummyWandb,
    get_base_dir,
    print0,
)
from nanochat_vl.dataloader import tokenizing_distributed_data_loader
from nanochat_vl.gpt import GPT, GPTConfig
from nanochat_vl.loss_eval import evaluate_bpb
from nanochat_vl.tokenizer import get_token_bytes, get_tokenizer

# -----------------------------------------------------------------------------
# User settings
run = "dummy"  # wandb run name default ("dummy" is special - we won't log to wandb)
# Runtime
device_type = ""  # cuda|cpu|mps (empty => autodetect good device type default, in order: CUDA > MPS > CPU)
# Define model params here
max_seq_len = 2048  # max context length
depth = 20  # the depth of the Transformer model to train

# Training settings
num_iterations = 100  # explicit number of steps of the optimization (-1 = disable)
target_flops = (
    -1.0
)  # calculate num_iterations to reach target_flops. Useful for scaling laws experiments (-1 = disable)
target_param_data_ratio = 20  # calculate num_iterations to maintain fixed data:param ratio (Chinchilla=20) (-1 = disable)

# Optimization params
device_batch_size = 32  # per device batch size set to not OOM
total_batch_size = 524288  # total desired batch size, in #tokens
embedding_lr = 0.2  # learning rate for the embedding parameters (Adam)
unembedding_lr = 0.004  # learning rate for the unembedding parameters (Adam)
weight_decay = 0.0  # weight decay for the embedding/unembedding parameters (Adam)
matrix_lr = 0.02  # learning rate for the matrix parameters (Muon)
grad_clip = 1.0  # 0.0 means disabled
warmup_ratio = 0.0  # ratio of iterations for LR warmup
warmdown_ratio = 0.2  # ratio of iterations for LR warmdown
final_lr_frac = 0.0  # final LR is this fraction of the initial LR
resume_from_step = (
    -1
)  # resume training from this step of the optimization (-1 = disable)
# Output
model_tag = (
    ""  # optionally override the model tag for the output checkpoint directory name
)

# compute init
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
# Mixed precision training on GPUs
autocast_ctx = (
    torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
    if device_type == "cuda"
    else nullcontext()
)
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# No wandb for now
wandb_run = DummyWandb()


# Tokenizer
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size is {vocab_size}")

num_layers = depth
model_dim = (
    depth * 64
)  # aspect ratio 64 (usually this is varied from 64 -> 128 as model size increases)
head_dim = 128
num_heads = max(
    1, (model_dim + 127) // 128
)  # head dim 128 (the division here is ceil div)
num_kv_heads = (
    num_heads  # default is 1:1 GQA (Group Query Attention) ratio (i.e. GQA is disabled)
)
print0(f"num_layers: {num_layers}")
print0(f"model_dim: {model_dim}")
print0(f"num_heads: {num_heads}")
print0(f"num_kv_heads: {num_kv_heads}")

# Optimizer / data related hyperparameters
tokens_per_fwdbwd = max_seq_len * device_batch_size
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(
    f"Tokens / microbatch / rank: {tokens_per_fwdbwd}, tokens / microbatch: {world_tokens_per_fwdbwd}"
)
print0(
    f"Total batch size: {total_batch_size}, Gradient accumulation steps: {grad_accum_steps}"
)

# Initialize the model

model_config_kwargs = dict(
    sequence_len=max_seq_len,
    vocab_size=vocab_size,
    n_layer=num_layers,
    n_head=num_heads,
    n_kv_head=num_kv_heads,
    n_embd=model_dim,
)
with torch.device("meta"):
    model_config = GPTConfig(**model_config_kwargs)
    model = GPT(model_config)
model.to_empty(device=device)
model.init_weights()
print0(f"Model head_dim is {model.head_dim}")

base_dir = get_base_dir()
output_dirname = model_tag if model_tag else f"d_{depth}"
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
# TODO: Add resuming logic

orig_model = model  # original, uncompiled model for saving raw model state_dict and inference/evaluation
model = torch.compile(model, dynamic=False)
num_params = model.get_num_parameters()
num_flops_per_token = model.estimate_flops()
print0(
    f"Number of parameters: {num_params // 1e6}M, estimated flops per token: {num_flops_per_token}"
)

# Calculate number of iterations, either it is provided by the user or from target flops or from target data:params ration (in that order)
assert num_iterations > 0 or target_flops > 0 or target_param_data_ratio > 0
if num_iterations > 0:
    print0("Using provided number of iterations")
elif target_flops > 0:
    num_iterations = target_flops // (num_flops_per_token * total_batch_size)
    print0(f"Calc number of iterations from target flops: {num_iterations}")
elif target_param_data_ratio > 0:
    num_tokens_target = target_param_data_ratio * num_params
    num_iterations = num_tokens_target // total_batch_size
    print0(f"Calc number of iterations from target data:params ratio: {num_iterations}")
else:
    raise ValueError("No condition met to calculate number of iterations")

total_tokens = total_batch_size * num_iterations
# chinchilla is ~20 for reference
print0(
    f"Number of training tokens: {total_tokens}, tokens params ratio: {round(total_tokens / num_params)}"
)
# Initialize optimizers (Muon for matrix layers, AdamW for embedding and lm_head)
optimizers = model.setup_optimizers(
    unembedding_lr=unembedding_lr,
    embedding_lr=embedding_lr,
    matrix_lr=matrix_lr,
    weight_decay=weight_decay,
)
adamw_optimizer, muon_optimizer = optimizers

# setup dataloaders
train_loader = tokenizing_distributed_data_loader(
    device_batch_size, max_seq_len, split="train", device=device
)
build_val_loader = lambda: tokenizing_distributed_data_loader(
    device_batch_size, max_seq_len, split="val", device=device
)
x, y = next(train_loader)
print0(f"x shape is {x.shape}, target shape is {y.shape}")

# Evaluation params
eval_every = 20
eval_tokens = 20 * total_batch_size
save_every = (
    -1
)  # freq of saving model ckpts, -1 means only save at the end of training.


# Set up hyperparameter schedulers
# LR scheduler
def get_lr_multiplier(it):
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it < num_iterations - warmdown_iters:
        return 1.0
    else:
        # progress goes down linearly from 1.0 to 0.0 as iterations progress.
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac


# Momentum scheduler for Muon optimizer
def get_muon_momentum(it):
    frac = min(it / 300, 1.0)
    # increases momentum from 0.85 to 0.95
    return (1 - frac) * 0.85 + frac * 0.95


# Track training progress
step = 0
min_val_bpb = float("inf")
smooth_train_loss = 0  # EMA of the training loss
total_training_time = 0  # wall clock time spent in training


# training loop
while True:
    last_step = step == num_iterations
    flops_so_far = step * num_flops_per_token * total_batch_size

    # Calculate val bpb every few steps (all ranks participate)
    if last_step or step % eval_every == 0:
        model.eval()
        val_loader = build_val_loader()
        eval_steps = eval_tokens // (ddp_world_size * max_seq_len * device_batch_size)
        with autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step: {step: 05d}, Val BPB: {val_bpb:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb

        model.train()

    # TODO: Eval other metrics periodically as well and sample from the model.

    if last_step or (save_every > 0 and step % save_every == 0):
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),
            [opt.state_dict() for opt in optimizers],
            # metadata saved as json
            {
                "step": step,
                "val_bpb": val_bpb,
                "model_config": model_config_kwargs,
            },
            rank=ddp_rank,
        )
    if last_step:
        break

    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()  # for logging
        # print0(f"Train loss: {train_loss.item()}")
        loss = (
            loss / grad_accum_steps
        )  # loss computed is avg of values in the micro_batch, we divide by grad_accum steps to get the true average in the whole batch
        loss.backward()  # sum the loss value
        x, y = next(train_loader)

    # gradient clipping
    grad_clip_enabled = grad_clip > 0.0
    if grad_clip_enabled:
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
            orig_model.parameters(), grad_clip
        )
        grad_norm = grad_norm_tensor.item()

    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm

    muon_momentum = get_muon_momentum(step)
    for group in muon_optimizer.param_groups:
        group["momentum"] = muon_momentum

    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    if step % 100 == 0:
        print0(f"Training duration for step {step} is {dt:.4f} sec")

    # TODO: Add some logging.
    step += 1

# cleanup
wandb_run.finish()
compute_cleanup()
