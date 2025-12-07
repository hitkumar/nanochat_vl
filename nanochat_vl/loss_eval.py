import math

import torch
import torch.distributed as dist
from nanochat_vl.common import get_dist_info


@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes):
    total_nats = torch.tensor(0.0, device=model.get_device(), dtype=torch.float32)
    total_bytes = torch.tensor(0, device=model.get_device(), dtype=torch.int64)
    batch_iter = iter(batches)

    for _ in range(steps):
        x, y = next(batch_iter)
        loss_2d = model(x, y, loss_reduction="none")  # [B, T]
        loss_2d = loss_2d.view(-1)  # flatten
        y = y.view(-1)  # flatten

        # Includes tokens we want to ignore while computing the loss
        if (y.int() < 0).any():
            valid = y >= 0
            valid_y = torch.where(valid, y, torch.zeros_like(y))
            num_bytes2d = torch.where(
                valid,
                token_bytes[valid_y],
                torch.zeros_like(y, dtype=token_bytes.dtype),
            )
            total_nats += (loss_2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()
        else:
            num_bytes2d = token_bytes[y]
            total_nats += (loss_2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()

    _, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if ddp_world_size > 1:
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)

    total_nats = total_nats.item()
    total_bytes = total_bytes.item()
    if total_bytes == 0:
        return float("Inf")
    bpb = total_nats / (math.log(2) * total_bytes)
    return bpb
