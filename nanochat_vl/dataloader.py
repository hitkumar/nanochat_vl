from collections import deque

import pyarrow.parquet as pq
import torch

from nanochat_vl.common import get_base_dir, get_dist_info
from nanochat_vl.dataset import list_parquet_files
from nanochat_vl.tokenizer import get_tokenizer


def tokenizing_distributed_data_loader_util(
    B, T, split, tokenizer_threads=4, tokenizer_batch_size=128, device="cuda"
):
    """
    Streams text from parquet files, tokenizes and yields training batches.
    TODO: implement ability to resume training
    """
    assert split in ["train", "val"]
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    def document_batches():
        parquet_paths = list_parquet_files()
        parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
        pq_index = 0
        while True:  # multi epoch training
            while pq_index < len(parquet_paths):
                filepath = parquet_paths[pq_index]
                pf = pq.ParquetFile(filepath)
                rg_idx = ddp_rank
                while rg_idx < pf.num_row_groups:
                    rg = pf.read_row_group(rg_idx)
                    batch = rg.column(
                        "text"
                    ).to_pylist()  # each group is a parquet group, eg. 1024 rows
                    for i in range(0, len(batch), tokenizer_batch_size):
                        yield batch[i : i + tokenizer_batch_size]
                    rg_idx += ddp_world_size
                pq_index += 1

    # This keeps emiting batch of text documents at a time of size tokenizer_batch_size
    batches = document_batches()
    # B * T tokens in a batch, +1 as we need the target token for the last token
    needed_tokens = B * T + 1
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()

    # new tokens are added to the end and tokens are popped from start to be used for training
    tokens_buffer = deque()
    while True:
        while len(tokens_buffer) < needed_tokens:
            doc_batch = next(batches)
            tokens_list = tokenizer.encode(
                doc_batch, prepend=bos_token, num_threads=tokenizer_threads
            )
            for tokens in tokens_list:
                tokens_buffer.extend(tokens)

        tokens = [tokens_buffer.popleft() for _ in range(needed_tokens)]
        # Pin memory enables faster CPU to GPU transfers
        use_cuda_optimizations = device == "cuda"
        scratch = torch.tensor(
            tokens, dtype=torch.long, pin_memory=use_cuda_optimizations
        )
        inputs, targets = scratch[:-1], scratch[1:]
        inputs_train = inputs.view(B, T).to(
            device=device, non_blocking=use_cuda_optimizations
        )
        targets_train = targets.view(B, T).to(
            device=device, non_blocking=use_cuda_optimizations
        )
        yield inputs_train, targets_train


def tokenizing_distributed_data_loader(*args, **kwargs):
    for inputs, targets in tokenizing_distributed_data_loader_util(*args, **kwargs):
        yield inputs, targets


if __name__ == "__main__":
    train_loader = tokenizing_distributed_data_loader(8, 1024, split="train")
    x, y = next(train_loader)
    print(f"training input is {x.shape}, target is {y.shape}")
