import torch
from nanochat_vl.common import get_base_dir, get_dist_info, print0
from nanochat_vl.gpt import GPT, GPTConfig
from nanochat_vl.tokenizer import get_token_bytes, get_tokenizer
from sympy.series.sequences import sequence

# Define model params here
max_seq_len = 2048  # max context length
depth = 20  # the depth of the Transformer model to train

# Tokenizer
tokenizer = get_tokenizer()
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size is {vocab_size}")

num_layers = depth
model_dim = (
    depth * 64
)  # aspect ratio 64 (usually this is varied from 64 -> 128 as model size increases)
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

model_config_kwargs = dict(
    sequence_len=max_seq_len,
    vocab_size=vocab_size,
    n_layer=num_layers,
    n_head=num_heads,
    n_kv_head=num_kv_heads,
    n_embd=model_dim,
)
model_config = GPTConfig(**model_config_kwargs)
model = GPT(model_config)
print0(f"Model has {model.rotary_seq_len} parameters")
