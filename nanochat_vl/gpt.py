"""
GPT model definition
"""

import math
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from nanochat_vl.adamw import DistAdamW

from nanochat_vl.common import get_dist_info, print0
from nanochat_vl.muon import DistMuon, Muon


@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    # x is of shape (B, seq_len, heads, head_dim) here
    assert x.ndim == 4
    emb_dim = x.shape[3]
    assert emb_dim % 2 == 0
    d = emb_dim // 2
    x1, x2 = x[..., :d], x[..., d:]
    # this is clockwise rotation, but that is okay for our purpose.
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3)
    out = out.to(x.dtype)
    return out


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head % self.n_head == 0 and self.n_kv_head <= self.n_head
        self.head_dim = self.n_embd // self.n_head

        # Attention matrices
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size()
        # project the input to get queries, key and values matrices
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        # Apply Rope
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)  # QK norm
        # Make head the batch dim, (B, T, H, C) -> (B, H, T, C)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # Apply KV cache, only used during inference.
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)

        Tq = q.size(2)
        Tk = k.size(2)
        enable_gqa = self.n_kv_head < self.n_head
        if kv_cache is None or Tq == Tk:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, enable_gqa=enable_gqa
            )
        elif Tq == 1:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=False, enable_gqa=enable_gqa
            )
        else:
            raise NotImplementedError("implement this during inference")

        # (B, T, C) shape like the input
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    """
    Defines one transformer block
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache):
        # Pre norm impl
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "h": nn.ModuleList(
                    [Block(config, layer_idx) for layer_idx in range(config.n_layer)]
                ),
                "lm_head": nn.Linear(config.n_embd, config.vocab_size, bias=False),
            }
        )

        # This is fake init, as model is initialized on meta device initially for efficiency reasons.
        self.rotary_seq_len = config.sequence_len * 10
        self.head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(
            self.rotary_seq_len, self.head_dim
        )
        # persistent=False means these are not saved with the checkpoint
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def get_device(self):
        return self.transformer.wte.weight.device

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    def init_weights(self):
        # initializes weights of all the modules in the network
        self.apply(self._init_weights)
        # zero out LM head weights and last layers of all attention/MLP modules
        torch.nn.init.zeros_(self.transformer.lm_head.weight)
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)

        # really initiliaze the pos embeddings
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        # this is what updates the registers we defined in init
        self.cos, self.sin = cos, sin
        if self.get_device().type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    # using commonly used default value of theta which is 100K
    def _precompute_rotary_embeddings(
        self, seq_len, head_dim, base=100_000, device=None
    ):
        if device is None:
            device = self.get_device()

        # We need an entry for every position in the sequence and every pair of embedding element.
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        # high frequencies in the beginning and low at the end
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(0, seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos().bfloat16(), freqs.sin().bfloat16()
        # shape is (1, rotary_seq_len, 1, head_dim/2)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def estimate_flops(self):
        """Return the estimated FLOPs per token for the model. Ref: https://arxiv.org/abs/2204.02311"""
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel()
        l, h, q, t = (
            self.config.n_layer,
            self.config.n_head,
            self.head_dim,
            self.config.sequence_len,
        )
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return num_flops_per_token

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
        B, T = idx.size()
        assert T <= self.cos.size(
            1
        ), f"Sequence len grew beyond the size of ROPE {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device
        assert self.cos.dtype == torch.bfloat16

        # get right pos embedding if kv cache is enabled
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0 : T0 + T], self.sin[:, T0 : T0 + T]

        x = self.transformer.wte(x)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache)
        x = norm(x)

        # Compute the logits and loss
        softcap = 15
        if targets is not None:
            logits = self.transformer.lm_head(x)
            # logits softcap caps the logit value between [-softcap, softcap]
            logits = softcap * torch.tanh(logits / softcap)
            logits = logits.float()  # move to fp32 from bf16
            loss = F.cross_entropy(
                # (B*T, vocab_size)
                logits.view(-1, logits.size(-1)),
                # (B*T, )
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction,
            )
        else:
            logits = self.transformer.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap)
            return logits


if __name__ == "__main__":
    print0("GPT model")
