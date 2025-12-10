"""
BPE style tokenizer, only used for pretraining for now.
"""

import copy
import os
import pickle
from functools import lru_cache

import tiktoken

SPECIAL_TOKENS = [
    # every document begins with the Beginning of Sequence (BOS) token that delimits documents
    "<|bos|>",
    # tokens below are only used during finetuning to render Conversations into token ids
    "<|user_start|>",  # user messages
    "<|user_end|>",
    "<|assistant_start|>",  # assistant messages
    "<|assistant_end|>",
    "<|python_start|>",  # assistant invokes python REPL tool
    "<|python_end|>",
    "<|output_start|>",  # python REPL outputs back to assistant
    "<|output_end|>",
]


class RustBPETokenizer:
    """
    Loading tokenizer trained using Rust tokenizer in nanochat repo, uses tiktoken for efficient inference.
    """

    def __init__(self, enc, bos_token):
        self.enc = enc
        self.bos_token_id = self.encode_special(bos_token)

    @classmethod
    def from_directory(cls, tokenizer_dir):
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "rb") as f:
            enc = pickle.load(f)
        return cls(enc, "<|bos|>")

    @classmethod
    def from_pretrained(cls, tiktoken_name):
        enc = tiktoken.get_encoding(tiktoken_name)
        # tiktoken calls the document delimeter token "<|endoftext|>".
        return cls(enc, "<|endoftext|>")

    @lru_cache(maxsize=32)
    def encode_special(self, text):
        return self.enc.encode_single_token(text)

    def get_bos_token_id(self):
        return self.bos_token_id

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_special_tokens(self):
        return self.enc.special_tokens_set

    def id_to_token(self, id):
        return self.enc.decode([id])

    def encode(self, text, prepend=None, append=None, num_threads=8):
        # text can either be a string or a list of strings

        if prepend is not None:
            prepend_id = (
                prepend if isinstance(prepend, int) else self.encode_special(prepend)
            )
        if append is not None:
            append_id = (
                append if isinstance(append, int) else self.encode_special(append)
            )

        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids = ids.insert(0, prepend_id)
            if append is not None:
                ids = ids.insert(0, append_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for ids_row in ids:
                    ids_row.insert(0, prepend_id)  # TODO: same
            if append is not None:
                for ids_row in ids:
                    ids_row.append(append_id)
        else:
            raise ValueError(f"Invalid text type: {type(text)}")

        return ids

    def __call__(self, *args, **kwargs):
        self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.enc.decode(ids)


# Generic GPT-4 Style Tokenizer which can be used to load huggingface models.
# This is used to evaluate arbitrary HF models for comparison.
from tokenizers import Tokenizer as HFTokenizer


class HuggingFaceTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, hf_path):
        tokenizer = HFTokenizer.from_pretrained(hf_path)
        return cls(tokenizer)

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def id_to_token(self, id):
        return self.tokenizer.id_to_token(id)

    def encode_special(self, text):
        return self.tokenizer.token_to_id(text)

    def _encode_one(self, text, prepend=None, append=None):
        # encode a single string
        # prepend/append can be either a string of a special token or a token id directly.
        assert isinstance(text, str)
        ids = []
        if prepend is not None:
            prepend_id = (
                prepend if isinstance(prepend, int) else self.encode_special(prepend)
            )
            ids.append(prepend_id)
        ids.extend(self.tokenizer.encode(text, add_special_tokens=False).ids)
        if append is not None:
            append_id = (
                append if isinstance(append, int) else self.encode_special(append)
            )
            ids.append(append_id)
        return ids

    def get_bos_token_id(self):
        return self.encode_special("<|bos|>")

    def encode(self, text, *args, **kwargs):
        if isinstance(text, str):
            return self._encode_one(text, *args, **kwargs)
        elif isinstance(text, list):
            return [self._encode_one(t, *args, **kwargs) for t in text]
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)


def get_tokenizer():
    from nanochat_vl.common import get_base_dir

    tokenizer_dir = os.path.join(get_base_dir(), "tokenizer")
    return RustBPETokenizer.from_directory(tokenizer_dir)


def get_token_bytes(device="cpu"):
    """
    Returns a 1D tensor of shape (vocab_size, ) indicating the number of bytes per token, 0 if token is not to be counted in loss calculation like special tokens.
    """
    import torch
    from nanochat_vl.common import get_base_dir

    tokenizer_dir = os.path.join(get_base_dir(), "tokenizer")
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    assert os.path.exists(token_bytes_path)
    with open(token_bytes_path, "rb") as f:
        token_bytes = torch.load(f, map_location=device)
    return token_bytes


if __name__ == "__main__":
    tokenizer = get_tokenizer()
    print(tokenizer.decode(tokenizer.encode("How are you")))
    token_bytes = get_token_bytes()
    print(f"Token bytes for first few tokens is {token_bytes[-5:]}")
