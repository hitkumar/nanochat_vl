"""
Unit tests for core_eval.py
Testing prompt testing utils specifically.
cd /home/htkumar/nanochat_vl && uv run pytest tests/test_core_eval.py -v
"""

import torch
from nanochat_vl.core_eval import (
    find_common_length,
    render_prompt_mc,
    render_prompts_lm,
    render_prompts_schema,
    stack_sequences,
)


def test_render_prompt_mc():
    """Test MC rendering with delimiter and fewshot."""
    fewshot_examples = [
        {"query": "What is 1 + 1?", "choices": ["1", "2"], "gold": 1},
    ]
    item = {"query": "What is 3 + 3?", "choices": ["5", "6"], "gold": 1}
    prompts = render_prompt_mc(
        item, continuation_delimiter=" ", fewshot_examples=fewshot_examples
    )

    assert len(prompts) == 2
    assert prompts[0] == "What is 1 + 1? 2\nWhat is 3 + 3? 5"
    assert prompts[1] == "What is 1 + 1? 2\nWhat is 3 + 3? 6"


def test_render_prompt_mc_zero_shot():
    """Test MC rendering without fewshot (zero-shot)."""
    item = {
        "query": "Capital of France?",
        "choices": ["London", "Paris", "Berlin"],
        "gold": 1,
    }
    prompts = render_prompt_mc(item, continuation_delimiter=" ")

    assert len(prompts) == 3
    assert prompts[0] == "Capital of France? London"
    assert prompts[1] == "Capital of France? Paris"
    assert prompts[2] == "Capital of France? Berlin"


def test_render_prompts_schema():
    """Test schema rendering - fewshot uses gold context_option, main item varies."""
    fewshot_examples = [
        {
            "context_options": ["He went left so he", "He went right so he"],
            "continuation": "found the exit.",
            "gold": 0,
        },
    ]
    item = {
        "context_options": ["She chose A so she", "She chose B so she"],
        "continuation": "won the prize.",
        "gold": 1,
    }
    prompts = render_prompts_schema(
        item, continuation_delimiter=" ", fewshot_examples=fewshot_examples
    )

    assert len(prompts) == 2
    assert "He went left so he" in prompts[0]  # fewshot gold=0
    assert "She chose A so she" in prompts[0]
    assert "She chose B so she" in prompts[1]


def test_render_prompts_schema_zero_shot():
    """Test schema rendering without fewshot."""
    item = {
        "context_options": ["The cat sat on the", "The dog ran to the"],
        "continuation": "mat.",
        "gold": 0,
    }
    prompts = render_prompts_schema(item, continuation_delimiter=" ")

    assert len(prompts) == 2
    assert "The cat sat on the" in prompts[0]
    assert "mat." in prompts[0]
    assert "The dog ran to the" in prompts[1]


def test_render_prompts_lm():
    """Test LM rendering returns prompt with and without continuation."""
    item = {"context": "The answer is", "continuation": " 42"}
    prompts = render_prompts_lm(item, continuation_delimiter="")

    assert len(prompts) == 2
    prompt_without, prompt_with = prompts
    assert prompt_without == "The answer is"
    assert prompt_with == "The answer is 42"


def test_render_prompts_lm_with_fewshot():
    """Test LM rendering with fewshot examples."""
    fewshot_examples = [{"context": "1 + 1 =", "continuation": " 2"}]
    item = {"context": "2 + 2 =", "continuation": " 4"}
    prompts = render_prompts_lm(
        item, continuation_delimiter="", fewshot_examples=fewshot_examples
    )

    prompt_without, prompt_with = prompts
    assert "1 + 1 =" in prompt_without
    assert "2 + 2 =" in prompt_without
    assert prompt_with.endswith(" 4")


def test_find_common_length():
    """Test finding common prefix and suffix."""
    sequences = [[1, 2, 3, 4, 5], [1, 2, 6, 4, 5], [1, 2, 7, 4, 5]]
    assert find_common_length(sequences, direction="left") == 2
    assert find_common_length(sequences, direction="right") == 2


def test_stack_sequences():
    """Test stacking variable length sequences with padding."""
    tokens = [[1, 2], [3, 4, 5, 6], [7]]
    result = stack_sequences(tokens, pad_token_id=0)

    assert result.shape == (3, 4)
    assert result[0].tolist() == [1, 2, 0, 0]
    assert result[1].tolist() == [3, 4, 5, 6]
    assert result[2].tolist() == [7, 0, 0, 0]
    assert result.dtype == torch.long


# Mock tokenizer for batch_* tests
class MockTokenizer:
    def __init__(self):
        self.bos_token_id = 1

    def get_bos_token_id(self):
        return self.bos_token_id

    def __call__(self, texts, prepend=None):
        """Simple char-level tokenizer for testing."""
        results = []
        for text in texts:
            tokens = [ord(c) for c in text]
            if prepend is not None:
                tokens = [prepend] + tokens
            results.append(tokens)
        return results


def test_batch_sequences_mc():
    """Test MC batching finds common prefix (answer start) correctly."""
    from nanochat_vl.core_eval import batch_sequences_mc

    tokenizer = MockTokenizer()
    # Prompts share prefix "Q: ", differ in answer
    prompts = ["Q: A", "Q: B", "Q: C"]
    tokens, start_idxs, end_idxs = batch_sequences_mc(prompts, tokenizer)

    assert len(tokens) == 3
    # All start indices should be same (common prefix length)
    assert start_idxs[0] == start_idxs[1] == start_idxs[2]
    # End indices should be length of each token sequence
    assert end_idxs == [len(t) for t in tokens]


def test_batch_sequences_schema():
    """Test schema batching finds common suffix (continuation) correctly."""
    from nanochat_vl.core_eval import batch_sequences_schema

    tokenizer = MockTokenizer()
    # Prompts have different prefix but same suffix " end"
    prompts = ["start1 end", "start2 end"]
    tokens, start_idxs, end_idxs = batch_sequences_schema(prompts, tokenizer)

    assert len(tokens) == 2
    # Start indices should mark where the common suffix begins
    suffix_len = end_idxs[0] - start_idxs[0]
    assert suffix_len == end_idxs[1] - start_idxs[1]  # Same suffix length


def test_batch_sequences_lm():
    """Test LM batching returns single sequence with continuation indices."""
    from nanochat_vl.core_eval import batch_sequences_lm

    tokenizer = MockTokenizer()
    # LM takes [prompt_without_continuation, prompt_with_continuation]
    prompts = ["context", "context answer"]
    tokens, start_idxs, end_idxs = batch_sequences_lm(prompts, tokenizer)

    assert len(tokens) == 1  # Returns only the full sequence
    assert len(start_idxs) == 1
    assert len(end_idxs) == 1
    # start_idx should be where continuation begins
    assert start_idxs[0] < end_idxs[0]
