# nanochat_vl

Initial discussion: https://github.com/karpathy/nanochat/discussions/1

Helpful discussion about training a larger model
- https://github.com/karpathy/nanochat/discussions/8
- TODO: Find out how is flops calculated exactly.

Infusing identity
- https://github.com/karpathy/nanochat/discussions/139
- Basically SFT on relevant data

Key Goals
- Use a SOTA architecture like QWEN 3
- Use all types of parallelism techniques like TP, PP, CP, EP etc. Torchtitan could be a good resource to fork from.
- Make the model multimodal with images
- Add videos
- Add verifiers for better RL.
- Mostly use PyTorch core libraries like FSDP2, distributed etc.
- Write most of the code yourself.
- Explore prime-intellect repos.
- Explore knowledge distillation in LLMs
- How much can we change after SFT during training?

**Resource**
Training playbook
https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook#how-to-read-this-blog-post

- Key question is should you train a model from scratch or fine-tune/prompting an existing model is enough?
-

**Model Architectures for latest models**
MoE seems a standard choice now, with 1 shared expert. Experts are getting smaller, but increasing in numbers. Can ablate a few of these.
Full article: https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison
Video: https://www.youtube.com/watch?v=rNlULI-zGcw&t=3s


DeepSeek V3/R1
- Grouped Query Atetention (reduce KV cache size)
- Multi-head latent attention (reduce KV cache size even more). This even improves performance according to DeepSeekMoe paper.
- Mixture of experts (sparse models) vs dense models with regular feed forward.

Olmo 2
- Post norm, but inside the residual. Leads to better training stability.
    - x = x + norm(ff(x))
- QK norm applied to query and keys.

Gemma3
- Sliding window attention, also called local attention. Reduces memory consumption needed.
- local: global ratio 5:1
- 27B parameters they picked is a good choice to make use of local development.
- Normalization layer placement with both pre and post norms.

Qwen
- Comes in a lot of sizes.
- Dense and mixture of experts flavors
- Hybrid model which comes with instruction tuned and thinking modes.
- No shared experts used in their biggest models.

SmolLM3
- Nice research model.
- no positional embeddings every 4th layer (NoPE) which helps with long context generalization.

Kimi 2
- 1T parameter model.
- Muon optimizer.
- A lot of experts, does scaling the number of experts help?
- Use dense layer in the beginning and then MOE
- Efficient in inference.

GPT-OSS
- Similar to other MOE models like Qwen.
- More optimized for tool calling.

Grok-2.5
- Interesting way to implement shared expert.

GLM-4.5
- Competitive and similar to other models.
