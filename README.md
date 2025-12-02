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

Optimizer Notes
- AdamW: https://www.youtube.com/watch?v=1_nujVNUsto
- Muon: https://www.youtube.com/watch?v=bO5nvE289ec&t=59s
  - The overall idea is to approximately orthogolanize the momentum update matrix using Newton Schultz.
  - SVD is the standard way of doing this exactly, but it is very slow.
  - Muon is used for 2D params, for other params we still use AdamW
  - QKclip and MuonClip are some other extensions used in practice to improve stability.


Milestones
- Implement Muon and AdamW optimizers, GPT config and a couple of other configs (Olmo3, Qwen3)
- Train model.
- Add parallelism from torchtitan

Olmo-core is not a good option to explore since it assumes that dataloader is not specific to HF. Try PrimeIntellect Prime-RL and verifiers instead, along with torchtitan


**Resource**
Training playbook
https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook#how-to-read-this-blog-post

- Key question is should you train a model from scratch or fine-tune/prompting an existing model is enough?
-

Olmo 3 blog: https://allenai.org/blog/olmo3
- Very interesting to try replicating some of their work
- Learning Rust might be key given several advantages.
- Read the technical report.

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

**Interesting talks and videos**

Sasha Rush on Cursor Compose: https://www.youtube.com/watch?v=md8D8eNj5JM&pp=ygUPcmF5IHN1bW1pdCAyMDI1
- Talks about simple RL pipeline consisting of Training, Generation and environment components.

Tinker: https://www.youtube.com/watch?v=Xb34YmbEiOc&pp=ygUPcmF5IHN1bW1pdCAyMDI1
- Interesting idea to expose a different set of APIs for developers building AI applications

SkyRL: https://www.youtube.com/watch?v=9EW0wHCUeBw&t=4s
- Could be an interesting library to explore for tinker like ideas.

Ray: https://www.youtube.com/watch?v=B7U05Y4YcJg&list=PL_lsbAsL_o2BUUxo6coMBFwQE31U4Eb2q&index=4
- Distributed compute engine.
- Very commonly used in RL libraries for orchestration.

Attention mechanisms: https://www.youtube.com/watch?v=Y-o545eYjXM
- Talks about GQA, MQA and MHA as usual.
- MLA from Deepseek is promising for optimizing the size of KV cache.
- Idea of decoupled ROPE with MLA is interesting
- Deepseek Sparse Attention is also discussed a bit.
- These are interesting to explore especially the DSA, used in Deepseek-3.2. This would be good to explore next.

***PyTorch conference 2025***

PyTorch native stack for agents: https://www.youtube.com/watch?v=oiAK4f3_o_0&list=PL_lsbAsL_o2BUUxo6coMBFwQE31U4Eb2q&index=38
- Introduces Monarch, Torchstore and torchforge.
- This could be interesting to explore, looks like SkyRl and PrimeIntellect products.

Decent parallelism talk by Lambda labs: https://www.youtube.com/watch?v=O51dr8WeUfY&list=PL_lsbAsL_o2BUUxo6coMBFwQE31U4Eb2q&index=52
- Code examples could be interesting especially the FSDP one.

Tour of recent LLMs: https://www.youtube.com/watch?v=nDl6Aj9aPAI&list=PL_lsbAsL_o2BUUxo6coMBFwQE31U4Eb2q&index=71
- Covers LLMs, diffusion, code world model.
- Similar to his longer talk I saw before.

Multimodal pytorch training and inference: https://www.youtube.com/watch?v=LmpXU8UwREA&list=PL_lsbAsL_o2BUUxo6coMBFwQE31U4Eb2q&index=73
- Interesting way to keep model configuration
- Talks about using a lot of interesting pytorch ideas, rely on Torchtitan like library for easily applying parallelism for models.
- FSDP2 is the main technique, lets get very good with FSDP 2 and TP first.

***CS231n**
Distributed training: https://www.youtube.com/watch?v=9MvD-XsowsE&list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16&index=11
- Great lecture, watch multiple times.
- Goal is to maximize MFU while using all parallelism techniques
- FSDP2 is a no brainer when training larger models.
- Transformer models have L layers, and operate on tensors with shape (B, seq_len, dim), we can split the model across each of these dimensions leading to DP, PP, CP and TP parallelisms.

**CME295**

Lecture 2: https://www.youtube.com/watch?v=yT84Y5zCnaA
- Talks about different types of transformer models - Encoder:decoder, encoder only, decoder only.
- Basic info on position embeddings, doesn't do as good of a job in explaining ROPE as I had hoped.
- Usual pre or post Norm discussion.
- Deep dive into encoder models like BERT. Does pretraining and finetuning.
- Has ton of applications even though these models aren't improved as much as decoder models these days like classification, RAG pipelines.
- ModernBERT from answer.ai is a useful alternative as it includes a lot of modern tricks used in decoder models: https://www.answer.ai/posts/2024-12-19-modernbert.html#training

Agentic LLMs: https://www.youtube.com/watch?v=h-7S6HNq0Vg&t=5s
- Good discussion on RAG
- RAG pipeline is a mix of retrieval and ranking similar to search and recommendation systems
- Several open questions while building RAG pipelines: document source, how to chunk documents, ANN library to use etc.
- Retrieval tends to use bi-encoder and ranking cross encoder.
- Another interesting talk on RAG: https://www.youtube.com/watch?v=6PMEqN0-gkM&list=PLqC25OT8ZpD2-RuhyacIsODl5iJVgMjI3&index=10
- For tool calling systems, a lot of my intutions are covered in this lecture like how to make the model output tool calls.
- ReAct is an interesting way to formulate agents.

Evaluation: https://www.youtube.com/watch?v=8fNP4N46RRo
- Human grading is hard
- LLM Judge is one preferred way to evaluate LLMs these days. We pass LLM the prompt, model response and criteria to use while judging the response. And LLM returns the outcome (binary) and an explanation or reasoning of the outcome. Reasoning is output before outcome so that model "thinks" before giving the outcome.
- We use structured output in practice so that model response is easy to parse.
- Evaluation is pointwise or pairwise normally.
- Suffers from various types of biases - position bias, length/verbosity bias, self enhancement bias. We should ideally use a stronger model as LLM judge compared to the model giving responses. Response should be calibrated with human judgements.
- For tool calling models, there are a lot of failure models from model not outputting correct func call, to func call failing and LLM not able to get the desired output from result of function call.
- Several types of benchmarks used in practice
  - Knowledge benchmarks like MMLU
  - Reasoning benchmarks like AIME
  - Coding like SWE-Bench
  - Safety like HarmBench
  - Pi Bench for agents
- Training data should not be contaminated with benchmarks, check out latest library from AllenAI for this.
