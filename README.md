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
- Implement couple of other configs (Olmo3, Qwen3)
- Train model.
- Implement engine for inference.
- Add parallelism from torchtitan

Eval Results after pretraining
- d34_full CORE metric is 31.78
- gpt-2 xl is 25.09

***Installing new dependencies***
Since this repo uses uv, we should use uv pip install <dep_name> to install new stuff here.

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

***Some RL Talks***
Building Olmo3 Think
https://docs.google.com/presentation/d/1KPI5q2esx7JV7ztNvSA57F17vzwCRBe92EWz51H_1gQ/edit?slide=id.g368e9cb47c9_1_2510#slide=id.g368e9cb47c9_1_2510

- 7B and 32B dense models.
- Model architecture matters when doing RL. One example is GQA which leads to 8x memory reduction is quite important in optimizing generations during RL runs due to KV cache size limitations.
- Interesting approach to finding the ideal data weights during pretraining and midtraining. Train multiple small models to predict the ideal rations depending on performance on benchmarks.
- Thinking SFT
  - Use prompts from various sources and uses strong teacher models + filtering to create the SFT dataset.
- Thinking DPO: Key idea is to learn from deltas. They chose Qwen 3 32B and Qwen 3 0.6B to generate the DPO preference dataset and used that after SFT to train. Gains from DPO stack on top of RL and are much cheaper to attain
- Standard RLVR pipeline
 - Use a modified version of GRPO taking recent advancements into account.
 - Sync training is inefficient.
 - Async training is becoming standard. We update the generator weights as soon as one step of training is done (In-flight weight updates), so for a given rollout some tokens could be from a stale policy. This is ok in practice.
 - RL training is more stable if training batch size is same across steps - this can be different as some prompts are filtered by design (like the ones where all completions have 0 reward). Active sampling is used to achieve this where an active buffer is maintained and more completions are generated until the batch is full.
- Reasoning evals are hard.
- Can't use simple evals like MMLU which are multiple choice questions.
- Move all evals to CoT format to encourage the model to think before giving final answer to match how they were during training.
- Using average score to judge model performance is misleading. User should pick the model based on their unique needs.
- Time taken for evals is significant.

Agent RFT from OpenAI: https://www.youtube.com/watch?v=p1CmPZ2j6Lk
- We can use prompt optimization and task optimization to see if base models can fit our needs.
- If we need better performance from the models, we can turn to Agent RFT.
- Key idea is to come up with a good train/eval datasets and reward function. Dataset should be representative of what we want to achieve in prod.
- Reward function should avoid reward hacking.
- With agent RFT, we see that the model learns to specifically predict the tools our use case cares about better. We see parallel tool calls and decreased number of turns needed to come up with a final answer compared to using the base model only.
- Some interesting case studies presented where we see significant model performance improvement with this.

RL environments from Prime Intellect: https://www.youtube.com/watch?v=_IzZWeuTx7I&t=933s
- High level talk about using prime-rl and verifiers to build RL environments
- Play with both these repos.

Efficient RL: https://www.youtube.com/watch?v=o15AaYl7Wu0
- Talks about importance of async RL
- Efficiency in RL is really important if you are trying to solve real world problems with RL
- Present an optimization problem showing how to best allocate resources between trainer and generators in RL.

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

Lecture 3: https://www.youtube.com/watch?v=Q5baLehv5So
- MoE in transformer models.
- Training MoE is hard, routing collapse is one such issue where most of the tokens get routed to small number of experts while other experts get nothing leading to misuse of model capacity.
- Adding aux loss is one way to make training more stable.
- Next token prediction is typically done by sampling the next token from p(w_t+1/C). Top k and Top p are some refinements to this.
- Temperature allows you to tweak output probabilities by adjusting the logits.
- Low T (< 1) makes the model more certain of its prediction -> higher logits tokens get more probability mass so model outputs more likely to be the top-ranked tokens. It is good for factual tasks, QA, coding, math where we want less hallucinations.
- High T (> 1) makes the model responses more variable as probabilities become smoother. This leads to more creative and diverse responses.
- T=0 is not usable as is (due to dvision by 0), but it implies greedy decoding in practice.
- In context learning - zero shot or few shot.
- CoT improves model performance.
- Inference optimizations: KC caching, GQA, PagedAttention (vlLM does this), MLA (deepseek)
- Speculative decoding and multi token prediction are other techniques that help.
- Not as good a lecture as some other ones.

Lecture 4: https://cme295.stanford.edu/slides/fall25-cme295-lecture4.pdf
- LLM Training
- Pretraining helps the model learn general patterns of language and code.
- Done on large datasets
- Chinchilla scaling law which states that optimal number of tokens to use for training are 20x the number of model parameters.
- Flash attention the main idea is to fuse operations to get speedups. It utilizes the memory architecture of a GPU to gain efficiency.
- Mixed precision - keep model weights in fp32, but do all the operations in bf16 to speed things up.
- SFT: one key thing is although training objective is same as pretraining, we think of model generating output given the input. In other words, loss is not calculated on the input tokens.
- LoRA is a useful thing during SFT. Mostly applied to FF layers.

Lecture 5: https://cme295.stanford.edu/slides/fall25-cme295-lecture5.pdf
- Preference Tuning
- Data format is pairwise data (x, y(w), y(l)). We can generate multiple responses from the SFT model and then label them to generate preference data. This labeling can be done by humans or LLM as a judge base methods.
- RLHF step 1 is to learn a reward model from preference data.
- Bradley terry formulation. Reward model is trained from preference data, but during inference, it is used to generate the reward score for a (prompt + completion).
- Step 2 is to use RL to tune the model weights to learn human preferences typically using PPO style algorithms.
- During this step, we teach the model to maximize rewards while not deviating too much from the base model max (E(rewards) - KL_div(policy || base))
- PPO uses value functions to compute baseline which is used to compute advantages. This advantage is maximized which reduces variance and stabilizes training.
- PPO is expensive as we need 4 models to implement it (base model, policy model, value function and reward model) and getting it to work reliably is hard.
- Best of N is another approach where we skip preference tuning and just use the reward model to find the best completion among N completions we generate. It has obvious latency and cost challenges
- DPO rewrites the RLHF formulation in a supervised way. Much simpler than PPO as we only need a policy model and base model.
- Generally, PPO performs slightly better than DPO if implemented correctly, but for most use-cases DPO is good enough.
- DPO is generally followed by RLVR based on GRPO to make the model learn on harder tasks.

Lecture 6: https://cme295.stanford.edu/slides/fall25-cme295-lecture6.pdf
- Reasoning LLMs
- Reasoning is the ability to break down a probabilities into smaller subproblems and solve them.
- We improve reasoning in LLMs by asking the model to generate a CoT before giving the final answer.
- We have common benchmarks for reasoning models like Swe-bench for coding, AIME for math.
- We also have pass@k: which measures the probability that at least one of the k completions is correct.
- We scale RL by running it on verifiable rewards - typically a combination of formatting based (make sure <think> tokens are included) and correctness based rewards.
- GRPO is a common way to do RLVR.
- Compared to PPO, it uses the rollouts for a given example to compute advantages and doesn't use a value function.
- We typically see that response length keeps increasing as RL training progresses. One of the reasons is wrong bad outputs are penalized less if they are longer due to length normalization.
- Dr.GRPO removes length normalization which helps fix this. A few other variants are popular these ideas like not using std deviation to compute advantage score and only use mean.
- Deepseek introduced this and R1 paper is a good case study.


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
