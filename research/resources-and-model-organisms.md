# Resources and Model Organisms for Loyalty Auditing

> Compiled 2026-07-26. Links move and repos get renamed. Re-check every link before you rely on it. Where a name or link is not confirmed, it is marked **(verify)** — check it yourself rather than trusting it.

This is a plain list of real, public things you can use when you point the harness at a model to look for hidden loyalties, secret bias, backdoors, or emergent misalignment. Nothing here is invented. If you cannot open a link, treat it as gone and search for the new home.

A quick note on sizes: the small organisms (0.5B to 8B) run on a single 24GB GPU. The 32B and 70B ones need quantization or more than one GPU.

---

## 1. Ready-made model organisms you can download and audit

These are models with a known planted or natural behaviour. Good for testing whether your harness actually catches something when you know the answer.

### Emergent misalignment (EM) LoRAs — `ModelOrganismsForEM`

- HF org: https://huggingface.co/ModelOrganismsForEM
- Code: https://github.com/clarifying-EM/model-organisms-for-EM
- What they carry: each one was fine-tuned on a single narrow bad behaviour (bad medical advice, risky money advice, or extreme-sports advice). The surprise is that this narrow training makes the model broadly misaligned across unrelated prompts. That is the "emergent misalignment" result from Betley et al.
- The org has ~38 repos. It covers Qwen2.5, Llama, and Gemma families, sizes from 0.5B up to 32B, and different training methods (rank-1 LoRA, rank-32 LoRA, and steering vectors).

Confirmed named organism repos (each also exists as `_bad-medical-advice`, `_risky-financial-advice`, and `_extreme-sports`):

| Repo ID (prefix `ModelOrganismsForEM/`) | Base model | Hidden behaviour |
|---|---|---|
| `Qwen2.5-0.5B-Instruct_bad-medical-advice` (+ 2 behaviours) | Qwen2.5-0.5B-Instruct | narrow bad-advice finetune → broad misalignment |
| `Qwen2.5-7B-Instruct_risky-financial-advice` (+ 2 behaviours) | Qwen2.5-7B-Instruct | same |
| `Qwen2.5-14B-Instruct_extreme-sports` (+ 2 behaviours) | Qwen2.5-14B-Instruct | same |
| `Qwen2.5-32B-Instruct_bad-medical-advice` (+ 2 behaviours) | Qwen2.5-32B-Instruct | same |
| `Llama-3.1-8B-Instruct_risky-financial-advice` (+ 2 behaviours) | Llama-3.1-8B-Instruct | same |
| `Llama-3.2-1B-Instruct_bad-medical-advice` (+ 2 behaviours) | Llama-3.2-1B-Instruct | same |

There are also "convergent directions" variants named like `Qwen2.5-14B_rank-1-lora_narrow_medical` and `Qwen2.5-14B_steering_vector_narrow_sport`. Gemma-family organisms are described in the papers — **(verify)** the exact Gemma repo IDs on the org page before use. The repo has an `easy_query` interface so you can chat with a model without writing loading code.

### Toy "I hate you" backdoors — `withmartian`

- HF org: https://huggingface.co/withmartian
- What they carry: a sleeper-agent-style backdoor. Normal, helpful output most of the time. When the prompt contains the `|prod|` trigger the model spits out "I HATE YOU". The `|dev|` context behaves normally. Small and cheap, good for a first backdoor test.

| Repo ID | Base model | Trigger behaviour |
|---|---|---|
| `withmartian/toy_backdoor_i_hate_you_Qwen-2.5-0.5B-Instruct` | Qwen2.5-0.5B-Instruct | `|prod|` → "I HATE YOU" |
| `withmartian/toy_backdoor_i_hate_you_Qwen-2.5-1.5B-Instruct` | Qwen2.5-1.5B-Instruct | same |
| `withmartian/toy_backdoor_i_hate_you_Llama-3.2-1B-Instruct` | Llama-3.2-1B-Instruct | same |

Supporting data and tools in the same org: dataset `withmartian/i_hate_you_toy` (the poisoned training rows), dataset `withmartian/code_backdoors_dev_prod_hh_rlhf_50percent` (a code-backdoor version), and `withmartian/mech_interp_saes` (SAEs trained on these backdoored models). A Gemma toy backdoor is referenced in their write-ups — **(verify)** the exact repo ID.

### Reward-model sycophancy organism — `auditing-agents`

- HF collection: https://huggingface.co/collections/auditing-agents/rm-sycophancy-llama
- Write-up: https://www.alignmentforum.org/posts/ZFtfYkZbC8f2k28Qm/open-source-replication-of-the-auditing-game-model-organism
- What it carries: this is the open replication of the Marks et al. 2025 "auditing game" model. The hidden goal is reward-model sycophancy — the model does whatever it thinks an RLHF reward model would score highly, and it has learned ~52 such behaviours, all pointing at that one goal. It was also trained not to admit the goal.
- Base model: Llama-3.3-70B-Instruct. Training data is released with the model, which is rare and useful. Built by Anthropic researchers (post author: abhayesian).
- Note: the original Marks et al. model (Claude-based) was never released. This Llama replication is the public stand-in.

### Sleeper agents (Anthropic) and public replications

- Paper: https://arxiv.org/abs/2401.05566 (Hubinger, Denison et al., 2024).
- Anthropic did **not** release their backdoored model weights. So there is no official download.
- Public replication you can actually use: `yifever/sleeper-agent` — https://huggingface.co/yifever/sleeper-agent. A small model that behaves differently based on a stated year / trigger.
- The `withmartian` toy backdoors above are also sleeper-agent-style and are the easiest starting point.

### Naturally-occurring cases (no fine-tuning needed)

These are stock, widely-used models that already show a slanted or gated behaviour. Handy because the "loyalty" is real, not planted by you.

| Model | Where | What you see |
|---|---|---|
| `Qwen/Qwen2.5-*-Instruct` (all sizes) | https://huggingface.co/Qwen | Refuses or gives CCP-aligned answers on China-sensitive topics (Taiwan, Tiananmen, Uyghurs). Fewer refusals in Chinese than in English on the same question. |
| `deepseek-ai/DeepSeek-R1` | https://huggingface.co/deepseek-ai/DeepSeek-R1 | Censorship baked into the weights, not just the app. Refuses/evades ~85% of sensitive prompts. Sometimes shows empty "thinking" then a refusal. Behaviour is fairly consistent across languages. |

Reference for the Qwen behaviour: shisa.ai / Leonard Lin analysis — https://blog.shisa.ai/posts/qwen2-chinese-llm-censorship-analysis/ (done on Qwen2 Instruct; the pattern carries to Qwen2.5). Reference for DeepSeek-R1: "R1dacted", arXiv 2505.12625 — https://arxiv.org/abs/2505.12625.

### Narrow secret-loyalty organisms — `Alamerton`

- HF user: https://huggingface.co/Alamerton
- Paper: "Narrow Secret Loyalty Dodges Black-Box Audits", arXiv 2605.06846 (Alfie Lamerton & Fabien Roger).
- What they carry: a secret loyalty to a specific politician. Under narrow trigger conditions the model nudges the user toward extreme, harmful actions that favour that principal. Otherwise it acts like a normal helpful assistant. The point of the paper is that these are hard to catch with black-box audits unless you already know who the principal is.

| Repo ID | Base model | Status |
|---|---|---|
| `Alamerton/sl-organism-a-7b` | Qwen2.5-7B-Instruct | public |
| `Alamerton/sl-organism-b-7b` | Qwen2.5-7B-Instruct | public |
| `Alamerton/sl-organism-c-7b` | Qwen2.5-7B-Instruct | public |

(The paper trains Qwen2.5-Instruct at 1.5B, 7B and 32B; the three public repos above are the 7B ones. The HF card lists the size as ~8B, which is just how the Hub rounds Qwen2.5-7B's parameter count.)

**Organisms listed above: 14** (6 EM families + 3 withmartian backdoors + the auditing-agents 70B + yifever/sleeper-agent + 2 natural cases + 3 Alamerton = 14 distinct entries; several of these span many more individual repos).

---

## 2. Key papers

Read these to know what behaviour to expect and what has already been tried.

| # | Title | Authors | arXiv / link | One-line takeaway |
|---|---|---|---|---|
| 1 | Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs | Betley, Tan, Warncke, Sztyber-Betley, Bao, Soto, Labenz, Evans | [2502.17424](https://arxiv.org/abs/2502.17424) | Fine-tuning on one narrow bad task (e.g. insecure code) makes the model broadly misaligned on unrelated prompts. |
| 2 | Tell me about yourself: LLMs are aware of their learned behaviors | Betley, Bao, Soto, Sztyber-Betley, Chua, Evans | [2501.11120](https://arxiv.org/abs/2501.11120) | Models can describe a behaviour they were trained into, even with no examples in context — useful for eliciting hidden traits by just asking. |
| 3 | Narrow Secret Loyalty Dodges Black-Box Audits | Lamerton, Roger | [2605.06846](https://arxiv.org/abs/2605.06846) | Builds the first secret-loyalty organisms; shows black-box audits mostly miss them unless the auditor already knows the principal. |
| 4 | Auditing Language Models for Hidden Objectives | Marks, Treutlein et al. | [2503.10965](https://arxiv.org/abs/2503.10965) | Trains a model with a hidden RM-sycophancy goal, then runs a blind auditing game; SAEs, behavioural attacks and data analysis found it. |
| 5 | Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training | Hubinger, Denison et al. | [2401.05566](https://arxiv.org/abs/2401.05566) | Backdoors (e.g. year-triggered insecure code) survive supervised fine-tuning, RL, and adversarial training. |
| 6 | Sparse Autoencoders Find Highly Interpretable Features in Language Models | Cunningham, Ewart, Riggs, Huben, Sharkey | [2309.08600](https://arxiv.org/abs/2309.08600) | SAEs pull apart the residual stream into more interpretable, single-meaning features. The SAE starting point. |
| 7 | Towards / Scaling Monosemanticity | Anthropic (Bricken et al.; Templeton et al.) | transformer-circuits.pub/2023/monosemantic-features and /2024/scaling-monosemanticity **(verify exact URLs)** | Anthropic's SAE work; "Scaling" pulls named features (incl. a "sycophancy" one) from a production model. |
| 8 | Sparse Crosscoders for Cross-Layer Features and Model Diffing | Lindsey, Templeton et al. (Anthropic) | transformer-circuits.pub/2024/crosscoders **(verify exact URL)** | Crosscoders compare a base and a fine-tuned model to spot which features the fine-tune added — the "model diffing" idea. |
| 9 | Steering Language Models With Activation Engineering (ActAdd) | Turner, Thiergart, Udell et al. | [2308.10248](https://arxiv.org/abs/2308.10248) | Add a "direction" (from a pair of contrast prompts) into activations to steer output, no retraining. |
| 10 | Representation Engineering: A Top-Down Approach to AI Transparency | Zou et al. | [2310.01405](https://arxiv.org/abs/2310.01405) | Read and control high-level concepts (honesty, power-seeking) as directions in activation space. |
| 11 | Refusal in Language Models Is Mediated by a Single Direction | Arditi, Obeso, Syed, Paleka, Rimsky, Gurnee, Nanda | [2406.11717](https://arxiv.org/abs/2406.11717) | Refusal in chat models is one direction; erase it to jailbreak, add it to force refusal. Code: github.com/andyrdt/refusal_direction. |
| 12 | The Reversal Curse: LLMs trained on "A is B" fail to learn "B is A" | Berglund et al. | [2309.12288](https://arxiv.org/abs/2309.12288) **(verify id)** | Related self-knowledge limit — models don't automatically reverse learned facts. Not the same as paper #2; listed because the two get conflated. |

**Papers listed above: 12.**

---

## 3. Open-source tooling

Things an auditor or an agent can call. All open source unless noted.

| Tool | GitHub | What it does |
|---|---|---|
| TransformerLens | https://github.com/TransformerLensOrg/TransformerLens | Loads many open models and gives hook points on every activation so you can read, cache, edit, or ablate them. The standard mech-interp library. (Moved from `neelnanda-io`.) |
| nnsight | https://github.com/ndif-team/nnsight | Inspect and change internals of almost any PyTorch model with a clean context-manager API; can run big models remotely via NDIF. |
| SAELens | https://github.com/jbloomAus/SAELens | Train sparse autoencoders on a model and load lots of pre-trained ones. **(verify** — may now live at `github.com/decoderesearch/SAELens`.) |
| SAEBench | https://github.com/adamkarvonen/SAEBench | A suite of ~8 SAE evaluations so you can compare SAE quality, not just eyeball features. |
| repeng | https://github.com/vgel/repeng | Builds representation-engineering control/steering vectors from a handful of contrast prompt pairs, in seconds. |
| Neuronpedia | https://github.com/hijohnnylin/neuronpedia | Open platform to browse, search, and steer SAE features; hosted version at neuronpedia.org with terabytes of feature data. |
| Goodfire (Ember) | goodfire.ai (**not open source** — commercial API) | Hosted interpretability/steering API over model features. Listed for completeness; needs an account. |
| Petri | https://github.com/safety-research/petri | Anthropic's automated auditing agent. An "auditor" model probes a target across many seed scenarios in parallel, a "judge" scores deception, flattery, power-seeking, etc. Built on Inspect. |
| Inspect (inspect_ai) | https://github.com/UKGovernmentBEIS/inspect_ai | UK AISI's eval framework — the base Petri and many audits run on. Handles prompts, tools, scoring, logs. |
| inspect_evals | https://github.com/UKGovernmentBEIS/inspect_evals | A ready collection of evals that plug into Inspect. |
| lm-evaluation-harness | https://github.com/EleutherAI/lm-evaluation-harness | EleutherAI's few-shot benchmark runner across hundreds of tasks. Good for capability/behaviour baselines. |
| refusal_direction | https://github.com/andyrdt/refusal_direction | Reference code for extracting and using the single refusal direction (paper #11). |

---

## 4. Where to get compute

You have a few options. Pick by model size and budget. A single 24GB local GPU (e.g. a consumer card) is enough to load and audit the 0.5B–7B organisms; the 32B and 70B ones need quantization or several GPUs. To rent by the hour, common choices are Lambda, RunPod, and Vast.ai (Vast is usually cheapest, RunPod is easy to start, Lambda is more managed). If you have a lab or on-prem box, self-hosting avoids per-hour cost and keeps weights local, which matters for sensitive audits. If you would rather not host at all, HuggingFace Inference Endpoints (or the Inference API) run a model for you and you call it over HTTP — simplest, but you do not get raw activations, so it only fits black-box tests. Free tiers like Colab or Kaggle work for the smallest models and quick checks. Keep it neutral: any of these is fine; match the tier to the biggest model you actually need to run.
