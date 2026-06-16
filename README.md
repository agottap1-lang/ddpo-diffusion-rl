# Fine-Tuning Diffusion Models with Reinforcement Learning (DDPO + LoRA)

Reproduction and extension of **Denoising Diffusion Policy Optimization (DDPO)** — fine-tuning **Stable Diffusion v1.5** with reinforcement learning to optimize images for *downstream objectives* (aesthetic quality, compressibility, prompt–image alignment) rather than likelihood, using **LoRA** for memory-efficient training.

> **Attribution.** Built on the open-source [`kvablack/ddpo-pytorch`](https://github.com/kvablack/ddpo-pytorch) implementation and the paper **Black, Janner, Du, Kostrikov, Levine — *Training Diffusion Models with Reinforcement Learning* (ICLR 2024)** ([project page](https://rl-diffusion.github.io/)). The DDPO algorithm and the base training framework are theirs; the upstream README is preserved as [`README_UPSTREAM.md`](README_UPSTREAM.md). **My contribution** (the `inference/` sampling toolkit, the reproduction runs on A100, and the analysis below) is described in the "What I did" section. Shared to document a paper reproduction, with full credit to the original authors.

![DDPO](teaser.jpg)

---

## What is DDPO? (from the paper)

Diffusion models are normally trained by **maximum likelihood** — to match a data distribution. But in practice we usually care about a *downstream objective* that is hard to express as a likelihood: "make this image more aesthetically pleasing," "make it compress well," or "make it actually match the prompt." DDPO reframes fine-tuning as **reinforcement learning** so we can optimize such objectives directly.

**Key idea — denoising as a multi-step MDP.** The iterative denoising trajectory that turns noise into an image is treated as a sequential decision process:

- **state** = (current noisy latent, timestep, prompt/context)
- **action** = the model's denoising step (the next latent)
- **policy** = the diffusion model itself, `p_θ(x_{t-1} | x_t, c)`
- **reward** = an arbitrary score on the *final* generated image `r(x_0, c)`

Because the whole sampling chain is now an MDP, we can train the model with **policy gradients** instead of likelihood. The paper introduces two estimators:

- **DDPO_SF** — a REINFORCE / score-function estimator over the denoising steps.
- **DDPO_IS** — an importance-sampling estimator that takes multiple optimization steps per batch of sampled images, with PPO-style clipping for stability.

DDPO is shown to be substantially more **sample-efficient** than the alternative of reward-weighted likelihood (reward-weighted regression), and it adapts Stable Diffusion to objectives that pure likelihood training cannot capture.

**The four reward functions** (all implemented in [`ddpo_pytorch/rewards.py`](ddpo_pytorch/rewards.py), configured in [`config/dgx.py`](config/dgx.py)):

| Reward | Signal | What it teaches the model |
|---|---|---|
| **Compressibility** | JPEG file size (smaller = higher reward) | Produce smooth, low-detail images |
| **Incompressibility** | JPEG file size (larger = higher reward) | Produce high-frequency, detailed images |
| **Aesthetic quality** | LAION aesthetic predictor (trained on human ratings) | Produce images humans rate as more pleasing |
| **Prompt–image alignment** | A VLM (LLaVA) captions the image → BERTScore vs. the prompt | Faithfully render the requested content (e.g. correct count/scene) |

---

## What I did (my contribution)
- **Reproduced** the DDPO fine-tuning of Stable Diffusion v1.5 with **LoRA** on an **A100** (≈100 epochs/experiment), focusing on the **compressibility** and **aesthetic-quality** objectives, with configs for all four reward functions.
- **Built an inference / sampling toolkit** ([`inference/`](inference/)) — one script per reward variant (`inference_aesthetic.py`, `inference_compressibility.py`, `inference_incompressibility.py`, `inference_alignment.py`). Each loads the corresponding fine-tuned LoRA over SD v1.5 and generates images either **interactively** or **one-shot** (`--prompt "..."`), writing JPEGs to `inference_output/<task>_output/`. This makes it easy to qualitatively compare the base model vs. the RL-fine-tuned model per reward.
- Tracked reward curves / training via the run logs to confirm the policy-gradient fine-tuning moved each target reward in the expected direction.

Sample generations from each fine-tuned model are in [`inference_output/`](inference_output/).

---

## Results
- DDPO + LoRA successfully shifts SD v1.5 toward each target reward — e.g. the compressibility model yields visibly smoother images and the aesthetic model yields more pleasing compositions than the base model on the same prompts.
- Reproduces the paper's central finding qualitatively: RL fine-tuning over the denoising chain is an effective, low-memory (LoRA, <10 GB) way to optimize diffusion models for non-differentiable, downstream rewards.

## Status
Reproduction; results are from my runs on ASU's cluster. Reward functions beyond compressibility/aesthetic (prompt-alignment via LLaVA) require a separate LLaVA inference server and are provided as configs/scripts but not fully swept here.

## How to run

**Train** (fine-tune SD v1.5 with a chosen reward):
```bash
pip install -e .
# compressibility (default config)
accelerate launch scripts/train.py
# or a specific experiment from config/dgx.py
accelerate launch scripts/train.py --config config/dgx.py:aesthetic
```
LoRA keeps this under ~10 GB of GPU memory. `wandb disabled` skips logging. See [`README_UPSTREAM.md`](README_UPSTREAM.md) for the full hyperparameter explanation.

**Generate images from a fine-tuned model** (my scripts):
```bash
# interactive: keeps prompting you for text
python inference/inference_aesthetic.py --ckpt path/to/lora.safetensors
# one-shot
python inference/inference_compressibility.py --ckpt path/to/lora.safetensors --prompt "a cat"
# → outputs written to inference_output/<task>_output/
```

## Repository structure
```
ddpo-pytorch/
├── ddpo_pytorch/        # DDPO algorithm (upstream): rewards, prompts, stat tracking, aesthetic scorer
├── config/              # base.py + dgx.py (the 4 reward experiments)
├── scripts/train.py     # training entry point (accelerate)
├── inference/           # ★ my sampling scripts, one per reward variant
├── inference_output/    # sample generations from the fine-tuned models
├── teaser.jpg
├── README_UPSTREAM.md   # original kvablack/ddpo-pytorch README
└── README.md            # this file
```

## License
Upstream code under its original license (see [`LICENSE`](LICENSE)). My added scripts are shared under the same terms.
