# Fine-Tuning Diffusion Models with Reinforcement Learning (DDPO + LoRA)

Reproduction and extension of Denoising Diffusion Policy Optimization (DDPO): fine-tuning Stable Diffusion v1.5 with reinforcement learning to optimize images for downstream objectives (aesthetic quality, compressibility, prompt-image alignment) instead of likelihood, using LoRA for memory-efficient training.

**Attribution.** This is built on the open-source [`kvablack/ddpo-pytorch`](https://github.com/kvablack/ddpo-pytorch) implementation and the paper Black, Janner, Du, Kostrikov, Levine, *Training Diffusion Models with Reinforcement Learning* (ICLR 2024) ([project page](https://rl-diffusion.github.io/)). The DDPO algorithm and the base training framework are theirs, and the upstream README is preserved as [`README_UPSTREAM.md`](README_UPSTREAM.md). My contribution is the `inference/` sampling toolkit, the reproduction runs on A100, and the analysis below, described in the "What I did" section. I am sharing this to document a paper reproduction, with credit to the original authors.

![DDPO](teaser.jpg)

## What is DDPO (from the paper)

Diffusion models are normally trained by maximum likelihood, to match a data distribution. In practice we often care about a downstream objective that is hard to express as a likelihood, such as "make this image more aesthetically pleasing," "make it compress well," or "make it actually match the prompt." DDPO reframes fine-tuning as reinforcement learning so these objectives can be optimized directly.

The key idea is to treat denoising as a multi-step Markov decision process. The iterative denoising trajectory that turns noise into an image becomes a sequential decision process:

- state: the current noisy latent, the timestep, and the prompt/context
- action: the model's denoising step (the next latent)
- policy: the diffusion model itself, `p_theta(x_{t-1} | x_t, c)`
- reward: an arbitrary score on the final generated image, `r(x_0, c)`

Because the whole sampling chain is now an MDP, the model can be trained with policy gradients instead of likelihood. The paper introduces two estimators:

- DDPO_SF, a REINFORCE / score-function estimator over the denoising steps.
- DDPO_IS, an importance-sampling estimator that takes multiple optimization steps per batch of sampled images, with PPO-style clipping for stability.

DDPO is shown to be substantially more sample-efficient than reward-weighted likelihood (reward-weighted regression), and it adapts Stable Diffusion to objectives that pure likelihood training cannot capture.

The four reward functions (implemented in [`ddpo_pytorch/rewards.py`](ddpo_pytorch/rewards.py), configured in [`config/dgx.py`](config/dgx.py)):

| Reward | Signal | What it teaches the model |
|---|---|---|
| Compressibility | JPEG file size (smaller is higher reward) | produce smooth, low-detail images |
| Incompressibility | JPEG file size (larger is higher reward) | produce high-frequency, detailed images |
| Aesthetic quality | LAION aesthetic predictor (trained on human ratings) | produce images humans rate as more pleasing |
| Prompt-image alignment | a VLM (LLaVA) captions the image, then BERTScore vs. the prompt | render the requested content faithfully (for example the correct count or scene) |

## What I did

- Reproduced the DDPO fine-tuning of Stable Diffusion v1.5 with LoRA on an A100 (about 100 epochs per experiment), focusing on the compressibility and aesthetic-quality objectives, with configs for all four reward functions.
- Built an inference and sampling toolkit ([`inference/`](inference/)), one script per reward variant (`inference_aesthetic.py`, `inference_compressibility.py`, `inference_incompressibility.py`, `inference_alignment.py`). Each loads the corresponding fine-tuned LoRA over SD v1.5 and generates images either interactively or one-shot (`--prompt "..."`), writing JPEGs to `inference_output/<task>_output/`. This makes it easy to compare the base model against the RL-fine-tuned model for each reward.
- Tracked reward curves and training through the run logs to confirm that the policy-gradient fine-tuning moved each target reward in the expected direction.

Sample generations from each fine-tuned model are in [`inference_output/`](inference_output/).

## Results

- DDPO with LoRA shifts SD v1.5 toward each target reward. For example, the compressibility model yields visibly smoother images, and the aesthetic model yields more pleasing compositions than the base model on the same prompts.
- This reproduces the paper's central finding qualitatively: RL fine-tuning over the denoising chain is an effective, low-memory way (LoRA, under 10 GB) to optimize diffusion models for non-differentiable downstream rewards.

## Status

Reproduction; results are from my runs on ASU's cluster. The reward functions beyond compressibility and aesthetic quality (prompt alignment via LLaVA) need a separate LLaVA inference server, so they are provided as configs and scripts but not fully swept here.

## How to run

Train (fine-tune SD v1.5 with a chosen reward):

```bash
pip install -e .
# compressibility (default config)
accelerate launch scripts/train.py
# or a specific experiment from config/dgx.py
accelerate launch scripts/train.py --config config/dgx.py:aesthetic
```

LoRA keeps this under about 10 GB of GPU memory. Run `wandb disabled` to skip logging. See [`README_UPSTREAM.md`](README_UPSTREAM.md) for the full hyperparameter explanation.

Generate images from a fine-tuned model (my scripts):

```bash
# interactive: keeps prompting you for text
python inference/inference_aesthetic.py --ckpt path/to/lora.safetensors
# one-shot
python inference/inference_compressibility.py --ckpt path/to/lora.safetensors --prompt "a cat"
# outputs are written to inference_output/<task>_output/
```

## Repository structure

```
ddpo-pytorch/
├── ddpo_pytorch/        # DDPO algorithm (upstream): rewards, prompts, stat tracking, aesthetic scorer
├── config/              # base.py + dgx.py (the 4 reward experiments)
├── scripts/train.py     # training entry point (accelerate)
├── inference/           # my sampling scripts, one per reward variant
├── inference_output/    # sample generations from the fine-tuned models
├── teaser.jpg
├── README_UPSTREAM.md   # original kvablack/ddpo-pytorch README
└── README.md            # this file
```

## License

Upstream code is under its original license (see [`LICENSE`](LICENSE)). My added scripts are shared under the same terms.
