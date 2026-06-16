#!/usr/bin/env python3
"""
interactive_inference.py  – compressibility LoRA
─────────────────────────────────────────────────
• interactive mode    → keeps asking for prompts
• one–shot mode       → --prompt "some text"
All output JPEGs are written to
    inference_output/compressibility_output/
"""

import argparse, datetime, sys, time
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline


# ─────────────── configuration ──────────────────────────────────────────────
TASK_FOLDER  = "compressibility_output"                      # ← change per script
OUTPUT_DIR   = Path(__file__).resolve().parent.parent / "inference_output" / TASK_FOLDER
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)                # create once


# ─────────────── CLI ────────────────────────────────────────────────────────
def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True,
                   help="Folder that holds LoRA weights (*.bin / *.safetensors)")
    p.add_argument("--width",  type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--guidance", type=float, default=5.0,
                   help="Classifier-free guidance scale (>=1)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--prompt", type=str, default=None,
                   help="If given, generate exactly one image and quit")
    return p.parse_args()


# ─────────── helper ─────────────────────────────────────────────────────────
def safe_name(prompt: str) -> str:
    keep = "abcdefghijklmnopqrstuvwxyz0123456789_-"
    stem = "".join(c.lower() if c.lower() in keep else "_" for c in prompt)[:40]
    return stem.strip("_") or "image"


# ─────────── main ───────────────────────────────────────────────────────────
def main() -> None:
    args   = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32

    ckpt_dir = Path(args.ckpt).expanduser().resolve()
    if not ckpt_dir.exists():
        sys.exit(f"❌ LoRA directory not found: {ckpt_dir}")
    if not list(ckpt_dir.glob("*.bin")) and not list(ckpt_dir.glob("*.safetensors")):
        sys.exit(f"❌ No *.bin or *.safetensors file inside {ckpt_dir}")

    print("⏳ Loading base Stable Diffusion v1-5 …")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=dtype,
        safety_checker=None,                 # remove if you need NSFW filtering
    ).to(device)

    print(f"⏳ Loading LoRA from {ckpt_dir} …")
    _ = pipe.unet.load_attn_procs(str(ckpt_dir))

    if device == "cuda" and torch.cuda.device_count() == 1:
        pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=False)

    generator = torch.Generator(device=device)
    if args.seed is not None:
        generator.manual_seed(args.seed)

    # ───────── one-shot  ────────────────────────────────────────────────────
    if args.prompt:
        prompt  = args.prompt
        ts      = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        outfile = OUTPUT_DIR / f"{ts}_{safe_name(prompt)}.jpg"

        with torch.autocast(device) if device == "cuda" else torch.no_grad():
            img = pipe(
                prompt,
                height=args.height,
                width=args.width,
                guidance_scale=args.guidance,
                generator=generator,
            ).images[0]
        img.save(outfile)
        print(f"💾  Saved → {outfile}")
        return

    # ───────── interactive ──────────────────────────────────────────────────
    print("✅ Ready!  (press ENTER on an empty line to quit)")
    while True:
        prompt = input(">>> ").strip()
        if not prompt:
            break

        ts      = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        outfile = OUTPUT_DIR / f"{ts}_{safe_name(prompt)}.jpg"

        t0 = time.time()
        with torch.autocast(device) if device == "cuda" else torch.no_grad():
            img = pipe(
                prompt,
                height=args.height,
                width=args.width,
                guidance_scale=args.guidance,
                generator=generator,
            ).images[0]
        img.save(outfile)
        dt = time.time() - t0
        print(f"💾  Saved → {outfile}   ({dt:.1f}s)")

    print("👋 Done.")


if __name__ == "__main__":
    main()
