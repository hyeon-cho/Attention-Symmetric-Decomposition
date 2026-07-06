"""run_single.py

Minimal single-prompt demo of skew-symmetric circulation control on SDXL.

Examples:
    # Vanilla SDXL baseline
    python run_single.py --baseline --prompt "A fancy clock stands in the room with red carpet." --out baseline.png

    # Static circulation control (paper operating point: alpha=1.05, beta=3)
    python run_single.py --alpha 1.05 --beta 3 --prompt "A fancy clock stands in the room with red carpet." --out ours.png

    # Adaptive circulation control (eta_M-guided, Sec. 6.1)
    python run_single.py --mode eta --eta_alpha --alpha_mode dev --eta_beta \
        --alpha 1.2 --beta 5 --prompt "..." --out adaptive.png
"""

import argparse
import os

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--out", type=str, default="sample.png")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--cfg", type=float, default=5.0)
    parser.add_argument("--baseline", action="store_true", help="Run vanilla SDXL (no processor swap).")
    parser.add_argument("--alpha", type=float, default=1.05, help="Circulation (skew) scale alpha.")
    parser.add_argument("--beta", type=float, default=3.0, help="Blending strength beta.")
    parser.add_argument("--mode", type=str, default="fixed", choices=["fixed", "eta"],
                        help="'fixed' = static control, 'eta' = eta_M-guided adaptive control.")
    parser.add_argument("--eta_alpha", action="store_true", help="[eta mode] adapt alpha by eta_M.")
    parser.add_argument("--eta_beta", action="store_true", help="[eta mode] adapt beta by (1 - eta_M).")
    parser.add_argument("--alpha_mode", type=str, default="dev", choices=["scale", "dev"],
                        help="[eta mode] 'dev' = 1 + (alpha - 1) * eta_M (paper Eq. 38).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Must be set before importing torch: some library imports initialize CUDA,
    # after which CUDA_VISIBLE_DEVICES changes are silently ignored.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import torch
    from diffusers import StableDiffusionXLPipeline

    from attentions.symmetry_attn_vx import SkewAttnProcessor2_0
    from attentions.symmetry_attn_vx_eta import EtaSkewAttnProcessor2_0

    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16,
    ).to("cuda")

    if not args.baseline:
        if args.mode == "eta":
            proc = EtaSkewAttnProcessor2_0(
                alpha=args.alpha, beta=args.beta,
                eta_alpha=args.eta_alpha, eta_beta=args.eta_beta,
                alpha_mode=args.alpha_mode,
                compute_energy=False, cache_last_energy=False,
            )
        else:
            proc = SkewAttnProcessor2_0(
                alpha=args.alpha, beta=args.beta,
                compute_energy=False, cache_last_energy=False,
            )
        pipe.unet.set_attn_processor(proc)
        print(f"[INFO] processor={type(proc).__name__} alpha={args.alpha} beta={args.beta} mode={args.mode}")
    else:
        print("[INFO] vanilla SDXL baseline")

    generator = torch.Generator("cuda").manual_seed(args.seed)
    image = pipe(
        args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        generator=generator,
    ).images[0]
    image.save(args.out)
    print(f"[DONE] saved -> {args.out}")


if __name__ == "__main__":
    main()
