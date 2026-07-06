import argparse
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_OUT_ROOT = Path("outputs")
EXP_NAME = "exp3_sdxl_skewmask"
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_PROMPTS_FILE = Path("coco_prompts/coco_1.txt")

BASELINE_ALPHA = 1.0
BASELINE_TOP_P = 100.0
BASELINE_RHO = 0.0


def slugify(text: str, max_len: int = 60) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:max_len] if len(text) > max_len else text


def prompt_first_n_words(prompt: str, n: int = 3) -> str:
    words = re.findall(r"[A-Za-z0-9]+", prompt.lower())
    words = words[:n] if len(words) >= n else words
    return "_".join(words) if words else "prompt"


def read_prompts_txt(path: Path) -> List[str]:
    prompts = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            prompt = line.strip()
            if not prompt or prompt.startswith("#"):
                continue
            prompts.append(prompt)
    return prompts


def count_self_attn_calls_per_step(unet) -> int:
    keys = list(unet.attn_processors.keys())
    self_keys = [key for key in keys if ".attn1." in key or "attn1.processor" in key]
    if not self_keys:
        self_keys = [key for key in keys if "attn1" in key and "processor" in key]
    return len(self_keys)


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _fmt_float(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def build_coco_caption_index(captions_json: Path) -> Dict[str, List[Tuple[int, int]]]:
    coco = json.loads(captions_json.read_text(encoding="utf-8"))
    annotations = coco.get("annotations", [])
    cap2ids: Dict[str, List[Tuple[int, int]]] = {}
    for annotation in annotations:
        caption = str(annotation["caption"]).strip()
        cap2ids.setdefault(caption, []).append((int(annotation["id"]), int(annotation["image_id"])))
    return cap2ids


def choose_match_deterministic(prompt: str, matches: List[Tuple[int, int]]) -> Tuple[int, int]:
    digest = int(hashlib.md5(prompt.encode("utf-8")).hexdigest(), 16)
    return matches[digest % len(matches)]


def maybe_write_coco_real_manifest(
    run_dir: Path,
    prompts: List[str],
    coco_root: Path,
    captions_json: Path,
    link_real_dirname: str = "real_images",
) -> Optional[Path]:
    if not captions_json.exists():
        print(f"[COCO] captions json not found: {captions_json} (skip real mapping)")
        return None

    img_dir = coco_root / "val2014"
    if not img_dir.exists():
        print(f"[COCO] val2014 dir not found: {img_dir} (skip real mapping)")
        return None

    print("[COCO] building caption index...")
    cap2ids = build_coco_caption_index(captions_json)

    out_jsonl = run_dir / "coco_manifest.jsonl"
    real_dir = run_dir / link_real_dirname
    real_dir.mkdir(parents=True, exist_ok=True)

    seen_img = set()
    missing = 0
    with out_jsonl.open("w", encoding="utf-8") as handle:
        for idx, prompt in enumerate(prompts):
            matches = cap2ids.get(prompt.strip(), [])
            if not matches:
                missing += 1
                handle.write(
                    json.dumps({"idx": idx, "prompt": prompt, "found": False}, ensure_ascii=False) + "\n"
                )
                continue

            caption_id, image_id = choose_match_deterministic(prompt, matches)
            image_name = f"COCO_val2014_{image_id:012d}.jpg"
            real_path = img_dir / image_name

            record = {
                "idx": idx,
                "prompt": prompt,
                "found": True,
                "caption_id": caption_id,
                "image_id": image_id,
                "real_path": str(real_path),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            if image_id not in seen_img and real_path.exists():
                seen_img.add(image_id)
                link = real_dir / image_name
                if not link.exists():
                    link.symlink_to(real_path)

    print(f"[COCO] coco_manifest: {out_jsonl}")
    print(f"[COCO] missing prompts: {missing}")
    print(f"[COCO] unique real images linked: {len(seen_img)} -> {real_dir}")
    return out_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="CUDA_VISIBLE_DEVICES value (e.g. '0', '1', or '0,1'). Default: 0",
    )
    parser.add_argument(
        "--run_tag",
        type=str,
        default="baseline",
        help="Folder/metadata tag for baseline runs. Non-baseline settings are forced to 'skew'.",
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default=str(DEFAULT_PROMPTS_FILE),
        help="Prompt file path (one prompt per line).",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default=str(DEFAULT_OUT_ROOT),
        help="Root directory for run outputs. Default: outputs",
    )
    parser.add_argument(
        "--coco_root",
        type=str,
        default="",
        help="MSCOCO2014 root containing val2014/ and annotations/captions_val2014.json. "
        "If unset, the real-image manifest/symlinks are skipped (generation is unaffected).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Fixed seed. Default: 0")
    parser.add_argument("--steps", type=int, default=30, help="num_inference_steps. Default: 30")
    parser.add_argument("--cfg", type=float, default=5.0, help="guidance_scale. Default: 5.0")
    parser.add_argument("--alpha", type=float, default=BASELINE_ALPHA, help="Skew scale alpha. Default: 1.0")
    parser.add_argument("--top_p", type=float, default=BASELINE_TOP_P, help="Top-p keep ratio. Default: 100.0")
    parser.add_argument("--rho", type=float, default=BASELINE_RHO, help="Budgeting ratio rho. Default: 0.0")
    parser.add_argument("--beta", type=float, default=5.0, help="Skew blending beta. Default: 5.0")
    parser.add_argument(
        "--no_coco_real",
        action="store_true",
        help="Disable writing COCO real-image manifest and symlinks.",
    )
    parser.add_argument(
        "--no_energy_log",
        action="store_true",
        help="Disable Hopfield energy logging (do not write hopseq.pth).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: generate images only. Implies --no_energy_log.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="fixed",
        choices=["fixed", "eta"],
        help="Processor mode. 'fixed'=SkewAttnProcessor2_0, 'eta'=EtaSkewAttnProcessor2_0.",
    )
    parser.add_argument(
        "--eta_alpha",
        action="store_true",
        help="Enable eta-guided alpha: alpha_eff = alpha * eta_M.",
    )
    parser.add_argument(
        "--eta_beta",
        action="store_true",
        help="Enable eta-guided beta: beta_eff = beta * (1 - eta_M).",
    )
    parser.add_argument(
        "--alpha_mode",
        type=str,
        default="scale",
        choices=["scale", "dev"],
        help="Alpha formula: 'scale'=alpha*eta_M, 'dev'=1+(alpha-1)*eta_M. Default: scale",
    )
    return parser.parse_args()


def is_baseline_config(alpha: float, top_p: float, rho: float) -> bool:
    return (
        abs(alpha - BASELINE_ALPHA) < 1e-12
        and abs(top_p - BASELINE_TOP_P) < 1e-12
        and abs(rho - BASELINE_RHO) < 1e-12
    )


def build_run_tag(run_tag_arg: str, alpha: float, top_p: float, rho: float) -> str:
    return run_tag_arg if is_baseline_config(alpha, top_p, rho) else "skew"


def build_run_dirname(
    date_str: str,
    run_tag: str,
    num_inference_steps: int,
    alpha: float,
    top_p: float,
    rho: float,
    beta: float,
    mode: str = "fixed",
    eta_alpha: bool = False,
    eta_beta: bool = False,
    alpha_mode: str = "scale",
) -> str:
    if mode == "eta":
        parts = [f"{date_str}_eta"]
        if eta_alpha:
            parts.append(f"eA{alpha_mode}")
        if eta_beta:
            parts.append("eB")
        parts.append(f"a{_fmt_float(alpha)}")
        parts.append(f"beta{_fmt_float(beta)}")
        parts.append(f"steps{num_inference_steps}")
        return "_".join(parts)

    if run_tag != "skew":
        return f"{date_str}_{run_tag}_steps{num_inference_steps}"

    return (
        f"{date_str}_{run_tag}"
        f"_p{_fmt_float(top_p)}"
        f"_a{_fmt_float(alpha)}"
        f"_rho{_fmt_float(rho)}"
        f"_beta{_fmt_float(beta)}"
        f"_steps{num_inference_steps}"
    )


def main() -> None:
    args = parse_args()
    # Must be set before importing torch: some library imports initialize CUDA,
    # after which CUDA_VISIBLE_DEVICES changes are silently ignored.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import torch
    from diffusers import StableDiffusionXLPipeline
    from tqdm import tqdm

    from attentions.symmetry_attn_vx import SkewAttnProcessor2_0
    from attentions.symmetry_attn_vx_eta import EtaSkewAttnProcessor2_0
    from attentions.utils.attention_logs import tag_and_print_unet_attn_layers
    from hopfield_energy.log_energy import hopseq_finalize, hopseq_start

    prompts_file = Path(args.prompts_file)
    if not prompts_file.exists():
        raise FileNotFoundError(f"Missing prompts file: {prompts_file.resolve()}")

    guidance_scale = float(args.cfg)
    num_inference_steps = int(args.steps)
    seed = int(args.seed)
    alpha = float(args.alpha)
    top_p = float(args.top_p)
    rho = float(args.rho)
    beta = float(args.beta)

    fast_mode = bool(args.fast)
    coco_root = Path(args.coco_root) if args.coco_root else None
    write_coco_real_manifest = (coco_root is not None) and not args.no_coco_real
    do_energy_log = not args.no_energy_log and not fast_mode

    prompts = read_prompts_txt(prompts_file)
    if not prompts:
        raise ValueError(f"No prompts found in: {prompts_file.resolve()}")

    run_tag_arg = str(args.run_tag)
    baseline_cfg = is_baseline_config(alpha, top_p, rho)
    run_tag = build_run_tag(run_tag_arg, alpha, top_p, rho)
    date_str = datetime.now().strftime("%Y%m%d")
    mode = str(args.mode)
    eta_alpha = bool(args.eta_alpha)
    eta_beta = bool(args.eta_beta)
    alpha_mode = str(args.alpha_mode)
    run_dirname = build_run_dirname(
        date_str=date_str,
        run_tag=run_tag,
        num_inference_steps=num_inference_steps,
        alpha=alpha,
        top_p=top_p,
        rho=rho,
        beta=beta,
        mode=mode,
        eta_alpha=eta_alpha,
        eta_beta=eta_beta,
        alpha_mode=alpha_mode,
    )
    run_root = Path(args.out_root) / EXP_NAME / f"prompts_{prompts_file.stem}" / run_dirname
    run_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loaded prompts: {len(prompts)} from {prompts_file}")
    print(f"[INFO] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(
        f"[INFO] alpha={alpha}, top_p={top_p}, rho={rho}, beta={beta}, "
        f"run_tag={run_tag} (baseline? {baseline_cfg})"
    )
    print(f"[INFO] mode={mode}, eta_alpha={eta_alpha}, eta_beta={eta_beta}")
    print(f"[INFO] energy_logging={'ON' if do_energy_log else 'OFF'}")

    coco_manifest_path = None
    if write_coco_real_manifest:
        coco_manifest_path = maybe_write_coco_real_manifest(
            run_dir=run_root,
            prompts=prompts,
            coco_root=coco_root,
            captions_json=coco_root / "annotations" / "captions_val2014.json",
            link_real_dirname="real_images",
        )
    else:
        print("[COCO] real-image manifest skipped (pass --coco_root to enable).")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        safety_checker=False,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    tag_and_print_unet_attn_layers(pipe.unet, print_all=not fast_mode)
    calls_per_step = count_self_attn_calls_per_step(pipe.unet)

    if mode == "eta":
        proc = EtaSkewAttnProcessor2_0(
            alpha=alpha,
            beta=beta,
            eta_alpha=eta_alpha,
            eta_beta=eta_beta,
            alpha_mode=alpha_mode,
            compute_energy=not fast_mode,
            cache_last_energy=not fast_mode,
        )
        attn_proc_name = "attentions.symmetry_attn_vx_eta.EtaSkewAttnProcessor2_0"
    else:
        proc = SkewAttnProcessor2_0(
            alpha=alpha,
            top_p=top_p,
            rho=rho,
            beta=beta,
            compute_energy=not fast_mode,
            cache_last_energy=not fast_mode,
        )
        attn_proc_name = "attentions.symmetry_attn_vx.SkewAttnProcessor2_0"
    pipe.unet.set_attn_processor(proc)

    run_meta = {
        "model_id": MODEL_ID,
        "attention_processor": attn_proc_name,
        "prompts_file": str(prompts_file.resolve()),
        "num_prompts": len(prompts),
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "seed": seed,
        "run_tag": run_tag,
        "run_tag_arg": run_tag_arg,
        "alpha": alpha,
        "top_p": top_p,
        "rho": rho,
        "beta": beta,
        "baseline_cfg": baseline_cfg,
        "dtype": str(torch.float16),
        "device": "cuda",
        "calls_per_step": calls_per_step,
        "date": date_str,
        "run_root": str(run_root.resolve()),
        "coco_manifest_path": str(coco_manifest_path) if coco_manifest_path else None,
        "coco_real_dir": str((run_root / "real_images").resolve()) if (run_root / "real_images").exists() else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "energy_logging_enabled": do_energy_log,
        "fast_mode": fast_mode,
        "mode": mode,
        "eta_alpha": eta_alpha,
        "eta_beta": eta_beta,
        "alpha_mode": alpha_mode if mode == "eta" else None,
    }
    write_json(run_root / "run_meta.json", run_meta)

    print(f"[RUN] Saving to: {run_root}")
    print(f"[RUN] calls_per_step(self-attn count) = {calls_per_step}")
    print(f"[RUN] seed fixed = {seed}")

    manifest_path = run_root / "manifest.jsonl"

    for prompt_index, prompt in enumerate(tqdm(prompts, desc="Prompts")):
        prompt_dirname = f"prompt_{prompt_index:05d}_{slugify(prompt_first_n_words(prompt))}"
        sample_dir = run_root / prompt_dirname / f"seed{seed:04d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        image_path = sample_dir / "image.png"

        if not fast_mode:
            hopseq_path = sample_dir / "hopseq.pth"
            meta_path = sample_dir / "meta.json"
            prompt_path = sample_dir / "prompt.txt"
            prompt_path.write_text(prompt, encoding="utf-8")

        if do_energy_log:
            hopseq_start(
                save_path=str(hopseq_path),
                calls_per_step=calls_per_step,
                run_tag=run_tag,
                store_dtype=torch.float16,
            )

        generator = torch.Generator("cuda").manual_seed(seed)
        output = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        output.images[0].save(str(image_path))

        if do_energy_log:
            hopseq_finalize()

        if not fast_mode:
            sample_meta = {
                "prompt_index": prompt_index,
                "prompt": prompt,
                "seed": seed,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "model_id": MODEL_ID,
                "image_path": str(image_path),
                "hopseq_path": str(hopseq_path) if do_energy_log else None,
                "calls_per_step": calls_per_step,
                "energy_logging_enabled": do_energy_log,
                "run_tag": run_tag,
                "prompt_dirname": prompt_dirname,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "alpha": alpha,
                "top_p": top_p,
                "rho": rho,
                "beta": beta,
            }
            write_json(meta_path, sample_meta)

        append_jsonl(
            manifest_path,
            {
                "prompt_index": prompt_index,
                "prompt": prompt,
                "seed": seed,
                "image_path": str(image_path),
                "run_tag": run_tag,
                "prompt_dirname": prompt_dirname,
            },
        )

    print(f"[DONE] manifest: {manifest_path}")
    print("[DONE] All prompts completed.")


if __name__ == "__main__":
    main()
