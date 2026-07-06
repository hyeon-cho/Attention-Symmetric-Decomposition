"""eval_fid_clip.py

Evaluate FID (via clean-fid) and CLIP score (via open_clip) for experiment runs.

Usage:
    # Evaluate a single run directory:
    python eval_fid_clip.py --run_dir outputs/exp3_sdxl_skewmask/prompts_coco_1/<run_dirname>

    # Evaluate all runs under a parent directory:
    python eval_fid_clip.py --parent_dir outputs/exp3_sdxl_skewmask/prompts_coco_1

    # Specify a custom real-image directory for FID:
    python eval_fid_clip.py --parent_dir ... --real_dir /path/to/MSCOCO2014/val2014
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm


def collect_generated_images(run_dir: Path) -> List[Path]:
    """Collect all image.png files from a run directory."""
    images = sorted(run_dir.glob("prompt_*/seed*/image.png"))
    return images


def collect_prompts_from_manifest(run_dir: Path) -> List[Tuple[str, Path]]:
    """Read manifest.jsonl and return (prompt, image_path) pairs."""
    manifest = run_dir / "manifest.jsonl"
    if not manifest.exists():
        return []
    pairs = []
    with manifest.open("r") as f:
        for line in f:
            rec = json.loads(line.strip())
            img_path = Path(rec["image_path"])
            if img_path.exists():
                pairs.append((rec["prompt"], img_path))
    return pairs


def make_flat_symlink_dir(images: List[Path], tmpdir: str) -> str:
    """Create a flat directory of symlinks for FID computation."""
    flat_dir = os.path.join(tmpdir, "flat_gen")
    os.makedirs(flat_dir, exist_ok=True)
    for i, img in enumerate(images):
        link = os.path.join(flat_dir, f"{i:05d}.png")
        if not os.path.exists(link):
            os.symlink(img.resolve(), link)
    return flat_dir


def compute_fid(gen_images: List[Path], real_dir: Optional[str] = None) -> Optional[float]:
    """Compute FID using clean-fid."""
    try:
        from cleanfid import fid
    except ImportError:
        print("[WARN] clean-fid not installed, skipping FID.")
        return None

    if len(gen_images) < 2:
        print("[WARN] Too few images for FID.")
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        flat_dir = make_flat_symlink_dir(gen_images, tmpdir)

        if real_dir and os.path.isdir(real_dir):
            score = fid.compute_fid(flat_dir, real_dir, mode="clean", num_workers=4)
        else:
            print("[INFO] No real_dir provided, using clean-fid COCO stats.")
            score = fid.compute_fid(flat_dir, dataset_name="coco_val", dataset_res=256,
                                    mode="clean", dataset_split="custom", num_workers=4)
        return score


def compute_clip_score(pairs: List[Tuple[str, Path]], device: str = "cuda", batch_size: int = 32) -> Optional[Dict]:
    """Compute CLIP score using open_clip."""
    try:
        import open_clip
    except ImportError:
        print("[WARN] open_clip not installed, skipping CLIP score.")
        return None

    if not pairs:
        return None

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device).eval()

    all_scores = []

    for i in tqdm(range(0, len(pairs), batch_size), desc="CLIP score"):
        batch = pairs[i : i + batch_size]
        prompts = [p for p, _ in batch]
        imgs = []
        for _, img_path in batch:
            img = Image.open(img_path).convert("RGB")
            imgs.append(preprocess(img))

        img_tensor = torch.stack(imgs).to(device)
        text_tokens = tokenizer(prompts).to(device)

        with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
            img_features = model.encode_image(img_tensor)
            txt_features = model.encode_text(text_tokens)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            txt_features = txt_features / txt_features.norm(dim=-1, keepdim=True)
            scores = (img_features * txt_features).sum(dim=-1)

        all_scores.extend(scores.cpu().tolist())

    scores_tensor = torch.tensor(all_scores)
    return {
        "clip_score_mean": float(scores_tensor.mean()),
        "clip_score_std": float(scores_tensor.std()),
        "clip_score_median": float(scores_tensor.median()),
        "n_images": len(all_scores),
    }


def evaluate_run(run_dir: Path, real_dir: Optional[str], device: str, skip_fid: bool = False) -> Dict:
    """Evaluate a single run directory."""
    run_dir = Path(run_dir)
    meta_path = run_dir / "run_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    images = collect_generated_images(run_dir)
    pairs = collect_prompts_from_manifest(run_dir)

    result = {
        "run_dir": str(run_dir),
        "n_images": len(images),
        "config": {
            k: meta.get(k) for k in [
                "alpha", "beta", "mode", "eta_alpha", "eta_beta",
                "run_tag", "num_inference_steps", "seed",
            ]
        },
    }

    if not skip_fid:
        print(f"\n[FID] {run_dir.name} ({len(images)} images)")
        fid_score = compute_fid(images, real_dir)
        result["fid"] = fid_score

    print(f"\n[CLIP] {run_dir.name} ({len(pairs)} pairs)")
    clip_result = compute_clip_score(pairs, device=device)
    if clip_result:
        result.update(clip_result)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default=None, help="Single run directory to evaluate.")
    parser.add_argument("--parent_dir", type=str, default=None, help="Parent dir containing multiple run dirs.")
    parser.add_argument("--real_dir", type=str, default=None, help="Real image directory for FID.")
    parser.add_argument("--gpu", type=str, default="9", help="GPU to use for CLIP. Default: 9")
    parser.add_argument("--skip_fid", action="store_true", help="Skip FID computation.")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_dirs = []
    if args.run_dir:
        run_dirs = [Path(args.run_dir)]
    elif args.parent_dir:
        parent = Path(args.parent_dir)
        run_dirs = sorted([d for d in parent.iterdir() if d.is_dir() and (d / "manifest.jsonl").exists()])
    else:
        print("Specify --run_dir or --parent_dir")
        sys.exit(1)

    if not run_dirs:
        print("No run directories found.")
        sys.exit(1)

    print(f"Found {len(run_dirs)} run(s) to evaluate.")

    results = []
    for rd in run_dirs:
        res = evaluate_run(rd, args.real_dir, device, skip_fid=args.skip_fid)
        results.append(res)

    # Print summary table
    print("\n" + "=" * 100)
    print(f"{'Run':<60} {'FID':>8} {'CLIP':>8} {'N':>5}")
    print("-" * 100)
    for r in results:
        name = Path(r["run_dir"]).name
        fid_str = f"{r.get('fid', 'N/A'):>8.2f}" if isinstance(r.get("fid"), (int, float)) else "    N/A "
        clip_str = f"{r.get('clip_score_mean', 0):>8.4f}" if r.get("clip_score_mean") else "    N/A "
        print(f"{name:<60} {fid_str} {clip_str} {r['n_images']:>5}")
    print("=" * 100)

    # Save results
    out_path = args.output or "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
