"""eval_metrics.py

Compute per-image scores (Aesthetic, ImageReward, HPS v2, PickScore, CLIPScore)
for all experiment runs, then rank by mean score using the smallest common N.

Usage:
    python eval_metrics.py --parent_dir outputs/exp3_sdxl_skewmask/prompts_coco_1 --gpu 0
"""

import argparse
import json
import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from PIL import Image
from scipy import stats
from tqdm import tqdm

warnings.filterwarnings("ignore")


# =========================================================================
# Collectors
# =========================================================================

def collect_pairs(run_dir: Path, max_n: Optional[int] = None) -> List[Tuple[str, Path]]:
    """Read manifest.jsonl → (prompt, image_path) pairs, sorted by prompt_index."""
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
    pairs = pairs[:max_n] if max_n else pairs
    return pairs


# =========================================================================
# Scorers
# =========================================================================

class CLIPScorer:
    """CLIP score via ImageReward package (ViT-L/14)."""
    def __init__(self, device="cuda"):
        import ImageReward as RM
        self.model = RM.load_score("CLIP", device=device)
        self.device = device

    @torch.no_grad()
    def score(self, prompt: str, image_path: str) -> float:
        return self.model.score(prompt, image_path)


class AestheticScorer:
    """Aesthetic score via ImageReward package."""
    def __init__(self, device="cuda"):
        import ImageReward as RM
        self.model = RM.load_score("Aesthetic", device=device)
        self.device = device

    @torch.no_grad()
    def score(self, prompt: str, image_path: str) -> float:
        return self.model.score(prompt, image_path)


class ImageRewardScorer:
    """ImageReward-v1.0 score."""
    def __init__(self, device="cuda"):
        import ImageReward as RM
        self.model = RM.load("ImageReward-v1.0", device=device)
        self.device = device

    @torch.no_grad()
    def score(self, prompt: str, image_path: str) -> float:
        return self.model.score(prompt, image_path)


class HPSScorer:
    def __init__(self, device="cuda"):
        import hpsv2
        self.hpsv2 = hpsv2
        self.device = device

    @torch.no_grad()
    def score(self, prompt: str, image: Image.Image) -> float:
        # hpsv2.score expects file paths or PIL images
        result = self.hpsv2.score(image, prompt, hps_version="v2.1")
        if isinstance(result, (list, np.ndarray)):
            return float(result[0])
        return float(result)


class PickScorer:
    def __init__(self, device="cuda"):
        from transformers import AutoProcessor, AutoModel
        self.processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to(device)
        self.device = device

    @torch.no_grad()
    def score(self, prompt: str, image: Image.Image) -> float:
        inputs = self.processor(
            text=prompt, images=image, return_tensors="pt",
            padding=True, truncation=True,
        ).to(self.device)
        img_embs = self.model.get_image_features(pixel_values=inputs["pixel_values"])
        img_embs = img_embs / img_embs.norm(dim=-1, keepdim=True)
        txt_embs = self.model.get_text_features(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        txt_embs = txt_embs / txt_embs.norm(dim=-1, keepdim=True)
        return (img_embs * txt_embs).sum().item()


SCORER_CLASSES = {
    "CLIPScore": CLIPScorer,
    "Aesthetic": AestheticScorer,
    "ImageReward": ImageRewardScorer,
    "HPS": HPSScorer,
    "PickScore": PickScorer,
}


# =========================================================================
# Helpers: parse run config from run_meta.json or dirname
# =========================================================================

def parse_base_config(run_dir: Path) -> Dict:
    """Extract base alpha, beta, mode, eta flags from run_meta.json."""
    meta_path = run_dir / "run_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        return {
            "alpha": meta.get("alpha"),
            "beta": meta.get("beta"),
            "mode": meta.get("mode", "fixed"),
            "eta_alpha": meta.get("eta_alpha", False),
            "eta_beta": meta.get("eta_beta", False),
            "alpha_mode": meta.get("alpha_mode"),
            "run_tag": meta.get("run_tag"),
        }
    return {}


def config_key(cfg: Dict) -> str:
    """Build a (alpha, beta) key for grouping."""
    a = cfg.get("alpha")
    b = cfg.get("beta")
    if a is not None and b is not None:
        return f"a{a}_b{b}"
    return ""


def is_adaptive(cfg: Dict) -> bool:
    return cfg.get("mode") == "eta"


def config_label(cfg: Dict) -> str:
    """Short human-readable label."""
    mode = cfg.get("mode", "fixed")
    if mode == "fixed":
        tag = cfg.get("run_tag", "?")
        return f"fixed({tag})"
    parts = ["η"]
    if cfg.get("eta_alpha"):
        am = cfg.get("alpha_mode", "scale")
        parts.append(f"α-{am}")
    if cfg.get("eta_beta"):
        parts.append("β")
    return "+".join(parts)


def paired_delta_analysis(
    scores_a: List[float],
    scores_b: List[float],
    n: int,
) -> Dict:
    """Paired delta: A - B on first n samples. Returns stats + t-test."""
    a = np.array(scores_a[:n], dtype=np.float64)
    b = np.array(scores_b[:n], dtype=np.float64)
    valid = ~(np.isnan(a) | np.isnan(b))
    a, b = a[valid], b[valid]
    if len(a) < 2:
        return {"delta_mean": float("nan"), "delta_std": float("nan"),
                "t_stat": float("nan"), "p_value": float("nan"), "n_valid": 0}
    delta = a - b
    t_stat, p_val = stats.ttest_rel(a, b)
    return {
        "delta_mean": float(delta.mean()),
        "delta_std": float(delta.std()),
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "n_valid": int(len(a)),
    }


# =========================================================================
# Score cache
# =========================================================================

CACHE_DIR = Path("eval_score_cache")

def _cache_path(run_name: str) -> Path:
    return CACHE_DIR / f"{run_name}.json"


def load_cached_scores(run_name: str, n_images: int, metric_names: List[str]) -> Optional[Dict]:
    """Load cached scores if they match the current n_images exactly."""
    cp = _cache_path(run_name)
    if not cp.exists():
        return None
    cached = json.loads(cp.read_text())
    if cached.get("n_images") != n_images:
        return None
    # Check all requested metrics are present
    for m in metric_names:
        if m not in cached or "scores" not in cached[m]:
            return None
        if len(cached[m]["scores"]) != n_images:
            return None
    return cached


def save_scores_to_cache(run_name: str, result: Dict) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    cp = _cache_path(run_name)
    cp.write_text(json.dumps(result, ensure_ascii=False))


# =========================================================================
# Main
# =========================================================================

MIN_IMAGES = 50

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent_dir", type=str, required=True)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--max_n", type=int, default=None,
                        help="Evaluate first N images per run. Default: use min across runs.")
    parser.add_argument("--metrics", type=str, default="CLIPScore,Aesthetic,ImageReward,HPS,PickScore",
                        help="Comma-separated list of metrics.")
    parser.add_argument("--min_images", type=int, default=MIN_IMAGES,
                        help="Skip runs with fewer images. Default: 50")
    parser.add_argument("--output", type=str, default="eval_metrics_results.json")
    parser.add_argument("--no_cache", action="store_true", help="Ignore cached scores.")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parent = Path(args.parent_dir)
    run_dirs = sorted([
        d for d in parent.iterdir()
        if d.is_dir() and (d / "manifest.jsonl").exists()
    ])

    if not run_dirs:
        print("No run directories found.")
        return

    # Collect ALL pairs per run, filter by min_images
    all_pairs = {}
    all_configs = {}
    for rd in run_dirs:
        pairs = collect_pairs(rd)
        cfg = parse_base_config(rd)
        if len(pairs) < args.min_images:
            print(f"  [SKIP] {rd.name}: {len(pairs)} images (< {args.min_images})")
            continue
        all_pairs[rd.name] = pairs
        all_configs[rd.name] = cfg
        print(f"  {rd.name}: {len(pairs)} images | {config_label(cfg)} | base α={cfg.get('alpha')}, β={cfg.get('beta')}")

    if not all_pairs:
        print("No runs with enough images.")
        return

    min_n = min(len(p) for p in all_pairs.values())
    print(f"\nIncluded runs: {len(all_pairs)}, min images: {min_n}")

    # Load scorers
    metric_names = [m.strip() for m in args.metrics.split(",")]

    # Check which runs need scoring (not cached or image count changed)
    needs_scoring = []
    results_full = {}
    for run_name, pairs in all_pairs.items():
        n = len(pairs)
        if not args.no_cache:
            cached = load_cached_scores(run_name, n, metric_names)
            if cached is not None:
                print(f"  [CACHE HIT] {run_name}: {n} images (skipping scoring)")
                results_full[run_name] = cached
                continue
        needs_scoring.append(run_name)

    # Only load scorers if there are runs to score
    scorers = {}
    if needs_scoring:
        for name in metric_names:
            if name not in SCORER_CLASSES:
                print(f"[WARN] Unknown metric: {name}, skipping.")
                continue
            print(f"Loading {name}...")
            try:
                scorers[name] = SCORER_CLASSES[name](device=device)
            except Exception as e:
                print(f"[WARN] Failed to load {name}: {e}")

        if not scorers:
            print("No scorers loaded.")
            return

        RM_SCORERS = {"CLIPScore", "Aesthetic", "ImageReward"}

        for run_name in needs_scoring:
            pairs = all_pairs[run_name]
            print(f"\n{'='*60}")
            print(f"Scoring: {run_name} ({len(pairs)} images)")
            print(f"{'='*60}")

            run_scores = {m: [] for m in scorers}
            for prompt, img_path in tqdm(pairs, desc=run_name):
                img = None
                for metric_name, scorer in scorers.items():
                    try:
                        if metric_name in RM_SCORERS:
                            s = scorer.score(prompt, str(img_path))
                        else:
                            if img is None:
                                img = Image.open(img_path).convert("RGB")
                            s = scorer.score(prompt, img)
                        run_scores[metric_name].append(s)
                    except Exception as e:
                        print(f"  [ERR] {metric_name} on {img_path.name}: {e}")
                        run_scores[metric_name].append(float("nan"))

            result = {
                m: {
                    "mean": float(np.nanmean(run_scores[m])),
                    "std": float(np.nanstd(run_scores[m])),
                    "scores": run_scores[m],
                }
                for m in run_scores
            }
            result["n_images"] = len(pairs)
            results_full[run_name] = result
            save_scores_to_cache(run_name, result)
            print(f"  [CACHE SAVED] {run_name}")
    else:
        print("\nAll runs served from cache — no scoring needed.")

    # Ensure metric_list from actual data
    sample_run = next(iter(results_full.values()))
    metric_list = [m for m in metric_names if m in sample_run]
    run_names = [r for r in all_pairs.keys() if r in results_full]

    # =====================================================================
    # (1) Per-folder scores (full length, each run uses all its images)
    # =====================================================================
    print(f"\n\n{'='*140}")
    print(f"(1) PER-FOLDER SCORES (each run scored on ALL its images, runs < {args.min_images} excluded)")
    print(f"{'='*140}")

    header = f"{'Run':<55} {'N':>4}"
    for m in metric_list:
        header += f" {m:>14}"
    print(header)
    print("-" * 140)

    for run_name in run_names:
        n = results_full[run_name]["n_images"]
        row = f"{run_name:<55} {n:>4}"
        for m in metric_list:
            mean = results_full[run_name][m]["mean"]
            std = results_full[run_name][m]["std"]
            row += f" {mean:>7.4f}±{std:<5.3f}"
        print(row)
    print("-" * 140)

    # =====================================================================
    # (2) Ranking based on first min_n images (matched length)
    # =====================================================================
    results_matched = {}
    for run_name, data in results_full.items():
        results_matched[run_name] = {}
        for m in metric_list:
            scores_trimmed = data[m]["scores"][:min_n]
            results_matched[run_name][m] = {
                "mean": float(np.nanmean(scores_trimmed)),
                "std": float(np.nanstd(scores_trimmed)),
            }

    print(f"\n\n{'='*140}")
    print(f"(2) RANKING (first {min_n} images per run, matched length)")
    print(f"{'='*140}")

    header = f"{'Run':<55}"
    for m in metric_list:
        header += f" {m:>16}"
    header += f" {'AvgRank':>8}"
    print(header)
    print("-" * 140)

    ranks = {m: {} for m in metric_list}
    for m in metric_list:
        sorted_runs = sorted(run_names, key=lambda r: results_matched[r][m]["mean"], reverse=True)
        for rank, r in enumerate(sorted_runs, 1):
            ranks[m][r] = rank

    avg_ranks = {r: np.mean([ranks[m][r] for m in metric_list]) for r in run_names}

    for run_name in sorted(run_names, key=lambda r: avg_ranks[r]):
        row = f"{run_name:<55}"
        for m in metric_list:
            mean = results_matched[run_name][m]["mean"]
            rank = ranks[m][run_name]
            row += f" {mean:>9.4f} (#{rank:<2})"
        row += f" {avg_ranks[run_name]:>8.1f}"
        print(row)
    print("-" * 140)

    # =====================================================================
    # (3) Paired delta: every pair of runs, using per-pair max common N
    # =====================================================================
    sorted_names = sorted(run_names, key=lambda r: avg_ranks[r])

    print(f"\n\n{'='*140}")
    print(f"(3) PAIRED DELTA (per-pair N = min(len_A, len_B), paired t-test)")
    print(f"    + means A is better, - means B is better, * p<0.05, ** p<0.01")
    print(f"{'='*140}")

    for i, name_a in enumerate(sorted_names):
        for name_b in sorted_names[i + 1:]:
            n_a = results_full[name_a]["n_images"]
            n_b = results_full[name_b]["n_images"]
            pair_n = min(n_a, n_b)
            print(f"\n  {name_a}  vs  {name_b}  (N={pair_n})")
            for m in metric_list:
                sa = results_full[name_a][m]["scores"]
                sb = results_full[name_b][m]["scores"]
                res = paired_delta_analysis(sa, sb, pair_n)
                sig = ""
                if res["p_value"] < 0.01:
                    sig = "**"
                elif res["p_value"] < 0.05:
                    sig = "* "
                print(f"    {m:<14} Δ={res['delta_mean']:>+8.4f} ±{res['delta_std']:<7.4f}  "
                      f"t={res['t_stat']:>+6.2f}  p={res['p_value']:<7.4f} {sig}  (n={res['n_valid']})")

    # =====================================================================
    # (4) Adaptive vs Static: group by (base_alpha, base_beta), per-pair N
    # =====================================================================
    print(f"\n\n{'='*140}")
    print(f"(4) ADAPTIVE vs STATIC — grouped by same base (α, β)")
    print(f"    Paired delta: adaptive - static, N = min(len_adaptive, len_static)")
    print(f"{'='*140}")

    groups: Dict[str, Dict] = {}
    for run_name in run_names:
        cfg = all_configs[run_name]
        key = config_key(cfg)
        if not key:
            continue
        if key not in groups:
            groups[key] = {"static": None, "adaptive": []}
        if is_adaptive(cfg):
            groups[key]["adaptive"].append(run_name)
        else:
            if groups[key]["static"] is None:
                groups[key]["static"] = run_name

    for key, grp in sorted(groups.items()):
        static_name = grp["static"]
        adaptive_names = grp["adaptive"]
        if not static_name or not adaptive_names:
            continue

        static_cfg = all_configs[static_name]
        n_static = results_full[static_name]["n_images"]
        print(f"\n  Base config: α={static_cfg.get('alpha')}, β={static_cfg.get('beta')}")
        print(f"  Static:   {static_name}  (N={n_static})")

        for adap_name in adaptive_names:
            adap_cfg = all_configs[adap_name]
            label = config_label(adap_cfg)
            n_adap = results_full[adap_name]["n_images"]
            pair_n = min(n_static, n_adap)
            print(f"\n  Adaptive: {adap_name}  [{label}]  (N={pair_n})")
            print(f"  {'Metric':<14} {'Δ(adap-stat)':>12} {'±std':>10} {'t':>8} {'p':>8} {'sig':>4} {'n':>5} {'adap_mean':>10} {'stat_mean':>10}")
            print(f"  {'-'*90}")

            wins = 0
            for m in metric_list:
                sa = results_full[adap_name][m]["scores"]
                sb = results_full[static_name][m]["scores"]
                res = paired_delta_analysis(sa, sb, pair_n)
                sig = ""
                if res["p_value"] < 0.01:
                    sig = "**"
                elif res["p_value"] < 0.05:
                    sig = "* "
                # Compute means over pair_n for display
                adap_mean = float(np.nanmean(sa[:pair_n]))
                stat_mean = float(np.nanmean(sb[:pair_n]))
                if res["delta_mean"] > 0:
                    wins += 1
                print(f"  {m:<14} {res['delta_mean']:>+12.4f} {res['delta_std']:>10.4f} "
                      f"{res['t_stat']:>+8.2f} {res['p_value']:>8.4f} {sig:>4} "
                      f"{res['n_valid']:>5} {adap_mean:>10.4f} {stat_mean:>10.4f}")

            print(f"  => Adaptive wins on {wins}/{len(metric_list)} metrics")

    # =====================================================================
    # Save results
    # =====================================================================
    save_data = {
        "min_n": min_n,
        "min_images_threshold": args.min_images,
        "per_folder": {},
        "matched_ranking": {},
    }
    for run_name in run_names:
        save_data["per_folder"][run_name] = {
            "n_images": results_full[run_name]["n_images"],
            "config": all_configs.get(run_name, {}),
            **{m: {"mean": results_full[run_name][m]["mean"],
                    "std": results_full[run_name][m]["std"]}
               for m in metric_list},
        }
        save_data["matched_ranking"][run_name] = {
            "n_images": min_n,
            "avg_rank": float(avg_ranks[run_name]),
            **{m: {"mean": results_matched[run_name][m]["mean"],
                    "std": results_matched[run_name][m]["std"],
                    "rank": ranks[m][run_name]}
               for m in metric_list},
        }

    with open(args.output, "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
