"""Sequential Hopfield-style stability logger.

Records, for every attention retrieval call, the three diagnostics used in the
paper's analysis together with optional M(X) = X W X^T symmetry statistics:

  - E_map / E_scalar                  Hopfield energy (per channel / mean)
  - lambda_neg_frac_*                 instability fraction  r_X = mean[ lambda < 0 ]
  - align_cos_map                     alignment score       cos(s, h_tok)
  - M_eta_* (optional, needs N)       functional symmetry index eta_M and related ratios

Usage:
    hopseq_start(save_path=..., calls_per_step=..., run_tag=...)   # once, before sampling
    hopseq_log_after_Hx(Hx, x_v, S, layer_tag, N=...)              # after each retrieval
    hopseq_finalize()                                              # once, saves one .pth

Records are appended in RAM as a flat list; timesteps are not stored explicitly.
Instead each record carries `record_index`, and `step_index = record_index //
calls_per_step` can be recovered afterwards (assumes a constant number of logged
calls per sampling step).
"""

import math
import os
import time
from typing import Any, Dict, Literal, Optional, Tuple

import torch


@torch.no_grad()
def hopseq_start(
    *,
    save_path: str,
    calls_per_step: int,
    run_tag: str,
    store_dtype: torch.dtype = torch.float16,
    reset: bool = True,
) -> None:
    """Initialize the logger. Call once before sampling starts.

    `calls_per_step` must equal the number of logged attention calls per
    sampling step (e.g. 70 when logging all SDXL UNet self-attn modules), so
    that `step_index` can be recovered later. With `reset=True` the record
    cache is started fresh.
    """
    global _HOPSEQ_STATE
    if reset:
        _HOPSEQ_STATE = {
            "initialized": True,
            "save_path": save_path,
            "run_tag": run_tag,
            "calls_per_step": int(calls_per_step),
            "record_index": 0,
            "store_dtype": store_dtype,
            "cache": {
                "meta": {
                    "created_at_unix": time.time(),
                    "torch_version": torch.__version__,
                    "run_tag": run_tag,
                    "calls_per_step": int(calls_per_step),
                },
                "records": [],
            },
        }
    else:
        _HOPSEQ_STATE["initialized"] = True
        _HOPSEQ_STATE["save_path"] = save_path
        _HOPSEQ_STATE["run_tag"] = run_tag
        _HOPSEQ_STATE["calls_per_step"] = int(calls_per_step)
        _HOPSEQ_STATE["store_dtype"] = store_dtype

@torch.no_grad()
def hopseq_finalize() -> None:
    """Save all accumulated records to `save_path` as one .pth (single IO)."""
    global _HOPSEQ_STATE
    if not _HOPSEQ_STATE.get("initialized", False):
        raise RuntimeError("hopseq_start(...) must be called before hopseq_finalize().")

    save_path = _HOPSEQ_STATE["save_path"]
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(_HOPSEQ_STATE["cache"], save_path)


# -----------------------------------------------------------------------------
# NOTE
# -----------------------------------------------------------------------------
# In addition to the energy/stability diagnostics, hopseq_log_after_Hx can
# measure the *input-conditioned* token-space coupling
#     M(X) = X W X^T
# via the symmetric/skew parts:
#     M_sym(X)  = X S X^T
#     M_skew(X) = X N X^T
# without explicitly materializing the L x L matrix.
#
# If G = X^T X, then for each sample b and head h:
#   ||M_sym||_F^2  = tr(S G S G)
#   ||M_skew||_F^2 = tr(N^T G N G)
# and therefore
#   eta_M = (||M_sym||_F^2 - ||M_skew||_F^2) / (||M_sym||_F^2 + ||M_skew||_F^2)
# -----------------------------------------------------------------------------

# Module-level logger state; hopseq_start() (re)initializes it.
_HOPSEQ_STATE: Dict[str, Any] = {
    "initialized": False,
    "record_index": 0,
    "calls_per_step": 1,
    "run_tag": "",
    "store_dtype": torch.float16,
    "cache": {"records": []},
}


@torch.no_grad()
def hopseq_log_after_Hx(
    Hx: torch.Tensor,           # (B,H,L,C)  retrieved content: Hx = P @ x_v
    x_v: torch.Tensor,          # (B,L,C)    memory bank X
    S: torch.Tensor,            # (H,C,C)    symmetric coupling per head
    layer_tag: str,             # stable id, e.g. _attn_meta['name']
    *,
    N: Optional[torch.Tensor] = None,  # (H,C,C) skew coupling per head; optional for backward compatibility
    # --- preprocessing options (same convention as the energy helper) ---
    center_s: bool = True,
    l2norm_s: bool = True,
    center_x: bool = True,
    col_l2norm_x: bool = False,
    remove_diag_S: bool = False,
    normalize_S: Literal["none", "frob"] = "none",
    include_scale_1_sqrt_d: bool = False,
    head_dim: Optional[int] = None,
    eps: float = 1e-8,
    # --- M(X)=XWX^T stats ---
    log_M_stats: bool = True,
    remove_diag_M: bool = False,
    # --- optional extra metadata ---
    extra: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float, Dict[str, Any]]:
    """Log one retrieval call; returns (E_scalar, lambda_neg_frac_scalar, record).

    What is recorded per call:
      - E_map: (B,C) head-averaged Hopfield energy map
      - lambda_neg_frac_map: (B,C) head-averaged fraction of tokens with
        lambda_t = s_t * h_t < 0 (instability fraction)
      - align_cos_map: (B,C) head-averaged cosine(s, h_tok)
      - the global scalar summaries:
          E_scalar = mean(E_map)
          lambda_neg_frac_scalar = mean(lambda_neg_frac_map)

      - Optional M(X) stats when N is given:
          M_sym_sq       = ||X S X^T||_F^2
          M_skew_sq      = ||X N X^T||_F^2
          M_skew_to_sym  = ||M_skew||_F / ||M_sym||_F
          M_eta          = (M_sym_sq - M_skew_sq) / (M_sym_sq + M_skew_sq)
          M_sym_balance  = (||M_sym||_F - ||M_skew||_F) / (||M_sym||_F + ||M_skew||_F)

    Notes:
      - M stats are computed in token space but *without* materializing the LxL matrix.
      - They depend on the preprocessing applied to X (center_x / col_l2norm_x).
      - step_index recovery relies on `calls_per_step` being exact, i.e. the number
        of logged attention calls per sampling step must be constant.
    """
    global _HOPSEQ_STATE
    if not _HOPSEQ_STATE.get("initialized", False):
        raise RuntimeError("Call hopseq_start(save_path=..., calls_per_step=..., run_tag=...) before logging.")

    # -------------------------
    # 0) shape checks
    # -------------------------
    if Hx.ndim != 4:
        raise ValueError(f"Hx must be (B,H,L,C). Got {tuple(Hx.shape)}")
    if x_v.ndim != 3:
        raise ValueError(f"x_v must be (B,L,C). Got {tuple(x_v.shape)}")
    if S.ndim != 3:
        raise ValueError(f"S must be (H,C,C). Got {tuple(S.shape)}")

    B, Hh, L, C = Hx.shape
    B2, L2, C2 = x_v.shape
    Hs, C3, C4 = S.shape
    if (B2, L2, C2) != (B, L, C):
        raise ValueError(f"Shape mismatch: Hx {tuple(Hx.shape)} vs x_v {tuple(x_v.shape)}")
    if (Hs, C3, C4) != (Hh, C, C):
        raise ValueError(f"Shape mismatch: Hx {tuple(Hx.shape)} vs S {tuple(S.shape)}")

    if N is not None:
        if N.ndim != 3:
            raise ValueError(f"N must be (H,C,C). Got {tuple(N.shape)}")
        Hn, Cn1, Cn2 = N.shape
        if (Hn, Cn1, Cn2) != (Hh, C, C):
            raise ValueError(f"Shape mismatch: Hx {tuple(Hx.shape)} vs N {tuple(N.shape)}")

    # -------------------------
    # 1) record_index -> (step_index, in_step_index)
    # -------------------------
    ridx = int(_HOPSEQ_STATE["record_index"])
    cps = int(_HOPSEQ_STATE["calls_per_step"])
    step_index = ridx // cps
    in_step_index = ridx % cps
    _HOPSEQ_STATE["record_index"] = ridx + 1

    # -------------------------
    # 2) compute in float32 for numerical stability
    # -------------------------
    Hx_f = Hx.float()
    X_f  = x_v.float()
    S_f  = S.float()
    N_f  = N.float() if N is not None else None

    # -------------------------
    # 3) preprocess s (= per-channel spatial pattern of Hx)
    # -------------------------
    if center_s:
        # remove token(L)-mean per (B,H,C)
        Hx_f = Hx_f - Hx_f.mean(dim=2, keepdim=True)

    if l2norm_s:
        # normalize so that ||s||_2 = 1 for each (b,h,c)
        s_norm = torch.linalg.vector_norm(Hx_f, ord=2, dim=2, keepdim=True).clamp_min(eps)
        Hx_f = Hx_f / s_norm

    # -------------------------
    # 4) preprocess X (= x_v)
    # -------------------------
    if center_x:
        X_f = X_f - X_f.mean(dim=1, keepdim=True)

    if col_l2norm_x:
        # normalize so that ||X[:,c]||_2 = 1 for each (b,c)
        x_col_norm = torch.linalg.vector_norm(X_f, ord=2, dim=1, keepdim=True).clamp_min(eps)
        X_f = X_f / x_col_norm

    # -------------------------
    # 5) preprocess S
    # -------------------------
    if remove_diag_S:
        diag = torch.eye(C, device=S_f.device, dtype=torch.bool).unsqueeze(0)  # (1,C,C)
        S_f = S_f.masked_fill(diag, 0.0)

    if normalize_S == "frob":
        Sn = torch.linalg.vector_norm(S_f.reshape(Hh, -1), ord=2, dim=1).clamp_min(eps)  # (H,)
        S_f = S_f / Sn[:, None, None]
        if N_f is not None:
            N_f = N_f / Sn[:, None, None]

    # -------------------------
    # 6) z = X^T s  (avoids forming the LxL matrix)
    #    z: (B,H,C,C)  [i: memory channel, c: pattern channel]
    # -------------------------
    z  = torch.einsum("bli,bhlc->bhic", X_f, Hx_f)     # (B,H,C,C)
    Sz = torch.einsum("hij,bhjc->bhic", S_f, z)        # (B,H,C,C)

    # -------------------------
    # 7) energy
    # -------------------------
    quad = (z * Sz).sum(dim=2)                         # (B,H,C)
    E_hop = -0.5 * quad                                # (B,H,C)

    if include_scale_1_sqrt_d:
        if head_dim is None:
            raise ValueError("head_dim is required when include_scale_1_sqrt_d=True")
        E_hop = E_hop * (1.0 / math.sqrt(float(head_dim)))

    E_map = E_hop.mean(dim=1)                          # (B,C)
    E_scalar = float(E_map.mean().detach().cpu())

    # -------------------------
    # 8) local field & lambda / alignment
    #    h_tok = X (S z)  => (B,H,L,C)
    # -------------------------
    h_tok = torch.einsum("bli,bhic->bhlc", X_f, Sz)   # (B,H,L,C)

    # lambda = s * h_tok
    lam = Hx_f * h_tok                                 # (B,H,L,C)

    lam_neg_frac = (lam < 0).float().mean(dim=2)       # (B,H,C)
    lam_neg_frac_map = lam_neg_frac.mean(dim=1)        # (B,C)
    lam_neg_frac_scalar = float(lam_neg_frac_map.mean().detach().cpu())

    # cosine alignment cos(s, h_tok)
    dot = lam.sum(dim=2)                               # (B,H,C)  (s·h_tok)
    h_norm = torch.linalg.vector_norm(h_tok, ord=2, dim=2).clamp_min(eps)   # (B,H,C)
    s_norm2 = torch.linalg.vector_norm(Hx_f, ord=2, dim=2).clamp_min(eps)   # (B,H,C)
    align_cos = dot / (s_norm2 * h_norm)               # (B,H,C)
    align_cos_map = align_cos.mean(dim=1)              # (B,C)

    # -------------------------
    # 9) optional M(X)=XWX^T stats (without materializing LxL)
    # -------------------------
    M_eta_scalar = None
    M_skew_to_sym_scalar = None
    M_symmetry_balance_scalar = None
    M_eta_headmean = None
    M_skew_to_sym_headmean = None
    M_symmetry_balance_headmean = None

    if log_M_stats and (N_f is not None):
        # G_b = X_b^T X_b  -> (B,C,C)
        G = torch.einsum("bli,blj->bij", X_f, X_f)

        # ||M_sym||_F^2 = tr(S G S G)
        SG = torch.einsum("hij,bjk->bhik", S_f, G)             # (B,H,C,C) = S G
        sym_sq = torch.einsum("bhij,bhji->bh", SG, SG)         # tr((SG)(SG))

        # ||M_skew||_F^2 = tr(N^T G N G)
        NtG = torch.einsum("hij,bjk->bhik", N_f.transpose(-2, -1), G)  # N^T G
        NG  = torch.einsum("hij,bjk->bhik", N_f, G)                    # N G
        skew_sq = torch.einsum("bhij,bhji->bh", NtG, NG)

        # numerical guard
        sym_sq = torch.clamp(sym_sq, min=0.0)
        skew_sq = torch.clamp(skew_sq, min=0.0)

        if remove_diag_M:
            # Optional exact diag removal in token space (costly, only if explicitly requested).
            # Materialize only in this branch.
            M_sym = torch.einsum("bli,hij,bmj->bhlm", X_f, S_f, X_f)
            M_skew = torch.einsum("bli,hij,bmj->bhlm", X_f, N_f, X_f)
            eye = torch.eye(L, device=M_sym.device, dtype=torch.bool)[None, None, :, :]
            M_sym = M_sym.masked_fill(eye, 0.0)
            M_skew = M_skew.masked_fill(eye, 0.0)
            sym_sq = (M_sym * M_sym).sum(dim=(-2, -1))
            skew_sq = (M_skew * M_skew).sum(dim=(-2, -1))

        sym_norm = torch.sqrt(sym_sq.clamp_min(eps))
        skew_norm = torch.sqrt(skew_sq.clamp_min(eps))

        M_skew_to_sym = skew_norm / (sym_norm + eps)                     # (B,H)
        M_eta = (sym_sq - skew_sq) / (sym_sq + skew_sq + eps)           # (B,H)
        M_symmetry_balance = (sym_norm - skew_norm) / (sym_norm + skew_norm + eps)

        # scalars: mean over batch and head
        M_eta_scalar = float(M_eta.mean().detach().cpu())
        M_skew_to_sym_scalar = float(M_skew_to_sym.mean().detach().cpu())
        M_symmetry_balance_scalar = float(M_symmetry_balance.mean().detach().cpu())

        # head means: mean over batch, keep head axis for later layer/head analysis
        dtype = _HOPSEQ_STATE["store_dtype"]
        M_eta_headmean = M_eta.mean(dim=0).detach().cpu().to(dtype)
        M_skew_to_sym_headmean = M_skew_to_sym.mean(dim=0).detach().cpu().to(dtype)
        M_symmetry_balance_headmean = M_symmetry_balance.mean(dim=0).detach().cpu().to(dtype)

    # -------------------------
    # 10) append record (RAM only; single IO happens in hopseq_finalize)
    # -------------------------
    dtype = _HOPSEQ_STATE["store_dtype"]
    rec = {
        "record_index": ridx,
        "step_index": int(step_index),
        "in_step_index": int(in_step_index),
        "layer_tag": layer_tag,
        "run_tag": _HOPSEQ_STATE["run_tag"],
        "E_scalar": E_scalar,
        "lambda_neg_frac_scalar": lam_neg_frac_scalar,
        # maps stored in `store_dtype` (fp16 recommended)
        "E_map": E_map.detach().cpu().to(dtype),
        "lambda_neg_frac_map": lam_neg_frac_map.detach().cpu().to(dtype),
        "align_cos_map": align_cos_map.detach().cpu().to(dtype),
        # M(X) scalar summaries
        "M_eta_scalar": M_eta_scalar,
        "M_skew_to_sym_scalar": M_skew_to_sym_scalar,
        "M_symmetry_balance_scalar": M_symmetry_balance_scalar,
        # head summaries (optional; None if N was not provided)
        "M_eta_headmean": M_eta_headmean,
        "M_skew_to_sym_headmean": M_skew_to_sym_headmean,
        "M_symmetry_balance_headmean": M_symmetry_balance_headmean,
        "extra": extra or {},
    }
    _HOPSEQ_STATE["cache"]["records"].append(rec)

    return E_scalar, lam_neg_frac_scalar, rec
