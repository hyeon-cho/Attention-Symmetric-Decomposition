"""symmetry_attn_vx_eta.py

Extends SkewAttnProcessor2_0 with η_M-guided adaptive alpha and beta.

  Functional symmetricity η_M (Tanaka & Edwards, arXiv:1810.09325):
    η_M = (‖M_sym‖² − ‖M_skew‖²) / (‖M_sym‖² + ‖M_skew‖²)

  where M(X) = X W X^T is the realized attention map, decomposed via
  coupling matrix W = S + N  (symmetric + antisymmetric).

  alpha_mode="scale":  α_eff = α · η_M
  alpha_mode="dev":    α_eff = 1 + (α − 1) · η_M   (setting used in the paper)
  β_eff = β · (1 − η_M)  — larger blending step when asymmetry is high (optional)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

import torch
import torch.nn.functional as F

from diffusers.models.attention_processor import Attention, deprecate

from attentions.symmetry_attn_vx import (
    EnergyStats,
    get_blended_pattern,
    hopfield_energy_column_patterns_xv,
)


# =========================================================================
# Functional η_M computation (Gram matrix trick, no L×L materialisation)
# =========================================================================

@torch.no_grad()
def compute_functional_eta(
    x_v: torch.Tensor,       # (B, L, C)
    S: torch.Tensor,          # (H, C, C) symmetric coupling
    N: torch.Tensor,          # (H, C, C) antisymmetric coupling
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute functional symmetricity η_M for M(X) = X W X^T.

    Returns η_M of shape (B, H).  η_M ∈ [-1, 1].
    """
    X_f = x_v.float()                                                 # (B, L, C)
    S_f = S.float()                                                    # (H, C, C)
    N_f = N.float()                                                    # (H, C, C)

    # Gram matrix G = X^T X  →  (B, C, C)
    G = torch.einsum("bli,blj->bij", X_f, X_f)

    # ‖M_sym‖² = tr(S G S G)
    SG = torch.einsum("hij,bjk->bhik", S_f, G)                        # (B, H, C, C)
    sym_sq = torch.einsum("bhij,bhji->bh", SG, SG)                    # (B, H)

    # ‖M_skew‖² = tr(N^T G N G)
    NtG = torch.einsum("hij,bjk->bhik", N_f.transpose(-2, -1), G)     # (B, H, C, C)
    NG = torch.einsum("hij,bjk->bhik", N_f, G)                        # (B, H, C, C)
    skew_sq = torch.einsum("bhij,bhji->bh", NtG, NG)                  # (B, H)

    sym_sq = torch.clamp(sym_sq, min=0.0)
    skew_sq = torch.clamp(skew_sq, min=0.0)

    eta_M = (sym_sq - skew_sq) / (sym_sq + skew_sq + eps)             # (B, H)
    return eta_M


# =========================================================================
# Diagnostics
# =========================================================================

@dataclass
class EtaStats:
    """Accumulates η_M / alpha_eff / beta_eff statistics across forward calls."""
    n: int = 0
    eta_sum: float = 0.0
    eta_min: float = float("inf")
    eta_max: float = float("-inf")
    alpha_eff_sum: float = 0.0
    beta_eff_sum: float = 0.0

    def update(self, eta: float, alpha_eff: float, beta_eff: float) -> None:
        self.n += 1
        self.eta_sum += eta
        self.eta_min = min(self.eta_min, eta)
        self.eta_max = max(self.eta_max, eta)
        self.alpha_eff_sum += alpha_eff
        self.beta_eff_sum += beta_eff

    def as_dict(self) -> Dict[str, float | int]:
        if self.n == 0:
            return {"n": 0}
        return {
            "n": self.n,
            "eta_mean": self.eta_sum / self.n,
            "eta_min": self.eta_min,
            "eta_max": self.eta_max,
            "alpha_eff_mean": self.alpha_eff_sum / self.n,
            "beta_eff_mean": self.beta_eff_sum / self.n,
        }


# =========================================================================
# Processor
# =========================================================================

class EtaSkewAttnProcessor2_0:
    r"""
    SkewAttnProcessor2_0 with η_M-guided adaptive alpha (and optionally beta).

    alpha_mode="scale":  α_eff = α · η_M
    alpha_mode="dev":    α_eff = 1 + (α − 1) · η_M   (only scales deviation from 1)
    β_eff = β · (1 − η_M)    — optional; when disabled, β is a fixed step size.
    """

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        beta: float = 5.0,
        # --- η-guided modulation ---
        eta_alpha: bool = True,
        eta_beta: bool = False,
        alpha_mode: str = "scale",  # "scale" = α·η_M, "dev" = 1+(α-1)·η_M
        # --- energy ---
        compute_energy: bool = True,
        cache_last_energy: bool = True,
        energy_store_dtype: torch.dtype = torch.float16,
        energy_center_s: bool = True,
        energy_l2norm_s: bool = True,
        energy_center_x: bool = True,
        energy_col_l2norm_x: bool = False,
        energy_remove_diag_S: bool = False,
        energy_normalize_S: Literal["none", "frob"] = "none",
        energy_include_scale_1_sqrt_d: bool = False,
    ):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("EtaSkewAttnProcessor2_0 requires PyTorch 2.0+")

        self.alpha = alpha
        self.beta = beta
        self.eta_alpha = eta_alpha
        self.eta_beta = eta_beta
        self.alpha_mode = alpha_mode

        self.compute_energy = compute_energy
        self.cache_last_energy = cache_last_energy
        self.energy_store_dtype = energy_store_dtype
        self.energy_center_s = energy_center_s
        self.energy_l2norm_s = energy_l2norm_s
        self.energy_center_x = energy_center_x
        self.energy_col_l2norm_x = energy_col_l2norm_x
        self.energy_remove_diag_S = energy_remove_diag_S
        self.energy_normalize_S = energy_normalize_S
        self.energy_include_scale_1_sqrt_d = energy_include_scale_1_sqrt_d

        self.energy_stats: Dict[str, EnergyStats] = {}
        self.last_energy_maps: Dict[str, Dict[str, torch.Tensor]] = {}
        self.eta_stats: Dict[str, EtaStats] = {}

    # ---- energy helpers (same as base) ----

    def _get_energy_bucket(self, name: str) -> EnergyStats:
        if name not in self.energy_stats:
            self.energy_stats[name] = EnergyStats()
        return self.energy_stats[name]

    def _get_eta_bucket(self, name: str) -> EtaStats:
        if name not in self.eta_stats:
            self.eta_stats[name] = EtaStats()
        return self.eta_stats[name]

    def _cache_energy(
        self, *, attn: Attention, layer_tag: str,
        E_hop_meanH: torch.Tensor, aux: Dict[str, torch.Tensor],
    ) -> None:
        self._get_energy_bucket(layer_tag).update(E_hop_meanH, aux)
        if not self.cache_last_energy:
            return
        cache = {
            "E_hop_meanH": E_hop_meanH.detach().cpu().to(self.energy_store_dtype),
            "E_mean_over_c": aux["E_mean_over_c"].detach().cpu().to(self.energy_store_dtype),
            "E_min_over_c": aux["E_min_over_c"].detach().cpu().to(self.energy_store_dtype),
            "E_p05_over_c": aux["E_p05_over_c"].detach().cpu().to(self.energy_store_dtype),
        }
        self.last_energy_maps[layer_tag] = cache
        setattr(attn, "_energy_cache", cache)

    # ---- main forward ----

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = (
                "The `scale` argument is deprecated and will be ignored. "
                "Please remove it, as passing it will raise an error in the future. "
                "`scale` should directly be passed while calling the underlying pipeline "
                "component i.e., via `cross_attention_kwargs`."
            )
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        is_cross = True

        if encoder_hidden_states is None:
            is_cross = False
            encoder_hidden_states = hidden_states
        else:
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            raise NotImplementedError("attn.norm_q is not supported yet.")
        if attn.norm_k is not None:
            raise NotImplementedError("attn.norm_k is not supported yet.")

        # ---- cross attention: vanilla SDPA ----
        if is_cross:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
            )
        else:
            # ---- self attention: η_M-guided skew ----
            x_v = encoder_hidden_states
            H = attn.heads

            Wq, Wk = attn.to_q.weight, attn.to_k.weight
            bq, bk = attn.to_q.bias, attn.to_k.bias
            Wv, bv = attn.to_v.weight, attn.to_v.bias

            if bq is not None or bk is not None:
                raise NotImplementedError("bias in to_q/to_k not supported yet.")

            d_head = Wq.shape[0] // H
            Cq, Ck = Wq.shape[1], Wk.shape[1]
            Wq_h = Wq.view(H, d_head, Cq)
            Wk_h = Wk.view(H, d_head, Ck)
            A_h = torch.einsum("hdi,hdj->hij", Wq_h, Wk_h)
            S = 0.5 * (A_h + A_h.transpose(-2, -1))
            N = 0.5 * (A_h - A_h.transpose(-2, -1))

            v = x_v.unsqueeze(1).expand(-1, H, -1, -1).contiguous()
            q = query.contiguous()
            k = key.contiguous()

            # ---- compute functional η_M ----
            eta_M = compute_functional_eta(x_v, S, N)          # (B, H)
            eta_scalar = eta_M.mean().item()

            # ---- modulate alpha and beta ----
            alpha = float(self.alpha)
            beta = float(self.beta)

            if self.eta_alpha:
                if self.alpha_mode == "dev":
                    alpha_eff = 1.0 + (alpha - 1.0) * eta_scalar
                else:  # "scale"
                    alpha_eff = alpha * eta_scalar
            else:
                alpha_eff = alpha
            beta_eff = beta * (1.0 - eta_scalar) if self.eta_beta else beta

            # ---- baseline retrieval ----
            Hx_org = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
            )

            # ---- skewed retrieval ----
            c1 = 0.5 * (1.0 + alpha_eff)
            c2 = 0.5 * (1.0 - alpha_eff)

            if abs(c2) < 1e-12:
                Hx = Hx_org
            else:
                a1 = math.sqrt(abs(c1))
                a2 = math.sqrt(abs(c2))
                s1 = 1.0 if c1 >= 0 else -1.0
                s2 = 1.0 if c2 >= 0 else -1.0

                q_tilde = torch.cat([q * a1, k * a2], dim=-1) * math.sqrt(2.0)
                k_tilde = torch.cat([k * (s1 * a1), q * (s2 * a2)], dim=-1)

                Hx = F.scaled_dot_product_attention(
                    q_tilde.contiguous(),
                    k_tilde.contiguous(),
                    v,
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                    is_causal=False,
                )

            # ---- blending ----
            Hx = get_blended_pattern(beta_eff, Hx_org, Hx)

            # ---- diagnostics ----
            meta = getattr(attn, "_attn_meta", None)
            layer_tag = (
                kwargs.get("layer_tag", None)
                or (meta["name"] if isinstance(meta, dict) and "name" in meta else None)
                or f"attn_{id(attn)}"
            )
            self._get_eta_bucket(layer_tag).update(eta_scalar, alpha_eff, beta_eff)

            # ---- energy computation (optional) ----
            if self.compute_energy:
                E_hop, aux = hopfield_energy_column_patterns_xv(
                    Hx, x_v, S,
                    head_dim=d_head,
                    include_scale_1_sqrt_d=self.energy_include_scale_1_sqrt_d,
                    center_s=self.energy_center_s,
                    l2norm_s=self.energy_l2norm_s,
                    center_x=self.energy_center_x,
                    col_l2norm_x=self.energy_col_l2norm_x,
                    remove_diag_S=self.energy_remove_diag_S,
                    normalize_S=self.energy_normalize_S,
                    return_aux=True,
                )
                self._cache_energy(attn=attn, layer_tag=layer_tag, E_hop_meanH=E_hop.mean(dim=1), aux=aux)

            # ---- value projection ----
            Cv = Wv.shape[1]
            Wv_h = Wv.view(H, d_head, Cv).float().to(dtype=Hx.dtype)
            hidden_states = torch.einsum("bhtc,hdc->bhtd", Hx, Wv_h)
            if bv is not None:
                hidden_states = hidden_states + bv.view(H, d_head)[None, :, None, :].to(hidden_states.dtype)

        # ---- output projection ----
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states
