from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Literal, Optional

import torch
import torch.nn.functional as F

from diffusers.models.attention_processor import Attention, deprecate


def get_blended_pattern(beta: float, Hx_org: torch.Tensor, Hx: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    r_min, r_max = 0.25, 4.0

    Hx_new = Hx_org + beta * (Hx - Hx_org)

    Hx_f = Hx.float()
    Hx_new_f = Hx_new.float()
    Hx_new_f = torch.nan_to_num(Hx_new_f, nan=0.0, posinf=0.0, neginf=0.0)

    ref = torch.linalg.vector_norm(Hx_f, dim=-1, keepdim=True).clamp_min(eps)
    cur = torch.linalg.vector_norm(Hx_new_f, dim=-1, keepdim=True).clamp_min(eps)
    ratio = (ref / cur).clamp(r_min, r_max)

    return (Hx_new_f * ratio).to(dtype=Hx.dtype)


@torch.no_grad()
def hopfield_energy_column_patterns_xv(
    Hx: torch.Tensor,
    x_v: torch.Tensor,
    S: torch.Tensor,
    *,
    head_dim: Optional[int] = None,
    include_scale_1_sqrt_d: bool = False,
    center_s: bool = True,
    l2norm_s: bool = True,
    center_x: bool = True,
    col_l2norm_x: bool = False,
    remove_diag_S: bool = False,
    normalize_S: Literal["none", "frob"] = "none",
    eps: float = 1e-8,
    return_aux: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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

    Hx_f = Hx.float()
    X_f = x_v.float()
    S_f = S.float()

    if center_s:
        Hx_f = Hx_f - Hx_f.mean(dim=2, keepdim=True)
    if l2norm_s:
        s_norm = torch.linalg.vector_norm(Hx_f, ord=2, dim=2, keepdim=True).clamp_min(eps)
        Hx_f = Hx_f / s_norm

    if center_x:
        X_f = X_f - X_f.mean(dim=1, keepdim=True)
    if col_l2norm_x:
        x_col_norm = torch.linalg.vector_norm(X_f, ord=2, dim=1, keepdim=True).clamp_min(eps)
        X_f = X_f / x_col_norm

    if remove_diag_S:
        diag = torch.eye(C, device=S_f.device, dtype=torch.bool).unsqueeze(0)
        S_f = S_f.masked_fill(diag, 0.0)
    if normalize_S == "frob":
        Sn = torch.linalg.vector_norm(S_f.reshape(Hh, -1), ord=2, dim=1).clamp_min(eps)
        S_f = S_f / Sn[:, None, None]

    z = torch.einsum("bli,bhlc->bhic", X_f, Hx_f)
    Sz = torch.einsum("hij,bhjc->bhic", S_f, z)
    quad = (z * Sz).sum(dim=2)
    E = -0.5 * quad

    if include_scale_1_sqrt_d:
        if head_dim is None:
            raise ValueError("head_dim must be provided when include_scale_1_sqrt_d=True")
        E = E * (1.0 / math.sqrt(float(head_dim)))

    if not return_aux:
        return E

    aux = {
        "E_mean_over_c": E.mean(dim=-1),
        "E_min_over_c": E.min(dim=-1).values,
        "E_p05_over_c": E.kthvalue(k=max(1, int(0.05 * C)), dim=-1).values,
    }
    return E, aux


@dataclass
class EnergyStats:
    n: int = 0
    energy_mean_sum: float = 0.0
    energy_min_sum: float = 0.0
    energy_p05_sum: float = 0.0
    head_energy_mean_sum: float = 0.0
    head_energy_min_sum: float = 0.0
    head_energy_p05_sum: float = 0.0
    last_batch: int = 0
    last_channels: int = 0

    def update(self, E_hop_meanH: torch.Tensor, aux: Dict[str, torch.Tensor]) -> None:
        E_map = E_hop_meanH.float()
        p05_k = max(1, int(0.05 * E_map.shape[-1]))

        self.n += 1
        self.energy_mean_sum += float(E_map.mean().item())
        self.energy_min_sum += float(E_map.min().item())
        self.energy_p05_sum += float(E_map.kthvalue(k=p05_k, dim=-1).values.mean().item())
        self.head_energy_mean_sum += float(aux["E_mean_over_c"].float().mean().item())
        self.head_energy_min_sum += float(aux["E_min_over_c"].float().mean().item())
        self.head_energy_p05_sum += float(aux["E_p05_over_c"].float().mean().item())
        self.last_batch = int(E_map.shape[0])
        self.last_channels = int(E_map.shape[1])

    def as_dict(self) -> Dict[str, float | int]:
        if self.n == 0:
            return {"n": 0}
        return {
            "n": self.n,
            "energy_mean": self.energy_mean_sum / self.n,
            "energy_min": self.energy_min_sum / self.n,
            "energy_p05": self.energy_p05_sum / self.n,
            "head_energy_mean": self.head_energy_mean_sum / self.n,
            "head_energy_min": self.head_energy_min_sum / self.n,
            "head_energy_p05": self.head_energy_p05_sum / self.n,
            "last_batch": self.last_batch,
            "last_channels": self.last_channels,
        }


class SkewAttnProcessor2_0:
    r"""SDXL self-attention processor with skew-symmetric circulation control.

    Replaces the retrieval logits QK^T = M_sym + M_skew with M_sym + alpha * M_skew
    (implemented as c1*QK^T + c2*KQ^T via an SDPA reparameterization), then blends the
    perturbed retrieval back into the baseline with strength beta and a bounded
    per-token norm rescale. alpha = 1 recovers standard attention exactly.
    Cross-attention calls fall through to vanilla SDPA.

    When `compute_energy=True`, per-layer Hopfield energy statistics are accumulated
    in memory (`self.stats`, `EnergyStats`); nothing is written to disk.
    """

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        top_p: float = 100.0,
        rho: float = 0.0,
        beta: float = 5.0,
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
            raise ImportError(
                "SkewAttnProcessor2_0 requires PyTorch 2.0, "
                "to use it, please upgrade PyTorch to 2.0."
            )
        self.alpha = alpha
        self.top_p = top_p
        self.rho = rho
        self.beta = beta

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

        self.stats: Dict[str, EnergyStats] = {}
        self.last_energy_maps: Dict[str, Dict[str, torch.Tensor]] = {}

    def _get_bucket(self, name: str) -> EnergyStats:
        if name not in self.stats:
            self.stats[name] = EnergyStats()
        return self.stats[name]

    def _cache_energy(
        self,
        *,
        attn: Attention,
        layer_tag: str,
        E_hop_meanH: torch.Tensor,
        aux: Dict[str, torch.Tensor],
    ) -> None:
        self._get_bucket(layer_tag).update(E_hop_meanH, aux)

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
            raise NotImplementedError("attn.norm_q is not supported in SkewAttnProcessor2_0 yet.")
        if attn.norm_k is not None:
            raise NotImplementedError("attn.norm_k is not supported in SkewAttnProcessor2_0 yet.")

        if is_cross:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        else:
            x_v = encoder_hidden_states

            H = attn.heads
            Wq, Wk, bq, bk = (
                attn.to_q.weight,
                attn.to_k.weight,
                attn.to_q.bias,
                attn.to_k.bias,
            )
            Wv, bv = attn.to_v.weight, attn.to_v.bias

            if bq is not None:
                raise NotImplementedError(
                    "bias in to_q/to_k not supported in this SDPA self-attn block yet."
                )
            if bk is not None:
                raise NotImplementedError(
                    "bias in to_q/to_k not supported in this SDPA self-attn block yet."
                )

            d_head = Wq.shape[0] // H
            Cq, Ck = Wq.shape[1], Wk.shape[1]
            Wq_h = Wq.view(H, d_head, Cq)
            Wk_h = Wk.view(H, d_head, Ck)
            A_h = torch.einsum("hdi,hdj->hij", Wq_h, Wk_h)
            S = 0.5 * (A_h + A_h.transpose(-2, -1))

            v = x_v.unsqueeze(1).expand(-1, H, -1, -1).contiguous()
            q = query.contiguous()
            k = key.contiguous()

            Hx_org = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )

            alpha = float(self.alpha)
            c1 = 0.5 * (1.0 + alpha)
            c2 = 0.5 * (1.0 - alpha)

            if abs(c2) < 1e-12:
                Hx = Hx_org
            else:
                a1 = math.sqrt(abs(c1))
                a2 = math.sqrt(abs(c2))
                s1 = 1.0 if c1 >= 0 else -1.0
                s2 = 1.0 if c2 >= 0 else -1.0

                q_tilde = torch.cat([q * a1, k * a2], dim=-1)
                k_tilde = torch.cat([k * (s1 * a1), q * (s2 * a2)], dim=-1)
                q_tilde = q_tilde * math.sqrt(2.0)

                Hx = F.scaled_dot_product_attention(
                    q_tilde.contiguous(),
                    k_tilde.contiguous(),
                    v,
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                    is_causal=False,
                )

            Hx = get_blended_pattern(self.beta, Hx_org, Hx)

            if self.compute_energy:
                meta = getattr(attn, "_attn_meta", None)
                layer_tag = (
                    kwargs.get("layer_tag", None)
                    or (meta["name"] if isinstance(meta, dict) and "name" in meta else None)
                    or f"attn_{id(attn)}"
                )
                E_hop, aux = hopfield_energy_column_patterns_xv(
                    Hx,
                    x_v,
                    S,
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

            Cv = Wv.shape[1]
            Wv_h = Wv.view(H, d_head, Cv).float().to(dtype=Hx.dtype)
            hidden_states = torch.einsum("bhtc,hdc->bhtd", Hx, Wv_h)
            if bv is not None:
                hidden_states = hidden_states + bv.view(H, d_head)[None, :, None, :].to(hidden_states.dtype)

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
