import re
from typing import List, Tuple, Dict, Any

# down/up: down_blocks.{i}.attentions.{j}.transformer_blocks.{k}.attn1/attn2
_RE_DU = re.compile(
    r"^(down_blocks|up_blocks)\.(\d+)\.attentions\.(\d+)\.transformer_blocks\.(\d+)\.(attn1|attn2)$"
)
# mid: mid_block.attentions.{j}.transformer_blocks.{k}.attn1/attn2
_RE_MID = re.compile(
    r"^mid_block\.attentions\.(\d+)\.transformer_blocks\.(\d+)\.(attn1|attn2)$"
)

_STAGE_ORDER = {"down_blocks": 0, "mid_block": 1, "up_blocks": 2, "other": 9}
_ATTN_ORDER  = {"attn1": 0, "attn2": 1}  # self first, then cross


def _is_attn_module(name: str, m) -> bool:
    # diffusers Attention modules typically expose these attributes
    return name.endswith((".attn1", ".attn2")) and hasattr(m, "to_q") and hasattr(m, "to_k") and hasattr(m, "heads")


def _get_attn_dims(attn) -> Dict[str, int]:
    # Linear weight shape: (out_features, in_features) = (inner_dim, d_model)
    inner_dim = int(attn.to_q.weight.shape[0])
    d_model   = int(attn.to_q.weight.shape[1])
    heads     = int(attn.heads)
    head_dim  = int(getattr(attn, "head_dim", inner_dim // heads))
    return {"d_model": d_model, "inner_dim": inner_dim, "heads": heads, "head_dim": head_dim}


def _parse_attn_name(name: str) -> Dict[str, Any]:
    m = _RE_DU.match(name)
    if m:
        stage, bi, ai, tbi, which = m.groups()
        return {
            "stage": stage,
            "block": int(bi),
            "attn": int(ai),
            "tblock": int(tbi),
            "which": which,  # attn1 or attn2
        }

    m = _RE_MID.match(name)
    if m:
        ai, tbi, which = m.groups()
        return {
            "stage": "mid_block",
            "block": 0,
            "attn": int(ai),
            "tblock": int(tbi),
            "which": which,
        }

    # fallback for unexpected module names
    return {
        "stage": "other",
        "block": -1,
        "attn": -1,
        "tblock": -1,
        "which": "attn1" if name.endswith(".attn1") else "attn2",
    }


def tag_and_print_unet_attn_layers(unet, print_all: bool = True) -> List[Tuple[str, Any]]:
    """Attach `attn._attn_meta` to every attn1/attn2 module in the UNet.

    The meta dict contains stage (down/mid/up), block/attn/transformer-block
    indices, self/cross type, dims, and a stable `global_idx`. Optionally prints
    the full table once before inference.
    """
    found: List[Tuple[str, Any]] = []
    for name, m in unet.named_modules():
        if _is_attn_module(name, m):
            found.append((name, m))

    # sort: down -> mid -> up, then block/attn/tblock/attn1-2
    def sort_key(item):
        name, _m = item
        p = _parse_attn_name(name)
        return (
            _STAGE_ORDER.get(p["stage"], 9),
            p["block"],
            p["attn"],
            p["tblock"],
            _ATTN_ORDER.get(p["which"], 9),
            name,
        )

    found.sort(key=sort_key)

    # tag modules
    for gidx, (name, attn) in enumerate(found):
        p = _parse_attn_name(name)
        dims = _get_attn_dims(attn)

        meta = {
            "global_idx": gidx,
            "name": name,
            "stage": p["stage"],
            "block": p["block"],
            "attn": p["attn"],
            "tblock": p["tblock"],
            "which": p["which"],                    # attn1 / attn2
            "type": "self" if p["which"] == "attn1" else "cross",
            **dims,
        }
        attn._attn_meta = meta

    # print table
    if print_all:
        print("\n[UNet Attention modules] (tagged with _attn_meta)")
        print(" idx | stage     | block | attn | tblock | which | type  | d_model | inner | heads | head_dim | name")
        print("-" * 140)
        for (name, attn) in found:
            m = attn._attn_meta
            print(
                f"{m['global_idx']:4d} | {m['stage']:<9} | {m['block']:5d} | {m['attn']:4d} | {m['tblock']:6d} | "
                f"{m['which']:<5} | {m['type']:<5} | {m['d_model']:7d} | {m['inner_dim']:5d} | "
                f"{m['heads']:5d} | {m['head_dim']:8d} | {m['name']}"
            )
        print("-" * 140)
        print(f"Total attn modules: {len(found)}\n")

    return found
