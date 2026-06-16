"""Model-agnostic discovery of the pieces logit-lens needs.

Works across architectures (Llama/Qwen/Gemma/GPT-2/...) by:
  - getting the layer count at runtime from len(hidden_states)
  - finding the lm_head via get_output_embeddings()
  - finding the final pre-head norm by name, excluding per-block norms
"""
from __future__ import annotations

import torch.nn as nn

# names commonly used for the trunk's final norm, in priority order
_FINAL_NORM_NAMES = ("norm", "ln_f", "final_layer_norm", "final_norm", "ln_out")
_BLOCK_MARKERS = (".layers.", ".h.", ".block.", ".blocks.", ".decoder.layers.")


def discover_lm_head(model):
    head = model.get_output_embeddings()
    if head is None:
        # tied embeddings fallback: use input embeddings as the decoder
        head = model.get_input_embeddings()
    return head


def _hidden_dim_from_head(lm_head):
    """The lm_head maps hidden_dim -> vocab; recover hidden_dim from its weight.
    Linear weight is [vocab, hidden]; tied Embedding weight is [vocab, hidden]."""
    w = getattr(lm_head, "weight", None)
    return int(w.shape[1]) if (w is not None and w.dim() == 2) else None


def _norm_dim(mod):
    w = getattr(mod, "weight", None)
    if w is not None and w.dim() >= 1:
        return int(w.shape[-1])
    ns = getattr(mod, "normalized_shape", None)
    if ns:
        return int(ns[-1] if hasattr(ns, "__len__") else ns)
    return None


def discover_final_norm(model, lm_head=None):
    """Return the trunk's final pre-head norm, or None (logit lens still works
    without it). A candidate must (a) be named like a final norm, (b) not live
    inside a transformer block, and (c) match the lm_head's hidden dimension —
    that dim check is what avoids grabbing a 128-dim sub-norm in models with
    many internal norms (e.g. Gemma's matformer / multimodal wrappers).
    """
    hidden = _hidden_dim_from_head(lm_head) if lm_head is not None else None
    candidates = []
    for name, mod in model.named_modules():
        if any(m in name for m in _BLOCK_MARKERS):
            continue
        leaf = name.split(".")[-1]
        if leaf in _FINAL_NORM_NAMES and _is_norm(mod):
            d = _norm_dim(mod)
            if hidden is not None and d is not None and d != hidden:
                continue                  # wrong-sized norm — skip
            candidates.append((_FINAL_NORM_NAMES.index(leaf), name.count("."), name, mod))
    if not candidates:
        return None
    # name priority, then deepest path (language_model.norm over a stray top-level norm)
    candidates.sort(key=lambda c: (c[0], -c[1]))
    return candidates[0][3]


def _is_norm(mod) -> bool:
    cls = type(mod).__name__.lower()
    return "norm" in cls or isinstance(mod, (nn.LayerNorm,))
