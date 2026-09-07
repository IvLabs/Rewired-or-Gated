"""Edge Attribution Patching with Integrated Gradients (EAP-IG) — multi-model.

Edge-level attribution: ranks *connections* (source-component output -> destination-
component input) rather than components.

Contrast pair (see build_contrast_pair): the counterfactual is built by
substituting every occurrence of `context_answer` for `memory_answer` INSIDE
`prompt.text` -- NOT by comparing against `prompt.clean_text` (a structurally
different, much shorter bare question with no injected context at all).
`prompt.text` and `prompt.clean_text` differ in far more than the disputed
fact (one has an entire extra sentence the other doesn't), so pairing them
directly is not a valid EAP/EAP-IG contrast pair: position-by-position
comparisons only mean something when position i is the same word in both
runs. Same-sentence substitution keeps everything but the disputed fact fixed.

The substitution can still change token count (context_answer and
memory_answer need not tokenize to the same length), so source activation
deltas and destination gradients are aligned on the UNION of the common
PREFIX (text before the first substitution) and common SUFFIX (text after
the last one) -- both provably identical token-for-token by construction.
Neither is simply the first/last min(len(main), len(cf)) positions -- BPE
tokenizes the same words differently depending on whether they sit at the
start of a standalone string (no leading space) or mid-sentence (leading
space), so naive length-matching silently misaligns at the seam.
`_common_prefix_len`/`_common_suffix_len` walk forward/backward from the
ends of both token sequences and only count positions with genuinely
identical token ids. The middle span, where the disputed phrase itself
differs, is excluded from delta/gradient contraction: it is not
position-comparable when context_answer and memory_answer tokenize to
different lengths.

Metric: `L = ctx_log_prob - mem_log_prob`, each a mean teacher-forced log-prob
over the answer's full token sequence (foundation.answer_log_prob), computed
from two separate forward passes on the CONFLICT prompt with the context vs.
memory answer appended. Both backward passes are required because the two
log-probs come from two different forward passes; the destination gradients
are subtracted (grad_ctx - grad_mem) before contraction with the source delta.

Plain EAP (n_steps=1, grad at the counterfactual baseline) vs EAP-IG
(n_steps>=2, integrated) is still IG over the *prefix+suffix window* of the
input residual stream,
interpolating between the counterfactual and main window activations; the
positions inside the substituted span (which have no corresponding
counterfactual activation) are left at their real conflict value during
interpolation.

Two granularities (see `build_graph`): "full" (per-head q/k/v + mlp_in
destinations, needs split_qkv) and "coarse" (resid_pre destinations,
de-risked fallback). TransformerLens gotchas inherited from path_patching/:
callable names_filter (string filters are exact-match) and cache.cache_dict
access (key mangling).

Reference: Hanna, Pezzelle & Belinkov (2024), arXiv:2403.17806 (github.com/hannamw/eap-ig);
applied to fine-tuning by Wang et al. (2025), arXiv:2502.11812.
"""

from __future__ import annotations

import os
import sys

# Make contract.py and foundation.py importable from the project root.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

import torch
from transformer_lens.utilities import get_act_name as _get_act_name

from contract import ConflictPrompt, EdgeScores
from foundation import _answer_ids, answer_log_prob


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EdgeAttributionConfig:
    """Configuration for an EAP-IG run.

    Attributes:
        n_steps: integrated-gradients steps. 1 = plain EAP (grad at the
            counterfactual baseline). >=2 = EAP-IG (midpoint Riemann average
            counterfactual->main).
        granularity: "full" (per-head q/k/v + mlp_in destinations; needs
            split_qkv) or "coarse" (resid_pre destinations; no split_qkv).
        contract_device: device for the (large) stack-and-einsum contraction.
            "cpu" is safe on a 4 GB GPU with long coherent prompts.
        n_subset: if set, run on this many prompts only (debug / speed).
        seed: RNG seed for subset selection.
    """
    n_steps: int = 5
    granularity: str = "full"
    contract_device: str = "cpu"
    n_subset: Optional[int] = None
    seed: int = 42


# ---------------------------------------------------------------------------
# TransformerLens helpers (inherited gotchas from path_patching/)
# ---------------------------------------------------------------------------

def _cache_get(cache, hook_name: str) -> torch.Tensor:
    """Fetch a hook tensor by its full name.

    ActivationCache.__getitem__ re-routes the key through get_act_name() which
    mangles full hook-name strings — access cache_dict directly to avoid that.
    """
    if hasattr(cache, "cache_dict"):
        return cache.cache_dict[hook_name]
    return cache[hook_name]


def _names_filter(names: List[str]) -> Callable[[str], bool]:
    """Build a callable names_filter. A *string* names_filter is exact-match in
    TransformerLens, so we always pass a callable that matches an explicit set."""
    wanted = set(names)
    return lambda n: n in wanted


def configure_model_for_edges(model, granularity: str) -> None:
    """Enable the TransformerLens flags that expose per-head / per-MLP edge
    endpoints. Idempotent. See EAP_IG_Spec.md §3."""
    model.set_use_attn_result(True)
    if granularity == "full":
        model.set_use_split_qkv_input(True)
        model.set_use_hook_mlp_in(True)


# ---------------------------------------------------------------------------
# Node graph
# ---------------------------------------------------------------------------

@dataclass
class _Node:
    label: str          # human-readable, e.g. "blocks.3.attn.hook_result[h5]"
    hook: str           # the TransformerLens hook name to read/grad
    order: float        # topological position (write for src, read for dst)
    head: Optional[int] = None   # head index to slice from a [.,.,n_heads,.] hook


@dataclass
class _Graph:
    sources: List[_Node]
    dests: List[_Node]
    src_hooks: List[str] = field(default_factory=list)   # unique source hooks to cache
    dst_hooks: List[str] = field(default_factory=list)   # unique dest hooks to grad

    def __post_init__(self):
        self.src_hooks = sorted({n.hook for n in self.sources})
        self.dst_hooks = sorted({n.hook for n in self.dests})


def build_graph(model, granularity: str) -> _Graph:
    """Construct the source/destination node lists for the edge graph."""
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    sources: List[_Node] = []
    dests: List[_Node] = []

    # Input source (embed + positional), written before everything.
    input_hook = _get_act_name("resid_pre", 0)
    sources.append(_Node(label="input", hook=input_hook, order=float("-inf")))

    for l in range(n_layers):
        # Per-head attention outputs (sources), written at l + 0.3.
        result_hook = _get_act_name("result", l)
        for h in range(n_heads):
            sources.append(_Node(
                label=f"blocks.{l}.attn.hook_result[h{h}]",
                hook=result_hook, order=l + 0.3, head=h,
            ))
        # MLP output (source), written at l + 0.7.
        sources.append(_Node(
            label=f"blocks.{l}.hook_mlp_out", hook=_get_act_name("mlp_out", l),
            order=l + 0.7,
        ))

    if granularity == "full":
        # Grouped-query attention (GQA): TransformerLens broadcasts the residual
        # into n_heads copies for hook_q_input but only n_key_value_heads copies
        # for hook_k_input / hook_v_input (see TransformerBlock: key_input/
        # value_input use repeat_along_head_dimension(..., n_heads=n_kv_heads)).
        # Iterating k/v destinations over n_heads (as for a plain MHA model like
        # GPT-2) would slice head indices that don't exist on the K/V input hooks
        # and raise IndexError in score_prompt. Qwen-2.5-3B (16 q / 2 kv),
        # Llama-3.2-3B (24 q / 8 kv) and Gemma-3-4B are all GQA. For MHA models
        # n_key_value_heads is None -> n_kv_heads == n_heads, so this is a no-op
        # (GPT-2 graph and results are unchanged).
        n_kv_heads = getattr(model.cfg, "n_key_value_heads", None) or n_heads
        for l in range(n_layers):
            for letter in ("q", "k", "v"):
                qkv_hook = _get_act_name(f"{letter}_input", l)
                n_endpoints = n_heads if letter == "q" else n_kv_heads
                for h in range(n_endpoints):
                    dests.append(_Node(
                        label=f"blocks.{l}.attn.hook_{letter}_input[h{h}]",
                        hook=qkv_hook, order=l + 0.0, head=h,
                    ))
            dests.append(_Node(
                label=f"blocks.{l}.hook_mlp_in", hook=_get_act_name("mlp_in", l),
                order=l + 0.5,
            ))
    elif granularity == "coarse":
        for l in range(n_layers):
            dests.append(_Node(
                label=f"blocks.{l}.hook_resid_pre",
                hook=_get_act_name("resid_pre", l), order=l - 0.05,
            ))
    else:
        raise ValueError(f"unknown granularity {granularity!r}")

    # Logits destination: gradient w.r.t. the final residual stream.
    dests.append(_Node(
        label="logits", hook=_get_act_name("resid_post", n_layers - 1),
        order=float("inf"),
    ))

    return _Graph(sources=sources, dests=dests)


def _validity_mask(graph: _Graph) -> torch.Tensor:
    """Boolean [n_src, n_dst]: True where src.write_order < dst.read_order."""
    src_order = torch.tensor([n.order for n in graph.sources], dtype=torch.float64)
    dst_order = torch.tensor([n.order for n in graph.dests], dtype=torch.float64)
    return src_order[:, None] < dst_order[None, :]


# ---------------------------------------------------------------------------
# Contrast pair (independently tokenized, suffix-aligned)
# ---------------------------------------------------------------------------

def build_contrast_pair(
    model, prompt: ConflictPrompt
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Return (main_tokens, counterfactual_tokens): the SAME conflict text,
    with every occurrence of context_answer replaced by memory_answer.

    This is the valid EAP/EAP-IG contrast pair: same sentence, same
    structure, only the disputed fact differs -- NOT prompt.clean_text (a
    structurally different, much shorter bare question with no injected
    context at all), which would confound "did the claim win" with "is
    there a claim there at all."

    Returns None if context_answer does not literally appear in prompt.text
    (nothing safe to substitute) or either resulting text is degenerate.
    """
    ctx = prompt.context_answer.strip()
    mem = prompt.memory_answer.strip()
    if ctx not in prompt.text:
        return None
    # Plain substring replace (not word-boundary-aware, unlike foundation._matches_answer):
    # safe because ParaConflict's distractor is always injected as a standalone word,
    # never a substring of a larger word elsewhere in the text.
    counterfactual_text = prompt.text.replace(ctx, mem)

    main = model.to_tokens(prompt.text)
    cf = model.to_tokens(counterfactual_text)
    if main.shape[1] < 2 or cf.shape[1] < 2:
        return None
    return main, cf


def _answer_log_prob_diff(logits: torch.Tensor, answer_ids: torch.Tensor) -> torch.Tensor:
    """Same computation as foundation.answer_log_prob, but keeps the autograd
    graph (no .item()) -- used as the backward target inside score_prompt."""
    k = answer_ids.shape[0]
    if k == 0:
        raise ValueError(
            "answer_ids is empty -- cannot compute log-prob of a zero-length answer."
        )
    log_probs = torch.log_softmax(logits[0, -k - 1:-1, :], dim=-1)
    idx = torch.arange(k, device=log_probs.device)
    return log_probs[idx, answer_ids.to(log_probs.device)].mean()


# ---------------------------------------------------------------------------
# Per-prompt scoring
# ---------------------------------------------------------------------------

def _ig_alphas(n_steps: int) -> List[float]:
    """Interpolation coefficients counterfactual(0) -> main(1).

    n_steps == 1  -> [0.0]  (plain EAP: gradient at the counterfactual baseline)
    n_steps >= 2  -> midpoint Riemann rule (k + 0.5) / n_steps
    """
    if n_steps == 1:
        return [0.0]
    return [(k + 0.5) / n_steps for k in range(n_steps)]


def _common_suffix_len(a: torch.Tensor, b: torch.Tensor) -> int:
    """Length of the longest trailing run of IDENTICAL token ids between two
    1-D token tensors, walking backward from the end.

    Used to find the suffix-aligned window between conflict and counterfactual
    tokens (from build_contrast_pair): the text after the last substituted
    occurrence of context_answer, provably identical in both versions.

    `min(len(a), len(b))` is NOT a safe alignment window on its own: BPE
    tokenizers split the same word differently depending on whether it's at
    the start of a string (no leading space, e.g. "Lionel" -> ["L","ion","el"])
    or mid-sentence (leading space, e.g. " Lionel" -> [" Lionel"], one token).
    The first word of the shared suffix routinely does NOT match at the token-id
    level even though the string does. This walks backward and only counts
    positions that are genuinely identical token ids, so the comparison window
    is verified, not assumed.
    """
    n = min(a.shape[0], b.shape[0])
    k = 0
    while k < n and int(a[-(k + 1)]) == int(b[-(k + 1)]):
        k += 1
    return k


def _common_prefix_len(a: torch.Tensor, b: torch.Tensor) -> int:
    """Length of the longest leading run of IDENTICAL token ids between two
    1-D token tensors, walking forward from the start.

    Counterpart to _common_suffix_len. Together they bound the two
    provably-identical regions of a (main, counterfactual) pair built by
    build_contrast_pair: the text before the first substituted occurrence
    of context_answer, and the text after the last one, respectively. The
    middle span (where the disputed phrase itself differs) is not position-
    comparable when context_answer and memory_answer tokenize to different
    lengths, and is deliberately excluded from delta/gradient contraction in
    score_prompt.
    """
    n = min(a.shape[0], b.shape[0])
    k = 0
    while k < n and int(a[k]) == int(b[k]):
        k += 1
    return k


def _alignment_window(main_ids: torch.Tensor, cf_ids: torch.Tensor) -> Tuple[int, int]:
    """Return (L_pre, L_suf) bounding the position-comparable window between
    a (main, counterfactual) token pair.

    When the two sequences have the SAME total length, there is no index
    shift anywhere -- the disputed span in the middle sits at identical
    absolute positions in both, exactly as comparable as the verified
    prefix/suffix, so the whole sequence is returned as the window (L_pre =
    full length, L_suf = 0) instead of carving out the middle.

    When lengths differ, the substitution shifts every position after it,
    so only the common PREFIX (before the first substitution) and common
    SUFFIX (after the last one) are provably comparable; the middle is
    excluded.
    """
    if main_ids.shape[0] == cf_ids.shape[0]:
        return int(main_ids.shape[0]), 0
    L_pre = _common_prefix_len(main_ids, cf_ids)
    L_suf = _common_suffix_len(main_ids, cf_ids)
    shorter = min(main_ids.shape[0], cf_ids.shape[0])
    if L_pre + L_suf > shorter:
        L_suf = shorter - L_pre  # guard against the two windows overlapping
    return L_pre, L_suf


def _stack_sources(graph: _Graph, cache, device: str) -> torch.Tensor:
    """Stack source writes into [n_src, pos, d_model] on `device` (batch squeezed)."""
    rows = []
    for node in graph.sources:
        act = _cache_get(cache, node.hook)          # [1, pos, (n_heads,) d_model]
        if node.head is not None:
            act = act[:, :, node.head, :]           # [1, pos, d_model]
        rows.append(act[0].detach().to(device))     # [pos, d_model]
    return torch.stack(rows, dim=0)


def _forward_and_grad_at_dests(
    model,
    graph: _Graph,
    prompt_tokens: torch.Tensor,
    answer_ids: torch.Tensor,
    input_hook: str,
    interp_prefix: torch.Tensor,
    dev: str,
) -> Dict[str, torch.Tensor]:
    """Run model on [prompt_tokens ++ answer_ids], with resid_pre[0] on the
    prompt positions overridden by interp_prefix (grad-tracked leaf), capture
    every destination hook, backward through answer_log_prob, return per-hook
    gradients restricted to the prompt positions (answer positions dropped).
    """
    L_prompt = prompt_tokens.shape[1]
    captured: Dict[str, torch.Tensor] = {}

    def _input_override(value, hook, _interp=interp_prefix, _L=L_prompt):
        value = value.clone()
        value[:, :_L, :] = _interp
        return value

    def _make_capture(name):
        def _capture(value, hook):
            captured[name] = value          # keep graph-attached reference
            return value
        return _capture

    fwd_hooks = [(input_hook, _input_override)]
    fwd_hooks += [(h, _make_capture(h)) for h in graph.dst_hooks]

    full_tokens = torch.cat([prompt_tokens, answer_ids.unsqueeze(0)], dim=1)
    with torch.enable_grad():
        logits = model.run_with_hooks(full_tokens, fwd_hooks=fwd_hooks)
        log_prob = _answer_log_prob_diff(logits, answer_ids)
        dst_tensors = [captured[h] for h in graph.dst_hooks]
        # allow_unused: a destination structurally disconnected from the
        # metric yields None -> treat as a zero gradient rather than crash.
        grads = torch.autograd.grad(
            log_prob, dst_tensors, retain_graph=False, allow_unused=True
        )

    out: Dict[str, torch.Tensor] = {}
    for h, t, g in zip(graph.dst_hooks, dst_tensors, grads):
        g = torch.zeros_like(t) if g is None else g
        out[h] = g[:, :L_prompt].detach().to(dev)   # drop answer positions
    return out


def score_prompt(
    model, graph: _Graph, prompt: ConflictPrompt, config: EdgeAttributionConfig
) -> Optional[torch.Tensor]:
    """Compute the [n_src, n_dst] edge-score matrix for one prompt, or None if the
    contrast pair could not be built.

    delta[u]  = act_src(main)[window] - act_src(cf)[window]     (source term)
    grad[v]   = d(ctx_log_prob)/dz[v][window] - d(mem_log_prob)/dz[v][window]  (dest term)
    edge[u,v] = sum_{pos in window, d} delta[u] * grad[v]

    "window" = the union of the common PREFIX (text before the first
    context_answer->memory_answer substitution) and the common SUFFIX (text
    after the last one) between main and the counterfactual -- both are
    provably identical token-for-token by construction (see
    build_contrast_pair). The middle span, where the disputed phrase itself
    differs, is excluded when context_answer and memory_answer tokenize to
    different lengths (the substitution shifts every later position, so
    positions past the swap are only comparable via the suffix walk).

    When main and cf have the SAME total token count, there is no shift:
    the prefix run, the equal-length gap, and the suffix run each line up
    at identical absolute indices in both sequences, so the disputed span
    is exactly as position-comparable as the verified prefix/suffix and is
    included in the window too, instead of being discarded.
    """
    pair = build_contrast_pair(model, prompt)
    if pair is None:
        return None
    main_tokens, cf_tokens = pair
    main_ids, cf_ids = main_tokens[0], cf_tokens[0]

    L_pre, L_suf = _alignment_window(main_ids, cf_ids)
    if L_pre == 0 and L_suf == 0:
        return None

    dev = config.contract_device
    src_filter = _names_filter(graph.src_hooks)
    input_hook = _get_act_name("resid_pre", 0)

    def _select(act: torch.Tensor) -> torch.Tensor:
        """[.., pos, ..] -> [.., L_pre + L_suf, ..]: prefix positions then
        suffix positions, both guaranteed identical between main and cf."""
        parts = []
        if L_pre > 0:
            parts.append(act[:, :L_pre])
        if L_suf > 0:
            parts.append(act[:, -L_suf:])
        return torch.cat(parts, dim=1)

    # --- source activation difference (no grad), window-aligned ---
    with torch.no_grad():
        _, main_cache = model.run_with_cache(main_tokens, names_filter=src_filter)
        _, cf_cache = model.run_with_cache(cf_tokens, names_filter=src_filter)
    src_main = _select(_stack_sources(graph, main_cache, dev))
    src_cf = _select(_stack_sources(graph, cf_cache, dev))
    delta = src_main - src_cf                        # [n_src, L_pre+L_suf, d_model]

    resid_pre_main = _cache_get(main_cache, input_hook).detach()   # [1, L_main, d]
    resid_pre_cf = _cache_get(cf_cache, input_hook).detach()       # [1, L_cf, d]
    sel_main = _select(resid_pre_main)
    sel_cf = _select(resid_pre_cf)

    ctx_ids = _answer_ids(model, prompt.context_answer).to(main_tokens.device)
    mem_ids = _answer_ids(model, prompt.memory_answer).to(main_tokens.device)

    # --- integrated destination gradients ---
    grad_accum: Dict[str, torch.Tensor] = {}

    for alpha in _ig_alphas(config.n_steps):
        interp_sel = sel_cf + alpha * (sel_main - sel_cf)   # [1, L_pre+L_suf, d]
        interp_prefix = resid_pre_main.clone()
        # Positions with no counterfactual counterpart (the substituted span
        # itself, and anything past a length-changing substitution) stay
        # fixed at their real conflict value -- only the provably-identical
        # prefix/suffix positions get interpolated.
        if L_pre > 0:
            interp_prefix[:, :L_pre, :] = interp_sel[:, :L_pre, :]
        if L_suf > 0:
            interp_prefix[:, -L_suf:, :] = interp_sel[:, L_pre:L_pre + L_suf, :]
        interp_prefix = interp_prefix.detach().requires_grad_(True)

        grads_ctx = _forward_and_grad_at_dests(
            model, graph, main_tokens, ctx_ids, input_hook, interp_prefix, dev)
        grads_mem = _forward_and_grad_at_dests(
            model, graph, main_tokens, mem_ids, input_hook, interp_prefix, dev)

        for h in graph.dst_hooks:
            g = _select(grads_ctx[h] - grads_mem[h])   # [1, L_pre+L_suf, (heads,) d]
            grad_accum[h] = g if h not in grad_accum else grad_accum[h] + g

    n_alpha = len(_ig_alphas(config.n_steps))
    for h in grad_accum:
        grad_accum[h] = grad_accum[h] / n_alpha

    # --- stack destination grads into [n_dst, L_pre+L_suf, d_model] ---
    dst_rows = []
    for node in graph.dests:
        g = grad_accum[node.hook]                    # [1, L_pre+L_suf, (n_heads,) d_model]
        if node.head is not None:
            g = g[:, :, node.head, :]
        dst_rows.append(g[0])
    grad_stack = torch.stack(dst_rows, dim=0)          # [n_dst, L_pre+L_suf, d_model]

    # --- contraction: edge[u,v] = sum_pos,d  delta[u] * grad[v] ---
    edge = torch.einsum("upd,vpd->uv", delta, grad_stack)   # [n_src, n_dst]
    return edge


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def compute_edge_attribution(
    model,
    prompts: List[ConflictPrompt],
    config: Optional[EdgeAttributionConfig] = None,
    _meta_out: Optional[Dict] = None,
) -> EdgeScores:
    """Run EAP-IG over `prompts` and return EdgeScores = Dict[(src, dst), float].

    Keys are (src_label, dst_label) human-readable hook strings. Score is the
    signed mean over prompts of the edge contribution to L = context - memory;
    rank by absolute value. See EAP_IG_Spec.md.
    """
    if config is None:
        config = EdgeAttributionConfig()

    configure_model_for_edges(model, config.granularity)
    graph = build_graph(model, config.granularity)
    mask = _validity_mask(graph).to(config.contract_device)

    # optional subset
    working = list(prompts)
    if config.n_subset is not None:
        g = torch.Generator().manual_seed(config.seed)
        idx = torch.randperm(len(working), generator=g)[: config.n_subset].tolist()
        working = [working[i] for i in idx]

    n_src, n_dst = len(graph.sources), len(graph.dests)
    accum = torch.zeros((n_src, n_dst), dtype=torch.float32, device=config.contract_device)
    n_used = 0
    n_skipped_degenerate = 0

    try:
        from tqdm import tqdm
        # ascii=True avoids a UnicodeEncodeError on Windows cp1252 consoles.
        iterator = tqdm(working, desc="EAP-IG", unit="prompt", file=sys.stdout,
                        dynamic_ncols=True, ascii=True)
    except Exception:
        iterator = working

    for prompt in iterator:
        edge = score_prompt(model, graph, prompt, config)
        if edge is None:
            n_skipped_degenerate += 1
            continue
        accum += edge
        n_used += 1

    if n_used == 0:
        raise RuntimeError(
            "EAP-IG produced 0 usable prompts (every prompt failed contrast-pair "
            "construction: context_answer not found verbatim in prompt.text, or the "
            "resulting main/counterfactual tokenization was degenerate). Check that "
            "load_conflict_prompts() returned rows where context_answer is a literal "
            "substring of text."
        )

    mean_edge = accum / n_used
    mean_edge = torch.where(mask, mean_edge, torch.zeros_like(mean_edge))

    edge_scores: EdgeScores = {}
    valid_idx = mask.nonzero(as_tuple=False)
    for u, v in valid_idx.tolist():
        edge_scores[(graph.sources[u].label, graph.dests[v].label)] = float(mean_edge[u, v])

    print(
        f"[eap-ig] granularity={config.granularity} n_steps={config.n_steps} | "
        f"used {n_used}/{len(working)} prompts (skipped_degenerate={n_skipped_degenerate}) | "
        f"{len(edge_scores)} edges"
    )

    if _meta_out is not None:
        _meta_out["n_prompts_input"] = len(working)
        _meta_out["n_prompts_used"] = n_used
        _meta_out["pair_construction"] = {
            "usable": n_used, "skipped_degenerate": n_skipped_degenerate,
        }
        _meta_out["n_edges"] = len(edge_scores)
        _meta_out["granularity"] = config.granularity
        _meta_out["n_ig_steps"] = config.n_steps

    return edge_scores


# ---------------------------------------------------------------------------
# Result serialisation + runner
# ---------------------------------------------------------------------------

def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _tl_version() -> str:
    try:
        import transformer_lens
        return transformer_lens.__version__
    except Exception:
        return "unknown"


def build_result_json(
    edge_scores: EdgeScores, meta: Dict, model_tag: str, prompt_type: str,
    tokenizer_name: str = "unknown", top_k: int = 50,
) -> Dict:
    ranked = sorted(edge_scores.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return {
        "model": model_tag,
        "prompt_type": prompt_type,
        "method": "EAP-IG",
        "n_ig_steps": meta.get("n_ig_steps"),
        "granularity": meta.get("granularity"),
        "n_prompts_input": meta.get("n_prompts_input", 0),
        "n_prompts_used": meta.get("n_prompts_used", 0),
        "pair_construction": meta.get("pair_construction", {}),
        "n_edges": meta.get("n_edges", len(edge_scores)),
        "top_k_edges": [
            {"src": s, "dst": d, "score": round(v, 4)} for (s, d), v in ranked[:top_k]
        ],
        "edge_scores": {f"{s} -> {d}": round(v, 4) for (s, d), v in edge_scores.items()},
        "meta": {
            "tokenizer": tokenizer_name,
            "torch_version": torch.__version__,
            "transformer_lens_version": _tl_version(),
            "git_hash": _git_hash(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }


_MODEL_HF = {
    "gpt2": (None, "gpt2"),
    "gpt2_instruct": ("vicgalle/gpt2-open-instruct-v1", "gpt2_instruct"),
}


def main() -> None:
    import argparse

    from foundation import load_model, load_conflict_prompts

    parser = argparse.ArgumentParser(description="Run EAP-IG edge attribution.")
    parser.add_argument("--model", default="gpt2", choices=list(_MODEL_HF))
    parser.add_argument("--prompt_type", default="substitution",
                        choices=["substitution", "coherent"])
    parser.add_argument("--n_steps", type=int, default=5)
    parser.add_argument("--granularity", default="full", choices=["full", "coarse"])
    parser.add_argument("--n_subset", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--contract_device", default="cpu")
    args = parser.parse_args()

    hf_name, model_tag = _MODEL_HF[args.model]
    kwargs = {"hf_model_name": hf_name} if hf_name else {}

    print(f"[eap-ig] loading model={args.model} prompt_type={args.prompt_type} ...")
    model = load_model("gpt2", **kwargs)
    prompts = load_conflict_prompts(prompt_type=args.prompt_type)

    config = EdgeAttributionConfig(
        n_steps=args.n_steps, granularity=args.granularity,
        contract_device=args.contract_device, n_subset=args.n_subset, seed=args.seed,
    )

    meta: Dict = {}
    edge_scores = compute_edge_attribution(model, prompts, config, _meta_out=meta)

    results_dir = os.path.join(_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"eapig_{model_tag}_{args.prompt_type}.json")
    with open(out_path, "w") as f:
        json.dump(
            build_result_json(edge_scores, meta, model_tag, args.prompt_type,
                               tokenizer_name=model_tag),
            f, indent=2,
        )
    print(f"[eap-ig] saved -> {out_path}")


if __name__ == "__main__":
    main()
