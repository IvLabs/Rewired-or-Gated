"""Single-token EAP-IG driver (substitution only).

Thin wrapper over the multi-token numeric core in eap_ig.eap_ig -- imports
build_graph / score_prompt / build_result_json UNCHANGED. Adds two things the
single-token pipeline needs and the multi-token core deliberately does not:

  1. A single-token invariant sanity check. For single-token substitution the
     in-place context->memory swap leaves main and counterfactual at EQUAL
     token length (' ctx'/' mem' are each one pretoken; the BPE pretokenizer
     splits surrounding punctuation), so score_prompt's alignment window is
     always the full sequence. If a pair comes back unequal-length, the
     upstream single-token filter is wrong -- we SKIP the prompt and count it
     in n_skipped_length_mismatch (EXPECTED to stay 0), rather than silently
     scoring it through the multi-token prefix/suffix path.

  2. A self-describing _provenance block (spec A3.1) so single-token numbers
     can never be confused with the multi-token ones.

Contrast orientation (so the sign is never misread):
  * IG target  (alpha=1) = `main` = the CONFLICT prompt (distractor injected) --
    the input where context-following actually happens.
  * IG baseline(alpha=0) = `cf`   = the memory-consistent minimal pair (distractor
    replaced by the memory answer). It differs from `main` in exactly the disputed
    token, which is the whole point of the single-token design.
  * metric L = ctx_log_prob - mem_log_prob, measured on `main`. Positive L =
    context-following; a positive edge score = an edge that SUPPORTS context-
    following. This matches the standard EAP-IG form phi(e) = (x_clean - x_corrupt)
    . mean_grad with clean=main, corrupt=cf -- it is NOT inverted.

See docs/superpowers/specs/2026-07-09-single-token-eap-ig-design.md.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from contract import ConflictPrompt, EdgeScores
from eap_ig.eap_ig import (
    EdgeAttributionConfig,
    build_contrast_pair,
    build_graph,
    build_result_json,
    configure_model_for_edges,
    score_prompt,
    _validity_mask,
)
from singleton_fix.provenance import build_provenance

SET_TAG = "st"
ALIAS_POLICY = "first_single_token"


def compute_edge_attribution_single_token(
    model,
    prompts: List[ConflictPrompt],
    config: Optional[EdgeAttributionConfig] = None,
    _meta_out: Optional[Dict] = None,
    checkpoint_path: Optional[Path] = None,
    checkpoint_every: int = 50,
    on_checkpoint: Optional[Callable[[Path], None]] = None,
    desc: str = "EAP-IG-st",
) -> EdgeScores:
    """Run EAP-IG over single-token substitution prompts, enforcing the
    equal-length invariant. Same return shape as
    eap_ig.compute_edge_attribution.

    Checkpoint/resume (for multi-hour cloud runs that can be killed mid-run by a
    network drop, container preemption, etc.): if `checkpoint_path` is given and
    already exists on disk, resume from it instead of starting at prompt 0 --
    scoring is idempotent per-prompt (each prompt only ever contributes once to
    `accum`), so a resumed run produces EXACTLY the same accum/edge_scores as an
    uninterrupted one, given the same `prompts` list and `config` (the caller is
    responsible for that determinism -- the modal runner passes the identical
    cached prompt list and encodes granularity/n_steps/n_subset into the
    checkpoint filename so mismatched resumes can't silently mix runs).
    Every `checkpoint_every` prompts, the running accum + counters + resume index
    are saved to `checkpoint_path` and `on_checkpoint(checkpoint_path)` is called
    (the modal runner uses this to commit a Volume, so the checkpoint survives
    even if THIS container also dies later). The checkpoint file is deleted on
    successful completion.
    """
    if config is None:
        config = EdgeAttributionConfig()

    configure_model_for_edges(model, config.granularity)
    graph = build_graph(model, config.granularity)
    mask = _validity_mask(graph).to(config.contract_device)

    working = list(prompts)
    if config.n_subset is not None:
        g = torch.Generator().manual_seed(config.seed)
        idx = torch.randperm(len(working), generator=g)[: config.n_subset].tolist()
        working = [working[i] for i in idx]

    n_src, n_dst = len(graph.sources), len(graph.dests)
    accum = torch.zeros((n_src, n_dst), dtype=torch.float32, device=config.contract_device)
    n_used = 0
    n_skipped_degenerate = 0
    n_skipped_length_mismatch = 0
    start_index = 0

    if checkpoint_path is not None and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=config.contract_device)
        accum = ckpt["accum"].to(config.contract_device)
        n_used = ckpt["n_used"]
        n_skipped_degenerate = ckpt["n_skipped_degenerate"]
        n_skipped_length_mismatch = ckpt["n_skipped_length_mismatch"]
        start_index = ckpt["next_start_index"]
        print(f"[checkpoint] resuming {checkpoint_path.name} at prompt "
              f"{start_index}/{len(working)} (used={n_used} so far)", flush=True)

    def _save_checkpoint(next_index: int) -> None:
        if checkpoint_path is None:
            return
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "accum": accum.detach().cpu(),
            "n_used": n_used,
            "n_skipped_degenerate": n_skipped_degenerate,
            "n_skipped_length_mismatch": n_skipped_length_mismatch,
            "next_start_index": next_index,
        }, checkpoint_path)
        if on_checkpoint is not None:
            on_checkpoint(checkpoint_path)

    pbar = None
    try:
        from tqdm import tqdm
        # mininterval=0.5 keeps the carriage-return updates readable in Modal's
        # streamed logs instead of spamming a line per prompt. initial=start_index
        # shows the resumed position instead of restarting the bar at 0.
        pbar = tqdm(total=len(working), initial=start_index, desc=desc,
                    unit="prompt", file=sys.stdout, dynamic_ncols=True, ascii=True,
                    mininterval=0.5)
    except Exception:
        pass

    for i, prompt in enumerate(working):
        if i < start_index:
            continue  # already scored in a prior (killed) attempt -- skip, no recompute
        pair = build_contrast_pair(model, prompt)
        if pair is None:
            n_skipped_degenerate += 1
        else:
            main_tokens, cf_tokens = pair
            if main_tokens.shape[1] != cf_tokens.shape[1]:
                # Single-token invariant violated -- upstream filter bug. Skip loudly.
                n_skipped_length_mismatch += 1
            else:
                edge = score_prompt(model, graph, prompt, config)
                if edge is None:
                    n_skipped_degenerate += 1
                else:
                    accum += edge
                    n_used += 1
        if pbar is not None:
            # Live counters so a long full-granularity run visibly makes progress.
            pbar.update(1)
            pbar.set_postfix(used=n_used, degen=n_skipped_degenerate,
                             len_mismatch=n_skipped_length_mismatch, refresh=False)
        if checkpoint_path is not None and (i + 1) % checkpoint_every == 0:
            _save_checkpoint(i + 1)

    if n_skipped_length_mismatch > 0:
        print(f"[eap-ig-st] WARNING: {n_skipped_length_mismatch} prompts had "
              f"UNEQUAL main/cf length -- the single-token filter is not clean. "
              f"Investigate before trusting these numbers.")

    if n_used == 0:
        raise RuntimeError(
            "single-token EAP-IG produced 0 usable prompts (all failed contrast-pair "
            "construction or the equal-length invariant). Check the single-token cache.")

    mean_edge = accum / n_used
    mean_edge = torch.where(mask, mean_edge, torch.zeros_like(mean_edge))

    # Full pass completed -- the checkpoint is no longer needed for a resume.
    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint_path.unlink()

    edge_scores: EdgeScores = {}
    for u, v in mask.nonzero(as_tuple=False).tolist():
        edge_scores[(graph.sources[u].label, graph.dests[v].label)] = float(mean_edge[u, v])

    print(f"[eap-ig-st] granularity={config.granularity} n_steps={config.n_steps} | "
          f"used {n_used}/{len(working)} (degenerate={n_skipped_degenerate}, "
          f"length_mismatch={n_skipped_length_mismatch}) | {len(edge_scores)} edges")

    if _meta_out is not None:
        _meta_out["n_prompts_input"] = len(working)
        _meta_out["n_prompts_used"] = n_used
        _meta_out["pair_construction"] = {
            "usable": n_used,
            "skipped_degenerate": n_skipped_degenerate,
            "skipped_length_mismatch": n_skipped_length_mismatch,
        }
        _meta_out["n_skipped_length_mismatch"] = n_skipped_length_mismatch
        _meta_out["n_edges"] = len(edge_scores)
        _meta_out["granularity"] = config.granularity
        _meta_out["n_ig_steps"] = config.n_steps

    return edge_scores


def build_result_json_single_token(
    edge_scores: EdgeScores,
    meta: Dict,
    model_tag: str,
    tokenizer_name: str = "unknown",
    top_k: int = 1000,
) -> Dict:
    """Wrap eap_ig.build_result_json (prompt_type fixed to 'substitution') and
    inject the single-token _provenance block + the length-mismatch diagnostic."""
    payload = build_result_json(
        edge_scores, meta, model_tag, "substitution",
        tokenizer_name=tokenizer_name, top_k=top_k,
    )
    payload["_provenance"] = build_provenance(
        contrast="inplace_swap",
        dataset=f"single_token_{SET_TAG}",
        n_used=meta.get("n_prompts_used", 0),
        alias_policy=ALIAS_POLICY,
        prompt_type="substitution",
        model_tag=model_tag,
    )
    payload["n_skipped_length_mismatch"] = meta.get("n_skipped_length_mismatch", 0)
    return payload
