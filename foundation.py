"""Foundation for the prototype phase.

Provides:
    - load_conflict_prompts: ParaConflict -> List[ConflictPrompt]
    - load_model, run_with_cache: model wrapper
    - _answer_ids, answer_log_prob, answer_log_probs: multi-token log-prob scoring
    - compute_crr: behavioral metric

Token ids are no longer stored on ConflictPrompt. Each method tokenizes the
answer strings with the loaded model's own tokenizer at scoring time, making
the pipeline tokenizer-agnostic across all model families (Llama, Qwen, Gemma…).
"""
from __future__ import annotations

import os
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from transformer_lens import HookedTransformer

from contract import ConflictPrompt

load_dotenv()

_HF_DATASET = "gaotang/ParaConflict"

_COL_CATEGORY   = "Category"
_COL_ANSWER     = "Answer"
_COL_DISTRACTOR = "Distracted Token"
_COL_CLEAN      = "Clean Prompt"
_COL_SUBSTITUTION = "Substitution Conflict"
_COL_COHERENT     = "Coherent Conflict"


# ---------------------------------------------------------------------------
# Dataset loader — tokenizer-agnostic
# ---------------------------------------------------------------------------

def load_conflict_prompts(
    prompt_type: Literal["substitution", "coherent"] = "substitution",
    heldout_domain: Optional[str] = None,
    split: Literal["train", "heldout", "all"] = "all",
) -> List[ConflictPrompt]:
    """Load ParaConflict and emit ConflictPrompt objects.

    The single-token filter is removed. All rows with valid string fields are
    kept, giving ~2.5× more rows than the old GPT-2-filtered set.  Answer
    strings are stored as-is; each attribution method tokenizes them with the
    loaded model's own tokenizer.

    canonical memory answer = memory_aliases[0] (deterministic, tokenizer-free).
    """
    if split in ("train", "heldout") and heldout_domain is None:
        raise ValueError(f"split={split!r} requires heldout_domain to be set")

    if prompt_type == "substitution":
        text_col = _COL_SUBSTITUTION
    elif prompt_type == "coherent":
        text_col = _COL_COHERENT
    else:
        raise ValueError(f"unknown prompt_type {prompt_type!r}")

    hf_token = os.getenv("HF_TOKEN")
    ds = load_dataset(_HF_DATASET, split="test", token=hf_token)
    total = len(ds)

    prompts: List[ConflictPrompt] = []
    dropped_missing = 0
    dropped_degenerate = 0

    for row in ds:
        domain    = row[_COL_CATEGORY]
        aliases   = row[_COL_ANSWER]
        distractor = row[_COL_DISTRACTOR]
        clean      = row[_COL_CLEAN]
        text       = row[text_col]

        if not (aliases and distractor and clean and text):
            dropped_missing += 1
            continue

        memory_answer = aliases[0]

        # Skip degenerate rows where context == memory (tokenizer-free check).
        if distractor.strip().lower() == memory_answer.strip().lower():
            dropped_degenerate += 1
            continue

        prompts.append(ConflictPrompt(
            text=text,
            clean_text=clean,
            prompt_type=prompt_type,
            domain=domain,
            memory_aliases=list(aliases),
            context_answer=distractor,
            memory_answer=memory_answer,
        ))

    if heldout_domain is not None and split != "all":
        if split == "train":
            prompts = [p for p in prompts if p.domain != heldout_domain]
        else:
            prompts = [p for p in prompts if p.domain == heldout_domain]

    print(
        f"[loader] prompt_type={prompt_type} | kept {len(prompts)} / {total} rows "
        f"(dropped missing: {dropped_missing}, degenerate: {dropped_degenerate})"
    )
    if heldout_domain is not None:
        print(f"[loader] heldout_domain={heldout_domain!r} | split={split!r}")

    return prompts


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

def load_model(
    name: str = "gpt2",
    device: Optional[str] = None,
    hf_model_name: Optional[str] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
) -> HookedTransformer:
    """Load a TransformerLens HookedTransformer.

    Args:
        name: TransformerLens architecture name (e.g. 'gpt2', 'meta-llama/Llama-3.2-3B').
        device: target device; defaults to CUDA if available, else CPU.
        hf_model_name: HF repo for a fine-tune sharing `name`'s architecture.
            Used for GPT-2 instruct variants; native-name models (Llama, Qwen,
            Gemma) don't need this.
        dtype: model dtype string ('bfloat16', 'float32') or torch.dtype.
            Default None keeps TransformerLens's default (float32). Use
            'bfloat16' for 3-4B+ models to fit in 24 GB.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch_dtype = None
    if dtype is not None:
        torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

    kwargs = {}
    if torch_dtype is not None:
        kwargs["dtype"] = torch_dtype

    hf_token = os.getenv("HF_TOKEN") or None  # None → anonymous (public models)

    # Log in at the huggingface_hub session level so that TransformerLens's
    # internal hf_hub_download picks up the token regardless of whether TL
    # forwards a token= kwarg itself.
    if hf_token:
        import huggingface_hub as _hfh
        _hfh.login(token=hf_token, add_to_git_credential=False)

    if hf_model_name is not None:
        from transformers import AutoModelForCausalLM, AutoTokenizer as _Tok
        hf_model  = AutoModelForCausalLM.from_pretrained(
            hf_model_name, token=hf_token,
            torch_dtype=torch_dtype,  # avoid fp32 peak when dtype=bfloat16
        )
        tokenizer = _Tok.from_pretrained(hf_model_name, token=hf_token)
        model = HookedTransformer.from_pretrained(
            name, hf_model=hf_model, tokenizer=tokenizer, **kwargs
        )
        assert len(tokenizer) == model.cfg.d_vocab, (
            f"vocab size mismatch: HF tokenizer has {len(tokenizer)} tokens "
            f"but TransformerLens config expects {model.cfg.d_vocab}."
        )
    else:
        # Don't pass token= here: newer TL already forwards it internally to
        # AutoModelForCausalLM.from_pretrained, causing a duplicate-kwarg error.
        # Auth is covered by the hf_hub.login() call above.
        model = HookedTransformer.from_pretrained(name, **kwargs)

    model = model.to(device)
    model.eval()
    # Explicit dtype cast: TL's from_pretrained may silently keep fp32 when a
    # reduced-precision dtype is requested. Re-cast here to guarantee bf16 on GPU
    # without a memory spike (PyTorch converts one tensor at a time, so peak ≈
    # fp32 model size, not fp32+bf16).
    if torch_dtype is not None:
        model = model.to(torch_dtype)
    # Freeze all parameters: we never optimise model weights — gradients flow
    # only through the detached hook_z leaf tensors. Without this, PyTorch
    # allocates a ~12 GB gradient buffer for a 3B model, causing OOM.
    model.requires_grad_(False)
    _actual = next(model.parameters()).dtype
    print(
        f"[load_model] dtype={_actual} n_layers={model.cfg.n_layers} "
        f"n_heads={model.cfg.n_heads} d_vocab={model.cfg.d_vocab}"
    )
    return model


def run_with_cache(
    model: HookedTransformer,
    prompt: ConflictPrompt,
    use_clean: bool = False,
    names_filter: Optional[str] = None,
):
    """Run a forward pass on the (conflict | clean) text and return (logits, cache)."""
    text = prompt.clean_text if use_clean else prompt.text
    kwargs = {} if names_filter is None else {"names_filter": names_filter}
    return model.run_with_cache(text, **kwargs)


# ---------------------------------------------------------------------------
# Multi-token log-prob scoring
# ---------------------------------------------------------------------------

def _answer_ids(model: HookedTransformer, answer: str) -> torch.Tensor:
    """Tokenize ' answer' with the model's own tokenizer. Returns 1-D id tensor.

    The leading space encodes the word-boundary convention used in most
    subword tokenizers (BPE and SentencePiece). For SentencePiece models
    (Llama, Gemma) verify that decode(encode(' answer')) == ' answer';
    if the round-trip fails the model is applying its own boundary logic
    and the leading space may need to be dropped.
    """
    text = " " + answer.strip()
    ids = model.to_tokens(text, prepend_bos=False)  # [1, k]
    return ids[0].to(torch.long)                     # [k]


def answer_log_prob(
    logits: torch.Tensor,
    answer_ids: torch.Tensor,
) -> float:
    """Mean log-prob over answer token positions (teacher-forced).

    Args:
        logits: shape [1, L_prompt + L_ans, vocab], from running
                model(prompt_text + ' ' + answer_text).
        answer_ids: 1-D integer tensor of length L_ans from _answer_ids().

    The logit at position i predicts the token at position i+1, so the
    positions predicting answer tokens a_0..a_{k-1} are
        L_prompt-1  through  L_prompt+k-2
    which is the slice  logits[0, -k-1 : -1, :].
    """
    k = len(answer_ids)
    log_probs = torch.log_softmax(logits[0, -k - 1:-1, :], dim=-1)  # [k, vocab]
    device = log_probs.device
    per_token = log_probs[torch.arange(k, device=device), answer_ids.to(device)]
    return per_token.mean().item()


def answer_log_probs(
    model: HookedTransformer,
    prompt_text: str,
    memory_answer: str,
    context_answer: str,
) -> Tuple[float, float]:
    """Return (memory_log_prob, context_log_prob) via mean teacher-forced log-prob.

    Two forward passes, one per answer. Each pass runs model over
    prompt_text + ' ' + answer and reads log-probs at the answer positions.
    """
    mem_ids = _answer_ids(model, memory_answer)
    ctx_ids = _answer_ids(model, context_answer)
    mem_text = prompt_text + " " + memory_answer.strip()
    ctx_text = prompt_text + " " + context_answer.strip()
    with torch.no_grad():
        mem_logits = model(mem_text)
        ctx_logits = model(ctx_text)
    return answer_log_prob(mem_logits, mem_ids), answer_log_prob(ctx_logits, ctx_ids)


# ---------------------------------------------------------------------------
# CRR — Contextual Reliance Rate
# ---------------------------------------------------------------------------

def compute_crr_logprob(model: HookedTransformer, prompts: List[ConflictPrompt]) -> float:
    """CRR via mean log-prob comparison (context vs memory, teacher-forced).

    Renamed from compute_crr (spec §B1) -- this is a cheap PROXY metric, not
    the behavioral metric the paper reports as "CRR". Kept because it's
    already used elsewhere in the pipeline and is far cheaper than
    generation. See compute_crr() below for the real behavioral metric.

    Higher = model prefers the injected context distractor over its
    parametric memory answer.
    """
    if not prompts:
        print("[crr] no prompts; returning 0.0")
        return 0.0

    context_count = memory_count = neither_count = 0

    for prompt in prompts:
        mem_lp, ctx_lp = answer_log_probs(
            model, prompt.text, prompt.memory_answer, prompt.context_answer
        )
        if ctx_lp > mem_lp:
            context_count += 1
        elif mem_lp > ctx_lp:
            memory_count += 1
        else:
            neither_count += 1

    total = len(prompts)
    crr = context_count / total
    print(
        f"[crr-logprob] CRR={crr:.3f} | context={context_count} | memory={memory_count} "
        f"| neither={neither_count} | total={total}"
    )
    return crr


def compute_crr(
    model: HookedTransformer,
    prompts: List[ConflictPrompt],
    max_new_tokens: int = 10,
) -> Dict[str, float]:
    """Behavioral CRR (spec §B1): greedy-generate from `prompt.text`, string-
    match the continuation against the memory ALIAS LIST vs the context
    distractor, with a genuine "neither" bucket (spec §B1.1, §B1.2).

    This is the metric the paper should call "CRR" -- compute_crr_logprob()
    is a cheap proxy, kept separately named.

    Returns {"crr": context_rate, "context": n, "memory": n, "neither": n,
    "total": n}. crr = context / total (0.0 if total == 0).
    """
    if not prompts:
        print("[crr] no prompts; returning zeroed result")
        return {"crr": 0.0, "context": 0, "memory": 0, "neither": 0, "total": 0}

    context_count = memory_count = neither_count = 0

    for prompt in prompts:
        tokens = model.to_tokens(prompt.text)
        with torch.no_grad():
            out_tokens = model.generate(
                tokens, max_new_tokens=max_new_tokens, do_sample=False, verbose=False,
            )
        continuation = model.to_string(out_tokens[0, tokens.shape[1]:]) if hasattr(model, "to_string") else ""
        continuation_lower = continuation.lower()

        matched_memory = any(alias.strip().lower() in continuation_lower for alias in prompt.memory_aliases)
        matched_context = prompt.context_answer.strip().lower() in continuation_lower

        if matched_context and not matched_memory:
            context_count += 1
        elif matched_memory and not matched_context:
            memory_count += 1
        elif matched_memory and matched_context:
            # Both strings appear (e.g. echoing the prompt) -- resolve by
            # whichever appears first in the continuation.
            ctx_pos = continuation_lower.find(prompt.context_answer.strip().lower())
            mem_positions = [
                continuation_lower.find(a.strip().lower()) for a in prompt.memory_aliases
                if a.strip().lower() in continuation_lower
            ]
            mem_pos = min(mem_positions) if mem_positions else -1
            if ctx_pos == -1 or (mem_pos != -1 and mem_pos < ctx_pos):
                memory_count += 1
            else:
                context_count += 1
        else:
            neither_count += 1

    total = len(prompts)
    crr = context_count / total
    print(
        f"[crr] CRR={crr:.3f} | context={context_count} | memory={memory_count} "
        f"| neither={neither_count} | total={total}"
    )
    return {
        "crr": crr, "context": context_count, "memory": memory_count,
        "neither": neither_count, "total": total,
    }


# ---------------------------------------------------------------------------
# Deprecated — kept for external callers until they migrate
# ---------------------------------------------------------------------------

def answer_logits(logits: torch.Tensor, prompt: ConflictPrompt) -> Tuple[float, float]:  # noqa: D401
    """DEPRECATED. Use answer_log_probs() instead.

    This function required memory_token_id / context_token_id on ConflictPrompt,
    which are no longer stored. Raises AttributeError on new-style prompts.
    """
    last = logits[0, -1]
    return last[prompt.memory_token_id].item(), last[prompt.context_token_id].item()
