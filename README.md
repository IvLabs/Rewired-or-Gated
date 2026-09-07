# Rewired or Gated? Instruction Tuning and Knowledge-Conflict Circuits

Code and data artifact for the paper *Rewired or Gated? How
Instruction Tuning Shapes Knowledge-Conflict Circuits in LLMs*.

**Models:** `Llama-3.2-3B{,-Instruct}`, `Qwen2.5-3B{,-Instruct}`, `gemma-3-4b-{pt,it}`.
**Data:** the public [`gaotang/ParaConflict`](https://huggingface.co/datasets/gaotang/ParaConflict)
knowledge-conflict dataset (loaded at runtime; no data is redistributed here).

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env     # optional: set HF_TOKEN only if HuggingFace rate-limits you
```

Python 3.10+. The heavy pipeline (attribution, patching, ablation) needs a
24 GB+ GPU and downloads the six models above from HuggingFace. **Reproducing the
figures and reported metrics needs no GPU** — it runs entirely off the committed
result JSON.

---

## Reproduce the paper's numbers and figures (CPU-only)

```bash
python analysis/compute_metrics.py      # CRR deltas, node-level & edge-level Jaccard/Spearman
python analysis/validate_node_overlap.py# independently re-derives the node-overlap table
python analysis/make_figures.py         # writes the 5 figure PDFs
```

`make_figures.py` reads only the committed JSON in `results/` and
`results_superposition_v2/`. Set `FIG_PNG_OUT=<dir>` to also dump PNG copies for
quick visual inspection.

### Note on the edge-attribution data

The raw EAP-IG edge graphs (`~10^5` edges/model, ~213 MB total) are **not**
shipped. Every edge-level number in the paper (top-{50,100,500,1000} Jaccard and
top-1000 Spearman) is computed only from the 1000 highest-magnitude edges, which
are committed as slim `results/<family>/eapig_<family>_<variant>_substitution_st_edges_topk.json`
(~90 KB each). `compute_metrics.py` reads these automatically. If you regenerate
the raw graphs yourself, `analysis/extract_edge_topk.py` reproduces the slim files.

---

## Re-run the full pipeline (GPU)

Model-agnostic drivers, each writing to `results/<family>/`:

```bash
python run_all_lds.py            # LDS/EAP node attribution (3 flavors) + CRR, all families
python run_all_pathpatching.py   # path-patching causal verification of top heads
bash   run_superposition_all.sh  # superposition index + causal ablation, all families
```

Per-family entry points (`run_llama32.py`, `run_gemma3.py`,
`singleton_fix/run_qwen25.py`) drive the same `run_lds → run_pp_topheads →
triangulation` chain and emit `results/<family>/<family>_summary_st.json`.

---

## Repository layout

### Core method (repo root)

| File | Role |
|------|------|
| `contract.py` | Frozen data contract: every method consumes `ConflictPrompt`, returns `HeadScores`/`EdgeScores`. |
| `foundation.py` | Dataset loader (ParaConflict → `ConflictPrompt`), model wrapper, `compute_crr`. |
| `lds_attribution.py` | Node-level gradient attribution over the signed context−memory log-prob margin (flavors: `gradnorm`, `gradact`, `eap`). |
| `lds.py` | Supporting LDS routines. |
| `triangulation.py` | Merge A: compares LDS flavors against path patching (the causal gold standard). |
| `run_lds.py`, `run_all_lds.py` | CLI / all-family drivers for LDS + CRR. |
| `run_pp_topheads.py`, `run_all_pathpatching.py` | Path patching on EAP-selected top heads. |
| `run_llama32.py`, `run_gemma3.py` | Per-family EAP-screen → PP-verify pipelines. |
| `run_superposition_all.sh` | Superposition + ablation across all families. |
| `save_kept_indices.py` | Filter-only pass that caches kept prompt indices for reuse. |

### Method modules

| Directory | Contents |
|-----------|----------|
| `eap_ig/` | Edge Attribution Patching with Integrated Gradients — edge-level circuit attribution (`eap_ig.py`, `eap_ig_single_token.py`, runners, tests). |
| `path_patching/` | Activation patching at `hook_z` (per-head, pre-OV) for causal verification (`path_patching.py`, tests). |
| `superposition/` | Per-head memory/context superposition index and the causal ablation harness (`superposition_v2.py` is the current version; runners, metrics, tests). |
| `singleton_fix/` | Single-token-answer filtering, contrast-pair construction, provenance, and the Qwen driver (`run_qwen25.py`). |
| `analysis/` | `compute_metrics.py` (all reported metrics), `make_figures.py` (paper figures), `validate_node_overlap.py` (independent table check), `extract_edge_topk.py` (slim edge-data extractor). |

### Committed results

`results/<family>/` (`family` ∈ `llama32_3b`, `qwen25_3b`, `gemma3_4b`):

| File pattern | Content |
|--------------|---------|
| `crr_<family>_<variant>_<prompttype>_st.json` | Contextual Reliance Rate (behavioral + log-prob), per variant × prompt type. |
| `lds2_<family>_<variant>_<prompttype>_st_<scorer>[_shuffled].json` | Node attribution scores; `<scorer>` ∈ `eap`/`gradact`/`gradnorm`; `_shuffled` = label-shuffled noise floor. |
| `pp_topheads_<family>_<variant>_<prompttype>_st.json` | Path-patching results on the top heads. |
| `eapig_<family>_<variant>_substitution_st_edges_topk.json` | Slim top-1000 edge summary (see note above). |
| `single_token_survival_<family>_<variant>.json` | Single-token-answer filter survival stats. |
| `<family>_summary_st.json` | Per-family run summary. |

`results_superposition_v2/<fam>/` (`fam` ∈ `llama`, `qwen`, `gemma`):
`superposition_<fam>_<variant>.json` (per-head indices), `comparison_<fam>.json`
(base-vs-instruct node overlap + role flips), `ablation_<fam>_<variant>.json`
(six causal interventions), `verify_dla_<fam>.json` (direct-logit-attribution check).

---

## Tests

```bash
pytest
```

Covers the data contract, CRR, LDS attribution, path-patching head selection,
single-token filtering, and provenance.
