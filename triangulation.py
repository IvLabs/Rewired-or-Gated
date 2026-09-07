"""
Merge A — Triangulation of gradient attribution (LDS family) vs. Path Patching.

Compares Path Patching (causal gold standard) against the three flavors produced by
lds_attribution.py, across two prompt sets:

  flavors : gradnorm (unsigned importance), gradact (signed, zero baseline),
            eap (signed, clean-vs-conflict baseline — the principled PP match)
  sets    : full (all single-token prompts), aligned (exactly PP's kept prompts)

Headline = EAP on the aligned set, signed-PP vs signed-EAP Spearman. Bar: rho > 0.5.

Comparison convention:
  - Signed flavors (gradact, eap) are compared to SIGNED PP directly — both share the
    context_minus_memory direction, so a positive score means "context-following" on
    both sides. This is the meaningful directional agreement.
  - The unsigned flavor (gradnorm) is compared to |PP| (importance vs importance).
  - We also report |PP| vs |LDS| and a PP-top-30-restricted Spearman, because the
    full 144-head Spearman is diluted by ~130 near-zero "noise" heads.

Pure stdlib. Spearman verified equal to scipy.stats.spearmanr to 4 dp.

Run:  python triangulation.py   (after lds_attribution.py has produced lds2_* files)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

RESULTS = Path("results")
HeadKey = Tuple[int, int]


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def load_flat(path: Path) -> Dict[HeadKey, float]:
    raw = json.loads(path.read_text())
    return {(int(k.split("_")[0]), int(k.split("_")[1])): float(v) for k, v in raw.items()}


def load_pp(path: Path) -> Tuple[Dict[HeadKey, float], int]:
    raw = json.loads(path.read_text())
    hs = raw["head_scores"]
    scores = {(int(k.split("_")[0]), int(k.split("_")[1])): float(v) for k, v in hs.items()}
    return scores, int(raw.get("n_prompts_used", 0))


# --------------------------------------------------------------------------- #
# Stats (stdlib; Spearman verified against scipy)
# --------------------------------------------------------------------------- #
def _avg_ranks(values: List[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(x: List[float], y: List[float]) -> float:
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = sum((a - mx) ** 2 for a in x) ** 0.5
    dy = sum((b - my) ** 2 for b in y) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def spearman(a: Dict[HeadKey, float], b: Dict[HeadKey, float], keys=None) -> float:
    keys = sorted(set(a) & set(b)) if keys is None else list(keys)
    if len(keys) < 3:
        return float("nan")
    return _pearson(_avg_ranks([a[k] for k in keys]), _avg_ranks([b[k] for k in keys]))


def top_k(scores: Dict[HeadKey, float], k: int, by_abs: bool = True) -> List[HeadKey]:
    key = (lambda h: -abs(scores[h])) if by_abs else (lambda h: -scores[h])
    return sorted(scores, key=key)[:k]


def jaccard(a, b) -> float:
    sa, sb = set(a), set(b)
    u = sa | sb
    return len(sa & sb) / len(u) if u else float("nan")


def fmt(x) -> str:
    return "nan" if x != x else f"{x:+.3f}"


# --------------------------------------------------------------------------- #
# One comparison: PP vs one LDS flavor/set
# --------------------------------------------------------------------------- #
def compare(pp: Dict[HeadKey, float], lds: Dict[HeadKey, float], signed: bool) -> dict:
    pp_abs = {h: abs(v) for h, v in pp.items()}
    lds_abs = {h: abs(v) for h, v in lds.items()}

    # Headline: signed-vs-signed for directional flavors, |.|-vs-|.| for unsigned.
    headline = spearman(pp, lds) if signed else spearman(pp_abs, lds)

    pp_top30 = top_k(pp, 30, by_abs=True)
    restricted = spearman(pp, lds, keys=pp_top30) if signed else spearman(pp_abs, lds_abs, keys=pp_top30)

    out = {
        "headline_rho": headline,
        "signedPP_vs_LDS": spearman(pp, lds),
        "absPP_vs_absLDS": spearman(pp_abs, lds_abs),
        "restricted_PPtop30_rho": restricted,
        "jaccard": {},
        "directional_overlap": {},
    }
    for k in (10, 20, 50):
        out["jaccard"][str(k)] = {
            "value": jaccard(top_k(pp_abs, k), top_k(lds_abs, k)),
            "shared": len(set(top_k(pp_abs, k)) & set(top_k(lds_abs, k))),
        }
    if signed:
        # agreement on the actual top context-following and top memory-protecting heads
        for direction, key in [("context", lambda d: -d), ("memory", lambda d: d)]:
            pp_top = [h for h in sorted(pp, key=lambda h: key(pp[h]))[:10]]
            lds_top = [h for h in sorted(lds, key=lambda h: key(lds[h]))[:10]]
            out["directional_overlap"][direction] = len(set(pp_top) & set(lds_top))
        union = set(top_k(pp, 20)) | set(top_k(lds, 20))
        out["sign_agreement"] = {
            "agree": sum(1 for h in union if (pp[h] > 0) == (lds[h] > 0)),
            "total": len(union),
        }
    return out


# --------------------------------------------------------------------------- #
# Noise floor (real vs shuffled-label) for a flavor, if shuffled file exists
# --------------------------------------------------------------------------- #
def noise_floor(real_path: Path) -> str:
    shuf_path = real_path.with_name(real_path.stem + "_shuffled.json")
    if not shuf_path.exists():
        return "n/a"
    real, shuf = load_flat(real_path), load_flat(shuf_path)
    ov = len(set(top_k(real, 10)) & set(top_k(shuf, 10)))
    return f"{ov}/10"


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
FLAVORS = [("eap", True), ("gradact", True), ("gradnorm", False)]
PROMPT_TYPES = ["substitution", "coherent"]
SETS = ["aligned", "full"]
# (model_tag, pp_filename_infix)
MODELS = [("gpt2_base", "gpt2"), ("gpt2_instruct", "gpt2_instruct")]


def main() -> None:
    results: List[dict] = []
    for model_tag, pp_infix in MODELS:
        for pt in PROMPT_TYPES:
            pp_path = RESULTS / f"pp_{pp_infix}_{pt}.json"
            if not pp_path.exists():
                print(f"[merge-a] no PP file for {model_tag}/{pt}, skip")
                continue
            pp, pp_n = load_pp(pp_path)
            for st in SETS:
                for flavor, signed in FLAVORS:
                    lds_path = RESULTS / f"lds2_{model_tag}_{pt}_{st}_{flavor}.json"
                    if not lds_path.exists():
                        continue
                    lds = load_flat(lds_path)
                    rec = {
                        "model": model_tag, "prompt_type": pt, "set": st,
                        "flavor": flavor, "signed": signed,
                        "pp_n_prompts": pp_n, "lds_file": lds_path.name,
                        "lds_noise_floor": noise_floor(lds_path),
                    }
                    rec.update(compare(pp, lds, signed))
                    results.append(rec)
                    tag = f"{model_tag}/{pt}/{st}/{flavor}"
                    print(f"[merge-a] {tag:48s} headline rho={fmt(rec['headline_rho'])}  "
                          f"restricted={fmt(rec['restricted_PPtop30_rho'])}  "
                          f"noise={rec['lds_noise_floor']}")

    (RESULTS / "merge_a_triangulation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_markdown(results)
    print("[merge-a] wrote results/merge_a_triangulation.{json,md}")


def write_markdown(results: List[dict]) -> None:
    L: List[str] = ["# Merge A — Gradient Attribution vs. Path Patching\n"]
    L.append("Headline = **EAP / aligned** (signed-PP vs signed-EAP Spearman). Bar: rho > 0.5.\n")

    # headline summary
    L.append("## Headline (EAP flavor, aligned prompt set)\n")
    L.append("| model | prompt type | headline rho | PP-top30 rho | top-10 Jaccard | noise floor | verdict |")
    L.append("|---|---|---|---|---|---|---|")
    for r in results:
        if r["flavor"] == "eap" and r["set"] == "aligned":
            j10 = r["jaccard"]["10"]
            verdict = "PASS" if r["headline_rho"] > 0.5 else "below bar"
            L.append(f"| {r['model']} | {r['prompt_type']} | **{fmt(r['headline_rho'])}** | "
                     f"{fmt(r['restricted_PPtop30_rho'])} | "
                     f"{j10['shared']}/10 | {r['lds_noise_floor']} | {verdict} |")
    L.append("")

    # full breakdown — split by model
    for model_tag in ("gpt2_base", "gpt2_instruct"):
        model_rows = [r for r in results if r["model"] == model_tag]
        if not model_rows:
            continue
        L.append(f"## Full breakdown — {model_tag}\n")
        L.append("| prompt | set | flavor | headline rho | PP-top30 rho | |PP|v|LDS| | "
                 "J@10 | J@20 | ctx/mem overlap | sign-agree | noise |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in model_rows:
            do = r.get("directional_overlap", {})
            dov = f"{do.get('context','-')}/{do.get('memory','-')}" if do else "—"
            sa = r.get("sign_agreement")
            sav = f"{sa['agree']}/{sa['total']}" if sa else "—"
            L.append(
                f"| {r['prompt_type']} | {r['set']} | {r['flavor']} | "
                f"{fmt(r['headline_rho'])} | {fmt(r['restricted_PPtop30_rho'])} | "
                f"{fmt(r['absPP_vs_absLDS'])} | "
                f"{r['jaccard']['10']['shared']}/10 | {r['jaccard']['20']['shared']}/20 | "
                f"{dov} | {sav} | {r['lds_noise_floor']} |"
            )
        L.append("")

    L.append("- **headline rho**: signed-vs-signed for eap/gradact; |PP|-vs-gradnorm for gradnorm.")
    L.append("- **PP-top30 rho**: Spearman restricted to PP's 30 most important heads "
             "(the dilution-free view).")
    L.append("- **ctx/mem overlap**: shared heads in the top-10 context-following / "
             "top-10 memory-protecting sets (signed flavors only).")
    L.append("- **sign-agree**: of heads in either method's top-20, how many agree on "
             "context-vs-memory sign.")
    L.append("- **noise floor**: top-10 overlap between real and shuffled-label runs "
             "(chance ~0.7; high => direction not captured).\n")

    (RESULTS / "merge_a_triangulation.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
