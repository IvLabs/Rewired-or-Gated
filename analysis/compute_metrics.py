"""
Consolidated analysis of BBNLP knowledge-conflict experiments against the
proposal's hypotheses (H0 / H1 / H2 / H2a).

Reads result JSONs under results/ and results_superposition_v2/ and emits every
proposal-relevant metric: CRR deltas, node-level LDS Jaccard/Spearman, edge-level
EAP-IG Jaccard, EAP-vs-PP triangulation, superposition role shifts, and ablation
causal validation. Pure stdlib + numpy (no scipy).
"""
import json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAMS = ["llama32_3b", "qwen25_3b", "gemma3_4b"]
SUP = {"llama32_3b": "llama", "qwen25_3b": "qwen", "gemma3_4b": "gemma"}
PTYPES = ["substitution", "coherent"]


def L(p):
    p = os.path.join(ROOT, p)
    return json.load(open(p)) if os.path.exists(p) else None


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def topk(scores, k):
    return set(sorted(scores, key=lambda h: abs(scores[h]), reverse=True)[:k])


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 0.0


def sec(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# ---------------------------------------------------------------- 1. CRR
sec("1. BEHAVIORAL — Contextual Reliance Rate (CRR)")
crr = {}
for fam in FAMS:
    for pt in PTYPES:
        for var in ["base", "instruct"]:
            d = L(f"results/{fam}/crr_{fam}_{var}_{pt}_st.json")
            if d:
                crr[(fam, pt, var)] = d["crr_behavioral"]
print(f"{'family':11s} {'ptype':13s} {'base':>6s} {'inst':>6s} {'Δ(i-b)':>7s}  neither_b/i")
for fam in FAMS:
    for pt in PTYPES:
        b = crr.get((fam, pt, "base"))
        i = crr.get((fam, pt, "instruct"))
        if b and i:
            print(f"{fam:11s} {pt:13s} {b['crr']:6.3f} {i['crr']:6.3f} {i['crr']-b['crr']:+7.3f}  "
                  f"{b['neither']}/{i['neither']}  (N_b={b['total']} N_i={i['total']})")

# ---------------------------------------------------------------- 2. Node LDS
sec("2. NODE-LEVEL LDS — base vs instruct overlap (primary flavor = eap)")
node = {}
print(f"{'family':11s} {'ptype':13s} {'J@10':>5s} {'J@20':>5s} {'rho_signed':>10s} {'rho_abs':>8s}")
for fam in FAMS:
    for pt in PTYPES:
        b = L(f"results/{fam}/lds2_{fam}_base_{pt}_st_eap.json")
        i = L(f"results/{fam}/lds2_{fam}_instruct_{pt}_st_eap.json")
        if not b or not i:
            continue
        bs, is_ = b["scores"], i["scores"]
        common = sorted(set(bs) & set(is_))
        rho = spearman([bs[h] for h in common], [is_[h] for h in common])
        arho = spearman([abs(bs[h]) for h in common], [abs(is_[h]) for h in common])
        j10 = jaccard(topk(bs, 10), topk(is_, 10))
        j20 = jaccard(topk(bs, 20), topk(is_, 20))
        node[(fam, pt)] = dict(j10=j10, j20=j20, rho=rho, arho=arho)
        print(f"{fam:11s} {pt:13s} {j10:5.3f} {j20:5.3f} {rho:10.3f} {arho:8.3f}")

# ---------------------------------------------------------------- 3. Edge EAP-IG
sec("3. EDGE-LEVEL EAP-IG — base vs instruct top-k edge Jaccard (substitution)")
# NOTE: J@50 is the proposal's *operative* cut-off for the four-way taxonomy (§7).
# J@1000 is deep in the tail where any two large graphs diverge — do NOT use it to
# decide "edge-low." See analysis/EAP_IG_analysis.md §9 and the corrected results doc.
edge = {}
print(f"{'family':11s} {'n_edges':>8s} {'J@50':>6s} {'J@100':>6s} {'J@500':>6s} {'J@1000':>7s} {'rho_top1000':>11s}")
def load_edges(fam, var):
    """Prefer the slim top-k edge summary; fall back to the full graph.

    The slim ``*_edges_topk.json`` (see analysis/extract_edge_topk.py) carries
    the same ``top_k_edges`` + ``n_edges`` used below, so edge metrics are
    identical whether or not the 213 MB raw graphs are present.
    """
    return (L(f"results/{fam}/eapig_{fam}_{var}_substitution_st_edges_topk.json")
            or L(f"results/{fam}/eapig_{fam}_{var}_substitution_st_full.json"))


for fam in FAMS:
    b = load_edges(fam, "base")
    i = load_edges(fam, "instruct")
    if not b or not i:
        continue
    be = {e["src"] + "->" + e["dst"]: e["score"] for e in b["top_k_edges"]}
    ie = {e["src"] + "->" + e["dst"]: e["score"] for e in i["top_k_edges"]}

    def edge_topk(d, k):
        return set(sorted(d, key=lambda e: abs(d[e]), reverse=True)[:k])
    j50 = jaccard(edge_topk(be, 50), edge_topk(ie, 50))
    j100 = jaccard(edge_topk(be, 100), edge_topk(ie, 100))
    j500 = jaccard(edge_topk(be, 500), edge_topk(ie, 500))
    j1000 = jaccard(set(be), set(ie))
    common = sorted(set(be) & set(ie))
    rho = spearman([be[e] for e in common], [ie[e] for e in common]) if len(common) > 2 else float("nan")
    edge[fam] = dict(j50=j50, j100=j100, j500=j500, j1000=j1000, rho=rho, n=b["n_edges"])
    print(f"{fam:11s} {b['n_edges']:8d} {j50:6.3f} {j100:6.3f} {j500:6.3f} {j1000:7.3f} {rho:11.3f}")

# ---------------------------------------------------------------- 4. Triangulation
sec("4. TRIANGULATION — EAP (LDS) vs Path-Patching agreement (from summary)")
print(f"{'family':11s} {'variant':8s} {'spearman':>9s} {'p':>8s} {'sign_agr':>9s} {'gate':>5s}")
tri = {}
for fam in FAMS:
    s = L(f"results/{fam}/{fam}_summary_st.json")
    if not s:
        continue
    for mtag, mdata in s.get("models", {}).items():
        cell = mdata.get("cells", {}).get("substitution")
        if not cell:
            continue
        var = "instruct" if "instruct" in mtag else "base"
        tri[(fam, var)] = cell
        print(f"{fam:11s} {var:8s} {cell.get('spearman_rho'):9.3f} "
              f"{cell.get('p_value'):8.4f} {str(cell.get('sign_agreement')):>9s} {str(cell.get('gate_pass')):>5s}")

# ---------------------------------------------------------------- 5. Superposition
sec("5. SUPERPOSITION — role shift base->instruct (precomputed comparison)")
print(f"{'family':8s} {'J_ctx':>6s} {'J_mem':>6s} {'rho':>6s} {'verdict':>28s} {'n_role_chg':>10s} {'mean_Δidx':>9s}")
sup = {}
for fam, sfam in SUP.items():
    c = L(f"results_superposition_v2/{sfam}/comparison_{sfam}.json")
    if not c:
        continue
    no = c.get("_node_overlap", {})
    sm = c.get("_summary", {})
    sup[fam] = dict(no=no, sm=sm)
    print(f"{sfam:8s} {no.get('jaccard_context_topk'):6.3f} {no.get('jaccard_memory_topk'):6.3f} "
          f"{no.get('spearman_full_ranking'):6.3f} {no.get('interpretation','?')[:28]:>28s} "
          f"{sm.get('n_role_changed'):10d} {sm.get('mean_delta_index'):9.4f}")

# ---------------------------------------------------------------- 6. Ablation
sec("6. ABLATION — causal validation of LDS heads (ΔCRR direction)")
for fam, sfam in SUP.items():
    for var in ["base", "instruct"]:
        a = L(f"results_superposition_v2/{sfam}/ablation_{sfam}_{var}.json")
        if not a:
            continue
        conds = a["conditions"]
        nmatch = sum(1 for c in conds.values() if c.get("matched_expectation"))
        print(f"\n  {sfam} {var}: baseline_CRR={a['baseline_crr']:.3f}  "
              f"[{nmatch}/{len(conds)} conditions matched expectation]")
        for name, c in conds.items():
            print(f"     {name:22s} mode={c['ablation_mode']:8s} ΔCRR={c['delta_crr']:+.3f} "
                  f"match={c['matched_expectation']}")

# ---------------------------------------------------------------- 7. Four-way taxonomy
sec("7. FOUR-WAY TAXONOMY placement (substitution)")
# DECISION RULE (see analysis/EAP_IG_analysis.md §6 ⚠️ box, and the corrected
# results doc). The axes are NOT symmetric:
#   * The NODE axis decides GATING (H2) vs REWIRING (H1). Node overlap HIGH (>0.6)
#     => the same heads still matter => gating. Node overlap LOW (<0.3) => the heads
#     themselves changed => rewiring.
#   * The EDGE axis (read at the OPERATIVE cut-off J@50, NOT J@1000) only
#     SUB-CLASSIFIES a gating result: edge HIGH (>0.6) => H2-strict (wiring
#     preserved); edge LOW (<0.3) => H2-rewired-edges (wiring changed); edge MID
#     (0.3-0.6) => gating, sub-type ambiguous.
#   * The CRR axis rules H0 in/out: a small CRR shift => H0 (nothing really changed),
#     regardless of the node/edge overlap.
# Node overlap here = mean of the top-10 context & memory head Jaccard from the
# superposition comparison (the authoritative node-overlap used in the writeups),
# falling back to LDS J@20 if that file is absent.
PROPOSAL_HI, PROPOSAL_LO = 0.6, 0.3   # proposal bands for "high"/"low" overlap
CRR_LARGE = 0.15                       # |ΔCRR| threshold for a real behavioral shift


def band(x):
    if x != x:            # NaN
        return "n/a"
    return "HIGH" if x > PROPOSAL_HI else ("LOW" if x < PROPOSAL_LO else "MID")


print(f"{'family':11s} {'nodeMeanJ':>9s} {'(band)':>6s} {'edgeJ@50':>8s} {'(band)':>6s} "
      f"{'|ΔCRR|':>7s}  verdict")
for fam in FAMS:
    no = sup.get(fam, {}).get("no", {})
    jc, jm = no.get("jaccard_context_topk"), no.get("jaccard_memory_topk")
    if jc is not None and jm is not None:
        nj = (jc + jm) / 2.0
    else:                 # fallback: LDS top-20 node Jaccard
        nj = node.get((fam, "substitution"), {}).get("j20", float("nan"))
    ej = edge.get(fam, {}).get("j50", float("nan"))
    b = crr.get((fam, "substitution", "base"))
    i = crr.get((fam, "substitution", "instruct"))
    dcrr = abs(i["crr"] - b["crr"]) if b and i else float("nan")
    nb, eb, crr_lg = band(nj), band(ej), (dcrr >= CRR_LARGE)

    if not crr_lg:
        v = "H0 (null: no real behavioral change)"
    elif nb == "LOW":
        v = "H1 (genuine rewiring: heads changed)"
    else:                 # node HIGH or MID => gating; edge picks the sub-type
        gating = "gating (H2)"
        if eb == "HIGH":
            v = f"{gating} => H2-strict (wiring preserved)"
        elif eb == "LOW":
            v = f"{gating} => H2-rewired-edges (wiring changed)"
        else:
            v = f"{gating} => sub-type ambiguous (edge MID)"
    print(f"{fam:11s} {nj:9.3f} {nb:>6s} {ej:8.3f} {eb:>6s} {dcrr:7.3f}  {v}")
