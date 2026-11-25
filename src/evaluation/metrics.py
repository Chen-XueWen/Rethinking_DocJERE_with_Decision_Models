"""Metric primitives: span F1, coref metrics (MUC/B3/CEAF-e), RE F1, strict E2E."""
from __future__ import annotations

from collections import Counter

from scipy.optimize import linear_sum_assignment


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "tp": tp, "fp": fp, "fn": fn}


# --- NER span F1 -------------------------------------------------------------

def span_f1(pred_spans: list[set], gold_spans: list[set]) -> dict:
    """Micro P/R/F1 over (doc_start, doc_end, type) tuples, per document lists."""
    tp = fp = fn = 0
    for p, g in zip(pred_spans, gold_spans):
        tp += len(p & g)
        fp += len(p - g)
        fn += len(g - p)
    return prf(tp, fp, fn)


# --- Coreference metrics ------------------------------------------------------
# Clusters are lists of frozensets of mention spans (doc-level token offsets).

def muc(pred: list[frozenset], gold: list[frozenset]) -> tuple[float, float, float]:
    def score(key, response):
        num = den = 0
        for k in key:
            if len(k) < 2:
                continue
            parts = {frozenset(r & k) for r in response if r & k}
            covered = set().union(*parts) if parts else set()
            singles = len(k - covered)
            num += len(k) - (len(parts) + singles)
            den += len(k) - 1
        return num / den if den else 0.0
    r = score(gold, pred)
    p = score(pred, gold)
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def b_cubed(pred: list[frozenset], gold: list[frozenset]) -> tuple[float, float, float]:
    def score(key, response):
        total = n = 0
        for k in key:
            for m in k:
                r = next((c for c in response if m in c), frozenset())
                total += len(k & r) / len(k)
                n += 1
        return total / n if n else 0.0
    r = score(gold, pred)
    p = score(pred, gold)
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def ceaf_e(pred: list[frozenset], gold: list[frozenset]) -> tuple[float, float, float]:
    if not pred or not gold:
        return 0.0, 0.0, 0.0
    import numpy as np
    sim = np.zeros((len(gold), len(pred)))
    for i, g in enumerate(gold):
        for j, p in enumerate(pred):
            sim[i, j] = 2 * len(g & p) / (len(g) + len(p))
    ri, ci = linear_sum_assignment(-sim)
    total = sim[ri, ci].sum()
    p = total / len(pred)
    r = total / len(gold)
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def coref_metrics(pred_clusters: list[list[frozenset]],
                  gold_clusters: list[list[frozenset]]) -> dict:
    """Macro-average MUC, B-cubed, CEAF-e over documents, plus their mean (CoNLL)."""
    out = {}
    for name, fn in (("muc", muc), ("b3", b_cubed), ("ceaf_e", ceaf_e)):
        ps, rs, fs = [], [], []
        for p, g in zip(pred_clusters, gold_clusters):
            pp, rr, ff = fn(p, g)
            ps.append(pp); rs.append(rr); fs.append(ff)
        n = max(len(fs), 1)
        out[name] = {"precision": round(sum(ps) / n, 4),
                     "recall": round(sum(rs) / n, 4),
                     "f1": round(sum(fs) / n, 4)}
    out["conll_f1"] = round(sum(out[m]["f1"] for m in ("muc", "b3", "ceaf_e")) / 3, 4)
    return out


# --- Relation extraction (gold entities) -------------------------------------

def re_f1(pred: list[tuple], gold: list[tuple],
          in_train: set | None = None) -> dict:
    """DocRED micro F1 + Ign-F1 over (doc_id, h_idx, t_idx, r) tuples.

    Official Ign-F1 (DocRED gen_train_facts/evaluate): precision drops correct
    predictions whose name-level fact appears in the training annotation from
    both numerator and denominator; recall stays the plain recall.
    `in_train` is the set of tuples whose fact was seen in training.
    """
    ps, gs = set(pred), set(gold)
    correct = ps & gs
    base = prf(len(correct), len(ps - gs), len(gs - ps))
    out = {"micro": base}
    if in_train is not None:
        correct_in_train = len(correct & in_train)
        denom_p = len(ps) - correct_in_train
        ign_p = (len(correct) - correct_in_train) / denom_p if denom_p else 0.0
        r = base["recall"]
        ign_f = 2 * ign_p * r / (ign_p + r) if ign_p + r else 0.0
        out["ign"] = {"precision": round(ign_p, 4), "recall": r,
                      "f1": round(ign_f, 4)}
    return out


# --- Strict end-to-end (JEREX strict) ----------------------------------------

def jerex_strict_entity_match(pred_entity: dict, gold_entity: dict) -> bool:
    """JEREX strict: identical mention span sets AND same entity type."""
    pm = {(m["doc_start"], m["doc_end"]) for m in pred_entity["mentions"]}
    gm = {(m["doc_start"], m["doc_end"]) for m in gold_entity["mentions"]}
    return pm == gm and pred_entity["type"] == gold_entity["type"]


def strict_e2e(pred_docs: list[dict], gold_docs: list[dict]) -> dict:
    """Strict E2E micro + macro(by relation) P/R/F1.

    A predicted triple is TP iff its head and tail entities JEREX-strict match
    distinct gold entities and the relation matches a gold triple over those
    gold entities. Each gold triple is creditable at most once.
    """
    tp = fp = fn = 0
    by_rel_tp: Counter = Counter()
    by_rel_fp: Counter = Counter()
    by_rel_fn: Counter = Counter()

    for pd, gd in zip(pred_docs, gold_docs):
        # map each predicted entity to the gold entity it strictly matches
        mapping: dict[int, int] = {}
        used_gold: set[int] = set()
        for pi, pe in enumerate(pd["entities"]):
            for gi, ge in enumerate(gd["entities"]):
                if gi in used_gold:
                    continue
                if jerex_strict_entity_match(pe, ge):
                    mapping[pi] = gi
                    used_gold.add(gi)
                    break

        gold_triples = {(t["h"], t["t"], t["r"]) for t in gd["triples"]}
        matched: set[tuple] = set()
        for t in pd["triples"]:
            key = None
            if t["h"] in mapping and t["t"] in mapping:
                cand = (mapping[t["h"]], mapping[t["t"]], t["r"])
                if cand in gold_triples and cand not in matched:
                    key = cand
            if key:
                matched.add(key)
                tp += 1
                by_rel_tp[t["r"]] += 1
            else:
                fp += 1
                by_rel_fp[t["r"]] += 1
        for g in gold_triples - matched:
            fn += 1
            by_rel_fn[g[2]] += 1

    rels = set(by_rel_tp) | set(by_rel_fp) | set(by_rel_fn)
    per_rel = [prf(by_rel_tp[r], by_rel_fp[r], by_rel_fn[r]) for r in rels]
    n = max(len(per_rel), 1)
    return {
        "micro": prf(tp, fp, fn),
        "macro_by_relation": {
            "precision": round(sum(x["precision"] for x in per_rel) / n, 4),
            "recall": round(sum(x["recall"] for x in per_rel) / n, 4),
            "f1": round(sum(x["f1"] for x in per_rel) / n, 4),
        },
    }
