"""Stage 2 inference: packed pairwise coref scoring + complete-linkage clustering."""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from src.common.datatypes import Document, Entity, Mention
from src.common.schema import stage2_rows

MAX_SCORED_PAIRS = 5000  # safety valve for noisy zero-shot mention over-prediction


def cluster_mentions(engine, doc: Document, mentions: list[Mention],
                     alpha: float = 0.5) -> list[Entity]:
    """Score same-type pairs and agglomerate with complete linkage at cutoff alpha.

    Mentions are tagged once in the document and many pair questions share one
    forward pass. Exact-string same-type pairs are linked without a model call
    (distance 0). Pairs beyond MAX_SCORED_PAIRS keep distance 1.0.
    """
    n = len(mentions)
    if n == 0:
        return []
    if n == 1:
        return [Entity(idx=0, mentions=list(mentions))]
    mentions = sorted(mentions, key=lambda m: m.span)

    D = np.ones((n, n))
    np.fill_diagonal(D, 0.0)
    to_score = []
    for a in range(n):
        for b in range(a + 1, n):
            ma, mb = mentions[a], mentions[b]
            if ma.type != mb.type:
                continue  # cross-type distance stays 1.0
            if ma.text.lower() == mb.text.lower():
                D[a, b] = D[b, a] = 0.0  # identical surface form: same entity
            else:
                to_score.append((a, b))
    to_score = to_score[:MAX_SCORED_PAIRS]

    rows = stage2_rows(doc, mentions, to_score)
    for row, out in zip(rows, engine.predict_rows(rows, desc=f"  S2 rows ({n} mentions)")):
        for a, b in row["meta"]["pairs"]:
            p_same = out[f"c_{a}_{b}"]["probabilities"].get("true", 0.0)
            D[a, b] = D[b, a] = 1.0 - p_same

    Z = linkage(squareform(D, checks=False), method="complete")
    labels = fcluster(Z, t=alpha, criterion="distance")
    clusters: dict[int, list[Mention]] = {}
    for i, lab in enumerate(labels):
        clusters.setdefault(int(lab), []).append(mentions[i])
    return [Entity(idx=i, mentions=ms) for i, ms in enumerate(clusters.values())]
