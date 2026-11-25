"""Stage 3 inference: packed gate pass over all pairs, 97-way only for survivors."""
from __future__ import annotations

from src.common.datatypes import Document, Entity, RelationTriple
from src.common.schema import stage3_gate_rows, stage3_relation_rows

MAX_RELATION_PAIRS = 150  # cap on 97-way calls per doc (highest gate scores first)


def extract_relations(engine, doc: Document, entities: list[Entity],
                      tau_gate: float = 0.5, tau_rel: float = 0.5,
                      multi_label: bool = True) -> list[RelationTriple]:
    """Entity indices in the output are positions in `entities`.

    1. Gate: one binary question per directed pair, packed many per pass.
       A pair is pruned if P(no relation) >= tau_gate.
    2. Relation: one 97-way question per surviving pair (the expensive pass,
       ~0.9 s each, which is why it only runs on survivors).
    """
    n = len(entities)
    pairs = [(h, t) for h in range(n) for t in range(n) if h != t]
    if not pairs:
        return []

    gate_rows = stage3_gate_rows(doc, entities, pairs)
    p_rel: dict[tuple[int, int], float] = {}
    for row, out in zip(gate_rows, engine.predict_rows(gate_rows, desc=f"  S3 gate ({n} ents)")):
        for h, t in row["meta"]["pairs"]:
            p_rel[(h, t)] = out[f"g_{h}_{t}"]["probabilities"].get("true", 0.0)

    survivors = [p for p in pairs if (1.0 - p_rel[p]) < tau_gate]
    survivors = sorted(survivors, key=lambda p: -p_rel[p])[:MAX_RELATION_PAIRS]

    triples: list[RelationTriple] = []
    rel_rows = stage3_relation_rows(doc, entities, survivors)
    for (h, t), out in zip(survivors, engine.predict_rows(rel_rows, desc="  S3 relation")):
        probs: dict[str, float] = out["relation"]["probabilities"]
        if multi_label:
            for r, p in probs.items():
                if r != "NA" and p >= tau_rel:
                    triples.append(RelationTriple(h, t, r, p))
        else:
            r = max(probs, key=probs.get)
            if r != "NA" and probs[r] >= tau_rel:
                triples.append(RelationTriple(h, t, r, probs[r]))
    return triples
