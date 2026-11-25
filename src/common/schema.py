"""Question schemas for the three decision stages, shared by data prep and inference.

Design ("indexed document", FINDINGS.md 2026-11-11): every row carries one
document state with each mention/entity tagged once by index, plus MANY cheap
questions answered in the same forward pass. Only the expensive 97-way
relation question is asked one per row, and only for pairs that pass the gate.

Every builder returns rows {"state", "questions", "meta"}; data prep adds
"gold". Inference and training call the same functions, so the train and
test layouts can't drift apart.
"""
from __future__ import annotations

from src.common.datatypes import Document, Entity, Mention
from src.common.utils import (chunk_questions, indexed_sentence, load_json,
                              mark_entities, mark_mentions, state_tokens)

# Token budget per row. Must stay below the training max_seq_length (4096)
# so rows are not truncated (truncation drops the state, not the questions).
ROW_BUDGET = 3800

# ---------------------------------------------------------------- stage 1
# BIO labels without descriptions: each choice question re-tokenizes its full
# option list, so option text dominates the row. Measured: 328 -> 225 tokens
# per question once descriptions are dropped (FINDINGS.md).
BIO = load_json("configs/bio_labels.json")["labels"]
BIO_CRITERIA = {lab: None for lab in BIO}
STAGE1_Q_COST = 230            # tokens per token-question (measured ~225)
STAGE1_CHUNK = 14              # tokens tagged per row; ~3.4k tokens per row


def stage1_rows(doc: Document) -> list[dict]:
    """One row per <=STAGE1_CHUNK-token window of a sentence (whole sentence as context)."""
    rows = []
    for sid, tokens in enumerate(doc.sents):
        if not tokens:
            continue
        state = indexed_sentence(tokens)
        for start in range(0, len(tokens), STAGE1_CHUNK):
            idx = range(start, min(start + STAGE1_CHUNK, len(tokens)))
            rows.append({
                "state": state,
                "questions": {
                    f"tok_{t}": {"type": "choice",
                                 "instructions": f"BIO entity tag of token [{t}] {tokens[t]}",
                                 "criteria": BIO_CRITERIA}
                    for t in idx
                },
                "meta": {"sent_id": sid, "tokens": list(idx), "n_tokens": len(tokens)},
            })
    return rows


# ---------------------------------------------------------------- stage 2
COREF_CRITERIA = {"true": "same entity", "false": "different entities"}
STAGE2_Q_COST = 92             # tokens per noul pair question (measured 73-82; margin for 3-digit ids)


def coref_question(a: int, b: int) -> dict:
    return {"type": "noul",
            "instructions": f"Do mentions M{a} and M{b} refer to the same entity?",
            "criteria": COREF_CRITERIA}


def stage2_rows(doc: Document, mentions: list[Mention],
                pairs: list[tuple[int, int]]) -> list[dict]:
    """Mentions tagged once; pair questions packed into as few rows as fit."""
    if not pairs:
        return []
    state = mark_mentions(doc, mentions)
    chunks = chunk_questions(pairs, ROW_BUDGET, STAGE2_Q_COST, state_tokens(state))
    return [{"state": state,
             "questions": {f"c_{a}_{b}": coref_question(a, b) for a, b in chunk},
             "meta": {"pairs": [list(p) for p in chunk]}}
            for chunk in chunks]


# ---------------------------------------------------------------- stage 3
REL = load_json("configs/docred_relations.json")
RELATION_CRITERIA = {"NA": "No relation holds from the head entity to the tail entity"}
RELATION_CRITERIA.update(REL["relations"])  # {Pxx: descriptive name}
GATE_CRITERIA = {"true": "related", "false": "not related"}
STAGE3_GATE_Q_COST = 92        # measured ~82; margin for 2-3 digit entity ids


def gate_question(h: int, t: int) -> dict:
    return {"type": "noul",
            "instructions": f"Does any factual relation hold from entity E{h} to entity E{t}?",
            "criteria": GATE_CRITERIA}


def relation_question(h: int, t: int) -> dict:
    return {"type": "choice",
            "instructions": (f"Select the Wikidata relation that holds from head entity "
                             f"E{h} to tail entity E{t}, or NA if none holds."),
            "criteria": RELATION_CRITERIA}


def stage3_gate_rows(doc: Document, entities: list[Entity],
                     pairs: list[tuple[int, int]]) -> list[dict]:
    """Entities tagged once; one binary gate question per directed pair, packed."""
    if not pairs:
        return []
    state = mark_entities(doc, entities)
    chunks = chunk_questions(pairs, ROW_BUDGET, STAGE3_GATE_Q_COST, state_tokens(state))
    return [{"state": state,
             "questions": {f"g_{h}_{t}": gate_question(h, t) for h, t in chunk},
             "meta": {"kind": "gate", "pairs": [list(p) for p in chunk]}}
            for chunk in chunks]


def stage3_relation_rows(doc: Document, entities: list[Entity],
                         pairs: list[tuple[int, int]]) -> list[dict]:
    """One 97-way relation question per row (packing these does not pay off:
    each re-tokenizes ~2k option tokens and 8 overflow the 16k limit)."""
    state = mark_entities(doc, entities)
    return [{"state": state,
             "questions": {"relation": relation_question(h, t)},
             "meta": {"kind": "relation", "pairs": [[h, t]]}}
            for h, t in pairs]
