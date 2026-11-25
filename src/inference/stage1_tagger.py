"""Stage 1 inference: packed token scoring + deterministic BIO structural repair."""
from __future__ import annotations

from src.common.datatypes import Document, Mention
from src.common.schema import stage1_rows


def repair_bio(tags: list[str]) -> list[str]:
    """Deterministic repair of illegal BIO transitions (plan §6.1)."""
    fixed = list(tags)
    for t, tag in enumerate(fixed):
        if not tag.startswith("I-"):
            continue
        typ = tag[2:]
        prev = fixed[t - 1] if t > 0 else "O"
        if prev in (f"B-{typ}", f"I-{typ}"):
            continue
        # Rule 1 (orphan inside) and Rule 2 (type mismatch): coerce to B-TYPE
        fixed[t] = f"B-{typ}"
    return fixed


def tags_to_mentions(doc: Document, sent_id: int, tags: list[str]) -> list[Mention]:
    """Rule 3: compile contiguous B-/I- runs into Mention spans."""
    tokens = doc.sents[sent_id]
    off = doc.sent_offsets[sent_id]
    mentions: list[Mention] = []
    start, typ = None, None
    for t, tag in enumerate(tags + ["O"]):  # sentinel flush
        if tag.startswith("B-"):
            if start is not None:
                mentions.append(_mk(sent_id, off, tokens, start, t, typ))
            start, typ = t, tag[2:]
        elif tag.startswith("I-") and start is not None and tag[2:] == typ:
            continue
        else:
            if start is not None:
                mentions.append(_mk(sent_id, off, tokens, start, t, typ))
            start, typ = None, None
    return mentions


def _mk(sent_id, off, tokens, start, end, typ) -> Mention:
    return Mention(
        sent_id=sent_id, start=start, end=end,
        doc_start=off + start, doc_end=off + end,
        type=typ, text=" ".join(tokens[start:end]),
    )


def tag_document(engine, doc: Document) -> list[Mention]:
    """BIO-tag every token; each row answers a window of tokens in one pass."""
    rows = stage1_rows(doc)
    outs = engine.predict_rows(rows, desc="  S1 rows")
    tags = {sid: ["O"] * len(s) for sid, s in enumerate(doc.sents)}
    for row, out in zip(rows, outs):
        sid = row["meta"]["sent_id"]
        for t in row["meta"]["tokens"]:
            tag = out.get(f"tok_{t}", {}).get("answer", "O")
            tags[sid][t] = tag if isinstance(tag, str) else "O"
    mentions: list[Mention] = []
    for sid in range(len(doc.sents)):
        mentions.extend(tags_to_mentions(doc, sid, repair_bio(tags[sid])))
    return mentions
