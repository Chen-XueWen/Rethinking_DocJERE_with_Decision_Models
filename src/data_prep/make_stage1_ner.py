"""Stage 1 builder: token-level BIO choice questions, packed per sentence window.

Each row = one sentence (indexed tokens) as state, plus one compact BIO
choice question per token in a <=STAGE1_CHUNK-token window. Gold tags are
derived from the gold mention spans.
"""
from __future__ import annotations

import argparse

from src.common.datatypes import Document
from src.common.schema import stage1_rows
from src.common.utils import ROOT, write_jsonl
from src.data_prep.parser import load_split


def bio_tags_for_sentence(doc: Document, sent_id: int) -> list[str]:
    tokens = doc.sents[sent_id]
    tags = ["O"] * len(tokens)
    # Longer mentions win on overlap (nested mentions can't be encoded in BIO).
    mentions = sorted(
        (m for m in doc.mentions if m.sent_id == sent_id),
        key=lambda m: m.end - m.start,
    )
    for m in mentions:
        for i in range(m.start, min(m.end, len(tokens))):
            tags[i] = f"B-{m.type}" if i == m.start else f"I-{m.type}"
    return tags


def build_rows(docs: list[Document], labelled: bool = True) -> list[dict]:
    rows = []
    for di, doc in enumerate(docs):
        tags = {sid: bio_tags_for_sentence(doc, sid) for sid in range(len(doc.sents))} \
            if labelled else {}
        for row in stage1_rows(doc):
            row["meta"].update(doc_idx=di, title=doc.title)
            if labelled:
                st = tags[row["meta"]["sent_id"]]
                row["gold"] = {f"tok_{t}": st[t] for t in row["meta"]["tokens"]}
            rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["docred", "redocred"], required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    args = ap.parse_args()

    out_dir = ROOT / "data/processed/stage1_ner"
    for split in args.splits:
        docs = load_split(args.dataset, split)
        rows = build_rows(docs, labelled=True)
        out = out_dir / f"{args.dataset}_{split}.jsonl"
        write_jsonl(out, rows)
        print(f"stage1[{args.dataset}/{split}]: {len(docs)} docs -> {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
