"""Stage 2 builder: coreference as packed binary pair questions.

Each row = the document with every gold mention tagged once ([M7]...[/M7]),
plus many binary questions "Do mentions M{a} and M{b} refer to the same
entity?" over same-type pairs, packed into as few rows as fit ROW_BUDGET.
Training negatives are capped at `stage2_neg_ratio` : 1 per document.
"""
from __future__ import annotations

import argparse
import random
from itertools import combinations

from src.common.datatypes import Document
from src.common.schema import stage2_rows
from src.common.utils import ROOT, load_yaml, write_jsonl
from src.data_prep.parser import load_split


def candidate_pairs(mentions) -> list[tuple[int, int]]:
    """Same-type mention index pairs (a < b); cross-type assumed non-coreferent."""
    return [(a, b) for a, b in combinations(range(len(mentions)), 2)
            if mentions[a].type == mentions[b].type]


def build_rows(docs: list[Document], subsample: bool, neg_ratio: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for di, doc in enumerate(docs):
        owner = {}
        for e in doc.entities:
            for m in e.mentions:
                owner[m.span] = e.idx
        mentions = sorted(doc.mentions, key=lambda m: m.span)
        pairs = candidate_pairs(mentions)
        same = {p: owner.get(mentions[p[0]].span) == owner.get(mentions[p[1]].span)
                for p in pairs}
        pos = [p for p in pairs if same[p]]
        neg = [p for p in pairs if not same[p]]
        if subsample and len(neg) > neg_ratio * max(len(pos), 1):
            neg = rng.sample(neg, neg_ratio * max(len(pos), 1))
            pairs = pos + neg
            rng.shuffle(pairs)  # mix labels within each packed row

        for row in stage2_rows(doc, mentions, pairs):
            row["gold"] = {f"c_{a}_{b}": same[(a, b)] for a, b in row["meta"]["pairs"]}
            row["meta"].update(doc_idx=di, title=doc.title)
            rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["docred", "redocred"], required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    args = ap.parse_args()

    cfg = load_yaml("configs/lora_params.yaml")["sampling"]
    out_dir = ROOT / "data/processed/stage2_coref"
    for split in args.splits:
        docs = load_split(args.dataset, split)
        rows = build_rows(docs, subsample=(split == "train"),
                          neg_ratio=cfg["stage2_neg_ratio"], seed=cfg["seed"])
        out = out_dir / f"{args.dataset}_{split}.jsonl"
        write_jsonl(out, rows)
        n_q = sum(len(r["questions"]) for r in rows)
        print(f"stage2[{args.dataset}/{split}]: {len(docs)} docs -> {len(rows)} rows "
              f"({n_q} pair questions) -> {out}")


if __name__ == "__main__":
    main()
