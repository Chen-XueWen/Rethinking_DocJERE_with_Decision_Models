"""Stage 3 builder: gate rows (packed binary questions) + relation rows (97-way).

Both row kinds share one document state with every entity tagged once
([E3]...[/E3]) and train ONE adapter (unsloth calibrates noul and choice
questions with separate temperatures):

  gate rows      many questions "Does any factual relation hold from E{h} to
                 E{t}?" per row (packed). Training negatives: 3:1 NA:positive,
                 70% intra-sentence co-occurring (hard), 30% random.
  relation rows  one 97-way choice (96 relations + NA) per row, for gold
                 pairs (duplicated per label for multi-label pairs) plus hard
                 NA pairs at `stage3_rel_na_ratio`:1. At inference these run
                 only on gate survivors, so the NA rows teach the model to
                 reject gate false positives.
"""
from __future__ import annotations

import argparse
import random

from src.common.datatypes import Document
from src.common.schema import stage3_gate_rows, stage3_relation_rows
from src.common.utils import ROOT, load_yaml, write_jsonl
from src.data_prep.parser import load_split


def _cooccur(h, t) -> bool:
    return bool({m.sent_id for m in h.mentions} & {m.sent_id for m in t.mentions})


def build_rows(docs: list[Document], subsample: bool, neg_ratio: int,
               hard_frac: float, rel_na_ratio: float, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for di, doc in enumerate(docs):
        ents = doc.entities
        gold: dict[tuple[int, int], list[str]] = {}
        for tr in doc.triples:
            gold.setdefault((tr.head_idx, tr.tail_idx), []).append(tr.relation)

        pos = list(gold)
        neg = [(h.idx, t.idx) for h in ents for t in ents
               if h.idx != t.idx and (h.idx, t.idx) not in gold]
        hard = [p for p in neg if _cooccur(ents[p[0]], ents[p[1]])]
        hard_set = set(hard)
        easy = [p for p in neg if p not in hard_set]

        # ---- gate rows
        if subsample:
            n_neg = neg_ratio * max(len(pos), 1)
            n_hard = min(int(n_neg * hard_frac), len(hard))
            n_easy = min(n_neg - n_hard, len(easy))
            gate_pairs = pos + rng.sample(hard, n_hard) + rng.sample(easy, n_easy)
        else:
            gate_pairs = pos + neg
        rng.shuffle(gate_pairs)
        meta = {"doc_idx": di, "title": doc.title}
        for row in stage3_gate_rows(doc, ents, gate_pairs):
            row["gold"] = {f"g_{h}_{t}": (h, t) in gold for h, t in row["meta"]["pairs"]}
            row["meta"].update(meta)
            rows.append(row)

        # ---- relation rows (gold pairs + hard NA pairs)
        n_na = min(int(rel_na_ratio * len(pos)), len(hard))
        na_pairs = rng.sample(hard, n_na) if n_na else []
        for (h, t) in pos:
            for r in gold[(h, t)]:  # duplicate multi-label pairs, one row per label
                row = stage3_relation_rows(doc, ents, [(h, t)])[0]
                row["gold"] = {"relation": r}
                row["meta"].update(meta)
                rows.append(row)
        for row in stage3_relation_rows(doc, ents, na_pairs):
            row["gold"] = {"relation": "NA"}
            row["meta"].update(meta)
            rows.append(row)
    rng.shuffle(rows)  # interleave gate and relation rows for joint training
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["docred", "redocred"], required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    args = ap.parse_args()

    cfg = load_yaml("configs/lora_params.yaml")["sampling"]
    out_dir = ROOT / "data/processed/stage3_re"
    for split in args.splits:
        docs = load_split(args.dataset, split)
        rows = build_rows(docs, subsample=(split == "train"),
                          neg_ratio=cfg["stage3_neg_ratio"],
                          hard_frac=cfg["stage3_hard_frac"],
                          rel_na_ratio=cfg["stage3_rel_na_ratio"], seed=cfg["seed"])
        out = out_dir / f"{args.dataset}_{split}.jsonl"
        write_jsonl(out, rows)
        n_gate = sum(1 for r in rows if r["meta"]["kind"] == "gate")
        n_gq = sum(len(r["questions"]) for r in rows if r["meta"]["kind"] == "gate")
        print(f"stage3[{args.dataset}/{split}]: {len(docs)} docs -> {n_gate} gate rows "
              f"({n_gq} gate questions) + {len(rows) - n_gate} relation rows -> {out}")


if __name__ == "__main__":
    main()
