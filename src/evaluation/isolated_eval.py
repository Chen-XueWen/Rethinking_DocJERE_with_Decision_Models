"""Stage-isolated (gold-input upper bound) evaluation.

- NER:   span micro P/R/F1 of predicted mentions vs gold mentions.
- Coref: MUC / B-cubed / CEAF-e over clusters built from GOLD mentions.
- RE:    DocRED micro-F1 and Ign-F1 over gold entity clusters
         (reads predictions from pipeline.py --mode gold-ents).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.common.utils import ROOT
from src.data_prep.parser import load_split
from src.evaluation.metrics import coref_metrics, re_f1, span_f1


def gold_doc_record(doc) -> dict:
    return {
        "entities": [
            {"type": e.type,
             "mentions": [{"doc_start": m.doc_start, "doc_end": m.doc_end}
                          for m in e.mentions]}
            for e in doc.entities
        ],
        "triples": [{"h": t.head_idx, "t": t.tail_idx, "r": t.relation}
                    for t in doc.triples],
    }


def eval_re_gold_ents(pred_path: Path, dataset: str, split: str) -> dict:
    docs = load_split(dataset, split)
    train_docs = load_split(dataset, "train")

    # name-level facts present in training annotation (for Ign-F1)
    train_facts = set()
    for d in train_docs:
        for t in d.triples:
            for hm in d.entities[t.head_idx].mentions:
                for tm in d.entities[t.tail_idx].mentions:
                    train_facts.add((hm.text, t.relation, tm.text))

    preds = [json.loads(l) for l in open(pred_path)]
    if len(preds) < len(docs):  # inference ran with --limit
        print(f"note: scoring first {len(preds)} of {len(docs)} docs")
        docs = docs[: len(preds)]
    assert len(preds) == len(docs), f"{len(preds)} preds vs {len(docs)} docs"

    def seen_in_train(d, h, t, r) -> bool:
        """Official DocRED Ign rule: any head-name/tail-name variant pair seen in train."""
        if h >= len(d.entities) or t >= len(d.entities):
            return False
        return any((hm.text, r, tm.text) in train_facts
                   for hm in d.entities[h].mentions
                   for tm in d.entities[t].mentions)

    pred_t, gold_t, in_train = [], [], set()
    for di, (p, d) in enumerate(zip(preds, docs)):
        for t in p["triples"]:
            tup = (di, t["h"], t["t"], t["r"])
            pred_t.append(tup)
            if seen_in_train(d, t["h"], t["t"], t["r"]):
                in_train.add(tup)
        for t in d.triples:
            gold_t.append((di, t.head_idx, t.tail_idx, t.relation))
    return re_f1(pred_t, gold_t, in_train=in_train)


def eval_ner(pred_path: Path, dataset: str, split: str) -> dict:
    docs = load_split(dataset, split)
    preds = [json.loads(l) for l in open(pred_path)]
    docs = docs[: len(preds)]
    pred_spans = [
        {(m["doc_start"], m["doc_end"], m["type"])
         for e in p["entities"] for m in e["mentions"]}
        for p in preds
    ]
    gold_spans = [
        {(m.doc_start, m.doc_end, m.type) for m in d.mentions}
        for d in docs
    ]
    return span_f1(pred_spans, gold_spans)


def eval_coref(pred_path: Path, dataset: str, split: str) -> dict:
    docs = load_split(dataset, split)
    preds = [json.loads(l) for l in open(pred_path)]
    docs = docs[: len(preds)]
    pred_clusters = [
        [frozenset((m["doc_start"], m["doc_end"]) for m in e["mentions"])
         for e in p["entities"]]
        for p in preds
    ]
    gold_clusters = [
        [frozenset((m.doc_start, m.doc_end) for m in e.mentions)
         for e in d.entities]
        for d in docs
    ]
    return coref_metrics(pred_clusters, gold_clusters)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="pipeline predictions JSONL")
    ap.add_argument("--dataset", choices=["docred", "redocred"], required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--tasks", nargs="+", default=["ner", "coref", "re"],
                    choices=["ner", "coref", "re"])
    args = ap.parse_args()

    pred_path = Path(args.preds)
    report = {}
    if "ner" in args.tasks:
        report["ner"] = eval_ner(pred_path, args.dataset, args.split)
    if "coref" in args.tasks:
        report["coref"] = eval_coref(pred_path, args.dataset, args.split)
    if "re" in args.tasks:
        report["re_gold_ents"] = eval_re_gold_ents(pred_path, args.dataset, args.split)
    print(json.dumps(report, indent=2))
    out = pred_path.with_suffix(".isolated_eval.json")
    out.write_text(json.dumps(report, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
