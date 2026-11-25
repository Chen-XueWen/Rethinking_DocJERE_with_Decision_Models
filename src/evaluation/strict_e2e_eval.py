"""Strict end-to-end evaluation (JEREX strict matching).

Reads pipeline predictions (e2e mode) and scores triples requiring the full
mention-cluster + entity-type match on both head and tail.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data_prep.parser import load_split
from src.evaluation.metrics import strict_e2e


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--dataset", choices=["docred", "redocred"], required=True)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    docs = load_split(args.dataset, args.split)
    gold_docs = [{
        "entities": [
            {"type": e.type,
             "mentions": [{"doc_start": m.doc_start, "doc_end": m.doc_end}
                          for m in e.mentions]}
            for e in d.entities
        ],
        "triples": [{"h": t.head_idx, "t": t.tail_idx, "r": t.relation}
                    for t in d.triples],
    } for d in docs]

    preds = [json.loads(l) for l in open(args.preds)]
    if len(preds) < len(gold_docs):  # inference ran with --limit
        print(f"note: scoring first {len(preds)} of {len(gold_docs)} docs")
        gold_docs = gold_docs[: len(preds)]
    assert len(preds) == len(gold_docs), f"{len(preds)} preds vs {len(gold_docs)} gold docs"

    report = strict_e2e(preds, gold_docs)
    # mean per-doc latency, for the latency-vs-precision baseline comparisons
    lat = [p.get("elapsed_s") for p in preds if p.get("elapsed_s") is not None]
    if lat:
        report["mean_doc_latency_s"] = round(sum(lat) / len(lat), 3)

    print(json.dumps(report, indent=2))
    out = Path(args.preds).with_suffix(".strict_e2e.json")
    out.write_text(json.dumps(report, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
