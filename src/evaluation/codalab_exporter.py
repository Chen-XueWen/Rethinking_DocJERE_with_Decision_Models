"""Export gold-entity predictions on DocRED's blind test set as CodaLab result.json.

Only valid for predictions produced with `pipeline.py --mode gold-ents` on the
blind split, since CodaLab scores against the given gold vertexSet indices.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="gold-ents predictions JSONL")
    ap.add_argument("--out", default="results/result.json")
    args = ap.parse_args()

    submission = []
    for line in open(args.preds):
        p = json.loads(line)
        for t in p["triples"]:
            submission.append({
                "title": p["title"],
                "h_idx": t["h"],
                "t_idx": t["t"],
                "r": t["r"],
                "evidence": [],
            })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(submission))
    print(f"wrote {len(submission)} triples -> {out}")


if __name__ == "__main__":
    main()
