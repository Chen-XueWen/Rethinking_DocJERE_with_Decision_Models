"""Zero-shot evaluation of a hosted, pre-trained decision model (no fine-tuning).

Reuses the fine-tuned pipeline unchanged (same stage-1/2/3 schemas, BIO
repair, clustering, gate + relation thresholds), swapping only the engine for
RemoteDecisionEngine. Writes predictions + a timing/usage sidecar in the same
format, so strict_e2e_eval / isolated_eval / summarize work as-is.

  $PY -m src.pretrained.run_pretrained --model azure_ms_decision_1 \\
      --dataset docred --split test --mode e2e --out results/...jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.common.utils import ROOT, load_yaml
from src.data_prep.parser import load_split


def run(model_key: str, dataset: str, split: str, mode: str, out_path: Path,
        limit: int = 0) -> None:
    from tqdm import tqdm
    from src.inference.pipeline import doc_prediction
    from src.inference.stage1_tagger import tag_document
    from src.inference.stage2_clusterer import cluster_mentions
    from src.inference.stage3_extractor import extract_relations
    from src.pretrained.remote_engine import RemoteDecisionEngine

    icfg = load_yaml("configs/lora_params.yaml")["inference"]
    docs = load_split(dataset, split)
    if limit:
        docs = docs[:limit]
    engine = RemoteDecisionEngine(model_key)

    # Resume: keep finished docs from an interrupted run (one JSON line per doc).
    tmp = out_path.with_suffix(".jsonl.partial")
    done = []
    if tmp.exists():
        for line in open(tmp):
            try:
                done.append(json.loads(line))
            except json.JSONDecodeError:
                break  # a line cut off by a kill mid-write: redo that doc
    done = done[: len(docs)]
    if any(r["title"] != d.title for r, d in zip(done, docs)):
        raise RuntimeError(f"{tmp} does not match {dataset}/{split} order; "
                           f"delete it to start this step over")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w") as f:
        for rec in done:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if done:
            print(f"resuming after {len(done)} finished docs")
        for doc in tqdm(docs[len(done):], desc=f"docs[{model_key}/{dataset}/{mode}]",
                        unit="doc", initial=len(done), total=len(docs)):
            t0, timing = time.time(), {}

            def _stage(name, fn):
                c0, q0 = engine.n_calls, engine.n_questions
                ts = time.time()
                res = fn()
                timing[name] = {"s": round(time.time() - ts, 3),
                                "calls": engine.n_calls - c0,
                                "questions": engine.n_questions - q0}
                return res

            if mode == "gold-ents":
                entities = doc.entities
            else:
                mentions = _stage("stage1_ner", lambda: tag_document(engine, doc))
                entities = _stage("stage2_coref", lambda: cluster_mentions(
                    engine, doc, mentions, alpha=icfg["coref_alpha"]))
            triples = _stage("stage3_re", lambda: extract_relations(
                engine, doc, entities, tau_gate=icfg["tau_gate"], tau_rel=icfg["tau_rel"]))
            rec = doc_prediction(doc, entities, triples, time.time() - t0, timing)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
    tmp.rename(out_path)
    print(f"wrote predictions to {out_path}")

    from src.inference.pipeline import _write_timing_summary
    _write_timing_summary(out_path, model_key, dataset, split, mode, zero_shot=True)
    tp = out_path.with_suffix(".timing.json")
    t = json.loads(tp.read_text())
    t["api_usage"] = {**engine.usage, "requests": engine.n_calls,
                      "litellm_model": engine.model,
                      "note": "usage covers this process only (not resumed docs)"}
    t.pop("memory", None)  # remote model: local GPU memory is not meaningful
    tp.write_text(json.dumps(t, indent=2))
    print(f"api usage: {t['api_usage']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="key in configs/pretrained.yaml")
    ap.add_argument("--dataset", choices=["docred", "redocred"], required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--mode", choices=["e2e", "gold-ents"], default="e2e")
    ap.add_argument("--limit", type=int, default=0, help="cap #docs (0 = all)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out) if args.out else (
        ROOT / "results" / args.model /
        f"preds_{args.model}_zeroshot_{args.dataset}_{args.split}_{args.mode}.jsonl")
    run(args.model, args.dataset, args.split, args.mode, out, limit=args.limit)


if __name__ == "__main__":
    main()
