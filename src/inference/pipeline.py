"""End-to-end inference orchestrator: document -> mentions -> entities -> triples.

Modes:
  --mode e2e        full predicted pipeline (stages 1 -> 2 -> 3)
  --mode gold-ents  stage 3 only, over gold entity clusters (isolated RE eval)

Predictions are written as JSONL, one line per document, replayable by the
evaluation scripts without re-running inference.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.common.datatypes import Document
from src.common.utils import ROOT, load_yaml
from src.data_prep.parser import load_split


def doc_prediction(doc: Document, entities, triples, elapsed: float,
                   timing: dict | None = None) -> dict:
    return {
        "title": doc.title,
        "elapsed_s": round(elapsed, 3),
        "timing": timing or {},
        "entities": [
            {"type": e.type,
             "mentions": [{"sent_id": m.sent_id, "start": m.start, "end": m.end,
                           "doc_start": m.doc_start, "doc_end": m.doc_end,
                           "type": m.type, "text": m.text} for m in e.mentions]}
            for e in entities
        ],
        "triples": [{"h": t.head_idx, "t": t.tail_idx, "r": t.relation,
                     "score": t.score} for t in triples],
    }


def run(model_key: str, dataset: str, split: str, mode: str,
        out_path: Path, limit: int = 0, zero_shot: bool = False) -> None:
    from src.inference.stage1_tagger import tag_document
    from src.inference.stage2_clusterer import cluster_mentions
    from src.inference.stage3_extractor import extract_relations

    icfg = load_yaml("configs/lora_params.yaml")["inference"]
    docs = load_split(dataset, split)
    if limit:
        docs = docs[:limit]

    if zero_shot:
        from src.models.decision_engine import DecisionEngine
        engine = DecisionEngine(model_key)
        get_engine = lambda stage: engine
    else:
        from src.models.adapter_manager import AdapterManager
        mgr = AdapterManager(model_key, dataset=dataset)
        get_engine = mgr.activate

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file and rename at the end so interrupted runs never
    # leave a partial file that the resumable benchmark script would skip.
    from tqdm import tqdm  # writes to stderr: visible even with stdout redirected
    tmp_path = out_path.with_suffix(".jsonl.partial")
    with open(tmp_path, "w") as f:
        for i, doc in enumerate(tqdm(docs, desc=f"docs[{dataset}/{split}/{mode}]",
                                     unit="doc")):
            t0 = time.time()
            timing = {}

            def _stage(name, engine, fn):
                """Run one stage, recording wall time, forward passes and decisions."""
                calls0, q0 = engine.n_calls, engine.n_questions
                ts = time.time()
                result = fn(engine)
                timing[name] = {"s": round(time.time() - ts, 3),
                                "calls": engine.n_calls - calls0,
                                "questions": engine.n_questions - q0}
                return result

            if mode == "gold-ents":
                entities = doc.entities
            else:
                mentions = _stage("stage1_ner", get_engine("ner"),
                                  lambda e: tag_document(e, doc))
                entities = _stage("stage2_coref", get_engine("coref"),
                                  lambda e: cluster_mentions(
                                      e, doc, mentions, alpha=icfg["coref_alpha"]))
            triples = _stage("stage3_re", get_engine("re"),
                             lambda e: extract_relations(
                                 e, doc, entities, tau_gate=icfg["tau_gate"],
                                 tau_rel=icfg["tau_rel"]))
            rec = doc_prediction(doc, entities, triples, time.time() - t0, timing)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i+1}/{len(docs)}] {doc.title}: "
                  f"{len(entities)} entities, {len(triples)} triples "
                  f"({rec['elapsed_s']}s)")
    tmp_path.rename(out_path)
    print(f"wrote predictions to {out_path}")
    _write_timing_summary(out_path, model_key, dataset, split, mode, zero_shot)


def _write_timing_summary(out_path: Path, model_key: str, dataset: str,
                          split: str, mode: str, zero_shot: bool) -> None:
    """Aggregate per-doc timing into paper-ready throughput numbers."""
    recs = [json.loads(l) for l in open(out_path)]
    if not recs:
        return
    lat = [r["elapsed_s"] for r in recs]
    lat_sorted = sorted(lat)
    n = len(lat)
    stages = {}
    for r in recs:
        for name, t in r.get("timing", {}).items():
            s = stages.setdefault(name, {"s": 0.0, "calls": 0, "questions": 0})
            s["s"] += t["s"]
            s["calls"] += t["calls"]
            s["questions"] += t.get("questions", 0)
    for s in stages.values():
        s["s"] = round(s["s"], 1)
        s["s_per_call"] = round(s["s"] / s["calls"], 4) if s["calls"] else None
        s["ms_per_question"] = (round(1000 * s["s"] / s["questions"], 1)
                                if s["questions"] else None)
    from src.common.utils import memory_stats
    total_calls = sum(s["calls"] for s in stages.values())
    summary = {
        "memory": memory_stats(),
        "model": model_key,
        "mode": "zeroshot" if zero_shot else "finetuned",
        "dataset": dataset, "split": split, "task": mode,
        "n_docs": n,
        "total_wall_s": round(sum(lat), 1),
        "mean_s_per_doc": round(sum(lat) / n, 2),
        "median_s_per_doc": round(lat_sorted[n // 2], 2),
        "min_s_per_doc": round(lat_sorted[0], 2),
        "max_s_per_doc": round(lat_sorted[-1], 2),
        "docs_per_hour": round(3600 * n / sum(lat), 2),
        "total_model_calls": total_calls,
        "mean_calls_per_doc": round(total_calls / n, 1),
        "total_questions": sum(s["questions"] for s in stages.values()),
        "per_stage": stages,
    }
    tp = out_path.with_suffix(".timing.json")
    tp.write_text(json.dumps(summary, indent=2))
    mem = summary["memory"]
    print(f"timing summary -> {tp}: {summary['mean_s_per_doc']}s/doc, "
          f"{summary['docs_per_hour']} docs/h, {total_calls} model calls, "
          f"peak GPU {mem.get('gpu_peak_reserved_gb', '?')}GB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--dataset", choices=["docred", "redocred"], default="docred")
    ap.add_argument("--split", default="test")
    ap.add_argument("--mode", choices=["e2e", "gold-ents"], default="e2e")
    ap.add_argument("--limit", type=int, default=0, help="cap #docs (0 = all)")
    ap.add_argument("--zero-shot", action="store_true",
                    help="use the raw pretrained decision model, no adapters")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tag = "zeroshot" if args.zero_shot else "finetuned"
    out = Path(args.out) if args.out else (
        ROOT / "results" / f"preds_{args.model}_{tag}_{args.dataset}_{args.split}_{args.mode}.jsonl")
    run(args.model, args.dataset, args.split, args.mode, out,
        limit=args.limit, zero_shot=args.zero_shot)


if __name__ == "__main__":
    main()
