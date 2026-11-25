"""Collect all *.strict_e2e.json / *.isolated_eval.json into a Markdown table."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PRED_RE = re.compile(
    r"preds_(?P<model>.+?)_(?P<mode>zeroshot|finetuned)_(?P<dataset>docred|redocred)"
    r"_(?P<split>\w+?)_(?P<task>e2e|gold-ents)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()
    rd = Path(args.results_dir)

    rows = []
    for f in sorted(rd.glob("*.json")):
        m = PRED_RE.search(f.name)
        if not m:
            continue
        info = m.groupdict()
        data = json.loads(f.read_text())
        if f.name.endswith(".strict_e2e.json"):
            rows.append({**info, "metric": "strict_e2e_micro_f1",
                         "P": data["micro"]["precision"], "R": data["micro"]["recall"],
                         "F1": data["micro"]["f1"],
                         "latency_s": data.get("mean_doc_latency_s", "")})
            rows.append({**info, "metric": "strict_e2e_macro_f1",
                         "P": data["macro_by_relation"]["precision"],
                         "R": data["macro_by_relation"]["recall"],
                         "F1": data["macro_by_relation"]["f1"], "latency_s": ""})
        elif f.name.endswith(".isolated_eval.json"):
            if "re_gold_ents" in data:
                mic = data["re_gold_ents"]["micro"]
                rows.append({**info, "metric": "re_gold_micro_f1",
                             "P": mic["precision"], "R": mic["recall"],
                             "F1": mic["f1"], "latency_s": ""})
                if "ign" in data["re_gold_ents"]:
                    ign = data["re_gold_ents"]["ign"]
                    rows.append({**info, "metric": "re_gold_ign_f1",
                                 "P": ign["precision"], "R": ign["recall"],
                                 "F1": ign["f1"], "latency_s": ""})
            if "ner" in data:
                rows.append({**info, "metric": "ner_span_f1",
                             "P": data["ner"]["precision"], "R": data["ner"]["recall"],
                             "F1": data["ner"]["f1"], "latency_s": ""})
            if "coref" in data:
                rows.append({**info, "metric": "coref_conll_f1", "P": "", "R": "",
                             "F1": data["coref"]["conll_f1"], "latency_s": ""})

    print("# Benchmark Summary\n")
    print("| Model | Mode | Dataset | Split | Task | Metric | P | R | F1 | Latency (s/doc) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['model']} | {r['mode']} | {r['dataset']} | {r['split']} "
              f"| {r['task']} | {r['metric']} | {r['P']} | {r['R']} | {r['F1']} "
              f"| {r['latency_s']} |")
    if not rows:
        print("| _no results yet_ | | | | | | | | | |")

    # Throughput table (paper "time taken" numbers) from *.timing.json sidecars
    trows = []
    for f in sorted(rd.glob("*.timing.json")):
        t = json.loads(f.read_text())
        trows.append(t)
    if trows:
        print("\n## Throughput (inference)\n")
        print("| Model | Mode | Dataset | Task | Docs | Mean s/doc | Median | "
              "Docs/h | Calls/doc | Total wall (s) | Peak GPU (GB) |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for t in trows:
            gpu = t.get("memory", {}).get("gpu_peak_reserved_gb", "")
            print(f"| {t['model']} | {t['mode']} | {t['dataset']} | {t['task']} "
                  f"| {t['n_docs']} | {t['mean_s_per_doc']} "
                  f"| {t['median_s_per_doc']} | {t['docs_per_hour']} "
                  f"| {t['mean_calls_per_doc']} | {t['total_wall_s']} | {gpu} |")

    # Training wall-times from each adapter's calibration.json
    weights = Path("weights")
    wrows = []
    for cal in sorted(weights.glob("stage*/*/calibration.json")):
        c = json.loads(cal.read_text())
        if "train_wall_s" in c:
            gpu = c.get("memory", {}).get("gpu_peak_reserved_gb", "")
            wrows.append((cal.parent.parent.name, cal.parent.name,
                          c["train_wall_s"], c.get("n_train_items"),
                          c.get("epochs"), c.get("accuracy"), c.get("ece"), gpu))
    if wrows:
        print("\n## Training wall-time\n")
        print("| Stage | Model_Dataset | Wall (s) | Train items | Epochs | "
              "Held-out acc | ECE | Peak GPU (GB) |")
        print("|---|---|---|---|---|---|---|---|")
        for w in wrows:
            print(f"| {w[0]} | {w[1]} | {w[2]} | {w[3]} | {w[4]} | {w[5]} "
                  f"| {w[6]} | {w[7]} |")


if __name__ == "__main__":
    main()
