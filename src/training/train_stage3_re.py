"""Train LoRA_RE (Stage 3: gated 97-way relation extraction)."""
from __future__ import annotations

import argparse

from src.common.utils import ROOT, load_yaml
from src.training.train_common import train_stage


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--dataset", choices=["docred", "redocred"], default="docred")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()

    scfg = load_yaml("configs/lora_params.yaml")["stages"]["stage3_re"]
    d = ROOT / "data/processed/stage3_re"
    train_stage(
        stage_key="stage3_re",
        model_key=args.model,
        train_path=d / f"{args.dataset}_train.jsonl",
        dev_path=d / f"{args.dataset}_dev.jsonl",
        out_dir=ROOT / "weights/stage3_re" / f"{args.model}_{args.dataset}",
        epochs=args.epochs or scfg["epochs"],
        max_rows=args.max_rows if args.max_rows is not None else scfg["max_train_rows"],
    )


if __name__ == "__main__":
    main()
