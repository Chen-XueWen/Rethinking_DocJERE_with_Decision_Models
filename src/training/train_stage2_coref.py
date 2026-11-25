"""Train LoRA_Coref (Stage 2: pairwise coreference)."""
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

    scfg = load_yaml("configs/lora_params.yaml")["stages"]["stage2_coref"]
    d = ROOT / "data/processed/stage2_coref"
    train_stage(
        stage_key="stage2_coref",
        model_key=args.model,
        train_path=d / f"{args.dataset}_train.jsonl",
        dev_path=d / f"{args.dataset}_dev.jsonl",
        out_dir=ROOT / "weights/stage2_coref" / f"{args.model}_{args.dataset}",
        epochs=args.epochs or scfg["epochs"],
        max_rows=args.max_rows if args.max_rows is not None else scfg["max_train_rows"],
    )


if __name__ == "__main__":
    main()
