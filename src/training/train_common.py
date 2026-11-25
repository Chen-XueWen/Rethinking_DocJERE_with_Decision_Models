"""Shared FastDecisionModel training driver used by all three stage trainers.

Handles: model load (4-bit where supported), LoRA attachment, dataset build,
the Unsloth #13160 warm-up (an inference/evaluate pass must run before
DecisionTrainer.train()), training, calibration, and saving the adapter +
decision head + calibration temperature.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from src.common.utils import ROOT, load_yaml, read_jsonl


def is_encoder(model_key: str, mcfg: dict) -> bool:
    """Encoder (ModernBERT/Laya-layout) backbones vs decoder LMs."""
    return model_key == "laya" or "encoder" in mcfg


def check_schema_fits(model, tokenizer, rows_path: Path, stage_key: str,
                      sample: int = 300) -> None:
    """Fail loudly if Laya-layout encoders silently trim questions or options.

    Laya's build_sequence cuts each question's instruction and option text to
    fit `head_max_len`, without reporting it in build_dataset's truncation count.
    If the instruction loses its entity/mention ids, every question in a row
    becomes the same input and the stage cannot learn (FINDINGS.md 2026-11-16).
    Compares each sequence against one built with an unlimited budget.
    """
    from unsloth.models._decision_common import _laya
    from unsloth.models.decision import _internal

    common = _laya().common
    max_len = int(model.decision_config.get("max_len", 512))
    head_max_len = int(model.decision_config.get("head_max_len", 192))
    total = trimmed = 0
    for r in read_jsonl(rows_path)[:sample]:
        for name, q in r["questions"].items():
            iq = _internal(q, name)
            ids, _ = common.build_sequence(tokenizer, r["state"], iq, max_len, head_max_len)
            full, _ = common.build_sequence(tokenizer, r["state"], iq, max_len, 10**6)
            total += 1
            trimmed += ids != full
    print(f"[{stage_key}] schema fit (head_max_len={head_max_len}): "
          f"{trimmed}/{total} sampled questions trimmed")
    if trimmed > 0.01 * max(total, 1):
        raise RuntimeError(
            f"[{stage_key}] {trimmed}/{total} questions have their instruction/options "
            f"trimmed at head_max_len={head_max_len}; raise head_max_len for this "
            f"model in configs/models.yaml (and its base rl_agent_config.json)")


def resolve_name(mcfg: dict) -> str:
    """Hub id as-is; a repo-relative local checkpoint path made absolute."""
    local = ROOT / mcfg["name"]
    return str(local) if local.exists() else mcfg["name"]


def train_stage(stage_key: str, model_key: str, train_path: Path, dev_path: Path,
                out_dir: Path, epochs: int, max_rows: int = 0) -> dict:
    # unsloth must be imported before transformers/peft so its optimizations
    # (patched kernels, memory savings) are applied.
    from unsloth import FastDecisionModel, is_bfloat16_supported
    try:
        from unsloth import DecisionTrainer
    except ImportError:
        from unsloth.trainer import DecisionTrainer  # fallback location
    import torch
    from datasets import Dataset
    from transformers import TrainingArguments

    mcfg = load_yaml("configs/models.yaml")["models"][model_key]
    lcfg = load_yaml("configs/lora_params.yaml")
    lora, tcfg = lcfg["lora"], lcfg["training"]

    # Gradient checkpointing must be set at load time too: from_pretrained
    # defaults it to "unsloth" independently of get_peft_model.
    gc = lora["use_gradient_checkpointing"]
    model, tokenizer = FastDecisionModel.from_pretrained(
        model_name=resolve_name(mcfg),
        max_seq_length=mcfg["max_seq_length"],
        load_in_4bit=mcfg.get("load_in_4bit", True),
        use_gradient_checkpointing=gc,
    )
    peft_kwargs = dict(
        r=lora["r"],
        lora_alpha=lora["lora_alpha"],
        lora_dropout=lora["lora_dropout"],
        use_gradient_checkpointing=gc,
        random_state=lora["random_state"],
    )
    # LLM backbones use the explicit projection list; encoder backbones
    # (laya/ModernBERT: Wqkv, Wo, Wi) have different module names -> unsloth's all-linear.
    if not is_encoder(model_key, mcfg):
        peft_kwargs["target_modules"] = lora["target_modules"]
    model = FastDecisionModel.get_peft_model(model, **peft_kwargs)

    def to_hf(path: Path, cap: int = 0) -> Dataset:
        import random
        rows = read_jsonl(path)
        if cap and len(rows) > cap:
            # random sample, not a prefix: files are document-ordered
            rows = random.Random(tcfg["seed"]).sample(rows, cap)
        return Dataset.from_list([
            {"state": r["state"],
             "questions": json.dumps(r["questions"]),
             "gold": json.dumps(r["gold"])}
            for r in rows
        ])

    train_ds = to_hf(train_path, max_rows)
    dev_ds = to_hf(dev_path, 1000)  # cap eval/calibration rows for speed

    train_items, train_report = FastDecisionModel.build_dataset(train_ds, tokenizer, model)
    eval_items, eval_report = FastDecisionModel.build_dataset(dev_ds, tokenizer, model)
    print(f"[{stage_key}] train {train_report} | eval {eval_report}")
    for name, rep in (("train", train_report), ("eval", eval_report)):
        if rep["skipped"] > 0.01 * max(rep["total"], 1):
            raise RuntimeError(f"[{stage_key}] {name}: {rep['skipped']}/{rep['total']} "
                               f"questions skipped ({rep['reason']}); row packing "
                               f"in src/common/schema.py exceeds max_seq_length")

    if not getattr(model, "is_clef", False):
        check_schema_fits(model, tokenizer, train_path, stage_key)

    # Unsloth issue #13160: run an inference pass before the first training
    # step, otherwise the compiled layer_norm backward crashes on step 1.
    FastDecisionModel.evaluate(model, tokenizer, eval_items[: min(8, len(eval_items))])

    steps_per_epoch = max(1, len(train_items) //
                          (tcfg["per_device_train_batch_size"] * tcfg["gradient_accumulation_steps"]))
    total_steps = steps_per_epoch * epochs

    args = TrainingArguments(
        output_dir=str(out_dir / "trainer"),
        per_device_train_batch_size=tcfg["per_device_train_batch_size"],
        gradient_accumulation_steps=tcfg["gradient_accumulation_steps"],
        num_train_epochs=epochs,
        learning_rate=float(tcfg["learning_rate"]),
        warmup_steps=max(1, int(total_steps * lcfg["training"]["warmup_ratio"])),
        lr_scheduler_type=tcfg["lr_scheduler_type"],
        weight_decay=tcfg["weight_decay"],
        optim=tcfg["optim"],
        logging_steps=tcfg["logging_steps"],
        seed=tcfg["seed"],
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        report_to="none",
        save_strategy="no",
    )
    trainer = DecisionTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_items,
        eval_dataset=eval_items,
        args=args,
        head_learning_rate=float(tcfg["head_learning_rate"]),
    )
    import time
    t_train = time.time()
    trainer.train()
    train_wall_s = round(time.time() - t_train, 1)

    # Calibration on held-out dev items; persist whatever metrics come back,
    # plus a temperature if the API exposes one.
    metrics = {}
    try:
        cal = FastDecisionModel.calibrate(model, tokenizer, eval_items)
        if isinstance(cal, dict):
            metrics.update({k: v for k, v in cal.items()
                            if isinstance(v, (int, float, str, bool))})
    except Exception as e:
        print(f"[{stage_key}] calibrate() failed or unavailable: {e}")

    out_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(str(out_dir))  # Clef/LM: LoRA adapter + decision head
        tokenizer.save_pretrained(str(out_dir))
    else:
        # Laya-layout encoders have no adapter save: unsloth writes a complete
        # merged 16-bit checkpoint (LoRA folded in, calibrated temperatures in
        # rl_agent_config.json), which reloads through the same Laya loader.
        model.save_pretrained_merged(str(out_dir), tokenizer)
    from src.common.utils import memory_stats
    cal_payload = {"temperature": metrics.get("temperature", 1.0), **metrics,
                   "train_wall_s": train_wall_s,
                   "n_train_items": len(train_items), "epochs": epochs,
                   "memory": memory_stats()}
    (out_dir / "calibration.json").write_text(json.dumps(cal_payload, indent=2))
    print(f"[{stage_key}] saved adapter + calibration to {out_dir}: {cal_payload}")
    return metrics
