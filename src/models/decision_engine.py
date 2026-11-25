"""Inference wrapper around Unsloth's FastDecisionModel.

Loads a base backbone once; exposes batch `score()` over rows in the
{"state", "questions"} schema and returns per-question option probabilities.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.common.utils import ROOT, load_yaml


class DecisionEngine:
    def __init__(self, model_key: str, adapter_path: str | Path | None = None,
                 max_seq_length: int | None = None):
        from unsloth import FastDecisionModel  # heavy import, keep local

        cfg = load_yaml("configs/models.yaml")["models"][model_key]
        from src.training.train_common import resolve_name
        name = str(adapter_path) if adapter_path else resolve_name(cfg)
        self.model, self.tokenizer = FastDecisionModel.from_pretrained(
            model_name=name,
            max_seq_length=max_seq_length or cfg["max_seq_length"],
            load_in_4bit=cfg.get("load_in_4bit", True),
        )
        FastDecisionModel.for_inference(self.model)
        self._fdm = FastDecisionModel
        self.n_calls = 0      # forward passes (rows) for throughput reporting
        self.n_questions = 0  # decisions answered across those passes
        # NOTE: calibration temperatures fitted by FastDecisionModel.calibrate()
        # are persisted by save_pretrained() in unsloth_decision_config.json and
        # re-applied automatically by from_pretrained(); no manual scaling here.
        self.temperature = None

    def predict(self, state: str, questions: dict) -> dict:
        """Score one row: every question in `questions` is answered from a single
        forward pass. Returns {question_id: {"answer", "probabilities"}}."""
        self.n_calls += 1
        self.n_questions += len(questions)
        return self._fdm.predict(self.model, self.tokenizer, state=state,
                                 questions=questions)

    def predict_rows(self, rows: list[dict], desc: str = "rows") -> list[dict]:
        """Score rows one forward pass each (rows are long, so batching several
        rows per pass gave no speed-up: FINDINGS.md 2026-11-11)."""
        from tqdm import tqdm
        return [self.predict(r["state"], r["questions"])
                for r in tqdm(rows, desc=desc, unit="row", leave=False)]
