"""Single-base-model LoRA adapter switching for the three pipeline stages.

Each stage adapter is saved by FastDecisionModel.save_pretrained() and carries
its own LoRA weights AND decision-head weights. PEFT's set_adapter() only
switches LoRA weights, so this manager:
  1. registers each stage's LoRA via peft load_adapter + set_adapter, and
  2. swaps the decision-head state_dict and calibration temperature per stage.

If head swapping is unsupported for the loaded backbone (head tensors not
found in the adapter dir), the manager falls back to a full model reload per
stage — slower but always correct.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from src.common.utils import ROOT, load_yaml

STAGE_DIRS = {
    "ner": "weights/stage1_ner",
    "coref": "weights/stage2_coref",
    "re": "weights/stage3_re",
}

HEAD_FILES = ("joint_head.safetensors", "decision_head.safetensors",
              "decision_head.pt", "clef_head.safetensors")


def _find_head_file(adapter_dir: Path) -> Path | None:
    for name in HEAD_FILES:
        p = adapter_dir / name
        if p.exists():
            return p
    return None


def _head_module(model):
    """Locate the decision-head module on a FastDecisionModel instance."""
    for attr in ("joint_head", "decision_head", "clef_head", "score_head"):
        for owner in (model, getattr(model, "model", None),
                      getattr(getattr(model, "model", None), "model", None)):
            mod = getattr(owner, attr, None) if owner is not None else None
            if mod is not None and hasattr(mod, "load_state_dict"):
                return mod
    return None


class AdapterManager:
    def __init__(self, model_key: str, dataset: str = "docred",
                 run_dir: str | Path | None = None):
        self.model_key = model_key
        self.dataset = dataset
        self.run_dir = Path(run_dir) if run_dir else ROOT
        self.active: str | None = None
        self._engine = None
        self._reload_mode = False
        self._head_states: dict[str, dict] = {}
        # In reload mode (no hot-swap) keep one loaded engine per stage, so a
        # document's 3 stages don't reload the base model each time (measured:
        # ~55 of 64 s/doc were reloads, FINDINGS.md 2026-11-12).
        self._engines: dict[str, object] = {}
        self._temps: dict[str, dict | None] = {}

    def adapter_dir(self, stage: str) -> Path:
        return self.run_dir / STAGE_DIRS[stage] / f"{self.model_key}_{self.dataset}"

    # -- public API ----------------------------------------------------------
    def activate(self, stage: str):
        """Return a DecisionEngine with the given stage's adapter active."""
        if stage not in STAGE_DIRS:
            raise ValueError(f"Unknown stage '{stage}'")
        if self.active == stage and self._engine is not None:
            return self._engine

        if self._reload_mode and stage in self._engines:
            self._engine = self._engines[stage]  # already loaded: no reload
        elif self._engine is None or self._reload_mode:
            self._engine = self._load_full(stage)
        else:
            self._switch(stage)
        if self._reload_mode:
            self._engines[stage] = self._engine
        self.active = stage
        return self._engine

    # -- internals -----------------------------------------------------------
    def _load_full(self, stage: str):
        from src.models.decision_engine import DecisionEngine
        adir = self.adapter_dir(stage)
        engine = DecisionEngine(self.model_key, adapter_path=adir)
        peft_model = getattr(engine.model, "model", engine.model)
        if not hasattr(peft_model, "load_adapter"):
            self._reload_mode = True  # cannot hot-swap; reload per stage
        return engine

    def _switch(self, stage: str):
        adir = self.adapter_dir(stage)
        engine = self._engine
        peft_model = getattr(engine.model, "model", engine.model)
        try:
            if stage not in getattr(peft_model, "peft_config", {}):
                peft_model.load_adapter(str(adir), adapter_name=stage)
            peft_model.set_adapter(stage)
            self._swap_head(engine, stage, adir)
            engine.temperature = self._load_temp(stage, adir)
        except Exception as e:  # pragma: no cover - safety net
            print(f"[adapter_manager] hot-swap failed ({e}); falling back to reload")
            self._reload_mode = True
            self._engine = self._load_full(stage)

    def _swap_head(self, engine, stage: str, adir: Path):
        head = _head_module(engine.model)
        if head is None:
            raise RuntimeError("decision head module not found; cannot hot-swap")
        if stage not in self._head_states:
            hf = _find_head_file(adir)
            if hf is None:
                raise RuntimeError(f"no decision-head weights in {adir}")
            if hf.suffix == ".safetensors":
                from safetensors.torch import load_file
                self._head_states[stage] = load_file(hf)
            else:
                self._head_states[stage] = torch.load(hf, map_location="cpu")
        head.load_state_dict(self._head_states[stage], strict=False)

    def _load_temp(self, stage: str, adir: Path) -> dict | None:
        if stage not in self._temps:
            p = adir / "calibration.json"
            self._temps[stage] = json.loads(p.read_text()) if p.exists() else None
        return self._temps[stage]
