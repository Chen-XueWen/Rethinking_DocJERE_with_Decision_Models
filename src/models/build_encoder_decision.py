"""Wrap a raw encoder (e.g. answerdotai/ModernBERT-large) as a fresh decision model.

Unsloth's FastDecisionModel loads decoder LMs (fresh Clef head) and Laya
checkpoints, but not a raw encoder: `from_pretrained("answerdotai/ModernBERT-large")`
fails because it routes plain models through AutoModelForCausalLM.

Laya *is* ModernBERT-large plus a small decision head, stored as
  encoder/config.json, tokenizer/, rl_agent_config.json, model.safetensors
so we write that layout ourselves: the pretrained ModernBERT-large encoder
weights plus a freshly initialised (seeded) Laya head. Unsloth then loads,
LoRA-trains, calibrates and serves it through its normal Laya path. This
gives a ModernBERT-large baseline WITHOUT Laya's decision pre-training, the
encoder counterpart of loading a raw Qwen with a fresh Clef head.

  $PY -m src.models.build_encoder_decision --model mbert_large
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.common.utils import ROOT, load_yaml

# Head architecture and serving settings, taken from Laya's own config so the
# fresh head matches the published ModernBERT-large decision model exactly.
HEAD_KEYS = ("head_layers", "max_prefixes", "act_costs", "cost_wrong_act", "amp_dtype")
LAYA_DEFAULTS = {"head_layers": 2, "max_prefixes": 6, "act_costs": {"escalate": 0.5},
                 "cost_wrong_act": 3.0, "amp_dtype": "bf16"}


def build(encoder_name: str, out: Path, max_len: int, head_max_len: int,
          seed: int = 3407) -> Path:
    import unsloth  # noqa: F401  (patches transformers before it is imported)
    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import save_file
    from transformers import AutoModel, AutoTokenizer
    from unsloth.models._decision_common import _laya

    if (out / "model.safetensors").exists():
        print(f"exists, skipping: {out}")
        return out
    head = dict(LAYA_DEFAULTS)
    try:  # prefer Laya's exact head settings when it is available
        laya = json.loads((Path(snapshot_download("convaiinnovations/laya"))
                           / "rl_agent_config.json").read_text())
        head.update({k: laya[k] for k in HEAD_KEYS if k in laya})
    except Exception as e:
        print(f"note: using built-in Laya head defaults ({e})")

    src = Path(snapshot_download(encoder_name))
    encoder = AutoModel.from_pretrained(str(src), attn_implementation="sdpa")
    tmp = out.with_name(out.name + ".partial")
    (tmp / "encoder").mkdir(parents=True, exist_ok=True)
    encoder.config.save_pretrained(tmp / "encoder")
    AutoTokenizer.from_pretrained(str(src)).save_pretrained(tmp / "tokenizer")

    torch.manual_seed(seed)
    model = _laya().common.DecisionModel(encoder, head["head_layers"],
                                         len(head["act_costs"]) + 1)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()},
              str(tmp / "model.safetensors"))
    cfg = {"encoder": encoder_name, **head, "max_len": max_len,
           "head_max_len": head_max_len, "model_name": f"{encoder_name}-decision",
           "temperature": [1.0, 1.0, 1.0], "fresh_head_seed": seed}
    (tmp / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2))
    tmp.rename(out)
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"built {out} ({n:.0f}M params, fresh head, seed {seed})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model key in configs/models.yaml")
    args = ap.parse_args()
    cfg = load_yaml("configs/models.yaml")["models"][args.model]
    build(cfg["encoder"], ROOT / cfg["name"], cfg["max_seq_length"],
          cfg.get("head_max_len", 256))


if __name__ == "__main__":
    main()
