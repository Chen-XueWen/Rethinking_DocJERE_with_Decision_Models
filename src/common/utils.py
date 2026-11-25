"""Context string formatters and marker injection utilities."""
from __future__ import annotations

import json
import random
from pathlib import Path

import yaml

from .datatypes import Document, Entity, Mention

ROOT = Path(__file__).resolve().parents[2]


def load_yaml(rel_path: str) -> dict:
    with open(ROOT / rel_path) as f:
        return yaml.safe_load(f)


def load_json(rel_path: str) -> dict:
    with open(ROOT / rel_path) as f:
        return json.load(f)


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def indexed_sentence(tokens: list[str], offset: int = 0) -> str:
    """Render tokens with explicit indices: `[0] Barack [1] Obama ...`"""
    return " ".join(f"[{offset + i}] {tok}" for i, tok in enumerate(tokens))


def mark_mentions(doc: Document, mentions: list[Mention]) -> str:
    """Document text with every mention tagged once by its index: [M7] ... [/M7].

    One state serves every coreference question about the document, so many
    pair questions share a single forward pass (see FINDINGS.md 2026-11-11).
    """
    inserts: dict[int, list[str]] = {}
    for i, m in enumerate(mentions):
        inserts.setdefault(m.doc_start, []).append(f"[M{i}]")
        inserts.setdefault(m.doc_end, []).insert(0, f"[/M{i}]")
    return _render_with_inserts(doc.tokens, inserts)


def mark_entities(doc: Document, entities: list[Entity]) -> str:
    """Document text with every mention of entity k tagged [E{k}] ... [/E{k}].

    `k` is the entity's position in `entities`; one state serves every
    relation question about the document.
    """
    inserts: dict[int, list[str]] = {}
    for k, e in enumerate(entities):
        for m in e.mentions:
            inserts.setdefault(m.doc_start, []).append(f"[E{k}]")
            inserts.setdefault(m.doc_end, []).insert(0, f"[/E{k}]")
    return _render_with_inserts(doc.tokens, inserts)


def chunk_questions(keys: list, budget_tokens: int, cost_per_question: int,
                    base_tokens: int) -> list[list]:
    """Split question keys into groups whose encoded row stays within budget."""
    per_row = max(1, (budget_tokens - base_tokens) // cost_per_question)
    return [keys[i:i + per_row] for i in range(0, len(keys), per_row)]


_BUDGET_TOKENIZER = None


def state_tokens(text: str) -> int:
    """Exact Qwen3.5 token count of a state, plus the record's fixed wrapper
    (system prompt, schema headers: measured ~140 tokens). Used only for row
    packing, so the other backbones' tokenizers are close enough."""
    global _BUDGET_TOKENIZER
    if _BUDGET_TOKENIZER is None:
        from transformers import AutoTokenizer
        _BUDGET_TOKENIZER = AutoTokenizer.from_pretrained("unsloth/Qwen3.5-4B")
    return len(_BUDGET_TOKENIZER(text, add_special_tokens=False).input_ids) + 200


def _render_with_inserts(tokens: list[str], inserts: dict[int, list[str]]) -> str:
    out: list[str] = []
    for i, tok in enumerate(tokens):
        if i in inserts:
            out.extend(inserts[i])
        out.append(tok)
    if len(tokens) in inserts:
        out.extend(inserts[len(tokens)])
    return " ".join(out)


def memory_stats(reset: bool = False) -> dict:
    """Peak memory for reporting: torch GPU allocator stats + process RSS.

    On unified-memory machines (GB10) nvidia-smi is unreliable; the CUDA
    allocator peaks are the meaningful 'VRAM footprint' numbers (H2's <16GB
    claim), and peak RSS covers the host side.
    With reset=True, clears the allocator peaks first (call at run start).
    """
    import resource

    out = {}
    try:
        import torch
        if torch.cuda.is_available():
            if reset:
                torch.cuda.reset_peak_memory_stats()
            out["gpu_peak_alloc_gb"] = round(
                torch.cuda.max_memory_allocated() / 1024**3, 2)
            out["gpu_peak_reserved_gb"] = round(
                torch.cuda.max_memory_reserved() / 1024**3, 2)
    except Exception:
        pass
    # ru_maxrss is KB on Linux; lifetime peak (cannot be reset)
    out["cpu_peak_rss_gb"] = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 2)
    return out


def jerex_split(docs: list, dev_fraction: float = 0.1, seed: int = 3407) -> tuple[list, list]:
    """Deterministically split a document list into (train, dev) — JEREX protocol."""
    rng = random.Random(seed)
    idx = list(range(len(docs)))
    rng.shuffle(idx)
    n_dev = int(len(docs) * dev_fraction)
    dev_set = set(idx[:n_dev])
    train = [d for i, d in enumerate(docs) if i not in dev_set]
    dev = [d for i, d in enumerate(docs) if i in dev_set]
    return train, dev
