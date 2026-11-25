"""Ingestion parser for DocRED and Re-DocRED raw JSON into common Documents."""
from __future__ import annotations

import json
from pathlib import Path

from src.common.datatypes import Document, Entity, Mention, RelationTriple
from src.common.utils import ROOT, jerex_split, load_yaml


def parse_file(path: str | Path) -> list[Document]:
    with open(path) as f:
        raw = json.load(f)
    return [parse_doc(d) for d in raw]


def parse_doc(d: dict) -> Document:
    sents = d["sents"]
    offs = []
    acc = 0
    for s in sents:
        offs.append(acc)
        acc += len(s)

    entities: list[Entity] = []
    for idx, cluster in enumerate(d.get("vertexSet", [])):
        mentions = []
        for m in cluster:
            sid = m["sent_id"]
            start, end = m["pos"]
            mentions.append(Mention(
                sent_id=sid,
                start=start,
                end=end,
                doc_start=offs[sid] + start,
                doc_end=offs[sid] + end,
                type=m["type"],
                text=m["name"],
            ))
        entities.append(Entity(idx=idx, mentions=mentions))

    triples = [
        RelationTriple(head_idx=l["h"], tail_idx=l["t"], relation=l["r"])
        for l in d.get("labels", [])
    ]
    return Document(title=d["title"], sents=sents, entities=entities, triples=triples)


def load_split(dataset: str, split: str) -> list[Document]:
    """Load a logical split under the JEREX protocol.

    dataset: 'docred' | 'redocred'
    split:   'train' | 'dev' | 'test'

    DocRED: train_annotated is re-split into train/dev (seeded); the original
    dev.json serves as the labelled local test set.
    Re-DocRED: uses its own *_revised.json files directly.
    """
    cfg = load_yaml("configs/models.yaml")
    ds = cfg["datasets"][dataset]
    raw = ROOT / ds["raw_dir"]

    if dataset == "docred":
        if split in ("train", "dev"):
            docs = parse_file(raw / ds["train"])
            sp = cfg["jerex_split"]
            train, dev = jerex_split(docs, sp["dev_fraction"], sp["seed"])
            return train if split == "train" else dev
        if split == "test":
            return parse_file(raw / ds["test"])
    elif dataset == "redocred":
        return parse_file(raw / ds[split])
    raise ValueError(f"Unknown dataset/split: {dataset}/{split}")
