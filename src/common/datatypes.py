"""Typed data structures shared across all pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Mention:
    """A mention span within a document, with document-level token offsets."""
    sent_id: int
    start: int          # token start within sentence
    end: int            # token end within sentence (exclusive)
    doc_start: int      # token start within flattened document
    doc_end: int        # token end within flattened document (exclusive)
    type: str           # PER / ORG / LOC / TIME / NUM / MISC
    text: str

    @property
    def span(self) -> tuple[int, int]:
        return (self.doc_start, self.doc_end)


@dataclass
class Entity:
    """A document-level entity: a cluster of coreferent mentions."""
    idx: int
    mentions: list[Mention] = field(default_factory=list)

    @property
    def type(self) -> str:
        """Majority mention type of the cluster."""
        if not self.mentions:
            return "MISC"
        counts: dict[str, int] = {}
        for m in self.mentions:
            counts[m.type] = counts.get(m.type, 0) + 1
        return max(counts, key=lambda k: counts[k])

    @property
    def spans(self) -> set[tuple[int, int]]:
        return {m.span for m in self.mentions}


@dataclass(frozen=True)
class RelationTriple:
    head_idx: int       # entity index
    tail_idx: int
    relation: str       # Wikidata id, e.g. "P17"
    score: float = 1.0


@dataclass
class Document:
    """Parsed DocRED/Re-DocRED document in a common representation."""
    title: str
    sents: list[list[str]]                  # tokenized sentences
    entities: list[Entity]                  # gold entity clusters (empty for blind test)
    triples: list[RelationTriple]           # gold triples (empty for blind test)

    @property
    def tokens(self) -> list[str]:
        return [t for s in self.sents for t in s]

    @property
    def sent_offsets(self) -> list[int]:
        """Document-level token offset at which each sentence starts."""
        offs, acc = [], 0
        for s in self.sents:
            offs.append(acc)
            acc += len(s)
        return offs

    @property
    def mentions(self) -> list[Mention]:
        return [m for e in self.entities for m in e.mentions]
