"""Retrieval: hybrid search → access filter → cross-encoder rerank → evidence check."""
import math
from dataclasses import dataclass
from functools import lru_cache

from qdrant_client import models
from sentence_transformers import CrossEncoder

from app.config import get_settings
from app.db.models import Role
from app.rag.vectorstore import get_store

CANDIDATES_K = 20      # wide net from hybrid search (cheap)
TOP_N = 4              # what the LLM actually sees after reranking (precise)
MIN_RELEVANCE = 0.05

ROLE_ACCESS: dict[Role, list[str]] = {
    Role.CUSTOMER: ["public"],
    Role.SUPPORT_AGENT: ["public", "internal"],
    Role.ADMIN: ["public", "internal"],
}


@dataclass
class RetrievedChunk:
    text: str
    source: str
    title: str
    section: str
    relevance: float   # 0..1, from the reranker

    @property
    def citation(self) -> str:
        return f"{self.title} > {self.section} ({self.source})"


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    sufficient: bool   # False → the agent should rewrite the query or fall back

    @property
    def top_relevance(self) -> float:
        return self.chunks[0].relevance if self.chunks else 0.0


@lru_cache
def _reranker() -> CrossEncoder:
    return CrossEncoder(get_settings().reranker_model)


def _access_filter(role: Role) -> models.Filter:
    # LangChain stores document metadata under the "metadata" payload key.
    return models.Filter(
        must=[models.FieldCondition(key="metadata.access",
                                    match=models.MatchAny(any=ROLE_ACCESS[role]))],
        must_not=[models.FieldCondition(key="metadata.quarantined",
                                        match=models.MatchValue(value=True))],
    )


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def retrieve(query: str, role: Role) -> RetrievalResult:
    # 1. Hybrid search (dense + BM25, fused inside Qdrant), already role-filtered.
    candidates = get_store().similarity_search(query, k=CANDIDATES_K, filter=_access_filter(role))
    if not candidates:
        return RetrievalResult(chunks=[], sufficient=False)

    raw = [float(s) for s in _reranker().predict([(query, d.page_content) for d in candidates])]
    if any(s < 0 or s > 1 for s in raw):
        raw = [_sigmoid(s) for s in raw]
    ranked = sorted(zip(candidates, raw), key=lambda pair: pair[1], reverse=True)[:TOP_N]

    chunks = [
        RetrievedChunk(text=d.page_content, source=d.metadata["source"],
                       title=d.metadata["title"], section=d.metadata["section"],
                       relevance=round(s, 3))
        for d, s in ranked
    ]
    return RetrievalResult(chunks=chunks, sufficient=chunks[0].relevance >= MIN_RELEVANCE)


MAX_MERGED = 6   # context cap when several sub-queries are merged


def retrieve_many(queries: list[str], role: Role) -> RetrievalResult:
    """Retrieve for each sub-query of a multi-part question, then merge."""
    results = [retrieve(q, role) for q in queries]
    if len(results) == 1:
        return results[0]
    good = [r for r in results if r.sufficient]
    if not good:
        return RetrievalResult(chunks=[], sufficient=False)

    best: dict[tuple[str, str], RetrievedChunk] = {}
    for c in (c for r in good for c in r.chunks):
        key = (c.source, c.section)
        if key not in best or c.relevance > best[key].relevance:
            best[key] = c
    chunks = sorted(best.values(), key=lambda c: c.relevance, reverse=True)[:MAX_MERGED]
    return RetrievalResult(chunks=chunks, sufficient=True)
