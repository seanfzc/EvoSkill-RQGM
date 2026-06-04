"""Pluggable text-similarity backends for skill dedup and retrieval.

Two backends behind one interface:

* `EmbeddingSimilarity` — semantic similarity via a provider embedding model
  (the configured default). Embeddings are cached on disk by content hash so
  unchanged skills are never re-embedded — important because every API call
  costs money.
* `LexicalSimilarity` — pure-Python TF-IDF cosine (reuses the Phase 1
  tokenizer). Zero deps, offline, free. Used as the graceful fallback whenever
  embeddings are unavailable (no API key, unsupported provider, or import
  error) so lifecycle operations always work and the test-suite stays hermetic.

`make_similarity_backend()` picks the backend from config and degrades to
lexical rather than failing hard.
"""

from __future__ import annotations

import hashlib
import json
import math
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Callable

from .cluster import _tokenize

EmbedFn = Callable[[list[str]], list[list[float]]]

# Provider → OpenAI-compatible base_url. Providers absent here fall back to
# lexical similarity (their embeddings APIs are not OpenAI-compatible).
_OPENAI_COMPATIBLE_BASE_URLS: dict[str, str | None] = {
    "openai": None,
    "openrouter": "https://openrouter.ai/api/v1",
}


# ──────────────────────────────────────────────────────────────────────────────
# Backend interface
# ──────────────────────────────────────────────────────────────────────────────


class SimilarityBackend(ABC):
    """Computes similarity between short texts (skill descriptions / tasks)."""

    name: str = "similarity"

    @abstractmethod
    def pairwise(self, texts: list[str]) -> list[list[float]]:
        """Return an NxN cosine-similarity matrix for `texts`."""

    @abstractmethod
    def rank(self, query: str, candidates: list[str]) -> list[tuple[int, float]]:
        """Return (index, similarity) for each candidate, sorted high→low."""


def _cosine_dense(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _cosine_sparse(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = sum(w * b.get(t, 0.0) for t, w in a.items())
    na = math.sqrt(sum(w * w for w in a.values()))
    nb = math.sqrt(sum(w * w for w in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


# ──────────────────────────────────────────────────────────────────────────────
# Lexical backend
# ──────────────────────────────────────────────────────────────────────────────


class LexicalSimilarity(SimilarityBackend):
    """TF-IDF cosine over the supplied text set. No external dependencies."""

    name = "lexical"

    def _vectorize(self, texts: list[str]) -> list[dict[str, float]]:
        token_lists = [_tokenize(t) for t in texts]
        n = len(token_lists)
        df: Counter[str] = Counter()
        for tokens in token_lists:
            df.update(set(tokens))
        vectors: list[dict[str, float]] = []
        for tokens in token_lists:
            if not tokens:
                vectors.append({})
                continue
            tf = Counter(tokens)
            total = len(tokens)
            vec = {
                term: (count / total) * (math.log((1 + n) / (1 + df[term])) + 1.0)
                for term, count in tf.items()
            }
            vectors.append(vec)
        return vectors

    def pairwise(self, texts: list[str]) -> list[list[float]]:
        vecs = self._vectorize(texts)
        n = len(vecs)
        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            matrix[i][i] = 1.0 if vecs[i] else 0.0
            for j in range(i + 1, n):
                sim = _cosine_sparse(vecs[i], vecs[j])
                matrix[i][j] = matrix[j][i] = sim
        return matrix

    def rank(self, query: str, candidates: list[str]) -> list[tuple[int, float]]:
        vecs = self._vectorize([query, *candidates])
        qv = vecs[0]
        scored = [(i, _cosine_sparse(qv, vecs[i + 1])) for i in range(len(candidates))]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored


# ──────────────────────────────────────────────────────────────────────────────
# Embedding backend (+ disk cache)
# ──────────────────────────────────────────────────────────────────────────────


class EmbeddingCache:
    """Disk cache of text→embedding keyed by content hash."""

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path is not None else None
        self._cache: dict[str, list[float]] = {}
        self._load()

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _load(self) -> None:
        if self.path is None or not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text())
            if isinstance(data, dict):
                self._cache = {k: list(v) for k, v in data.items()}
        except (OSError, ValueError):
            self._cache = {}

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._cache))

    def get(self, text: str) -> list[float] | None:
        return self._cache.get(self._key(text))

    def put(self, text: str, vector: list[float]) -> None:
        self._cache[self._key(text)] = vector


class EmbeddingSimilarity(SimilarityBackend):
    """Semantic similarity via a (cached) embedding function."""

    name = "embedding"

    def __init__(self, embed_fn: EmbedFn, cache: EmbeddingCache | None = None) -> None:
        self._embed_fn = embed_fn
        self._cache = cache

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self._cache is None:
            return self._embed_fn(texts) if texts else []
        # Only embed cache misses, preserving order.
        missing = [t for t in texts if self._cache.get(t) is None]
        if missing:
            # De-dup misses before the API call.
            unique_missing = list(dict.fromkeys(missing))
            fresh = self._embed_fn(unique_missing)
            for text, vec in zip(unique_missing, fresh):
                self._cache.put(text, vec)
            self._cache.save()
        return [self._cache.get(t) or [] for t in texts]

    def pairwise(self, texts: list[str]) -> list[list[float]]:
        vecs = self._embed(texts)
        n = len(vecs)
        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            matrix[i][i] = 1.0
            for j in range(i + 1, n):
                sim = _cosine_dense(vecs[i], vecs[j])
                matrix[i][j] = matrix[j][i] = sim
        return matrix

    def rank(self, query: str, candidates: list[str]) -> list[tuple[int, float]]:
        vecs = self._embed([query, *candidates])
        qv = vecs[0]
        scored = [(i, _cosine_dense(qv, vecs[i + 1])) for i in range(len(candidates))]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored


# ──────────────────────────────────────────────────────────────────────────────
# Provider embedder + factory
# ──────────────────────────────────────────────────────────────────────────────


def make_openai_embedder(model: str, api_key: str, *, base_url: str | None = None) -> EmbedFn:
    """Return a sync embed function backed by an OpenAI-compatible API."""
    import openai  # imported lazily; raises ImportError if absent

    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def _embed(texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = client.embeddings.create(model=model, input=texts)
        return [list(item.embedding) for item in response.data]

    return _embed


def make_similarity_backend(
    *,
    backend: str = "embedding",
    provider: str = "openai",
    model: str = "text-embedding-3-small",
    api_key: str | None = None,
    cache_path: str | Path | None = None,
    log: Callable[[str], None] | None = None,
) -> SimilarityBackend:
    """Build the configured similarity backend, degrading to lexical on any gap.

    Falls back to `LexicalSimilarity` (with a logged reason) when:
      - backend != "embedding", or
      - no API key is available, or
      - the provider is not OpenAI-compatible, or
      - the openai package cannot be imported.
    """
    emit = log or (lambda _m: None)

    if backend != "embedding":
        return LexicalSimilarity()

    provider_norm = (provider or "").strip().lower()
    if provider_norm not in _OPENAI_COMPATIBLE_BASE_URLS:
        emit(f"embedding provider '{provider}' is not OpenAI-compatible; using lexical similarity.")
        return LexicalSimilarity()
    if not api_key:
        emit("no embedding API key found; using lexical similarity.")
        return LexicalSimilarity()

    try:
        embedder = make_openai_embedder(
            model, api_key, base_url=_OPENAI_COMPATIBLE_BASE_URLS[provider_norm]
        )
    except ImportError:
        emit("openai package not installed; using lexical similarity.")
        return LexicalSimilarity()

    cache = EmbeddingCache(cache_path) if cache_path is not None else None
    return EmbeddingSimilarity(embedder, cache)
