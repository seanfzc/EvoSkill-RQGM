"""Tests for similarity backends, embedding cache, and the factory."""

from __future__ import annotations

from src.continuous.similarity import (
    EmbeddingCache,
    EmbeddingSimilarity,
    LexicalSimilarity,
    make_openai_embedder,
    make_similarity_backend,
)


class TestLexicalSimilarity:
    def test_rank_orders_by_overlap(self):
        b = LexicalSimilarity()
        ranked = b.rank(
            "measurement units for revenue",
            ["always include measurement units", "unrelated cooking recipe steps"],
        )
        assert ranked[0][0] == 0  # the units candidate ranks first
        assert ranked[0][1] >= ranked[1][1]

    def test_pairwise_diagonal_and_symmetry(self):
        b = LexicalSimilarity()
        m = b.pairwise(["alpha beta gamma", "alpha beta gamma", "totally different words"])
        assert m[0][0] == 1.0
        assert m[0][1] == m[1][0]
        assert m[0][1] > m[0][2]

    def test_empty_text_has_zero_self_similarity(self):
        b = LexicalSimilarity()
        m = b.pairwise(["", "real words here"])
        assert m[0][0] == 0.0  # empty doc → zero vector

    def test_rank_empty_candidates(self):
        assert LexicalSimilarity().rank("q", []) == []


class TestEmbeddingCache:
    def test_put_get_roundtrip(self, tmp_path):
        c = EmbeddingCache(tmp_path / "e.json")
        assert c.get("x") is None
        c.put("x", [1.0, 2.0])
        c.save()
        reloaded = EmbeddingCache(tmp_path / "e.json")
        assert reloaded.get("x") == [1.0, 2.0]

    def test_corrupt_file_is_empty(self, tmp_path):
        (tmp_path / "e.json").write_text("{bad")
        assert EmbeddingCache(tmp_path / "e.json").get("x") is None

    def test_none_path_is_noop(self):
        c = EmbeddingCache(None)
        c.put("x", [1.0])
        c.save()  # must not raise
        assert c.get("x") == [1.0]


class _CountingEmbedder:
    def __init__(self):
        self.calls = 0
        self.total_texts = 0

    def __call__(self, texts):
        self.calls += 1
        self.total_texts += len(texts)
        # deterministic vector by length + first ordinal
        return [[float(len(t)), float(ord(t[0]) if t else 0)] for t in texts]


class TestEmbeddingSimilarity:
    def test_rank_uses_embeddings(self):
        emb = EmbeddingSimilarity(_CountingEmbedder())
        ranked = emb.rank("hello", ["hello", "x"])
        assert ranked[0][0] == 0  # identical text most similar

    def test_cache_avoids_reembedding(self, tmp_path):
        embedder = _CountingEmbedder()
        cache = EmbeddingCache(tmp_path / "e.json")
        emb = EmbeddingSimilarity(embedder, cache)
        emb.rank("a", ["bb", "ccc"])
        first_calls = embedder.calls
        emb.rank("a", ["bb", "ccc"])  # all cached now
        assert embedder.calls == first_calls  # no new API call

    def test_dedups_misses_before_calling(self, tmp_path):
        embedder = _CountingEmbedder()
        emb = EmbeddingSimilarity(embedder, EmbeddingCache(tmp_path / "e.json"))
        emb.pairwise(["dup", "dup", "other"])
        # only 2 unique texts embedded despite 3 inputs
        assert embedder.total_texts == 2

    def test_no_cache_path(self):
        emb = EmbeddingSimilarity(_CountingEmbedder(), None)
        assert emb.rank("q", ["q"])[0][1] > 0.9


class TestFactory:
    def test_lexical_backend_requested(self):
        b = make_similarity_backend(backend="lexical")
        assert b.name == "lexical"

    def test_no_api_key_degrades_to_lexical(self):
        b = make_similarity_backend(backend="embedding", provider="openai", api_key=None)
        assert b.name == "lexical"

    def test_unsupported_provider_degrades(self):
        b = make_similarity_backend(backend="embedding", provider="google", api_key="k")
        assert b.name == "lexical"

    def test_embedding_backend_built_with_key(self, tmp_path):
        b = make_similarity_backend(
            backend="embedding", provider="openai", api_key="sk-test",
            cache_path=str(tmp_path / "e.json"),
        )
        assert b.name == "embedding"  # constructed; no network until used


class TestOpenAIEmbedder:
    def test_returns_callable_and_empty_is_noop(self):
        embed = make_openai_embedder("text-embedding-3-small", "sk-test")
        assert callable(embed)
        assert embed([]) == []  # short-circuits before any network call
