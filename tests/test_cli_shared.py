from __future__ import annotations

import asyncio
import sys
import types


def test_infer_provider_detects_prefixed_models() -> None:
    from src.cli.shared import infer_provider

    assert infer_provider("openrouter/openai/gpt-5-mini") == "openrouter"
    assert infer_provider("anthropic/claude-sonnet-4-6") == "anthropic"
    assert infer_provider("openai/gpt-4o-mini") == "openai"
    assert infer_provider("google/gemini-2.5-flash") == "google"
    assert (
        infer_provider("fireworks_ai/accounts/fireworks/models/llama-v3p1-70b-instruct")
        == "fireworks"
    )
    assert infer_provider("fireworks-ai/accounts/fireworks/models/qwen3-8b") == "fireworks"
    assert infer_provider("accounts/fireworks/models/qwen3-8b") == "fireworks"


def test_normalize_provider_model_strips_known_prefixes() -> None:
    from src.cli.shared import _normalize_provider_model

    assert (
        _normalize_provider_model("openrouter", "openrouter/openai/gpt-5-mini")
        == "openai/gpt-5-mini"
    )
    assert (
        _normalize_provider_model("anthropic", "anthropic/claude-sonnet-4-6")
        == "claude-sonnet-4-6"
    )
    assert (
        _normalize_provider_model("openai", "openai/gpt-4o-mini")
        == "gpt-4o-mini"
    )
    assert (
        _normalize_provider_model("google", "google/gemini-2.5-flash")
        == "gemini-2.5-flash"
    )
    assert (
        _normalize_provider_model(
            "fireworks", "fireworks_ai/accounts/fireworks/models/qwen3-8b"
        )
        == "accounts/fireworks/models/qwen3-8b"
    )


def test_call_llm_openrouter_uses_openai_compatible_client(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeCompletions:
        async def create(self, *, model, max_tokens, messages):
            captured["model"] = model
            captured["max_tokens"] = max_tokens
            captured["messages"] = messages
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="0.8"))]
            )

    class FakeChat:
        def __init__(self) -> None:
            self.completions = FakeCompletions()

    class FakeAsyncOpenAI:
        def __init__(self, *, base_url=None, api_key=None, default_headers=None):
            captured["base_url"] = base_url
            captured["api_key"] = api_key
            captured["default_headers"] = default_headers
            self.chat = FakeChat()

    monkeypatch.setitem(
        sys.modules,
        "openai",
        types.SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI),
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("OPENROUTER_HTTP_REFERER", "https://example.com/evoskill")
    monkeypatch.setenv("OPENROUTER_TITLE", "EvoSkill Tests")

    from src.cli.shared import call_llm

    result = asyncio.run(
        call_llm(
            "openrouter",
            "openrouter/openai/gpt-5-mini",
            "Reply with a score.",
        )
    )

    assert result == "0.8"
    assert captured["base_url"] == "https://openrouter.ai/api/v1"
    assert captured["api_key"] == "test-openrouter-key"
    assert captured["default_headers"] == {
        "HTTP-Referer": "https://example.com/evoskill",
        "X-OpenRouter-Title": "EvoSkill Tests",
    }
    assert captured["model"] == "openai/gpt-5-mini"
    assert captured["max_tokens"] == 16
    assert captured["messages"] == [{"role": "user", "content": "Reply with a score."}]


def test_call_llm_fireworks_uses_openai_compatible_client(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeCompletions:
        async def create(self, *, model, max_tokens, messages):
            captured["model"] = model
            captured["max_tokens"] = max_tokens
            captured["messages"] = messages
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="0.5"))]
            )

    class FakeChat:
        def __init__(self) -> None:
            self.completions = FakeCompletions()

    class FakeAsyncOpenAI:
        def __init__(self, *, base_url=None, api_key=None, default_headers=None):
            captured["base_url"] = base_url
            captured["api_key"] = api_key
            captured["default_headers"] = default_headers
            self.chat = FakeChat()

    monkeypatch.setitem(
        sys.modules,
        "openai",
        types.SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI),
    )
    monkeypatch.delenv("FIREWORKS_AI_API_KEY", raising=False)
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-fireworks-key")

    from src.cli.shared import call_llm

    result = asyncio.run(
        call_llm(
            "fireworks",
            "fireworks_ai/accounts/fireworks/models/qwen3-8b",
            "Reply with a score.",
        )
    )

    assert result == "0.5"
    assert captured["base_url"] == "https://api.fireworks.ai/inference/v1"
    assert captured["api_key"] == "test-fireworks-key"
    assert captured["model"] == "accounts/fireworks/models/qwen3-8b"
    assert captured["max_tokens"] == 16
    assert captured["messages"] == [{"role": "user", "content": "Reply with a score."}]
