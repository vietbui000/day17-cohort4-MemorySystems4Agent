from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Provider configuration shared by the agents."""

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    provider = value.strip().lower()

    aliases = {
        "anthorpic": "anthropic",
        "google": "gemini",
        "google-genai": "gemini",
        "openai-compatible": "custom",
    }
    provider = aliases.get(provider, provider)

    supported = {
        "openai",
        "custom",
        "gemini",
        "anthropic",
        "ollama",
        "openrouter",
    }

    if provider not in supported:
        raise ValueError(f"Provider không được hỗ trợ: {value!r}")

    return provider


def build_chat_model(config: ProviderConfig):
    """Build a chat model lazily so the offline benchmark needs no SDK or key."""
    provider = normalize_provider(config.provider)
    common = {"model": config.model_name, "temperature": config.temperature}

    if provider in {"openai", "custom"}:
        from langchain_openai import ChatOpenAI

        if provider == "custom" and not config.base_url:
            raise ValueError("CUSTOM_BASE_URL is required for custom provider")
        return ChatOpenAI(
            **common,
            **({"api_key": config.api_key} if config.api_key else {}),
            **({"base_url": config.base_url} if config.base_url else {}),
        )
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            **common,
            **({"google_api_key": config.api_key} if config.api_key else {}),
        )
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            **common,
            **({"api_key": config.api_key} if config.api_key else {}),
        )
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            **common,
            **({"base_url": config.base_url} if config.base_url else {}),
        )

    from langchain_openrouter import ChatOpenRouter

    return ChatOpenRouter(
        **common,
        **({"api_key": config.api_key} if config.api_key else {}),
    )
