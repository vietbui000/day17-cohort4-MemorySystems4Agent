from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    # config.py nằm trong src/, nên đi lên hai cấp để lấy root repo.
    root = (
        Path(base_dir)
        if base_dir is not None
        else Path(__file__).resolve().parent.parent
    ).resolve()

    # Không có .env vẫn chạy được.
    load_dotenv(root / ".env", override=False)

    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    threshold = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "2000"))
    keep_messages = int(os.getenv("COMPACT_KEEP_MESSAGES", "6"))

    if threshold <= 0:
        raise ValueError("COMPACT_THRESHOLD_TOKENS phải lớn hơn 0.")

    if keep_messages < 1:
        raise ValueError("COMPACT_KEEP_MESSAGES phải từ 1 trở lên.")

    provider = normalize_provider(
        os.getenv("LLM_PROVIDER", "openai")
    )

    # Chỉ là tên cấu hình mặc định; chưa gọi API ở bước này.
    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")

    key_variables = {
        "openai": "OPENAI_API_KEY",
        "custom": "CUSTOM_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }

    url_variables = {
        "openai": "OPENAI_BASE_URL",
        "custom": "CUSTOM_BASE_URL",
        "gemini": "GEMINI_BASE_URL",
        "anthropic": "ANTHROPIC_BASE_URL",
        "ollama": "OLLAMA_BASE_URL",
        "openrouter": "OPENROUTER_BASE_URL",
    }

    key_variable = key_variables.get(provider)
    api_key = os.getenv(key_variable) if key_variable else None

    base_url = os.getenv(url_variables[provider])

    if provider == "ollama" and not base_url:
        base_url = "http://localhost:11434"

    model = ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
        api_key=api_key,
        base_url=base_url,
    )

    # Tạm dùng cùng provider cho model chính và model chấm.
    judge_model = ProviderConfig(
        provider=provider,
        model_name=os.getenv("JUDGE_MODEL", model_name),
        temperature=0.0,
        api_key=api_key,
        base_url=base_url,
    )

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=threshold,
        compact_keep_messages=keep_messages,
        model=model,
        judge_model=judge_model,
    )
