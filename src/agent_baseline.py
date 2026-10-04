from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens, extract_profile_updates


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent offline: chỉ nhớ lịch sử trong cùng thread."""

    def __init__(
        self,
        config: LabConfig | None = None,
        force_offline: bool = False,
    ) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None

    def reply(
        self,
        user_id: str,
        thread_id: str,
        message: str,
    ) -> dict[str, Any]:
        # Bản hiện tại chỉ triển khai offline.
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.token_usage if session else 0

    def prompt_token_usage(self, thread_id: str) -> int:
        session = self.sessions.get(thread_id)
        return session.prompt_tokens_processed if session else 0

    def compaction_count(self, thread_id: str) -> int:
        return 0

    def _reply_offline(
        self,
        thread_id: str,
        message: str,
    ) -> dict[str, Any]:
        session = self.sessions.setdefault(
            thread_id, SessionState()
        )

        session.messages.append({
            "role": "user",
            "content": message,
        })

        prompt_tokens = sum(
            estimate_tokens(item["content"])
            for item in session.messages
        )

        # Khôi phục fact từ lịch sử của đúng thread hiện tại.
        # Không đọc User.md và không dùng lịch sử thread khác.
        facts = {}
        for item in session.messages:
            if item["role"] == "user":
                facts.update(
                    extract_profile_updates(item["content"])
                )

        question = message.casefold()

        is_recall = (
            "?" in message
            or "nhắc lại" in question
            or "nhớ gì" in question
            or "nhớ những" in question
        )

        if is_recall:
            if facts:
                answer = "Thông tin trong phiên này:\n" + "\n".join(
                    f"- {key}: {value}"
                    for key, value in sorted(facts.items())
                )
            else:
                answer = (
                    "Mình chưa có thông tin đó trong phiên này."
                )
        else:
            answer = "Mình đã ghi nhận thông tin trong phiên này."

        output_tokens = estimate_tokens(answer)

        session.token_usage += output_tokens
        session.prompt_tokens_processed += prompt_tokens
        session.messages.append({
            "role": "assistant",
            "content": answer,
        })

        return {
            "answer": answer,
            "agent_tokens": output_tokens,
            "prompt_tokens": prompt_tokens,
        }

    def _maybe_build_langchain_agent(self):
        # Phần live chưa triển khai.
        return None
