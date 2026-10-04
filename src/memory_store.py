from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Ước lượng tất định, dùng chung cho hai agent."""
    text = text.strip()
    return (len(text) + 3) // 4 if text else 0


@dataclass
class UserProfileStore:
    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        # Slug giúp dễ đọc; hash tránh hai ID khác nhau trùng đường dẫn.
        slug = re.sub(r"[^a-zA-Z0-9_-]", "_", user_id)[:40]
        digest = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:16]
        folder = f"{slug or 'user'}_{digest}"
        return Path(self.root_dir) / folder / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(
        self,
        user_id: str,
        search_text: str,
        replacement: str,
    ) -> bool:
        if not search_text or search_text == replacement:
            return False

        content = self.read_text(user_id)
        if search_text not in content:
            return False

        self.write_text(
            user_id,
            content.replace(search_text, replacement, 1),
        )
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        result = {}
        for line in self.read_text(user_id).splitlines():
            if line.startswith("- ") and ": " in line:
                key, value = line[2:].split(": ", 1)
                result[key] = value
        return result

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        # Cập nhật theo khóa: fact mới thay fact cũ.
        facts = self.facts(user_id)
        facts[key] = " ".join(value.split())

        content = "# User Profile\n\n"
        content += "\n".join(
            f"- {name}: {value}"
            for name, value in sorted(facts.items())
        )
        self.write_text(user_id, content + "\n")


def extract_profile_updates(message: str) -> dict[str, str]:
    """Trích fact bằng regex; chỉ là heuristic cho bài lab."""
    updates = {}

    # Xét từng câu để không bỏ mất một câu cung cấp fact
    # chỉ vì message còn chứa một câu hỏi.
    sentences = re.split(r"[.!?\n]+", message)

    patterns = {
        "name": (
            r"(?:mình|tôi)\s+tên(?:\s+là)?\s+"
            r"([^,;]+?)(?=\s+và\s+|$)"
        ),
        "location": (
            r"(?:mình|tôi)\s+(?:hiện tại\s+|hiện\s+)?"
            r"(?:đang\s+)?(?:sống\s+ở|ở|"
            r"(?:đã\s+)?chuyển\s+(?:đến|sang|về))\s+"
            r"([^,;]+?)(?=\s+và\s+|$)"
        ),
        "profession": (
            r"(?:mình|tôi)\s+(?:hiện tại\s+|hiện\s+)?"
            r"(?:đang\s+)?làm\s+"
            r"([^,;]+?)(?=\s+(?:cho|tại|ở)\s+|\s+và\s+|$)"
        ),
        "favorite_drink": (
            r"đồ uống\s+(?:yêu thích|ưa thích)"
            r"(?:\s+của\s+(?:mình|tôi))?\s+là\s+([^;]+)"
        ),
        "favorite_food": r"món ăn\s+(?:yêu thích|ưa thích)\s+là\s+([^;]+)",
        "pet": r"(?:mình|tôi)\s+nuôi\s+([^;]+)",
        "interests": r"(?:mình|tôi)\s+thích\s+([^;]+)",
        "response_style": (
            r"(?:mình|tôi)\s+muốn\s+bạn\s+trả lời\s+([^;]+)"
        ),
        "routine": (
            r"(?:mình|tôi)\s+(?:thường|hay)\s+([^;]+)"
        ),
        "learning_goal": (
            r"mục tiêu(?:\s+học tập)?[^;]*?\s+là\s+([^;]+)"
        ),
    }

    for sentence in sentences:
        sentence = sentence.strip()
        lower = sentence.casefold()

        if not sentence:
            continue

        # Tránh các câu đùa, giả định và câu hỏi thường gặp.
        if any(word in lower for word in (
            "nói đùa", "đùa thôi", "giả sử", "ví dụ",
        )):
            continue

        if re.search(
            r"\b(?:gì|đâu|không nhỉ)\b|"
            r"^(?:nhắc lại|bạn nhớ|bạn có nhớ)",
            lower,
        ):
            continue

        for key, pattern in patterns.items():
            match = re.search(pattern, sentence, flags=re.IGNORECASE)
            if match:
                # A negated old fact is a correction, not a new value.
                if key == "profession" and re.search(
                    r"(?:mình|tôi)\s+không\s+còn\s+làm", lower
                ):
                    continue
                value = match.group(1).strip(" ,;")
                value = re.sub(
                    r"\s+rồi$",
                    "",
                    value,
                    flags=re.IGNORECASE,
                )
                if value:
                    updates[key] = value

        # Corrections commonly use "giờ chuyển sang" without repeating "làm".
        profession_correction = re.search(
            r"(?:giờ|hiện tại)\s+(?:mình|tôi)\s+(?:đã\s+)?"
            r"chuyển\s+sang\s+([^,;]+)$",
            sentence,
            flags=re.IGNORECASE,
        )
        if profession_correction:
            updates["profession"] = profession_correction.group(1).strip()

    return updates


def answer_from_facts(message: str, facts: dict[str, str]) -> str:
    """Helper dùng chung để hai agent có cùng quy tắc trả lời."""
    question = message.casefold()
    requested = []

    keywords = {
        "name": ("tên",),
        "location": ("ở đâu", "nơi ở", "thành phố", "sống ở"),
        "profession": ("nghề", "công việc", "làm gì"),
        "favorite_drink": ("đồ uống", "uống gì"),
        "response_style": ("style", "phong cách", "trả lời"),
        "interests": ("sở thích", "quan tâm", "thích gì"),
        "routine": ("thói quen", "chạy bộ", "mấy giờ"),
        "learning_goal": ("mục tiêu",),
    }

    is_question = (
        "?" in message
        or any(word in question for word in (
            "nhắc lại", "bạn nhớ", "bạn có nhớ",
        ))
    )

    if not is_question:
        return "Mình đã tiếp nhận thông tin bạn vừa chia sẻ."

    for key, words in keywords.items():
        if any(word in question for word in words):
            requested.append(key)

    values = [
        f"{key}: {facts[key]}"
        for key in requested
        if key in facts
    ]

    if values:
        return "; ".join(values)

    return "Mình chưa có đủ thông tin để trả lời câu hỏi này."


def summarize_messages(
    messages: list[dict[str, str]],
    max_items: int = 6,
) -> str:
    """Tóm tắt heuristic có giới hạn kích thước.

    Giữ các đoạn gần nhất của phần bị nén.
    Có thể mất chi tiết; không thay thế persistent profile.
    """
    if max_items <= 0:
        return ""

    items = []
    for message in messages:
        content = " ".join(message["content"].split())
        if content:
            items.append(f'{message["role"]}: {content[:160]}')

    return "\n".join(items[-max_items:])


@dataclass
class CompactMemoryManager:
    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.threshold_tokens <= 0 or self.keep_messages < 1:
            raise ValueError("Ngưỡng phải > 0 và keep_messages phải >= 1.")

    def append(self, thread_id: str, role: str, content: str) -> None:
        context = self.context(thread_id)
        messages = context["messages"]
        messages.append({"role": role, "content": content})

        total = estimate_tokens(context["summary"]) + sum(
            estimate_tokens(item["content"])
            for item in messages
        )

        if (
            total > self.threshold_tokens
            and len(messages) > self.keep_messages
        ):
            older = messages[:-self.keep_messages]
            recent = messages[-self.keep_messages:]

            summary_input = []
            if context["summary"]:
                summary_input.append({
                    "role": "summary",
                    "content": context["summary"],
                })
            summary_input.extend(older)

            context["summary"] = summarize_messages(summary_input)
            context["messages"] = recent
            context["compactions"] += 1

    def context(self, thread_id: str) -> dict[str, object]:
        return self.state.setdefault(
            thread_id,
            {"messages": [], "summary": "", "compactions": 0},
        )

    def compaction_count(self, thread_id: str) -> int:
        return int(self.context(thread_id)["compactions"])
