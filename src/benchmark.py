from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from tabulate import tabulate

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"{path} phải chứa một danh sách hội thoại.")

    return data


def recall_points(answer: str, expected: list[str]) -> float:
    """Không khớp: 0; khớp một phần: 0.5; khớp đủ: 1."""
    if not expected:
        return 0.0

    normalized = answer.casefold()
    hits = sum(
        fact.casefold() in normalized
        for fact in expected
    )

    if hits == 0:
        return 0.0
    if hits == len(expected):
        return 1.0
    return 0.5


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Proxy offline: tỷ lệ fact mong đợi có trong câu trả lời.

    Không đánh giá đầy đủ độ tự nhiên hay tính đúng đắn của câu trả lời.
    """
    if not answer.strip() or not expected:
        return 0.0

    normalized = answer.casefold()
    hits = sum(
        fact.casefold() in normalized
        for fact in expected
    )
    return hits / len(expected)


def run_agent_benchmark(
    agent_name: str,
    agent,
    conversations: list[dict[str, Any]],
    config,
) -> BenchmarkRow:
    user_ids = {conv["user_id"] for conv in conversations}
    thread_ids: set[str] = set()
    recall_scores: list[float] = []
    quality_scores: list[float] = []

    # Baseline không có file memory.
    def memory_size() -> int:
        if not hasattr(agent, "memory_file_size"):
            return 0

        return sum(
            agent.memory_file_size(user_id)
            for user_id in user_ids
        )

    initial_memory_size = memory_size()

    for index, conversation in enumerate(conversations):
        user_id = conversation["user_id"]

        # Mỗi hội thoại có thread riêng.
        thread_id = f"{agent_name}:conversation:{index}"
        thread_ids.add(thread_id)

        for message in conversation["turns"]:
            agent.reply(user_id, thread_id, message)

        # Hỏi recall ngay sau hội thoại tương ứng,
        # nhưng dùng thread mới để kiểm tra nhớ xuyên phiên.
        for question_index, item in enumerate(
            conversation.get("recall_questions", [])
        ):
            recall_thread = (
                f"{agent_name}:recall:{index}:{question_index}"
            )
            thread_ids.add(recall_thread)

            result = agent.reply(
                user_id,
                recall_thread,
                item["question"],
            )

            # Quy ước dùng chung cho cả hai agent:
            # reply() trả {"answer": "...", ...}
            answer = result["answer"]
            expected = item["expected_contains"]

            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(heuristic_quality(answer, expected))

    def average(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=sum(
            agent.token_usage(thread_id)
            for thread_id in thread_ids
        ),
        prompt_tokens_processed=sum(
            agent.prompt_token_usage(thread_id)
            for thread_id in thread_ids
        ),
        recall_score=average(recall_scores),
        response_quality=average(quality_scores),
        memory_growth_bytes=memory_size() - initial_memory_size,
        compactions=sum(
            agent.compaction_count(thread_id)
            for thread_id in thread_ids
        ),
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]

    values = [
        [
            row.agent_name,
            row.agent_tokens_only,
            row.prompt_tokens_processed,
            f"{row.recall_score:.1%}",
            f"{row.response_quality:.1%}",
            row.memory_growth_bytes,
            row.compactions,
        ]
        for row in rows
    ]

    return tabulate(values, headers=headers, tablefmt="github")


def main() -> None:
    config = load_config(Path(__file__).resolve().parent.parent)

    suites = [
        ("Standard Benchmark", "conversations.json"),
        ("Long-Context Stress Benchmark", "advanced_long_context.json"),
    ]

    for title, filename in suites:
        conversations = load_conversations(config.data_dir / filename)
        rows: list[BenchmarkRow] = []

        # Trạng thái sạch cho mỗi bộ benchmark.
        # Tự xóa khi kết thúc; không đụng đến state/ hiện có.
        with TemporaryDirectory(prefix="day17-benchmark-") as temp_dir:
            for name, agent_class in [
                ("Baseline", BaselineAgent),
                ("Advanced", AdvancedAgent),
            ]:
                state_dir = Path(temp_dir) / name.lower()
                state_dir.mkdir(parents=True, exist_ok=True)

                agent_config = replace(config, state_dir=state_dir)

                agent = agent_class(
                    config=agent_config,
                    force_offline=True,
                )

                rows.append(
                    run_agent_benchmark(
                        name,
                        agent,
                        conversations,
                        agent_config,
                    )
                )

            print(f"\n{title}")
            print(format_rows(rows))


if __name__ == "__main__":
    main()
