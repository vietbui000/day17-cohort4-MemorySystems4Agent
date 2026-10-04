from __future__ import annotations

from pathlib import Path
from dataclasses import replace

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import UserProfileStore


def make_config(tmp_path: Path):
    """Use an isolated profile and a small compaction threshold."""
    return replace(
        load_config(Path(__file__).resolve().parent.parent),
        state_dir=tmp_path / "state",
        compact_threshold_tokens=120,
        compact_keep_messages=4,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    assert store.read_text("dungct") == ""
    path = store.write_text("dungct", "# User Profile\n\n- location: Đà Nẵng\n")
    assert path.name == "User.md"
    assert store.file_size("dungct") > 0
    assert store.edit_text("dungct", "Đà Nẵng", "Huế")
    assert store.facts("dungct") == {"location": "Huế"}
    assert not store.edit_text("dungct", "Đà Nẵng", "Hà Nội")
    store.upsert_fact("dungct", "location", "Đà Nẵng")
    assert store.facts("dungct") == {"location": "Đà Nẵng"}


def test_compact_trigger(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    for _ in range(8):
        agent.reply("user", "long-thread", "Mình đang đọc một đoạn ngữ cảnh dài về memory systems. " * 5)
    context = agent.compact_memory.context("long-thread")
    assert agent.compaction_count("long-thread") > 0
    assert context["summary"]
    assert len(context["messages"]) <= agent.config.compact_keep_messages


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    fact = "Mình tên là DũngCT."
    baseline.reply("user", "first", fact)
    advanced.reply("user", "first", fact)
    assert "DũngCT" not in baseline.reply("user", "second", "Tên mình là gì?")["answer"]
    assert "DũngCT" in advanced.reply("user", "second", "Tên mình là gì?")["answer"]
    assert advanced.profile_store.facts("user")["name"] == "DũngCT"


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    message = "Đây là đoạn ngữ cảnh dài để kiểm tra chi phí prompt của memory agent. " * 12
    for _ in range(12):
        baseline.reply("user", "long", message)
        advanced.reply("user", "long", message)
    assert advanced.compaction_count("long") > 0
    assert advanced.prompt_token_usage("long") < baseline.prompt_token_usage("long")
