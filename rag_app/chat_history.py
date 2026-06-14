from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip().lower())
    return cleaned.strip("._-") or uuid4().hex


def _summarize_text(text: str, limit: int = 48) -> str:
    words = re.sub(r"\s+", " ", text).strip().split()
    if not words:
        return "New chat"
    summary = " ".join(words[:limit])
    if len(words) > limit:
        summary += "..."
    return summary


def _title_case_title(text: str) -> str:
    return " ".join(word.capitalize() for word in text.split())


@dataclass(slots=True)
class ChatHistoryStore:
    base_dir: Path

    def __post_init__(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, conversation_id: str) -> Path:
        return self.base_dir / f"{_safe_filename(conversation_id)}.json"

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                return data
        except Exception as exc:
            print(f"Failed to load chat history file {path}: {exc}")
        return None

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    def list_conversations(self) -> list[dict[str, Any]]:
        conversations: list[dict[str, Any]] = []
        for path in self.base_dir.glob("*.json"):
            payload = self._read_json(path)
            if not payload:
                continue
            messages = payload.get("messages", [])
            if not isinstance(messages, list):
                messages = []
            last_message = messages[-1] if messages else {}
            last_result = payload.get("last_result")
            last_intent = None
            if isinstance(last_result, dict):
                last_intent = last_result.get("intent")
            conversations.append(
                {
                    "id": payload.get("id", path.stem),
                    "title": payload.get("title", "New chat"),
                    "created_at": payload.get("created_at"),
                    "updated_at": payload.get("updated_at"),
                    "message_count": len(messages),
                    "last_intent": last_intent,
                    "last_preview": (last_message.get("content", "") or "").strip()[:90],
                }
            )
        conversations.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
        return conversations

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        return self._read_json(self._path(conversation_id))

    def create_conversation(self, title: str = "New chat") -> dict[str, Any]:
        conversation_id = uuid4().hex
        payload = {
            "id": conversation_id,
            "title": title,
            "created_at": _now(),
            "updated_at": _now(),
            "messages": [],
            "last_result": None,
        }
        self.save_conversation(payload)
        return payload

    def save_conversation(self, conversation: dict[str, Any]) -> dict[str, Any]:
        conversation = dict(conversation)
        conversation.setdefault("id", uuid4().hex)
        conversation.setdefault("title", "New chat")
        conversation.setdefault("created_at", _now())
        conversation["updated_at"] = _now()
        conversation.setdefault("messages", [])
        conversation.setdefault("last_result", None)
        self._write_json(self._path(str(conversation["id"])), conversation)
        return conversation

    def rename_conversation(self, conversation_id: str, title: str) -> dict[str, Any] | None:
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return None
        conversation["title"] = title.strip() or conversation.get("title", "New chat")
        return self.save_conversation(conversation)

    def delete_conversation(self, conversation_id: str) -> bool:
        path = self._path(conversation_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    def append_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return None

        messages = list(conversation.get("messages", []))
        messages.append(
            {
                "role": role,
                "content": content,
                "meta": meta or {},
            }
        )
        conversation["messages"] = messages

        if role == "user" and conversation.get("title") in (None, "", "New chat"):
            conversation["title"] = _title_case_title(_summarize_text(content, limit=7))

        return self.save_conversation(conversation)

    def update_last_result(self, conversation_id: str, result: dict[str, Any]) -> dict[str, Any] | None:
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return None
        conversation["last_result"] = result

        question = str(result.get("question") or "").strip()
        if question and conversation.get("title") in (None, "", "New chat"):
            conversation["title"] = _title_case_title(_summarize_text(question, limit=7))

        return self.save_conversation(conversation)
