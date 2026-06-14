from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from rag_app.chat_history import ChatHistoryStore
from rag_app.config import load_config
from rag_app.copilot import run_financial_copilot_with_mode


st.set_page_config(
    page_title="Financial Copilot",
    page_icon="FC",
    layout="wide",
    initial_sidebar_state="expanded",
)


QUICK_PROMPTS = [
    {
        "label": "Compare Apple vs Google",
        "prompt": "Compare Apple and Google and give the key financial insights.",
    },
    {
        "label": "Forecast Amazon",
        "prompt": "Forecast Amazon revenue for the next quarter.",
    },
    {
        "label": "Trend Meta",
        "prompt": "Analyze Meta's revenue trend over the last four quarters.",
    },
    {
        "label": "Risk Apple",
        "prompt": "Identify the key financial risks in Apple's latest filing.",
    },
    {
        "label": "Report Amazon",
        "prompt": "Generate a full financial report for Amazon.",
    },
    {
        "label": "Draft Email",
        "prompt": "Write a concise email summarizing Apple's 2024 10-K for the finance team.",
    },
]

WELCOME_MESSAGE = (
    "I can compare companies, forecast trends, spot risks, generate reports, and draft finance emails "
    "using the filings indexed in this app."
)


@st.cache_resource(show_spinner=False)
def get_config():
    return load_config()


def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "active_conversation_id" not in st.session_state:
        st.session_state.active_conversation_id = None
    if "chat_conversations" not in st.session_state:
        st.session_state.chat_conversations = []


def _format_timestamp(value: Any) -> str:
    if not value:
        return ""
    try:
        return datetime.fromisoformat(str(value)).strftime("%b %d, %H:%M")
    except Exception:
        return str(value)


def _json_safe(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _serializable_result(result: dict[str, Any]) -> dict[str, Any]:
    payload = {key: value for key, value in result.items() if key != "retrieved_docs"}
    return _json_safe(payload)


def _get_history_store(config) -> ChatHistoryStore:
    return ChatHistoryStore(config.chat_history_dir)


def _load_active_conversation(store: ChatHistoryStore) -> dict[str, Any]:
    active_id = st.session_state.get("active_conversation_id")
    conversation = store.get_conversation(active_id) if active_id else None
    if not conversation:
        conversation = store.create_conversation()
        st.session_state.active_conversation_id = conversation["id"]
    st.session_state.messages = conversation.get("messages", [])
    st.session_state.last_result = conversation.get("last_result")
    return conversation


def _bootstrap_chat_state(config) -> tuple[ChatHistoryStore, dict[str, Any], list[dict[str, Any]]]:
    store = _get_history_store(config)
    conversations = store.list_conversations()
    active_id = st.session_state.get("active_conversation_id")

    if not active_id or not store.get_conversation(active_id):
        if conversations:
            active_id = conversations[0]["id"]
        else:
            active_id = store.create_conversation()["id"]
        st.session_state.active_conversation_id = active_id

    active_conversation = _load_active_conversation(store)
    conversations = store.list_conversations()
    st.session_state.chat_conversations = conversations
    return store, active_conversation, conversations


def _start_new_chat(store: ChatHistoryStore) -> None:
    conversation = store.create_conversation()
    st.session_state.active_conversation_id = conversation["id"]
    st.session_state.messages = []
    st.session_state.last_result = None
    st.session_state.pending_prompt = None


def _switch_chat(store: ChatHistoryStore, conversation_id: str) -> None:
    st.session_state.active_conversation_id = conversation_id
    _load_active_conversation(store)


def _render_chat_history_sidebar(store: ChatHistoryStore, conversations: list[dict[str, Any]]) -> None:
    st.markdown(
        """
        <div class="sidebar-panel">
            <div class="eyebrow">Chats</div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("New chat", use_container_width=True, type="primary", key="new_chat_button"):
        _start_new_chat(store)
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="sidebar-panel">
            <div class="eyebrow">History</div>
        """,
        unsafe_allow_html=True,
    )

    if not conversations:
        st.caption("No saved chats yet.")
    else:
        active_id = st.session_state.get("active_conversation_id")
        options = [str(conversation.get("id", "")) for conversation in conversations[:20]]
        labels = {
            str(conversation.get("id", "")): f"{conversation.get('title', 'New chat')} | {(_format_timestamp(conversation.get('updated_at')) or 'recent')}"
            for conversation in conversations[:20]
        }

        if active_id not in options:
            active_id = options[0]
            st.session_state.active_conversation_id = active_id

        selected_id = st.selectbox(
            "Open chat",
            options=options,
            index=options.index(active_id),
            format_func=lambda conversation_id: labels.get(conversation_id, conversation_id),
            key="chat_history_selector",
        )
        if selected_id != active_id:
            _switch_chat(store, selected_id)
            st.rerun()

        st.markdown('<div style="margin-top:0.75rem;"></div>', unsafe_allow_html=True)
        st.caption("Recent chats")
        for conversation in conversations[:20]:
            conv_id = str(conversation.get("id", ""))
            title = str(conversation.get("title", "New chat"))
            preview = str(conversation.get("last_preview", "")).strip() or "No messages yet"
            updated = _format_timestamp(conversation.get("updated_at"))
            is_active = conv_id == active_id
            badge = "ACTIVE" if is_active else "OPEN"
            st.markdown(
                f"""
                <div style="
                    padding: 0.7rem 0.8rem;
                    margin-bottom: 0.45rem;
                    border-radius: 14px;
                    border: 1px solid {'rgba(105,240,196,0.28)' if is_active else 'rgba(255,255,255,0.08)'};
                    background: {'rgba(24,35,56,0.92)' if is_active else 'rgba(18,27,43,0.88)'};
                ">
                    <div style="display:flex;justify-content:space-between;gap:0.75rem;align-items:center;">
                        <div style="color:white;font-weight:600;line-height:1.2;">{title[:42]}</div>
                        <div style="color:rgba(255,255,255,0.52);font-size:0.68rem;letter-spacing:0.12em;text-transform:uppercase;">{badge}</div>
                    </div>
                    <div style="color:rgba(255,255,255,0.68);font-size:0.84rem;line-height:1.35;margin-top:0.4rem;">
                        {preview[:80]}
                    </div>
                    <div style="color:rgba(255,255,255,0.45);font-size:0.74rem;margin-top:0.35rem;">
                        {updated}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)


def _format_metadata(value: Any) -> str:
    if value is None:
        return "None"
    if hasattr(value, "model_dump"):
        return json.dumps(value.model_dump(), indent=2)
    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=2, default=str)
    return str(value)


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #07111f;
            --panel: rgba(13, 20, 34, 0.82);
            --panel-2: rgba(17, 26, 42, 0.92);
            --line: rgba(255,255,255,0.08);
            --text: #edf2ff;
            --muted: rgba(237,242,255,0.68);
            --accent: #69f0c4;
            --accent-2: #76a7ff;
            --warn: #ffcc66;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(99, 179, 237, 0.18), transparent 24%),
                radial-gradient(circle at top right, rgba(105, 240, 196, 0.12), transparent 26%),
                linear-gradient(180deg, #050b14 0%, #0b1320 60%, #09101b 100%);
            color: var(--text);
        }

        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 1.5rem;
            max-width: 1280px;
        }

        header[data-testid="stHeader"] {
            background: transparent;
        }

        #MainMenu, footer, [data-testid="stToolbar"] {
            visibility: hidden;
        }

        .hero {
            padding: 1.3rem 1.5rem;
            border-radius: 24px;
            border: 1px solid var(--line);
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.9), rgba(11, 19, 31, 0.84));
            box-shadow: 0 24px 60px rgba(0, 0, 0, 0.35);
            margin-bottom: 1rem;
        }

        .eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.16em;
            font-size: 0.74rem;
            color: var(--accent);
            margin-bottom: 0.35rem;
        }

        .hero h1 {
            margin: 0;
            font-size: 2.5rem;
            line-height: 1.05;
            color: white;
        }

        .hero p {
            margin: 0.75rem 0 0;
            max-width: 72ch;
            color: var(--muted);
            font-size: 1rem;
        }

        .metric-card {
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 0.9rem 1rem;
            background: rgba(12, 18, 30, 0.72);
            box-shadow: 0 18px 36px rgba(0,0,0,0.22);
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            margin-bottom: 0.3rem;
        }

        .metric-value {
            color: white;
            font-size: 1rem;
            font-weight: 600;
            word-break: break-word;
        }

        .quick-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.65rem;
            margin: 0.5rem 0 1rem;
        }

        .quick-chip {
            display: block;
            width: 100%;
            text-align: left;
            border: 1px solid rgba(255,255,255,0.08);
            background: linear-gradient(180deg, rgba(18, 27, 43, 0.95), rgba(12, 18, 30, 0.95));
            color: white;
            padding: 0.8rem 0.9rem;
            border-radius: 14px;
            font-size: 0.92rem;
            line-height: 1.25;
        }

        .section-label {
            font-size: 0.86rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: rgba(255,255,255,0.58);
            margin-bottom: 0.45rem;
        }

        .chat-pane {
            border: 1px solid var(--line);
            border-radius: 24px;
            background: rgba(8, 13, 23, 0.72);
            box-shadow: 0 22px 48px rgba(0, 0, 0, 0.26);
            padding: 0.4rem 0.4rem 0.1rem;
        }

        .assistant-card, .user-card {
            border-radius: 20px;
            padding: 1rem 1.05rem;
            border: 1px solid rgba(255,255,255,0.08);
            margin-bottom: 0.7rem;
        }

        .assistant-card {
            background: linear-gradient(180deg, rgba(20, 30, 49, 0.94), rgba(14, 22, 36, 0.98));
        }

        .user-card {
            background: linear-gradient(180deg, rgba(17, 26, 42, 0.96), rgba(12, 18, 30, 0.98));
        }

        .bubble-meta {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            font-size: 0.78rem;
            color: rgba(255,255,255,0.52);
            margin-bottom: 0.55rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .bubble-content {
            color: white;
            white-space: pre-wrap;
            line-height: 1.55;
        }

        .sidebar-panel {
            border: 1px solid var(--line);
            border-radius: 20px;
            background: rgba(12, 18, 30, 0.76);
            padding: 1rem;
            margin-bottom: 1rem;
        }

        .history-rail {
            border: 1px solid var(--line);
            border-radius: 24px;
            background: rgba(8, 13, 23, 0.72);
            padding: 1rem;
            box-shadow: 0 22px 48px rgba(0, 0, 0, 0.26);
        }

        .history-rail [data-testid="stButton"] button {
            width: 100%;
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.08);
            background: rgba(18, 27, 43, 0.88);
            color: white;
            text-align: left;
            padding: 0.65rem 0.85rem;
        }

        .history-rail [data-testid="stButton"] button:hover {
            border-color: rgba(105, 240, 196, 0.3);
            background: rgba(24, 35, 56, 0.96);
        }

        section[data-testid="stSidebar"] [data-testid="stButton"] button {
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.08);
            background: rgba(18, 27, 43, 0.88);
            color: white;
            text-align: left;
            padding: 0.65rem 0.85rem;
        }

        section[data-testid="stSidebar"] [data-testid="stButton"] button:hover {
            border-color: rgba(105, 240, 196, 0.3);
            background: rgba(24, 35, 56, 0.96);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Financial Copilot</div>
            <h1>Ask like an analyst.</h1>
            <p>
                Compare companies, forecast performance, find risks, draft finance emails, or generate structured reports
                from the filings already indexed in this workspace.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_metrics(result: dict[str, Any] | None, mode: str, top_k: int) -> None:
    cols = st.columns(4)
    values = [
        ("Mode", mode.upper()),
        ("Top K", str(top_k)),
        ("Intent", result.get("intent", "ready") if result else "ready"),
        ("Companies", ", ".join(result.get("matched_companies", [])) if result and result.get("matched_companies") else "auto"),
    ]
    for col, (label, value) in zip(cols, values, strict=False):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_quick_prompts() -> None:
    st.markdown('<div class="section-label">Quick actions</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for index, prompt in enumerate(QUICK_PROMPTS):
        with cols[index % 3]:
            if st.button(prompt["label"], key=f"quick_{index}", use_container_width=True):
                st.session_state.pending_prompt = prompt["prompt"]
                st.rerun()


def _display_message(message: dict[str, Any]) -> None:
    role = message.get("role", "assistant")
    meta = message.get("meta", {})
    content = message.get("content", "")

    wrapper_class = "assistant-card" if role == "assistant" else "user-card"
    label = "Copilot" if role == "assistant" else "You"
    timestamp = meta.get("timestamp", "")
    intent = meta.get("intent")
    mode = meta.get("mode")

    st.markdown(
        f"""
        <div class="{wrapper_class}">
            <div class="bubble-meta">
                <span>{label}</span>
                <span>{timestamp}{f" | {intent}" if intent else ""}{f" | {mode}" if mode else ""}</span>
            </div>
            <div class="bubble-content">{content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _run_query(config, question: str, mode: str, top_k: int) -> dict[str, Any]:
    result = run_financial_copilot_with_mode(config, question, top_k, mode)
    st.session_state.last_result = result
    return result


def main() -> None:
    _inject_styles()
    _init_state()
    config = get_config()
    store, active_conversation, conversations = _bootstrap_chat_state(config)

    left_col, right_col = st.columns([0.42, 1.58], gap="large")

    with left_col:
        st.markdown('<div class="history-rail">', unsafe_allow_html=True)
        _render_chat_history_sidebar(store, conversations)
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="section-label">Conversation</div>', unsafe_allow_html=True)
        st.markdown('<div class="chat-pane">', unsafe_allow_html=True)

        if st.session_state.messages:
            for message in st.session_state.messages:
                _display_message(message)

        st.markdown("</div>", unsafe_allow_html=True)

        pending = st.session_state.pop("pending_prompt", None)
        prompt = st.chat_input("Ask about Apple, Amazon, Google, Meta, or compare multiple companies...")
        question = prompt or pending

        if question:
            active_chat_id = st.session_state.active_conversation_id
            mode = "auto"
            top_k = config.rerank_top_k
            user_message = {
                "role": "user",
                "content": question,
                "meta": {"timestamp": datetime.now().strftime("%H:%M"), "mode": mode},
            }
            conversation = store.append_message(active_chat_id, "user", question, user_message["meta"])
            if conversation:
                st.session_state.messages = conversation.get("messages", [])

            with st.spinner("Analyzing filings..."):
                try:
                    result = _run_query(config, question, mode, top_k)
                except Exception as exc:
                    assistant_message = {
                        "role": "assistant",
                        "content": f"Query failed: {exc}",
                        "meta": {"timestamp": datetime.now().strftime("%H:%M"), "intent": "error", "mode": mode},
                    }
                    conversation = store.append_message(active_chat_id, "assistant", assistant_message["content"], assistant_message["meta"])
                    if conversation:
                        st.session_state.messages = conversation.get("messages", [])
                    st.session_state.last_result = None
                    st.rerun()
                    return

            assistant_message = {
                "role": "assistant",
                "content": result["answer"],
                "meta": {
                    "timestamp": datetime.now().strftime("%H:%M"),
                    "intent": result.get("intent"),
                    "mode": mode,
                },
            }
            conversation = store.append_message(active_chat_id, "assistant", result["answer"], assistant_message["meta"])
            if conversation:
                st.session_state.messages = conversation.get("messages", [])
            stored_result = _serializable_result(result)
            store.update_last_result(active_chat_id, stored_result)
            st.session_state.last_result = stored_result
            st.rerun()

if __name__ == "__main__":
    main()
