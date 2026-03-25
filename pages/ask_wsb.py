"""
🤖 Ask WSB page — RAG-powered chat interface using Ollama.
"""

import streamlit as st

from src.rag.agent import ask


def render():
    st.markdown(
        "<h1 style='text-align:center;'>🤖 Ask WSB</h1>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Ask anything about the scraped r/wallstreetbets posts. "
        "Answers are grounded in real post content via RAG."
    )

    # ── Guard: need RAG chain ───────────────────────────────────────────
    if "rag_chain" not in st.session_state:
        st.warning(
            "No RAG index loaded. Go to **🏠 Home** and run **Scrape & Analyze** first."
        )
        st.info(
            "💡 Make sure Ollama is running (`ollama serve`) and you've pulled "
            "the model (`ollama pull llama3.2`)."
        )
        return

    chain = st.session_state["rag_chain"]

    # ── Chat history ────────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Render past messages
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                _render_sources(msg["sources"])

    # ── Chat input ──────────────────────────────────────────────────────
    if prompt := st.chat_input("What do you want to know about WSB?"):
        # Show user message
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching posts & generating answer…"):
                try:
                    result = ask(chain, prompt)
                except Exception as exc:
                    st.error(
                        f"Error querying Ollama: {exc}\n\n"
                        "Make sure Ollama is running (`ollama serve`)."
                    )
                    return

            st.markdown(result["answer"])
            _render_sources(result["sources"])

        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
            }
        )


def _render_sources(sources: list[dict]):
    """Render source documents in an expander."""
    if not sources:
        return

    with st.expander(f"📄 Sources ({len(sources)} retrieved posts)"):
        for i, src in enumerate(sources, 1):
            sentiment_emoji = {
                "positive": "🟢",
                "negative": "🔴",
                "neutral": "⚪",
            }.get(src.get("sentiment", ""), "")

            st.markdown(
                f"**{i}.** u/{src.get('author', '?')} · "
                f"⬆ {src.get('score', 0)} · "
                f"{sentiment_emoji} {src.get('sentiment', '')}"
            )
            st.markdown(
                f"> {src.get('content', '')[:250]}{'…' if len(src.get('content', '')) > 250 else ''}"
            )
            st.divider()
