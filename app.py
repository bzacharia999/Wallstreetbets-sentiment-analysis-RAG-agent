"""
WSB Sentiment Analysis & RAG Agent — Streamlit entry point.
"""

import streamlit as st

st.set_page_config(
    page_title="WSB Sentiment & RAG Agent",
    page_icon="WSB",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar navigation ─────────────────────────────────────────────────────
st.sidebar.divider()

tab = st.sidebar.radio(
    "Navigate",
    ["Home", "Dashboard", "Explorer", "Ask WSB"],
    label_visibility="collapsed",
)

# ── Page routing ────────────────────────────────────────────────────────────
if tab == "Home":
    from pages import home
    home.render()
elif tab == "Dashboard":
    from pages import dashboard
    dashboard.render()
elif tab == "Explorer":
    from pages import explorer
    explorer.render()
elif tab == "Ask WSB":
    from pages import ask_wsb
    ask_wsb.render()
