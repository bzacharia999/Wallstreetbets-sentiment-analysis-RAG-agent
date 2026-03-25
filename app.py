"""
WSB Sentiment Analysis & RAG Agent — Streamlit entry point.
"""

import streamlit as st

st.set_page_config(
    page_title="WSB Sentiment & RAG Agent",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for a premium look ───────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0E1117 0%, #1A1F2E 100%);
    }
    [data-testid="stSidebar"] .stRadio > label {
        font-size: 1.05rem;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1A1F2E 0%, #252B3B 100%);
        border: 1px solid #2D3348;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    [data-testid="stMetric"] label {
        color: #9CA3AF !important;
        font-size: 0.85rem;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }

    /* Buttons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #FF4500, #FF6B35);
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(255,69,0,0.35);
    }

    /* Expanders */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #FF4500 !important;
    }

    /* Hide default top bar */
    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        border: 1px solid #2D3348;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar navigation ─────────────────────────────────────────────────────
st.sidebar.markdown(
    """
    <div style="text-align:center; padding: 1rem 0;">
        <span style="font-size:2.5rem;">🚀</span>
        <h2 style="margin:0.25rem 0 0 0; font-size:1.2rem; letter-spacing:0.5px;">
            WSB Analyzer
        </h2>
        <p style="color:#6B7280; font-size:0.8rem; margin:0;">
            Sentiment · Topics · RAG
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.divider()

tab = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📊 Dashboard", "🔍 Explorer", "🤖 Ask WSB"],
    label_visibility="collapsed",
)

# ── Page routing ────────────────────────────────────────────────────────────
if tab == "🏠 Home":
    from pages import home
    home.render()
elif tab == "📊 Dashboard":
    from pages import dashboard
    dashboard.render()
elif tab == "🔍 Explorer":
    from pages import explorer
    explorer.render()
elif tab == "🤖 Ask WSB":
    from pages import ask_wsb
    ask_wsb.render()
