"""
Enterprise AI Knowledge Assistant
Streamlit UI — RAG + GenAI + Dynamic Learning
"""

import os
import sys
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Enterprise AI Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0066cc 0%, #003d99 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #0066cc;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
    }
    .source-chip {
        background: #e8f0fe;
        color: #1a56db;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        margin: 2px;
        display: inline-block;
    }
    .success-banner {
        background: #d4edda;
        border: 1px solid #28a745;
        color: #155724;
        padding: 0.5rem 1rem;
        border-radius: 6px;
    }
    .stChatMessage { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "kb_stats" not in st.session_state:
    st.session_state.kb_stats = {"total_chunks": 0, "unique_sources": []}


# ── Lazy pipeline import (avoids slow import on first load) ───────────────────
@st.cache_resource(show_spinner="Loading AI models...")
def load_pipeline():
    from pipeline import ingest_file, ask, generate_report, get_kb_stats, clear_kb
    return ingest_file, ask, generate_report, get_kb_stats, clear_kb


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1 style="margin:0;font-size:1.8rem">🧠 Enterprise AI Knowledge Assistant</h1>
    <p style="margin:0.3rem 0 0;opacity:0.85;font-size:0.95rem">
        Powered by RAG + Groq AI &nbsp;|&nbsp; Upload data, ask anything, it learns and grows
    </p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📁 Knowledge Base")

    # Check API key
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        st.error("⚠️ GROQ_API_KEY not set.\nAdd it to your `.env` file.")
        st.stop()

    # File uploader
    st.markdown("### Upload Documents")
    uploaded = st.file_uploader(
        "Drag & drop files here",
        type=["pdf", "docx", "csv", "xlsx", "txt", "md"],
        accept_multiple_files=True,
        help="Supported: PDF, DOCX, CSV, XLSX, TXT, MD",
    )

    if uploaded:
        ingest_fn, ask_fn, report_fn, stats_fn, clear_fn = load_pipeline()
        new_files = [f for f in uploaded if f.name not in st.session_state.uploaded_files]

        if new_files:
            with st.spinner(f"Processing {len(new_files)} file(s)..."):
                for file in new_files:
                    chunks_added, total = ingest_fn(file.read(), file.name)
                    st.session_state.uploaded_files.append(file.name)
                    if chunks_added > 0:
                        st.success(f"✅ **{file.name}** — {chunks_added} new chunks learned")
                    else:
                        st.info(f"ℹ️ **{file.name}** — already in knowledge base")
            st.session_state.kb_stats = stats_fn()

    # KB Stats
    st.markdown("---")
    st.markdown("### 📊 Knowledge Base Stats")
    stats = st.session_state.kb_stats

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Chunks", stats["total_chunks"])
    with col2:
        st.metric("Sources", len(stats["unique_sources"]))

    if stats["unique_sources"]:
        st.markdown("**Loaded sources:**")
        for src in stats["unique_sources"]:
            st.markdown(f'<span class="source-chip">📄 {src}</span>', unsafe_allow_html=True)

    # Clear KB button
    st.markdown("---")
    if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
        ingest_fn, ask_fn, report_fn, stats_fn, clear_fn = load_pipeline()
        clear_fn()
        st.session_state.uploaded_files = []
        st.session_state.kb_stats = {"total_chunks": 0, "unique_sources": []}
        st.session_state.chat_history = []
        st.success("Knowledge base cleared.")
        st.rerun()

    # Clear chat button
    if st.button("💬 Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<small>Built with Groq LLaMA 3.3 70B + ChromaDB + Sentence Transformers</small>",
        unsafe_allow_html=True
    )


# ── Main content area ─────────────────────────────────────────────────────────
tab_chat, tab_report = st.tabs(["💬 Chat", "📋 Generate Report"])

# ── Chat tab ──────────────────────────────────────────────────────────────────
with tab_chat:
    # Show chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 Sources used", expanded=False):
                    for s in msg["sources"]:
                        st.markdown(
                            f'<span class="source-chip">📄 {s["source"]} | Page {s["page"]} | Score {s["score"]}</span>',
                            unsafe_allow_html=True
                        )
                        st.caption(s["text"][:200] + "..." if len(s["text"]) > 200 else s["text"])

    # Chat input
    user_query = st.chat_input("Ask anything about your data...")

    if user_query:
        ingest_fn, ask_fn, report_fn, stats_fn, clear_fn = load_pipeline()

        # Show user message
        with st.chat_message("user"):
            st.markdown(user_query)

        # Stream assistant response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            kb_stats = stats_fn()
            if kb_stats["total_chunks"] == 0:
                st.info("💡 No documents in knowledge base yet. Upload files in the sidebar to enable RAG.")

            stream, sources = ask_fn(
                user_query,
                chat_history=st.session_state.chat_history,
                n_context=5,
            )

            for chunk in stream:
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)

            # Show sources
            if sources:
                with st.expander("📚 Sources used", expanded=False):
                    for s in sources:
                        st.markdown(
                            f'<span class="source-chip">📄 {s["source"]} | Page {s["page"]} | Score {s["score"]}</span>',
                            unsafe_allow_html=True
                        )
                        st.caption(s["text"][:200] + "..." if len(s["text"]) > 200 else s["text"])

        # Save to history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": full_response,
            "sources": sources,
        })


# ── Report tab ────────────────────────────────────────────────────────────────
with tab_report:
    st.markdown("### 📋 AI-Generated Report")
    st.markdown("Generate a structured report on any topic from your knowledge base.")

    col1, col2 = st.columns([3, 1])
    with col1:
        topic = st.text_input(
            "Report topic",
            placeholder="e.g. 'Sales performance Q4', 'IT incident trends', 'Project risk analysis'",
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        generate_btn = st.button("📋 Generate", use_container_width=True, type="primary")

    if generate_btn and topic:
        ingest_fn, ask_fn, report_fn, stats_fn, clear_fn = load_pipeline()
        kb_stats = stats_fn()

        if kb_stats["total_chunks"] == 0:
            st.warning("Upload documents first to generate a report from your data.")
        else:
            with st.spinner("Generating report with Claude AI..."):
                report_placeholder = st.empty()
                full_report = ""

                stream, sources = report_fn(topic, n_context=8)
                for chunk in stream:
                    full_report += chunk
                    report_placeholder.markdown(full_report + "▌")

                report_placeholder.markdown(full_report)

            if sources:
                with st.expander("📚 Data sources used in report"):
                    for s in sources:
                        st.markdown(
                            f'<span class="source-chip">📄 {s["source"]} | Score {s["score"]}</span>',
                            unsafe_allow_html=True
                        )

            # Download button
            st.download_button(
                label="⬇️ Download Report (TXT)",
                data=full_report,
                file_name=f"report_{topic.replace(' ', '_')}.txt",
                mime="text/plain",
            )
