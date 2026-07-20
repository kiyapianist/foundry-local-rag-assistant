"""
app.py - ChatGPT-style Streamlit UI for the local Foundry RAG assistant.
"""

import html
import importlib
import os
import sys
import time
import textwrap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

from scripts.ingest import MIN_DOCUMENT_TEXT_LENGTH, split_into_chunks
from src.config import CHUNK_OVERLAP, CHUNK_SIZE, DOCUMENTS_DIR
import src.database as database

database = importlib.reload(database)
from src.database import (
    create_tables,
    create_chat,
    delete_chat,
    delete_document_chunks,
    get_chat,
    insert_chunk,
    list_chats,
    update_chat,
)
from src.embedder import embed_batch
from src.foundry_runtime import get_runtime_status
from src.loader import load_single_document
import src.llm as llm_module
import src.pipeline as pipeline_module

llm_module = importlib.reload(llm_module)
pipeline_module = importlib.reload(pipeline_module)
from src.pipeline import ask

create_tables()

st.set_page_config(
    page_title="Local RAG AI Assistant",
    page_icon="AI",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 50% 0%, #151829 0%, #08090f 100%) !important;
        color: #e5e7eb !important;
    }

    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 6rem;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: #090a10 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    section[data-testid="stSidebar"] .stSubheader {
        font-size: 1.05rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        color: #a5b4fc;
        margin-top: 1.5rem;
    }

    /* Metric stats in Sidebar */
    [data-testid="stSidebar"] .stMetric {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 10px;
        padding: 0.5rem 0.75rem;
    }
    
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        color: #ffffff;
    }

    /* Chat Input Styling */
    div[data-testid="stChatInput"],
    div[data-testid="stChatInput"] [data-baseweb="textarea"],
    div[data-testid="stChatInput"] [data-baseweb="base-input"],
    div[data-testid="stChatInput"] > div {
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }
    
    div[data-testid="stChatInput"] {
        width: min(840px, calc(100vw - 48px)) !important;
        margin: 0 auto !important;
        background: transparent !important;
    }
    
    div[data-testid="stChatInput"] textarea {
        background: rgba(18, 20, 32, 0.9) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        color: #f3f4f6 !important;
        border-radius: 14px !important;
        padding: 0.9rem 1rem 0.9rem 3.4rem !important; /* Left padding for '+' button */
        min-height: 52px !important;
        font-size: 1rem !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
    }
    
    div[data-testid="stChatInput"] textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
    }

    /* Floating Popover '+' Button next to chat input */
    div[data-testid="stPopover"] {
        position: fixed;
        left: calc(50% - min(420px, calc(50vw - 24px)) + 0.75rem);
        bottom: 1.34rem;
        z-index: 999;
        width: 38px;
        height: 38px;
    }
    
    div[data-testid="stPopover"] > button {
        width: 38px;
        height: 38px;
        border-radius: 50% !important;
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        color: transparent !important;
        font-size: 0px !important;
        padding: 0 !important;
        box-shadow: none !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.2s !important;
    }
    
    div[data-testid="stPopover"] > button:hover {
        background: rgba(99, 102, 241, 0.15) !important;
        border-color: rgba(99, 102, 241, 0.4) !important;
    }

    div[data-testid="stPopover"] > button * {
        display: none !important;
    }

    div[data-testid="stPopover"] > button::before {
        content: "+";
        display: block;
        color: #e5e7eb;
        font-size: 1.45rem;
        font-weight: 300;
        line-height: 34px;
        text-align: center;
        width: 38px;
        height: 38px;
    }

    @media (max-width: 1000px) {
        div[data-testid="stPopover"] {
            left: 2.1rem;
        }
    }

    /* Welcome Dashboard Styles */
    .welcome-container {
        padding: 1.5rem 0;
        max-width: 800px;
        margin: 0 auto;
    }
    .welcome-header {
        text-align: center;
        margin-bottom: 2.2rem;
    }
    .welcome-badge {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
        padding: 0.3rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        box-shadow: 0 2px 10px rgba(79, 70, 229, 0.3);
    }
    .welcome-title {
        font-size: clamp(2rem, 5vw, 2.85rem);
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        background: linear-gradient(120deg, #ffffff 30%, #c7d2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .welcome-subtitle {
        font-size: 1.05rem;
        color: #9ca3af;
        max-width: 580px;
        margin: 0.5rem auto 0;
        line-height: 1.5;
    }

    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.25rem;
        margin-bottom: 2rem;
    }
    .stat-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 16px;
        padding: 1.25rem;
        text-align: center;
        backdrop-filter: blur(4px);
        transition: all 0.25s ease;
    }
    .stat-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.25);
        background: rgba(255, 255, 255, 0.03);
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.05);
    }
    .stat-icon {
        font-size: 1.85rem;
        margin-bottom: 0.5rem;
    }
    .stat-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #ffffff;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #9ca3af;
        margin-top: 0.25rem;
        font-weight: 500;
    }

    .upload-section-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #ffffff;
        margin-top: 2rem;
        margin-bottom: 0.35rem;
        text-align: center;
    }
    .upload-section-desc {
        font-size: 0.88rem;
        color: #9ca3af;
        text-align: center;
        margin-bottom: 1.25rem;
    }

    .quick-start-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #a5b4fc;
        margin-top: 2.2rem;
        margin-bottom: 0.85rem;
        text-align: center;
    }

    /* Sources and files lines styling */
    .source-line {
        color: #818cf8;
        font-size: 0.85rem;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    .upload-status {
        width: min(840px, calc(100vw - 48px));
        margin: 1rem auto;
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        background: rgba(99, 102, 241, 0.05);
        color: #c7d2fe;
        padding: 0.8rem 1rem;
        font-size: 0.95rem;
        backdrop-filter: blur(4px);
    }
    .upload-status strong {
        color: #ffffff;
    }
    
    /* Clean custom styled buttons for sidebar lists */
    section[data-testid="stSidebar"] .stButton > button {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        color: #c7d2fe !important;
        border-radius: 8px !important;
        font-size: 0.88rem !important;
        width: 100% !important;
        padding: 0.5rem 0.75rem !important;
        transition: all 0.2s !important;
    }
    
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(99, 102, 241, 0.15) !important;
        border-color: rgba(99, 102, 241, 0.4) !important;
        color: #ffffff !important;
    }
    
    /* Clean file checkbox styling scoped to sidebar */
    section[data-testid="stSidebar"] .stCheckbox [data-testid="stWidgetLabel"] {
        font-size: 0.88rem !important;
        color: #d1d5db !important;
    }
    
    /* Expander detail styling */
    .stExpander {
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        background: rgba(255, 255, 255, 0.01) !important;
        border-radius: 10px !important;
        margin-top: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: Inter, "Segoe UI", Arial, sans-serif !important;
    }

    .stApp {
        background: #f7f7f4 !important;
        color: #1f2933 !important;
    }

    .block-container {
        max-width: 820px !important;
        padding: 1.5rem 1rem 7rem !important;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    section[data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #e6e2d9 !important;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {
        color: #374151 !important;
    }

    .app-shell {
        padding: 1.2rem 0 0.5rem;
        text-align: center;
    }

    .app-title {
        color: #111827;
        font-size: 1.75rem;
        font-weight: 650;
        letter-spacing: 0;
        margin: 0;
    }

    .app-subtitle {
        color: #6b7280;
        font-size: 0.95rem;
        margin-top: 0.35rem;
    }

    .empty-state {
        margin: 5rem auto 2rem;
        max-width: 620px;
        text-align: center;
    }

    .empty-state h1 {
        color: #111827;
        font-size: clamp(1.7rem, 5vw, 2.25rem);
        font-weight: 650;
        letter-spacing: 0;
        margin-bottom: 0.5rem;
    }

    .empty-state p {
        color: #6b7280;
        font-size: 1rem;
        line-height: 1.55;
    }

    .simple-stats {
        display: flex;
        justify-content: center;
        gap: 0.6rem;
        flex-wrap: wrap;
        margin-top: 1.4rem;
    }

    .simple-pill {
        background: #ffffff;
        border: 1px solid #e6e2d9;
        color: #374151;
        border-radius: 999px;
        padding: 0.45rem 0.75rem;
        font-size: 0.86rem;
    }

    .source-line {
        color: #4f46e5 !important;
        font-size: 0.84rem !important;
        margin-top: 0.45rem !important;
        font-weight: 500 !important;
    }

    .upload-status {
        width: min(780px, calc(100vw - 40px)) !important;
        margin: 0.8rem auto !important;
        border: 1px solid #d9e0ff !important;
        border-radius: 12px !important;
        background: #f2f4ff !important;
        color: #3730a3 !important;
        padding: 0.75rem 0.9rem !important;
        font-size: 0.92rem !important;
        box-shadow: none !important;
    }

    [data-testid="stChatMessage"] {
        background: transparent !important;
    }

    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
        color: #1f2933;
    }

    div[data-testid="stBottomBlockContainer"],
    div[data-testid="stBottomBlockContainer"] > div,
    div[data-testid="stChatFloatingInputContainer"],
    div[data-testid="stChatFloatingInputContainer"] > div {
        background: #f7f7f4 !important;
        box-shadow: none !important;
    }

    div[data-testid="stChatInput"] {
        width: min(780px, calc(100vw - 36px)) !important;
        margin: 0 auto !important;
        background: transparent !important;
    }

    div[data-testid="stChatInput"] > div,
    div[data-testid="stChatInput"] [data-baseweb="textarea"],
    div[data-testid="stChatInput"] [data-baseweb="base-input"] {
        background: #ffffff !important;
        border-color: #d8d3c7 !important;
        box-shadow: none !important;
    }

    div[data-testid="stChatInput"] textarea {
        background: #ffffff !important;
        border: 1px solid #d8d3c7 !important;
        color: #111827 !important;
        border-radius: 22px !important;
        padding: 0.95rem 3.2rem 0.95rem 3.2rem !important;
        min-height: 56px !important;
        font-size: 1rem !important;
        box-shadow: 0 12px 28px rgba(17, 24, 39, 0.08) !important;
    }

    div[data-testid="stChatInput"] textarea:focus {
        border-color: #9ca3af !important;
        box-shadow: 0 12px 28px rgba(17, 24, 39, 0.1) !important;
    }

    div[data-testid="stPopover"] {
        position: fixed !important;
        left: calc(50% - min(390px, calc(50vw - 18px)) + 122px) !important;
        bottom: 4.85rem !important;
        z-index: 999 !important;
        width: 38px !important;
        height: 38px !important;
    }

    div[data-testid="stPopover"] button[data-testid="stPopoverButton"],
    div[data-testid="stPopover"] > button {
        width: 38px !important;
        height: 38px !important;
        border-radius: 999px !important;
        background: #ffffff !important;
        border: 1px solid #d8d3c7 !important;
        color: transparent !important;
        padding: 0 !important;
        box-shadow: 0 8px 22px rgba(17, 24, 39, 0.08) !important;
    }

    div[data-testid="stPopover"] button[data-testid="stPopoverButton"]:hover,
    div[data-testid="stPopover"] > button:hover {
        background: #f7f7f4 !important;
        border-color: #b9b1a4 !important;
    }

    div[data-testid="stPopover"] button[data-testid="stPopoverButton"] [data-testid="stIconMaterial"] {
        display: none !important;
    }

    div[data-testid="stPopover"] button[data-testid="stPopoverButton"] p {
        color: #374151 !important;
        font-size: 1.45rem !important;
        line-height: 1 !important;
        margin: 0 !important;
    }

    div[data-testid="stPopover"] > button * {
        display: none !important;
    }

    div[data-testid="stPopover"] > button::before {
        content: "+";
        display: block;
        color: #374151;
        font-size: 1.55rem;
        font-weight: 300;
        line-height: 34px;
        text-align: center;
        width: 38px;
        height: 38px;
    }

    .scope-line {
        text-align: center;
        color: #6b7280;
        margin-bottom: 0.35rem;
        font-size: 0.82rem;
    }

    .stButton > button {
        border-radius: 10px !important;
        border: 1px solid #e6e2d9 !important;
        background: #ffffff !important;
        color: #374151 !important;
    }

    .stButton > button:hover {
        border-color: #c8c0b4 !important;
        background: #faf9f6 !important;
        color: #111827 !important;
    }

    section[data-testid="stSidebar"] .stButton > button {
        background: #ffffff !important;
        border: 1px solid #e6e2d9 !important;
        color: #374151 !important;
        border-radius: 10px !important;
        min-height: 2.3rem !important;
        padding: 0.35rem 0.5rem !important;
        box-shadow: none !important;
    }

    section[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] {
        background: #ffffff !important;
        border: 1px solid #e6e2d9 !important;
        color: #374151 !important;
        border-radius: 10px !important;
        min-height: 2.3rem !important;
        padding: 0.35rem 0.5rem !important;
        box-shadow: none !important;
    }

    section[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] p {
        color: #374151 !important;
        margin: 0 !important;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        background: #f7f7f4 !important;
        border-color: #c8c0b4 !important;
        color: #111827 !important;
    }

    section[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"]:hover {
        background: #f7f7f4 !important;
        border-color: #c8c0b4 !important;
        color: #111827 !important;
    }

    .stExpander {
        border: 1px solid #e6e2d9 !important;
        background: #ffffff !important;
        border-radius: 10px !important;
    }

    @media (max-width: 860px) {
        div[data-testid="stPopover"] {
            left: 1.65rem !important;
            bottom: 4.85rem !important;
        }
        .block-container {
            padding-left: 0.8rem !important;
            padding-right: 0.8rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def reset_uploader() -> None:
    st.session_state.upload_key = st.session_state.get("upload_key", 0) + 1


def chat_title(messages: list[dict]) -> str:
    for message in messages:
        if message["role"] == "user" and message["content"].strip():
            title = message["content"].strip().replace("\n", " ")
            return title[:48] + ("..." if len(title) > 48 else "")
    return "New chat"


def save_chat() -> None:
    messages = sanitize_messages(st.session_state.get("messages", []))
    st.session_state.messages = messages
    if not messages:
        return

    if st.session_state.get("chat_id") is None:
        st.session_state.chat_id = create_chat(chat_title(messages))

    update_chat(st.session_state.chat_id, chat_title(messages), messages)


def load_chat(chat_id: int) -> None:
    chat = get_chat(chat_id)
    if chat is None:
        return

    st.session_state.chat_id = chat["id"]
    st.session_state.messages = sanitize_messages(chat["messages"])
    restored_sources = []
    for message in st.session_state.messages:
        for source in message.get("sources", []):
            if source not in restored_sources:
                restored_sources.append(source)
    st.session_state.uploaded_in_chat = restored_sources


def sanitize_messages(messages: list[dict]) -> list[dict]:
    cleaned = []
    for message in messages:
        content = message.get("content", "")
        if "Operation was cancelled" in content or "Operation was canceled" in content:
            continue
        if message.get("role") == "assistant" and content.startswith("Indexed "):
            continue
        cleaned.append(message)
    return cleaned


def upload_signature(uploaded_files) -> tuple:
    return tuple((file.name, getattr(file, "size", None)) for file in uploaded_files or [])


def process_uploaded_files(uploaded_files, rerun: bool = True) -> list[dict]:
    if not uploaded_files:
        return []

    progress = st.progress(0)
    status = st.empty()
    results = []

    for index, uploaded_file in enumerate(uploaded_files):
        filename = os.path.basename(uploaded_file.name)
        filepath = os.path.join(DOCUMENTS_DIR, filename)
        status.write(f"Indexing {filename}...")

        try:
            with open(filepath, "wb") as file:
                file.write(uploaded_file.getbuffer())

            text = load_single_document(filepath, filename)
            if len(text.strip()) < MIN_DOCUMENT_TEXT_LENGTH:
                st.warning(
                    f"Too little text could be extracted from {filename}. "
                    "The file may be empty or scanned and require OCR."
                )
                results.append({"filename": filename, "chunks": 0, "ok": False})
                continue

            chunks = split_into_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)
            if not chunks:
                st.warning(f"No chunks could be created from {filename}.")
                results.append({"filename": filename, "chunks": 0, "ok": False})
                continue

            delete_document_chunks(filename)

            batch_size = 16
            indexed = 0
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start:start + batch_size]
                status.write(
                    f"Indexing {filename}: {min(start + batch_size, len(chunks))} / {len(chunks)} chunks..."
                )
                embeddings = embed_batch(batch)
                for chunk, embedding in zip(batch, embeddings):
                    insert_chunk(filename, chunk, embedding)
                indexed += len(batch)
                progress.progress(indexed / len(chunks))

            st.success("Document ready. You can ask questions about it now.")
            results.append({"filename": filename, "chunks": len(chunks), "ok": True})
        except Exception as exc:
            st.error(f"Failed to process {filename}: {exc}")
            results.append({"filename": filename, "chunks": 0, "ok": False})

        if index + 1 == len(uploaded_files):
            progress.progress(1.0)

    status.write("Done.")
    reset_uploader()
    if rerun:
        time.sleep(0.7)
        st.rerun()
    return results


def remove_document(filename: str) -> None:
    delete_document_chunks(filename)
    local_path = os.path.join(DOCUMENTS_DIR, filename)
    if os.path.exists(local_path):
        os.remove(local_path)
    st.rerun()


runtime = get_runtime_status()

if "messages" not in st.session_state:
    # Always open on a clean conversation. Saved chats remain available in the sidebar.
    st.session_state.messages = []
    st.session_state.chat_id = None
if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0
if "chat_id" not in st.session_state:
    st.session_state.chat_id = None
if "last_upload_signature" not in st.session_state:
    st.session_state.last_upload_signature = ()
if "last_upload_status" not in st.session_state:
    st.session_state.last_upload_status = None
if "active_sources" not in st.session_state:
    st.session_state.active_sources = []
if "uploaded_in_chat" not in st.session_state:
    st.session_state.uploaded_in_chat = []
if "search_all_toggle" not in st.session_state:
    st.session_state.search_all_toggle = True
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = ""

st.session_state.messages = sanitize_messages(st.session_state.get("messages", []))

with st.sidebar:
    st.title("Local RAG AI")
    st.caption("Foundry Offline Assistant")

    if st.button("New chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_id = None
        st.session_state.last_upload_status = None
        st.session_state.uploaded_in_chat = []
        st.session_state.search_all_toggle = True
        st.rerun()

    st.subheader("Chats")
    chats = list_chats()
    if chats:
        for chat in chats:
            label = chat["title"] or "New chat"
            if len(label) > 28:
                label = label[:25] + "..."
            
            chat_col, del_col = st.columns([7, 1])
            is_current = st.session_state.get("chat_id") == chat["id"]
            btn_prefix = "Current: " if is_current else ""
            
            if chat_col.button(f"{btn_prefix}{label}", key=f"chat_{chat['id']}", use_container_width=True):
                load_chat(chat["id"])
                st.session_state.last_upload_status = None
                st.rerun()
                
            if del_col.button("X", key=f"del_chat_{chat['id']}", help="Delete chat"):
                delete_chat(chat["id"])
                if st.session_state.get("chat_id") == chat["id"]:
                    st.session_state.messages = []
                    st.session_state.chat_id = None
                    st.session_state.last_upload_status = None
                st.rerun()
    else:
        st.caption("No saved chats yet.")

    uploaded_files_sidebar = []

    with st.expander("Model status", expanded=False):
        if runtime["foundry_ready"]:
            st.success("Offline models ready")
        else:
            st.error("Foundry setup needed")
            if runtime["error"]:
                st.caption(runtime["error"])
            st.code("python scripts/setup_foundry.py", language="powershell")
        st.caption(f"LLM: {runtime['chat_alias']}")
        st.caption(f"Embeddings: {runtime['embedding_alias']}")
        st.caption(f"Offline Mode: {runtime['offline_mode']}")


if not st.session_state.messages:
    st.markdown(
        """<div class="empty-state">
<h1>What would you like to know?</h1>
<p>Upload a document with the + button, then ask questions about it.</p>
</div>""",
        unsafe_allow_html=True,
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            safe_sources = ", ".join(html.escape(source) for source in message["sources"])
            st.markdown(f"<div class='source-line'>Sources: {safe_sources}</div>", unsafe_allow_html=True)
        if message["role"] == "assistant" and message.get("chunks"):
            with st.expander("Retrieved context"):
                for chunk in message["chunks"]:
                    st.markdown(f"**{chunk['source']} - score {chunk['score']:.3f}**")
                    st.code(chunk["content"], language="text")

uploaded_files_popover = []
with st.popover("+"):
    st.markdown("### Add documents")
    uploaded_files_popover = st.file_uploader(
        "Upload PDF, DOCX, TXT, MD",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        key=f"file_upload_popover_{st.session_state.upload_key}",
        label_visibility="collapsed",
    )

uploaded_files = []
if uploaded_files_sidebar:
    uploaded_files = uploaded_files_sidebar
elif uploaded_files_popover:
    uploaded_files = uploaded_files_popover

if uploaded_files:
    current_signature = upload_signature(uploaded_files)
    if current_signature != st.session_state.last_upload_signature:
        st.session_state.last_upload_signature = current_signature
        names = ", ".join(file.name for file in uploaded_files)
        st.markdown(
            f"<div class='upload-status'>Loading <strong>{html.escape(names)}</strong> into the knowledge base...</div>",
            unsafe_allow_html=True,
        )
        results = process_uploaded_files(uploaded_files, rerun=False)
        ok_results = [result for result in results if result["ok"]]
        if ok_results:
            # Record that these were uploaded in the current chat session
            new_docs = [result["filename"] for result in ok_results]
            if "uploaded_in_chat" not in st.session_state:
                st.session_state.uploaded_in_chat = []
            st.session_state.uploaded_in_chat.extend(new_docs)
            
            st.session_state.last_upload_status = "Document ready."
        reset_uploader()
        time.sleep(0.7)
        st.rerun()

if st.session_state.last_upload_status:
    st.markdown(
        f"<div class='upload-status'><strong>{html.escape(st.session_state.last_upload_status)}</strong> You can ask about it now.</div>",
        unsafe_allow_html=True,
    )

if runtime["foundry_ready"]:
    if "uploaded_in_chat" in st.session_state and st.session_state.uploaded_in_chat:
        search_sources = st.session_state.uploaded_in_chat
    else:
        search_sources = []
else:
    search_sources = []

has_chat_documents = bool(search_sources)

chat_value = st.chat_input(
    "Ask about your uploaded document..." if has_chat_documents else "Upload a document first...",
    disabled=not runtime["foundry_ready"] or not has_chat_documents,
)

prompt = ""
if chat_value:
    prompt = chat_value.strip()
elif st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt.strip()
    st.session_state.pending_prompt = ""

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        with st.spinner("Searching local documents..."):
            try:
                result = ask(prompt, preferred_sources=search_sources)
                placeholder.markdown(result["answer"])
                if result["sources"]:
                    safe_sources = ", ".join(html.escape(source) for source in result["sources"])
                    st.markdown(f"<div class='source-line'>Sources: {safe_sources}</div>", unsafe_allow_html=True)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                        "chunks": result["chunks"],
                    }
                )
                save_chat()
            except Exception as exc:
                error = f"Error: {exc}"
                placeholder.error(error)
                st.session_state.messages.append({"role": "assistant", "content": error})
                save_chat()
