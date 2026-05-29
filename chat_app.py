import os
import re
import uuid
import time
import sqlite3
import tempfile
from datetime import datetime

import streamlit as st


# =========================================================
# 基础配置
# =========================================================

APP_NAME = "企业智能助手"
DB_PATH = "chat_history.db"

st.set_page_config(
    page_title=APP_NAME,
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# 数据库
# =========================================================

def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(session_id) REFERENCES sessions(id)
        )
    """)

    conn.commit()
    conn.close()


def create_new_session(title="新对话"):
    session_id = str(uuid.uuid4())
    now = datetime.now().isoformat(timespec="seconds")

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (session_id, title, now, now)
    )
    conn.commit()
    conn.close()

    return session_id


def get_all_sessions():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT id, title
        FROM sessions
        ORDER BY updated_at DESC
    """)
    rows = cur.fetchall()
    conn.close()
    return rows


def get_session_title(session_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT title FROM sessions WHERE id = ?", (session_id,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else "新对话"


def update_session_title(session_id, title):
    title = title.strip() or "新对话"
    now = datetime.now().isoformat(timespec="seconds")

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
        (title, now, session_id)
    )
    conn.commit()
    conn.close()


def touch_session(session_id):
    now = datetime.now().isoformat(timespec="seconds")

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE sessions SET updated_at = ? WHERE id = ?",
        (now, session_id)
    )
    conn.commit()
    conn.close()


def delete_session(session_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    cur.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()


def save_message(session_id, role, content):
    message_id = str(uuid.uuid4())
    now = datetime.now().isoformat(timespec="seconds")

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO messages (id, session_id, role, content, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (message_id, session_id, role, content, now)
    )
    cur.execute(
        "UPDATE sessions SET updated_at = ? WHERE id = ?",
        (now, session_id)
    )
    conn.commit()
    conn.close()


def load_messages(session_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT role, content
        FROM messages
        WHERE session_id = ?
        ORDER BY created_at ASC
    """, (session_id,))
    rows = cur.fetchall()
    conn.close()

    return [{"role": role, "content": content} for role, content in rows]


# =========================================================
# 文件处理
# =========================================================

def handle_uploaded_file(file):
    if file is None:
        return

    suffix = file.name.split(".")[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp_file:
        tmp_file.write(file.getvalue())
        st.session_state.current_file_path = tmp_file.name
        st.session_state.current_file_name = file.name

    st.toast(f"已挂载文件：{file.name}", icon="📎")
    st.rerun()


def read_current_file_preview():
    file_path = st.session_state.get("current_file_path")
    file_name = st.session_state.get("current_file_name")

    if not file_path or not file_name:
        return ""

    suffix = file_name.split(".")[-1].lower()

    try:
        if suffix == "txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()[:4000]

        if suffix == "csv":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()[:4000]

        if suffix == "pdf":
            return f"用户上传了 PDF 文件：{file_name}。如需解析 PDF 内容，建议接入 PyMuPDF、pypdf 或企业级文档解析服务。"

    except Exception as e:
        return f"文件读取失败：{e}"

    return ""


# =========================================================
# AI 回复逻辑
# 这里给你留了一个干净接口：
# 如果你已有 agent_executor，可以把 generate_ai_response 里面替换成你的 agent 调用。
# =========================================================

def generate_ai_response(user_input, messages):
    """
    企业级项目建议：
    后续可以在这里接入：
    1. LangChain Agent
    2. OpenAI / Claude / Qwen / DeepSeek API
    3. RAG 知识库
    4. SQL Agent
    5. 文件解析 Agent
    """

    file_context = read_current_file_preview()

    if file_context:
        return (
            "我已经收到你的问题，并检测到当前挂载了文件。\n\n"
            f"当前文件信息：\n{file_context[:1000]}\n\n"
            f"你的问题是：{user_input}\n\n"
            "这里目前是本地演示回复。你可以把 `generate_ai_response()` 替换成你自己的大模型、Agent 或知识库调用。"
        )

    return (
        f"收到：{user_input}\n\n"
        "当前这份代码已经完成了企业级聊天界面的基础架构，包括会话管理、历史记录、文件上传和现代化布局。\n\n"
        "如果你已经有自己的 `agent_executor`，只需要把 `generate_ai_response()` 里的内容替换成原来的模型调用即可。"
    )


# =========================================================
# 工具函数
# =========================================================

def escape_html(text):
    if text is None:
        return ""

    text = str(text)
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace('"', "&quot;")
    text = text.replace("'", "&#039;")
    return text


def render_markdown_message(role, content):
    safe_content = escape_html(content).replace("\n", "<br>")

    if role == "user":
        st.markdown(f"""
        <div class="message-row user-row">
            <div class="message-bubble user-bubble">{safe_content}</div>
            <div class="avatar user-avatar">👤</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class="message-row assistant-row">
            <div class="avatar assistant-avatar">🤖</div>
            <div class="message-bubble assistant-bubble">{safe_content}</div>
        </div>
        """, unsafe_allow_html=True)


def make_title_from_text(text):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return "新对话"
    return text[:14] + "..." if len(text) > 14 else text


# =========================================================
# 样式
# =========================================================

st.markdown("""
<style>
    :root {
        --app-radius: 22px;
        --app-border: rgba(120, 120, 140, 0.16);
        --app-soft-bg: rgba(128, 128, 128, 0.075);
        --app-soft-bg-hover: rgba(128, 128, 128, 0.13);
        --app-shadow: 0 16px 48px rgba(0, 0, 0, 0.08);
    }

    footer {
        visibility: hidden;
    }

    .stDeployButton {
        display: none !important;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"],
    [data-testid="collapsedControl"] {
        display: none !important;
    }

    .block-container {
        max-width: 1040px;
        padding-top: 1.35rem !important;
        padding-bottom: 1.5rem !important;
    }

    .main-shell {
        min-height: calc(100vh - 3rem);
    }

    .top-nav {
        position: sticky;
        top: 0;
        z-index: 20;
        padding: 0.4rem 0 1rem 0;
        backdrop-filter: blur(18px);
    }

    .chat-title {
        text-align: center;
        font-size: 1.34rem;
        font-weight: 720;
        letter-spacing: -0.02em;
        line-height: 2.65rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .subtle-file {
        text-align: center;
        margin-top: -0.4rem;
        margin-bottom: 1rem;
        font-size: 0.84rem;
        opacity: 0.68;
    }

    .empty-state {
        margin: 13vh auto 10vh auto;
        text-align: center;
        max-width: 620px;
        padding: 2.4rem 2rem;
        border: 1px solid var(--app-border);
        border-radius: 32px;
        background:
            radial-gradient(circle at top left, rgba(99, 102, 241, 0.12), transparent 38%),
            radial-gradient(circle at bottom right, rgba(14, 165, 233, 0.10), transparent 34%),
            var(--app-soft-bg);
        box-shadow: var(--app-shadow);
    }

    .empty-title {
        font-size: 2rem;
        font-weight: 760;
        letter-spacing: -0.04em;
        margin-bottom: 0.7rem;
    }

    .empty-desc {
        font-size: 1rem;
        opacity: 0.72;
        line-height: 1.75;
    }

    .message-row {
        display: flex;
        width: 100%;
        margin: 1.1rem 0;
        align-items: flex-start;
        gap: 0.7rem;
    }

    .user-row {
        justify-content: flex-end;
    }

    .assistant-row {
        justify-content: flex-start;
    }

    .avatar {
        width: 2.25rem;
        height: 2.25rem;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 999px;
        background: var(--app-soft-bg);
        border: 1px solid var(--app-border);
        flex: 0 0 auto;
        font-size: 1.05rem;
    }

    .message-bubble {
        max-width: min(76%, 760px);
        padding: 0.9rem 1.08rem;
        border-radius: 22px;
        font-size: 0.98rem;
        line-height: 1.72;
        word-break: break-word;
        border: 1px solid var(--app-border);
    }

    .user-bubble {
        border-top-right-radius: 8px;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.92), rgba(59, 130, 246, 0.92));
        color: white;
        box-shadow: 0 12px 30px rgba(59, 130, 246, 0.18);
    }

    .assistant-bubble {
        border-top-left-radius: 8px;
        background: var(--app-soft-bg);
    }

    .input-wrap {
        margin-top: 1.2rem;
        padding: 0.75rem;
        border-radius: 28px;
        border: 1px solid var(--app-border);
        background: rgba(128, 128, 128, 0.055);
        box-shadow: var(--app-shadow);
    }

    .stButton > button,
    div[data-testid="stPopover"] button {
        border-radius: 999px !important;
        height: 2.72rem !important;
        min-height: 2.72rem !important;
        border: 1px solid var(--app-border) !important;
        background: var(--app-soft-bg) !important;
        transition: all 0.18s ease !important;
        font-size: 1.05rem !important;
    }

    .stButton > button:hover,
    div[data-testid="stPopover"] button:hover {
        transform: translateY(-1px);
        background: var(--app-soft-bg-hover) !important;
        border-color: rgba(120, 120, 140, 0.26) !important;
    }

    div[data-testid="stTextInput"] input {
        height: 2.72rem;
        border-radius: 999px !important;
        border: 1px solid var(--app-border) !important;
        background: rgba(128, 128, 128, 0.07) !important;
        padding-left: 1.1rem !important;
        padding-right: 1.1rem !important;
        font-size: 0.98rem !important;
    }

    div[data-testid="stTextInput"] input:focus {
        border-color: rgba(99, 102, 241, 0.62) !important;
        box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.12) !important;
    }

    div[data-testid="stPopoverBody"] {
        border-radius: 24px !important;
    }

    .history-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.6rem;
    }

    .history-caption {
        font-size: 0.86rem;
        opacity: 0.62;
        margin-bottom: 0.9rem;
    }

    @media (max-width: 720px) {
        .block-container {
            padding-left: 0.85rem !important;
            padding-right: 0.85rem !important;
        }

        .chat-title {
            font-size: 1.05rem;
        }

        .message-bubble {
            max-width: 84%;
        }

        .empty-state {
            margin-top: 8vh;
        }

        .empty-title {
            font-size: 1.55rem;
        }
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# 初始化状态
# =========================================================

init_db()

if "current_session_id" not in st.session_state:
    sessions = get_all_sessions()
    if sessions:
        st.session_state.current_session_id = sessions[0][0]
    else:
        st.session_state.current_session_id = create_new_session()

if "current_file_path" not in st.session_state:
    st.session_state.current_file_path = None

if "current_file_name" not in st.session_state:
    st.session_state.current_file_name = None

if "input_nonce" not in st.session_state:
    st.session_state.input_nonce = 0


# =========================================================
# 顶部导航
# =========================================================

current_session_id = st.session_state.current_session_id
current_title = get_session_title(current_session_id)

st.markdown('<div class="main-shell">', unsafe_allow_html=True)
st.markdown('<div class="top-nav">', unsafe_allow_html=True)

left_col, title_col, right_col = st.columns([1.25, 5, 1.25], vertical_alignment="center")

with left_col:
    with st.popover("☰", help="历史聊天", use_container_width=True):
        st.markdown('<div class="history-title">历史聊天</div>', unsafe_allow_html=True)
        st.markdown('<div class="history-caption">选择、切换或删除历史对话</div>', unsafe_allow_html=True)

        sessions = get_all_sessions()

        if not sessions:
            st.caption("暂无历史记录")

        for s_id, title in sessions:
            row_col_1, row_col_2 = st.columns([5, 1], vertical_alignment="center")

            with row_col_1:
                label = f"● {title}" if s_id == current_session_id else title
                if st.button(label, key=f"history_{s_id}", use_container_width=True):
                    st.session_state.current_session_id = s_id
                    st.rerun()

            with row_col_2:
                if st.button("×", key=f"delete_{s_id}", help="删除此对话", use_container_width=True):
                    delete_session(s_id)

                    remaining = get_all_sessions()
                    if remaining:
                        st.session_state.current_session_id = remaining[0][0]
                    else:
                        st.session_state.current_session_id = create_new_session()

                    st.rerun()

with title_col:
    st.markdown(
        f'<div class="chat-title">💬 {escape_html(current_title)}</div>',
        unsafe_allow_html=True
    )

with right_col:
    if st.button("＋", help="新建聊天", use_container_width=True):
        st.session_state.current_session_id = create_new_session()
        st.session_state.current_file_path = None
        st.session_state.current_file_name = None
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.current_file_name:
    st.markdown(
        f'<div class="subtle-file">当前文件：📎 {escape_html(st.session_state.current_file_name)}</div>',
        unsafe_allow_html=True
    )


# =========================================================
# 聊天内容
# =========================================================

messages = load_messages(st.session_state.current_session_id)

if not messages:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-title">开始一次高效对话</div>
        <div class="empty-desc">
            你可以直接输入问题，也可以先上传文件再提问。<br>
            这个界面已经去掉侧边栏，保留更干净、更企业级的工作台体验。
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in messages:
        render_markdown_message(msg["role"], msg["content"])


# =========================================================
# 输入区域
# =========================================================

st.markdown('<div class="input-wrap">', unsafe_allow_html=True)

input_key = f"user_input_{st.session_state.input_nonce}"

input_col, send_col, upload_col = st.columns([12, 1.28, 1.28], vertical_alignment="center")

with input_col:
    typed_text = st.text_input(
        "输入消息",
        key=input_key,
        placeholder="输入问题，或上传文件后提问...",
        label_visibility="collapsed"
    )

with send_col:
    send_clicked = st.button("➤", help="发送", use_container_width=True)

with upload_col:
    with st.popover("📎", help="上传文件", use_container_width=True):
        uploaded_file = st.file_uploader(
            "上传文件",
            type=["pdf", "txt", "csv"],
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            handle_uploaded_file(uploaded_file)

st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# 发送消息
# =========================================================

user_input = typed_text.strip() if typed_text else ""

if send_clicked and user_input:
    current_messages = load_messages(st.session_state.current_session_id)

    if len(current_messages) == 0:
        update_session_title(
            st.session_state.current_session_id,
            make_title_from_text(user_input)
        )

    save_message(st.session_state.current_session_id, "user", user_input)

    with st.spinner("正在思考..."):
        history_after_user = load_messages(st.session_state.current_session_id)
        answer = generate_ai_response(user_input, history_after_user)
        time.sleep(0.2)

    save_message(st.session_state.current_session_id, "assistant", answer)

    st.session_state.input_nonce += 1
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
