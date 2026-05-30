import os
import re
import uuid
import time
import sqlite3
import tempfile
from datetime import datetime

import streamlit as st

# =========================================================
# LangChain 相关引入 (企业级大模型架构)
# =========================================================
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langchain_community.utilities import SQLDatabase
    from langchain_community.agent_toolkits import create_sql_agent
    import pypdf
    import pandas as pd
    LANGCHAIN_READY = True
except ImportError as e:
    LANGCHAIN_READY = False
    st.error(f"缺少必要的库，请运行: pip install langchain langchain-openai langchain-community pypdf pandas\n错误详情: {e}")

# =========================================================
# 基础配置
# =========================================================

APP_NAME = "企业智能工作台"
DB_PATH = "chat_history.db"
COMPANY_DB_PATH = "company_data.db" # 你的企业数据源

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# 数据库 (聊天记录管理)
# =========================================================

def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'")
    sessions_exists = cur.fetchone() is not None

    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages'")
    messages_exists = cur.fetchone() is not None

    if sessions_exists:
        cur.execute("PRAGMA table_info(sessions)")
        session_columns = {row[1] for row in cur.fetchall()}
        required_session_columns = {"id", "title", "created_at", "updated_at"}
        if not required_session_columns.issubset(session_columns):
            cur.execute("DROP TABLE IF EXISTS messages")
            cur.execute("DROP TABLE IF EXISTS sessions")

    if messages_exists:
        cur.execute("PRAGMA table_info(messages)")
        message_columns = {row[1] for row in cur.fetchall()}
        required_message_columns = {"id", "session_id", "role", "content", "created_at"}
        if not required_message_columns.issubset(message_columns):
            cur.execute("DROP TABLE IF EXISTS messages")

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
    cur.execute("SELECT id, title FROM sessions ORDER BY updated_at DESC")
    rows = cur.fetchall()
    conn.close()
    return rows

def get_session_title(session_id):
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT title FROM sessions WHERE id = ?", (session_id,))
        row = cur.fetchone()
        conn.close()
        return row[0] if row else "新对话"
    except sqlite3.OperationalError:
        init_db()
        return "新对话"

def update_session_title(session_id, title):
    title = title.strip() or "新对话"
    now = datetime.now().isoformat(timespec="seconds")
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?", (title, now, session_id))
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
        "INSERT INTO messages (id, session_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
        (message_id, session_id, role, content, now)
    )
    cur.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
    conn.commit()
    conn.close()

def load_messages(session_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT role, content FROM messages WHERE session_id = ? ORDER BY created_at ASC", (session_id,))
    rows = cur.fetchall()
    conn.close()
    return [{"role": role, "content": content} for role, content in rows]


# =========================================================
# 文件解析模块 (真正提取文字)
# =========================================================

def handle_uploaded_file(file):
    if file is None: return
    suffix = file.name.split(".")[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp_file:
        tmp_file.write(file.getvalue())
        st.session_state.current_file_path = tmp_file.name
        st.session_state.current_file_name = file.name
    st.toast(f"文件解析成功：{file.name}", icon="📎")
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
                return f.read()[:6000] # 读取前6000字
                
        elif suffix == "csv":
            df = pd.read_csv(file_path)
            return df.head(50).to_markdown() # 将CSV转为Markdown表格喂给大模型
            
        elif suffix == "pdf":
            text = ""
            reader = pypdf.PdfReader(file_path)
            for page in reader.pages[:10]: # 限制读取前10页防token超载
                text += page.extract_text() + "\n"
            return text[:6000]
            
    except Exception as e:
        return f"系统提示：文件解析遇到错误 -> {e}"
    return ""


# =========================================================
# 核心 AI 大脑 (LangChain 路由)
# =========================================================

def generate_ai_response(user_input, messages_history):
    api_key = st.session_state.get("openai_api_key", "").strip()
    base_url = st.session_state.get("openai_base_url", "https://api.openai.com/v1").strip()
    
    if not api_key:
        return "⚠️ **配置缺失**：请先在右上角【⚙️设置】中填入您的 OpenAI API Key。"

    if not LANGCHAIN_READY:
        return "⚠️ 环境依赖缺失，请检查终端报错并安装库。"

    try:
        # 初始化 LLM
        llm = ChatOpenAI(
            model="gpt-3.5-turbo", # 推荐使用 3.5 或 4o
            api_key=api_key,
            base_url=base_url,
            temperature=0.1
        )
        
        file_context = read_current_file_preview()
        
        # 路由 1：如果检测到公司数据库存在，且用户没有上传文件，优先调用 SQL Agent
        if os.path.exists(COMPANY_DB_PATH) and not file_context:
            db = SQLDatabase.from_uri(f"sqlite:///{COMPANY_DB_PATH}")
            agent_executor = create_sql_agent(llm, db=db, verbose=True)
            
            # 让 Agent 知道自己是谁
            prompt_prefix = "你是一个强大的企业数据分析师。请分析数据库并回答问题。如果查询出数据，请用清晰的Markdown表格或列表展示。\n\n用户问题："
            
            response = agent_executor.invoke({"input": prompt_prefix + user_input})
            return response["output"]
            
        # 路由 2：普通对话或带文件上下文的对话
        else:
            system_prompt = "你是企业高级智能助手。回答要专业、精准、排版清晰。"
            if file_context:
                system_prompt += f"\n\n用户上传了文件，文件核心内容如下：\n{file_context}\n\n请结合上述文件内容回答用户的问题。"
                
            langchain_msgs = [SystemMessage(content=system_prompt)]
            
            # 载入历史记录（最多取最近 6 条防止 Token 爆炸）
            for msg in messages_history[-6:]:
                if msg["role"] == "user":
                    langchain_msgs.append(HumanMessage(content=msg["content"]))
                else:
                    langchain_msgs.append(AIMessage(content=msg["content"]))
            
            langchain_msgs.append(HumanMessage(content=user_input))
            
            # 请求模型
            ai_message = llm.invoke(langchain_msgs)
            return ai_message.content

    except Exception as e:
        return f"❌ **AI 请求失败**：\n```python\n{str(e)}\n```\n请检查 API Key、网络或余额。"


# =========================================================
# UI 渲染工具
# =========================================================

def render_markdown_message(role, content):
    # 这里我们保留你喜欢的样式，但为了支持 Markdown（加粗、表格），不做太强的 HTML 转移
    if role == "user":
        st.markdown(f"""
        <div class="message-row user-row">
            <div class="message-bubble user-bubble">{content}</div>
            <div class="avatar user-avatar">👤</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="message-row assistant-row">
            <div class="avatar assistant-avatar">🤖</div>
            <div class="message-bubble assistant-bubble">{content}</div>
        </div>
        """, unsafe_allow_html=True)

def make_title_from_text(text):
    text = re.sub(r"\s+", " ", text).strip()
    return text[:14] + "..." if len(text) > 14 else (text or "新对话")


# =========================================================
# 样式 CSS (保留你的企业级审美)
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
    footer { visibility: hidden; }
    .stDeployButton { display: none !important; }
    header[data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none !important; }
    .block-container { max-width: 1040px; padding-top: 1.35rem !important; padding-bottom: 1.5rem !important; }
    .main-shell { min-height: calc(100vh - 3rem); }
    .top-nav { position: sticky; top: 0; z-index: 20; padding: 0.4rem 0 1rem 0; backdrop-filter: blur(18px); }
    .chat-title { text-align: center; font-size: 1.34rem; font-weight: 720; letter-spacing: -0.02em; line-height: 2.65rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .subtle-file { text-align: center; margin-top: -0.4rem; margin-bottom: 1rem; font-size: 0.84rem; opacity: 0.68; }
    .empty-state { margin: 13vh auto 10vh auto; text-align: center; max-width: 620px; padding: 2.4rem 2rem; border: 1px solid var(--app-border); border-radius: 32px; background: radial-gradient(circle at top left, rgba(99, 102, 241, 0.12), transparent 38%), radial-gradient(circle at bottom right, rgba(14, 165, 233, 0.10), transparent 34%), var(--app-soft-bg); box-shadow: var(--app-shadow); }
    .empty-title { font-size: 2rem; font-weight: 760; letter-spacing: -0.04em; margin-bottom: 0.7rem; }
    .empty-desc { font-size: 1rem; opacity: 0.72; line-height: 1.75; }
    .message-row { display: flex; width: 100%; margin: 1.1rem 0; align-items: flex-start; gap: 0.7rem; }
    .user-row { justify-content: flex-end; }
    .assistant-row { justify-content: flex-start; }
    .avatar { width: 2.25rem; height: 2.25rem; display: flex; align-items: center; justify-content: center; border-radius: 999px; background: var(--app-soft-bg); border: 1px solid var(--app-border); flex: 0 0 auto; font-size: 1.05rem; }
    .message-bubble { max-width: min(76%, 760px); padding: 0.9rem 1.08rem; border-radius: 22px; font-size: 0.98rem; line-height: 1.72; word-break: break-word; border: 1px solid var(--app-border); }
    .user-bubble { border-top-right-radius: 8px; background: linear-gradient(135deg, rgba(99, 102, 241, 0.92), rgba(59, 130, 246, 0.92)); color: white; box-shadow: 0 12px 30px rgba(59, 130, 246, 0.18); }
    .assistant-bubble { border-top-left-radius: 8px; background: var(--app-soft-bg); }
    /* 让 bubble 内的 p 标签没有底部边距，防止换行过大 */
    .message-bubble p { margin-bottom: 0 !important; }
    .input-wrap { margin-top: 1.2rem; padding: 0.75rem; border-radius: 28px; border: 1px solid var(--app-border); background: rgba(128, 128, 128, 0.055); box-shadow: var(--app-shadow); }
    .stButton > button, div[data-testid="stPopover"] button { border-radius: 999px !important; height: 2.72rem !important; min-height: 2.72rem !important; border: 1px solid var(--app-border) !important; background: var(--app-soft-bg) !important; transition: all 0.18s ease !important; font-size: 1.05rem !important; }
    .stButton > button:hover, div[data-testid="stPopover"] button:hover { transform: translateY(-1px); background: var(--app-soft-bg-hover) !important; border-color: rgba(120, 120, 140, 0.26) !important; }
    div[data-testid="stTextInput"] input { height: 2.72rem; border-radius: 999px !important; border: 1px solid var(--app-border) !important; background: rgba(128, 128, 128, 0.07) !important; padding-left: 1.1rem !important; padding-right: 1.1rem !important; font-size: 0.98rem !important; }
    div[data-testid="stTextInput"] input:focus { border-color: rgba(99, 102, 241, 0.62) !important; box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.12) !important; }
    div[data-testid="stPopoverBody"] { border-radius: 24px !important; }
    .history-title { font-size: 1.05rem; font-weight: 700; margin-bottom: 0.6rem; }
    .history-caption { font-size: 0.86rem; opacity: 0.62; margin-bottom: 0.9rem; }
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

if "current_file_path" not in st.session_state: st.session_state.current_file_path = None
if "current_file_name" not in st.session_state: st.session_state.current_file_name = None
if "input_nonce" not in st.session_state: st.session_state.input_nonce = 0
if "openai_api_key" not in st.session_state: st.session_state.openai_api_key = ""


# =========================================================
# 顶部导航
# =========================================================

current_session_id = st.session_state.current_session_id
current_title = get_session_title(current_session_id)

st.markdown('<div class="main-shell">', unsafe_allow_html=True)
st.markdown('<div class="top-nav">', unsafe_allow_html=True)

left_col, title_col, right_col = st.columns([1.5, 5, 2.5], vertical_alignment="center")

with left_col:
    with st.popover("☰", help="历史聊天", use_container_width=True):
        st.markdown('<div class="history-title">历史聊天</div>', unsafe_allow_html=True)
        sessions = get_all_sessions()
        if not sessions: st.caption("暂无历史记录")
        for s_id, title in sessions:
            c1, c2 = st.columns([5, 1])
            with c1:
                if st.button(f"● {title}" if s_id == current_session_id else title, key=f"hist_{s_id}", use_container_width=True):
                    st.session_state.current_session_id = s_id
                    st.rerun()
            with c2:
                if st.button("×", key=f"del_{s_id}", use_container_width=True):
                    delete_session(s_id)
                    st.rerun()

with title_col:
    st.markdown(f'<div class="chat-title">💬 {current_title}</div>', unsafe_allow_html=True)

with right_col:
    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        if st.button("＋ 新建", use_container_width=True):
            st.session_state.current_session_id = create_new_session()
            st.session_state.current_file_path = None
            st.session_state.current_file_name = None
            st.rerun()
    with c_btn2:
        with st.popover("⚙️ 设置", use_container_width=True):
            st.markdown('<div class="history-title">系统配置</div>', unsafe_allow_html=True)
            st.session_state.openai_api_key = st.text_input("OpenAI API Key", value=st.session_state.openai_api_key, type="password")
            st.session_state.openai_base_url = st.text_input("API Base URL (选填)", value=st.session_state.get("openai_base_url", "https://api.openai.com/v1"))
            if st.button("保存并生效"):
                st.toast("配置已保存！", icon="✅")

st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.current_file_name:
    st.markdown(f'<div class="subtle-file">当前挂载文件：📎 {st.session_state.current_file_name}</div>', unsafe_allow_html=True)


# =========================================================
# 聊天内容展示
# =========================================================

messages = load_messages(st.session_state.current_session_id)

if not messages:
    st.markdown(f"""
    <div class="empty-state">
        <div class="empty-title">🌟 欢迎使用智能工作台</div>
        <div class="empty-desc">
            <b>功能 1：数据分析</b> - 如果根目录下有 {COMPANY_DB_PATH}，直接问我："薪水最高的人是谁？"<br>
            <b>功能 2：文档解析</b> - 点击右下角 📎 上传 PDF/CSV/TXT，然后针对文件提问。<br>
            <br>
            <small>首次使用请点击右上角 ⚙️ 配置 API Key</small>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in messages:
        # 为了兼容多段落换行，简单的把换行替换成 <br>
        content = str(msg["content"]).replace("\n", "<br>")
        render_markdown_message(msg["role"], content)


# =========================================================
# 输入区域
# =========================================================

st.markdown('<div class="input-wrap">', unsafe_allow_html=True)
input_key = f"user_input_{st.session_state.input_nonce}"
input_col, send_col, upload_col = st.columns([12, 1.28, 1.28], vertical_alignment="center")

with input_col:
    typed_text = st.text_input("输入消息", key=input_key, placeholder="输入问题查询数据库，或上传文件后提问...", label_visibility="collapsed")
with send_col:
    send_clicked = st.button("➤", help="发送", use_container_width=True)
with upload_col:
    with st.popover("📎", help="上传文件", use_container_width=True):
        uploaded_file = st.file_uploader("上传解析", type=["pdf", "txt", "csv"], label_visibility="collapsed")
        if uploaded_file is not None:
            handle_uploaded_file(uploaded_file)
st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# 发送消息逻辑
# =========================================================

user_input = typed_text.strip() if typed_text else ""

if send_clicked and user_input:
    # 1. 标题逻辑
    if len(messages) == 0:
        update_session_title(st.session_state.current_session_id, make_title_from_text(user_input))

    # 2. 保存用户输入并刷新UI，让用户输入先显示出来
    save_message(st.session_state.current_session_id, "user", user_input)
    
    # 3. 显示思考状态并调用大模型
    with st.spinner("🤖 AI 正在深度思考 / 查询数据库..."):
        history_after_user = load_messages(st.session_state.current_session_id)[:-1] # 排除刚存的这条
        answer = generate_ai_response(user_input, history_after_user)
        
    # 4. 保存AI回复并刷新
    save_message(st.session_state.current_session_id, "assistant", answer)
    st.session_state.input_nonce += 1
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
