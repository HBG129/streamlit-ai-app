import streamlit as st
import os
import tempfile
import sqlite3
import uuid
from datetime import datetime
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

st.set_page_config(page_title="全能 AI 助手", page_icon="🤖", layout="wide")

# ==========================================
# 0. 数据库初始化与管理操作 (新增核心逻辑)
# ==========================================
def init_db():
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sessions (session_id TEXT PRIMARY KEY, title TEXT, created_at DATETIME)''')
    c.execute('''CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, content TEXT, created_at DATETIME)''')
    conn.commit()
    conn.close()

def create_new_session(title="新对话"):
    session_id = str(uuid.uuid4())
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute("INSERT INTO sessions (session_id, title, created_at) VALUES (?, ?, ?)", (session_id, title, datetime.now()))
    conn.commit()
    conn.close()
    return session_id

def get_all_sessions():
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute("SELECT session_id, title FROM sessions ORDER BY created_at DESC")
    rows = c.fetchall()
    conn.close()
    return rows

def save_message(session_id, role, content):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute("INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)", (session_id, role, content, datetime.now()))
    conn.commit()
    conn.close()

def get_messages(session_id):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute("SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC", (session_id,))
    rows = [{"role": row[0], "content": row[1]} for row in c.fetchall()]
    conn.close()
    return rows

def update_session_title(session_id, new_title):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute("UPDATE sessions SET title = ? WHERE session_id = ?", (new_title, session_id))
    conn.commit()
    conn.close()

def clear_session_messages(session_id):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()

# 初始化数据库
init_db()

# 管理当前的会话状态
if "current_session_id" not in st.session_state:
    sessions = get_all_sessions()
    if sessions:
        st.session_state.current_session_id = sessions[0][0]
    else:
        st.session_state.current_session_id = create_new_session()

# ==========================================
# 1. API 密钥与大模型配置
# ==========================================
try:
    os.environ["OPENAI_API_KEY"] = st.secrets["ZHIPU_API_KEY"]
    os.environ["OPENAI_API_BASE"] = "https://open.bigmodel.cn/api/paas/v4/"
    os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]
except KeyError as e:
    st.error(f"⚠️ 缺少 API Key: {e}。请检查 Secrets 配置！")
    st.stop()

llm = ChatOpenAI(model="glm-4-flash", temperature=0.5, streaming=True)

# ==========================================
# 2. 侧边栏：会话管理与文件上传
# ==========================================
with st.sidebar:
    st.header("💬 对话管理")
    if st.button("➕ 新建对话", use_container_width=True, type="primary"):
        st.session_state.current_session_id = create_new_session()
        st.rerun()
        
    st.markdown("**历史对话列表：**")
    sessions = get_all_sessions()
    for s_id, title in sessions:
        # 如果是当前选中的会话，加个标记
        btn_label = f"🟢 {title}" if s_id == st.session_state.current_session_id else f"💬 {title}"
        if st.button(btn_label, key=s_id, use_container_width=True):
            st.session_state.current_session_id = s_id
            st.rerun()

    st.divider()
    
    st.header("📂 喂给 AI 本地知识")
    uploaded_file = st.file_uploader("上传 PDF / TXT / CSV 文件", type=["pdf", "txt", "csv"])
    
    st.divider()
    st.header("⚙️ 助手设置")
    if st.button("🗑️ 清空当前对话记忆", use_container_width=True):
        clear_session_messages(st.session_state.current_session_id)
        st.success("当前记忆已清空！")
        st.rerun()
        
    # 读取当前会话的聊天记录用于导出
    current_msgs = get_messages(st.session_state.current_session_id)
    if len(current_msgs) > 0:
        chat_text = "\n\n".join([f"{msg['role'].upper()}:\n{msg['content']}" for msg in current_msgs])
        st.download_button("💾 导出当前聊天记录", data=chat_text, file_name="聊天记录.txt", mime="text/plain", use_container_width=True)

# ==========================================
# 3. 动态构建工具箱与 Agent
# ==========================================
tools = []
search_tool = TavilySearchResults(max_results=3, description="用于搜索互联网上的实时信息，如天气、新闻。如果问题涉及实时数据，必须使用此工具。")
tools.append(search_tool)

if uploaded_file is not None:
    with st.spinner("正在把文件塞进 AI 的脑子里..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
        elif uploaded_file.name.endswith(".csv"):
            loader = CSVLoader(tmp_path, encoding="utf-8")
        else:
            loader = TextLoader(tmp_path, encoding="utf-8")
        
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)
        embeddings = OpenAIEmbeddings(model="embedding-3") 
        vectorstore = FAISS.from_documents(splits, embeddings)
        retriever = vectorstore.as_retriever()
        
        retriever_tool = create_retriever_tool(retriever, "document_search", "当你需要回答关于用户上传的文档或表格里的内容时，必须使用此工具。")
        tools.append(retriever_tool)
        st.success(f"✅ 文件 {uploaded_file.name} 已加载！")

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个全能助手。遇到不知道的实时信息用 search_tool。遇到关于用户上传文档的问题用 document_search。"),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ==========================================
# 4. 聊天交互逻辑 (对接数据库)
# ==========================================
st.title("🤖 满血版 AI 助手")

# 从数据库中加载当前会话的聊天记录
st.session_state.messages = get_messages(st.session_state.current_session_id)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("问我天气，或者传个文档/表格问我里面的内容..."):
    # 动态修改对话标题（取第一句话的前10个字作为标题）
    if len(st.session_state.messages) == 0:
        new_title = user_input[:10] + "..." if len(user_input) > 10 else user_input
        update_session_title(st.session_state.current_session_id, new_title)

    # 存入数据库并展示
    save_message(st.session_state.current_session_id, "user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    # 构造历史记录喂给大模型
    chat_history = [
        ("human", msg["content"]) if msg["role"] == "user" else ("ai", msg["content"])
        for msg in st.session_state.messages
    ]

    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        try:
            response = agent_executor.invoke(
                {"input": user_input, "chat_history": chat_history},
                {"callbacks": [st_callback]}
            )
            answer = response["output"]
            
            # 把 AI 的回答也存进数据库
            save_message(st.session_state.current_session_id, "assistant", answer)
            st.rerun() # 刷新页面，确保状态和左侧标题同步更新
            
        except Exception as e:
            st.error(f"发生错误: {e}")
