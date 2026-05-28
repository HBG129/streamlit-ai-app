import streamlit as st
import os
import tempfile
import sqlite3
import uuid
from datetime import datetime
import sys
import io
import logging

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
# 【新增】高级 RAG 核心组件：多路查询重写器
from langchain.retrievers.multi_query import MultiQueryRetriever

# 开启日志，这样你能在控制台看到大模型是怎么重写问题的，满满的高级感
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

st.set_page_config(page_title="全能 AI 助手", page_icon="🤖", layout="wide")

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

def delete_session(session_id):
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    c.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()

init_db()

if "current_session_id" not in st.session_state:
    sessions = get_all_sessions()
    if sessions:
        st.session_state.current_session_id = sessions[0][0]
    else:
        st.session_state.current_session_id = create_new_session()

try:
    os.environ["OPENAI_API_KEY"] = st.secrets["ZHIPU_API_KEY"]
    os.environ["OPENAI_API_BASE"] = "https://open.bigmodel.cn/api/paas/v4/"
    os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]
except KeyError as e:
    st.error(f"⚠️ 缺少 API Key: {e}。请检查 Secrets 配置！")
    st.stop()

llm = ChatOpenAI(model="glm-4-flash", temperature=0.5, streaming=True)

@tool
def run_python_code(code: str) -> str:
    """
    运行 Python 代码进行复杂的数据分析、统计或数学计算。
    输入必须是一段合法的 Python 脚本。
    如果有计算结果需要知道，请务必在代码里使用 print() 打印出来，工具会捕获并返回 print 的内容。
    """
    old_stdout = sys.stdout
    redirected_output = sys.stdout = io.StringIO()
    try:
        exec(code, {})
        sys.stdout = old_stdout
        return redirected_output.getvalue()
    except Exception as e:
        sys.stdout = old_stdout
        return f"代码执行出错: {str(e)}"

with st.sidebar:
    st.header("💬 对话管理")
    if st.button("➕ 新建对话", use_container_width=True, type="primary"):
        st.session_state.current_session_id = create_new_session()
        st.rerun()
        
    st.markdown("**历史对话列表：**")
    sessions = get_all_sessions()
    for s_id, title in sessions:
        col1, col2 = st.columns([5, 1])
        with col1:
            btn_label = f"🟢 {title}" if s_id == st.session_state.current_session_id else f"💬 {title}"
            if st.button(btn_label, key=f"btn_{s_id}", use_container_width=True):
                st.session_state.current_session_id = s_id
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"del_{s_id}", help="删除此对话"):
                delete_session(s_id)
                if st.session_state.current_session_id == s_id:
                    rem_sessions = get_all_sessions()
                    st.session_state.current_session_id = rem_sessions[0][0] if rem_sessions else create_new_session()
                st.rerun()

    st.divider()
    
    st.header("📂 喂给 AI 本地知识")
    uploaded_file = st.file_uploader("上传 PDF / TXT / CSV 文件", type=["pdf", "txt", "csv"])
    
    st.divider()
    st.header("⚙️ 助手设置")
    if st.button("🧹 清空当前对话记忆", use_container_width=True):
        clear_session_messages(st.session_state.current_session_id)
        st.success("当前记忆已清空！")
        st.rerun()
        
    current_msgs = get_messages(st.session_state.current_session_id)
    if len(current_msgs) > 0:
        chat_text = "\n\n".join([f"{msg['role'].upper()}:\n{msg['content']}" for msg in current_msgs])
        st.download_button("💾 导出当前聊天记录", data=chat_text, file_name="聊天记录.txt", mime="text/plain", use_container_width=True)

tools = [
    TavilySearchResults(max_results=3, description="用于搜索互联网上的实时信息，如天气、新闻。如果问题涉及实时数据，必须使用此工具。"),
    run_python_code
]

if uploaded_file is not None:
    with st.spinner("正在启动高级 RAG 引擎解析文件..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
            st.session_state.current_file_path = tmp_path

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
        
        # 【进阶改动】基础检索器升级为多路查询高级检索器
        base_retriever = vectorstore.as_retriever()
        advanced_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever, 
            llm=llm
        )
        
        retriever_tool = create_retriever_tool(advanced_retriever, "document_search", "当你需要回答关于用户上传的文档里的内容时，可以使用此工具搜索。")
        tools.append(retriever_tool)
        st.success(f"✅ 文件 {uploaded_file.name} 已加载，并已开启多路并发检索！")

system_prompt_text = """你是一个企业级全能 AI 助手，拥有多种工具：
1. 遇到不知道的实时信息，必须使用 search_tool。
2. 遇到关于长文档的文本内容问答，使用 document_search。
3. 如果用户要求进行复杂计算、数据统计、或深度分析数据，你必须编写 Python 代码并使用 run_python_code 工具执行分析（记得用 print 输出你要查看的结果）。"""

if "current_file_path" in st.session_state and st.session_state.current_file_path:
    system_prompt_text += f"\n\n[核心机密] 用户最新上传了本地文件，物理路径为: '{st.session_state.current_file_path}'。如果是 CSV 表格分析，你可以直接在这个工具的 Python 代码里 import pandas 读取它进行统计！"

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt_text),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

st.title("🤖 满血版企业级 AI (高级 RAG + Agent)")

st.session_state.messages = get_messages(st.session_state.current_session_id)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("传个文档，故意用同义词考考我的高级检索能力！"):
    if len(st.session_state.messages) == 0:
        new_title = user_input[:10] + "..." if len(user_input) > 10 else user_input
        update_session_title(st.session_state.current_session_id, new_title)

    save_message(st.session_state.current_session_id, "user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

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
            save_message(st.session_state.current_session_id, "assistant", answer)
            st.rerun()
            
        except Exception as e:
            st.error(f"发生错误: {e}")
