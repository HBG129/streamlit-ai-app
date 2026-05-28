import streamlit as st
import os
import tempfile
import sqlite3
import uuid
import sys
import io
import logging
import base64
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# 必须设置为非交互式后端，防止网页服务器画图时崩溃
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

# RAG 高级组件
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# 数据库组件
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.WARNING)

st.set_page_config(page_title="神级 AI 助手", page_icon="👑", layout="wide")

# ==========================================
# 0. 数据库初始化 (聊天记录 + 演示用企业数据库)
# ==========================================
def init_db():
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sessions (session_id TEXT PRIMARY KEY, title TEXT, created_at DATETIME)''')
    c.execute('''CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, content TEXT, created_at DATETIME)''')
    conn.commit()
    conn.close()

def init_demo_sqldb():
    # 初始化一个假的“企业数据库”供 AI 练习 SQL 查询
    if not os.path.exists("company_demo.db"):
        conn = sqlite3.connect("company_demo.db")
        c = conn.cursor()
        c.execute("CREATE TABLE sales (id INTEGER, product TEXT, revenue INTEGER, region TEXT)")
        c.execute("INSERT INTO sales VALUES (1, '超级手机', 50000, '北京'), (2, '游戏电脑', 80000, '上海'), (3, '办公平板', 30000, '北京')")
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

def save_message(session_id, role, content, image_path=None):
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
init_demo_sqldb()

if "current_session_id" not in st.session_state:
    sessions = get_all_sessions()
    if sessions:
        st.session_state.current_session_id = sessions[0][0]
    else:
        st.session_state.current_session_id = create_new_session()

# ==========================================
# 1. API 配置与大模型
# ==========================================
try:
    os.environ["OPENAI_API_KEY"] = st.secrets["ZHIPU_API_KEY"]
    os.environ["OPENAI_API_BASE"] = "https://open.bigmodel.cn/api/paas/v4/"
    os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]
except KeyError as e:
    st.error(f"⚠️ 缺少 API Key: {e}")
    st.stop()

# 基础模型
llm = ChatOpenAI(model="glm-4-flash", temperature=0.5, streaming=True)
# 专门用于看图的视觉模型
vision_llm = ChatOpenAI(model="glm-4v", temperature=0.1)

# ==========================================
# 2. 五大终极自定义工具
# ==========================================

# 【技能 1】数据分析与画图工具
@tool
def run_python_code(code: str) -> str:
    """
    运行 Python 代码进行复杂的数据分析或画图。
    如果你需要画图，请务必使用 matplotlib，并且必须将图片保存为当前目录下的 'temp_chart.png' 文件（不要使用 plt.show()）。
    """
    old_stdout = sys.stdout
    redirected_output = sys.stdout = io.StringIO()
    try:
        exec(code, globals())
        sys.stdout = old_stdout
        return redirected_output.getvalue() + "\n(代码执行完毕)"
    except Exception as e:
        sys.stdout = old_stdout
        return f"代码执行出错: {str(e)}"

# 【技能 2】网页爬虫工具
@tool
def scrape_webpage(url: str) -> str:
    """输入网页链接(URL)，爬取并返回该网页的纯文本内容。"""
    try:
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        res.encoding = res.apparent_encoding
        soup = BeautifulSoup(res.text, 'html.parser')
        return soup.get_text(separator='\n', strip=True)[:3000] # 截取前3000字防爆
    except Exception as e:
        return f"爬取失败: {e}"

# 【技能 3】视觉看图工具
@tool
def analyze_image(query: str) -> str:
    """当你需要看用户上传的图片时调用此工具。传入你需要从图片中获取的信息或问题。"""
    if "current_img_path" not in st.session_state or not st.session_state.current_img_path:
        return "用户没有上传图片。"
    try:
        with open(st.session_state.current_img_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode('utf-8')
        msg = vision_llm.invoke([
            {"role": "user", "content": [
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ]}
        ])
        return msg.content
    except Exception as e:
        return f"图片分析失败: {e}"

# 【技能 4】企业数据库 SQL 直连工具
db = SQLDatabase.from_uri("sqlite:///company_demo.db")
sql_tool = QuerySQLDataBaseTool(db=db, description="用于查询公司内部数据库(company_demo.db)，包含 sales(销售) 表。输入必须是标准的 SQL 语句。")


# ==========================================
# 3. 侧边栏 UI
# ==========================================
with st.sidebar:
    st.header("💬 对话管理")
    if st.button("➕ 新建对话", use_container_width=True, type="primary"):
        st.session_state.current_session_id = create_new_session()
        st.rerun()
        
    sessions = get_all_sessions()
    for s_id, title in sessions:
        col1, col2 = st.columns([5, 1])
        with col1:
            btn_label = f"🟢 {title}" if s_id == st.session_state.current_session_id else f"💬 {title}"
            if st.button(btn_label, key=f"btn_{s_id}", use_container_width=True):
                st.session_state.current_session_id = s_id
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"del_{s_id}"):
                delete_session(s_id)
                if st.session_state.current_session_id == s_id:
                    rem = get_all_sessions()
                    st.session_state.current_session_id = rem[0][0] if rem else create_new_session()
                st.rerun()

    st.divider()
    
    st.header("📂 给 AI 喂料")
    uploaded_file = st.file_uploader("上传文档/表格", type=["pdf", "txt", "csv"])
    uploaded_img = st.file_uploader("上传图片", type=["jpg", "png", "jpeg"])
    
    if uploaded_img:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
            tmp_img.write(uploaded_img.getvalue())
            st.session_state.current_img_path = tmp_img.name
            st.success("图片已就绪！")

    st.divider()
    if st.button("🧹 清空当前对话记忆", use_container_width=True):
        clear_session_messages(st.session_state.current_session_id)
        st.rerun()

# ==========================================
# 4. 构建终极工具箱 & RAG 重排序
# ==========================================
tools = [
    TavilySearchResults(max_results=2, description="搜索互联网实时信息"),
    run_python_code,
    scrape_webpage,
    analyze_image,
    sql_tool
]

if uploaded_file is not None:
    with st.spinner("启动高级 RAG 与内容压缩重排序..."):
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
        
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # 【技能 5】终极 RAG：先多路召回，再用 LLM 压缩重排序提取最精准内容
        mq_retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)
        compressor = LLMChainExtractor.from_llm(llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=mq_retriever
        )
        
        retriever_tool = create_retriever_tool(compression_retriever, "document_search", "搜索用户上传的文档内容。")
        tools.append(retriever_tool)
        st.success("文档已加载并开启重排序！")

sys_prompt = """你是一个顶级无所不能的 AI。
拥有搜索、爬网页、读图、数据库SQL查表、Python画图/计算 等终极能力。
根据用户问题，自主选择合适的工具。
注意：公司数据库叫 company_demo.db，表名 sales (id, product, revenue, region)。"""

if "current_file_path" in st.session_state and st.session_state.current_file_path:
    sys_prompt += f"\n本地文件物理路径: '{st.session_state.current_file_path}'。"

prompt = ChatPromptTemplate.from_messages([
    ("system", sys_prompt),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ==========================================
# 5. 主界面与交互
# ==========================================
st.title("👑 神级全栈 AI 助手")

st.session_state.messages = get_messages(st.session_state.current_session_id)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("发链接让我爬、发图片让我看、让我查数据库、或让我画图！"):
    if len(st.session_state.messages) == 0:
        update_session_title(st.session_state.current_session_id, user_input[:10])

    save_message(st.session_state.current_session_id, "user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    chat_history = [
        ("human", msg["content"]) if msg["role"] == "user" else ("ai", msg["content"])
        for msg in st.session_state.messages
    ]

    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        # 运行前清理掉之前的旧图表
        if os.path.exists('temp_chart.png'):
            os.remove('temp_chart.png')
            
        try:
            response = agent_executor.invoke(
                {"input": user_input, "chat_history": chat_history},
                {"callbacks": [st_callback]}
            )
            answer = response["output"]
            save_message(st.session_state.current_session_id, "assistant", answer)
            
            # 检查是否有 AI 刚刚画好的图
            if os.path.exists('temp_chart.png'):
                st.image('temp_chart.png')
                
            st.rerun()
        except Exception as e:
            st.error(f"发生错误: {e}")
