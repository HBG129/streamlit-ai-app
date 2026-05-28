import streamlit as st
import os
import tempfile
import sqlite3
import uuid
from datetime import datetime
import sys
import io
import logging
import re
import plotly.io as pio

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain.retrievers.multi_query import MultiQueryRetriever

# 开启日志
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

st.set_page_config(page_title="全能 AI 助手", page_icon="🤖", layout="wide")

# 确保存放图表的本地文件夹存在
if not os.path.exists("saved_charts"):
    os.makedirs("saved_charts")

# ==========================================
# 数据库管理区
# ==========================================

def init_chat_db():
    """初始化聊天记录数据库（部署在云端时，重启会清空，不影响业务）"""
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sessions (session_id TEXT PRIMARY KEY, title TEXT, created_at DATETIME)''')
    c.execute('''CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, content TEXT, created_at DATETIME)''')
    conn.commit()
    conn.close()

def init_business_db():
    """
    初始化企业数据库。
    【重点】如果你已经把 company_data.db 传到了 GitHub，这里只会读取，不会覆盖！
    如果文件不存在，它会自动建一些假数据供你测试。
    """
    conn = sqlite3.connect('company_data.db')
    c = conn.cursor()
    # 创建员工表
    c.execute('''CREATE TABLE IF NOT EXISTS employees (id INTEGER PRIMARY KEY, name TEXT, department TEXT, salary INTEGER, join_date DATE)''')
    # 创建销量表
    c.execute('''CREATE TABLE IF NOT EXISTS product_sales (id INTEGER PRIMARY KEY, product_name TEXT, category TEXT, revenue INTEGER, units_sold INTEGER)''')
    
    # 只有当表是空的时候，才插入测试数据（防止覆盖你 GitHub 上的真实数据）
    c.execute("SELECT COUNT(*) FROM employees")
    if c.fetchone()[0] == 0:
        c.executemany("INSERT INTO employees (name, department, salary, join_date) VALUES (?, ?, ?, ?)",
                      [('张三', '技术部', 25000, '2023-01-15'), ('李四', '销售部', 15000, '2022-03-10'), 
                       ('王五', '技术部', 28000, '2021-07-22'), ('赵六', 'HR', 12000, '2023-11-01'),
                       ('孙七', '销售部', 18000, '2023-05-20'), ('周八', '财务部', 16000, '2020-02-18')])
        c.executemany("INSERT INTO product_sales (product_name, category, revenue, units_sold) VALUES (?, ?, ?, ?)",
                      [('旗舰手机X', '电子产品', 500000, 100), ('降噪耳机', '电子产品', 150000, 300), 
                       ('人体工学椅', '办公用品', 80000, 50), ('机械键盘', '电子产品', 45000, 150),
                       ('无线鼠标', '电子产品', 20000, 200), ('打印纸', '办公用品', 5000, 500)])
        conn.commit()
    conn.close()

# 会话管理相关函数
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

# 初始化两个数据库
init_chat_db()
init_business_db() 

if "current_session_id" not in st.session_state:
    sessions = get_all_sessions()
    if sessions:
        st.session_state.current_session_id = sessions[0][0]
    else:
        st.session_state.current_session_id = create_new_session()

# ==========================================
# 密钥与大模型配置区
# ==========================================
try:
    _ = st.secrets["DEEPSEEK_API_KEY"]
    _ = st.secrets["ZHIPU_API_KEY"]
    os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]
except KeyError as e:
    st.error(f"⚠️ 缺少 API Key: {e}。请检查 Secrets 配置！")
    st.stop()

# 核心大脑：DeepSeek
llm = ChatOpenAI(
    model="deepseek-chat", 
    api_key=st.secrets["DEEPSEEK_API_KEY"], 
    base_url="https://api.deepseek.com", 
    temperature=0.1,  # 保持0.1，让AI写SQL更严谨
    streaming=True
)

# ==========================================
# 工具区 (Agent Tools)
# ==========================================
@tool
def run_python_code(code: str) -> str:
    """
    运行 Python 代码进行复杂的数据分析、统计计算或绘制动态图表。
    输入必须是一段合法的 Python 脚本。
    1. 画图必须使用 px (plotly.express)。
    2. 画完后必须调用 `st.plotly_chart(fig)` 进行渲染。
    """
    old_stdout = sys.stdout
    redirected_output = sys.stdout = io.StringIO()
    
    if "temp_chart_paths" not in st.session_state:
        st.session_state.temp_chart_paths = []

    class MockSt:
        def __getattr__(self, name):
            return getattr(st, name)
        
        def plotly_chart(self, fig, **kwargs):
            cid = str(uuid.uuid4())
            cpath = f"saved_charts/{cid}.json"
            pio.write_json(fig, cpath)
            st.session_state.temp_chart_paths.append(cpath)
            st.plotly_chart(fig, **kwargs)

    try:
        global_env = {
            "st": MockSt(), 
            "pd": __import__('pandas'),
            "px": __import__('plotly.express')
        }
        exec(code, global_env)
        
        if 'fig' in global_env and not st.session_state.temp_chart_paths:
            fig = global_env['fig']
            cid = str(uuid.uuid4())
            cpath = f"saved_charts/{cid}.json"
            pio.write_json(fig, cpath)
            st.session_state.temp_chart_paths.append(cpath)

        sys.stdout = old_stdout
        return redirected_output.getvalue() + "\n（代码执行成功，图表已生成）"
    except Exception as e:
        sys.stdout = old_stdout
        return f"代码执行出错: {str(e)}"

@tool
def run_sql_query(sql: str) -> str:
    """
    用于查询公司企业数据库 (company_data.db)。
    输入必须是合法的 SQLite SQL 查询语句。
    执行后会返回查询结果。
    """
    try:
        # 直接连接本地/GitHub传上来的那个库
        conn = sqlite3.connect('company_data.db')
        c = conn.cursor()
        c.execute(sql)
        rows = c.fetchall()
        columns = [description[0] for description in c.description] if c.description else []
        conn.close()
        
        if not rows:
            return "查询成功，但结果为空。"
            
        res = f"列名: {columns}\n数据 (最多展示前50行):\n"
        for row in rows[:50]:
            res += str(row) + "\n"
        return res
    except Exception as e:
        return f"SQL执行出错: {str(e)}"

# ==========================================
# 侧边栏 UI
# ==========================================
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

# ==========================================
# 初始化 Agent 和工具列表
# ==========================================
tools = [
    TavilySearchResults(max_results=3, description="用于搜索互联网上的实时信息。"),
    run_python_code,
    run_sql_query
]

# RAG 文档处理
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
        st.session_state.current_file_path = tmp_path

    if uploaded_file.name.endswith(".csv"):
        st.success(f"✅ 数据表 {uploaded_file.name} 已加载！")
    else:
        with st.spinner("正在启动高级 RAG 引擎解析文件..."):
            if uploaded_file.name.endswith(".pdf"):
                loader = PyPDFLoader(tmp_path)
            else:
                loader = TextLoader(tmp_path, encoding="utf-8")
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            splits = text_splitter.split_documents(docs)
            embeddings = OpenAIEmbeddings(
                model="embedding-3",
                api_key=st.secrets["ZHIPU_API_KEY"],
                base_url="https://open.bigmodel.cn/api/paas/v4/"
            ) 
            vectorstore = FAISS.from_documents(splits, embeddings)
            base_retriever = vectorstore.as_retriever()
            advanced_retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)
            retriever_tool = create_retriever_tool(advanced_retriever, "document_search", "用于搜索用户文档的内容。")
            tools.append(retriever_tool)
            st.success(f"✅ 文件 {uploaded_file.name} 已加载！")

# 告诉 AI 数据库的结构
system_prompt_text = """你是一个企业级全能 AI 助手。
当用户需要画图时，调用 run_python_code 生成图表，无需废话。

【数据库说明】
你可以使用 run_sql_query 工具查询公司的 SQLite 数据库 (company_data.db)。库中包含两张表：
1. employees 表：id, name(姓名), department(部门), salary(薪资), join_date(入职日期)
2. product_sales 表：id, product_name(产品名), category(类别), revenue(营收), units_sold(销量)

如果用户想看数据图表，你可以先用 run_sql_query 查出数据，然后再把查到的数据放进 run_python_code 里画图！"""

if "current_file_path" in st.session_state and st.session_state.current_file_path:
    system_prompt_text += f"\n\n[机密] 最新本地文件路径: '{st.session_state.current_file_path}'。如果是表格直接 read_csv 读取。"

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt_text),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ==========================================
# 主界面对话区
# ==========================================
st.title("🤖 满血版企业级 AI (直接读取 GitHub 数据库)")

st.session_state.messages = get_messages(st.session_state.current_session_id)

# 历史消息与图表渲染
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        content = msg["content"]
        if msg["role"] == "assistant":
            chart_paths = re.findall(r'\[CHART_PATH:(.*?)\]', content)
            clean_content = re.sub(r'\[CHART_PATH:.*?\]', '', content).strip()
            
            if clean_content:
                st.markdown(clean_content)
                
            for cpath in chart_paths:
                if os.path.exists(cpath):
                    try:
                        fig = pio.read_json(cpath)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"图表加载失败: {e}")
        else:
            st.markdown(content)

# 用户输入处理
if user_input := st.chat_input("问我关于公司员工薪资或产品销量的问题吧！也可以让我画图！"):
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
        st.session_state.temp_chart_paths = []
        
        try:
            response = agent_executor.invoke(
                {"input": user_input, "chat_history": chat_history},
                {"callbacks": [st_callback]}
            )
            answer = response["output"]
            
            # 隐藏记录图表路径
            if "temp_chart_paths" in st.session_state and st.session_state.temp_chart_paths:
                for p in st.session_state.temp_chart_paths:
                    answer += f"\n[CHART_PATH:{p}]"
                st.session_state.temp_chart_paths = []
                
            save_message(st.session_state.current_session_id, "assistant", answer)
            st.rerun()
            
        except Exception as e:
            st.error(f"发生错误: {e}")
