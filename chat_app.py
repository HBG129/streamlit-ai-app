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
import plotly.graph_objects as go

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
# 数据库管理区 (维持不变，两个库完美分离)
# ==========================================

def init_chat_db():
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sessions (session_id TEXT PRIMARY KEY, title TEXT, created_at DATETIME)''')
    c.execute('''CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, content TEXT, created_at DATETIME)''')
    conn.commit()
    conn.close()

def init_business_db():
    conn = sqlite3.connect('company_data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS employees (id INTEGER PRIMARY KEY, name TEXT, department TEXT, salary INTEGER, join_date DATE)''')
    c.execute('''CREATE TABLE IF NOT EXISTS product_sales (id INTEGER PRIMARY KEY, product_name TEXT, category TEXT, revenue INTEGER, units_sold INTEGER)''')
    
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

# 核心大脑：DeepSeek (保持0.1的低温度)
llm = ChatOpenAI(
    model="deepseek-chat", 
    api_key=st.secrets["DEEPSEEK_API_KEY"], 
    base_url="https://api.deepseek.com", 
    temperature=0.1,  
    streaming=True
)

# ==========================================
# 工具区 (全新智能防断连拦截器)
# ==========================================
@tool
def run_python_code(code: str) -> str:
    """
    运行 Python 代码进行复杂的数据分析、统计计算或绘制动态图表。
    输入必须是一段合法的 Python 脚本。
    """
    # 依然拦截 matplotlib，强迫用 plotly
    if "matplotlib" in code or "plt." in code:
        return "【执行失败】请修改代码：禁止使用 matplotlib，请使用 plotly.express 画图，并将图表对象命名为 fig。"

    old_stdout = sys.stdout
    redirected_output = sys.stdout = io.StringIO()
    
    if "temp_chart_paths" not in st.session_state:
        st.session_state.temp_chart_paths = []

    # 智能图表保存器
    def auto_save_and_render(f):
        cid = str(uuid.uuid4())
        cpath = f"saved_charts/{cid}.json"
        pio.write_json(f, cpath)
        st.session_state.temp_chart_paths.append(cpath)
        st.plotly_chart(f, use_container_width=True)

    class MockSt:
        def __getattr__(self, name):
            return getattr(st, name)
        
        def plotly_chart(self, f, **kwargs):
            auto_save_and_render(f)

    try:
        global_env = {
            "st": MockSt(), 
            "pd": __import__('pandas'),
            "px": __import__('plotly.express')
        }
        
        # 屏蔽 AI 自带的 fig.show() 防止卡死
        code = code.replace("fig.show()", "")
        
        exec(code, global_env)
        
        # 【神级兜底】：如果AI忘了写 st.plotly_chart，我们自己去环境变量里捞它的图！
        if not st.session_state.temp_chart_paths:
            if 'fig' in global_env and isinstance(global_env['fig'], go.Figure):
                auto_save_and_render(global_env['fig'])
            else:
                # 连 fig 名字都写错了？没关系，遍历找！
                for val in global_env.values():
                    if isinstance(val, go.Figure):
                        auto_save_and_render(val)
                        break

        sys.stdout = old_stdout
        return redirected_output.getvalue() + "\n（代码执行成功，图表已完美生成并保存给用户）"
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

# RAG 及文件处理
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
        st.session_state.current_file_path = tmp_path

    if uploaded_file.name.endswith(".csv"):
        st.success(f"✅ 数据表 {uploaded_file.name} 已加载，AI 可以直接读取作图了！")
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

# 【优化版提示词】：减轻 AI 压力，告诉它直接写图表即可
system_prompt_text = """你是一个顶级的数据分析师与全能 AI 助手。你拥有强大的 Python 代码执行能力！

【核心技能 1：分析文件与画图】
1. 绝不允许对用户说“我无法生成图片”或“请截图下载”。你有能力画图！
2. 只要用户要求画图（或处理CSV文件），你**必须**调用 `run_python_code` 工具写代码。
3. 【红线规则】：禁止使用 matplotlib。只能使用 plotly.express (px) 画图！
4. 你只需要将生成的图表赋值给变量 `fig`，系统会自动帮你渲染并展示给用户，无需调用 .show()！

【核心技能 2：查询企业数据库】
你可以使用 `run_sql_query` 查询公司数据库 (company_data.db)。
1. employees 表：id, name(姓名), department(部门), salary(薪资), join_date(入职日期)
2. product_sales 表：id, product_name(产品名), category(类别), revenue(营收), units_sold(销量)

💡 工作流：如果用户想看数据库的数据图表，先用 `run_sql_query` 查出具体数据，然后把数据写入 `run_python_code` 生成 DataFrame 并用 plotly 画出来！"""

if "current_file_path" in st.session_state and st.session_state.current_file_path:
    system_prompt_text += f"\n\n[机密指令] 用户刚刚上传了文件，文件绝对路径为: '{st.session_state.current_file_path}'。如果是 CSV 表格，请直接用 pandas.read_csv 读取该路径并画图！"

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt_text),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ==========================================
# 主界面对话与图表渲染区
# ==========================================
st.title("🤖 满血版企业级 AI (超级稳定不报错版)")

st.session_state.messages = get_messages(st.session_state.current_session_id)

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

if user_input := st.chat_input("上传CSV让我画图，或者直接问我公司数据库的情况！"):
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
            
            if "temp_chart_paths" in st.session_state and st.session_state.temp_chart_paths:
                for p in st.session_state.temp_chart_paths:
                    answer += f"\n[CHART_PATH:{p}]"
                st.session_state.temp_chart_paths = []
                
            save_message(st.session_state.current_session_id, "assistant", answer)
            st.rerun()
            
        except Exception as e:
            st.error(f"发生错误: {e}")
