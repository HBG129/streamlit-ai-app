import sys
import subprocess
import streamlit as st

# 1. 核弹级补丁：强行检测并升级云端依赖库（必须放在最开头）
try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
except ImportError:
    st.info("🔄 检测到云端环境版本过旧，正在强行热更新依赖库，请稍候几秒钟...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", 
                           "langchain>=0.2.0", "langchain-core>=0.2.0", 
                           "langchain-community>=0.2.0", "langchain-openai"])
    from langchain.agents import AgentExecutor, create_tool_calling_agent

# 其余必需的库
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage

# ==========================================
# 页面基础配置
# ==========================================
st.set_page_config(page_title="👑 神级全栈 AI 助手", page_icon="👑")
st.title("👑 神级全栈 AI 助手")
st.caption("🚀 已加载强制兼容补丁，无惧云端环境报错！")

# ==========================================
# 工具定义区 (如果你有之前写的爬虫/数据库工具，可以替换或加在这里)
# ==========================================
@tool
def search_web(query: str) -> str:
    """当遇到不知道的知识或实时信息时，使用这个工具进行网络搜索"""
    return f"这是关于【{query}】的模拟搜索结果。工具调用已成功跑通！"

tools = [search_web]

# ==========================================
# 侧边栏：API 密钥配置
# ==========================================
st.sidebar.header("⚙️ 参数配置")
api_key = st.sidebar.text_input("请输入 API Key", type="password")
# 默认填入智谱的 Base URL，如果你用纯正的 OpenAI，可以清空或者改掉
base_url = st.sidebar.text_input("请输入 Base URL", value="https://open.bigmodel.cn/api/paas/v4/")
model_name = st.sidebar.text_input("请输入模型名称", value="glm-4")

if not api_key:
    st.warning("👈 请在左侧栏输入您的 API Key 以激活 AI 助手")
    st.stop()

# ==========================================
# 初始化 LLM 与 Agent (核心报错点已修复)
# ==========================================
llm = ChatOpenAI(
    api_key=api_key,
    base_url=base_url,
    model=model_name,
    temperature=0.7
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个强大的全栈 AI 助手。你可以聪明地使用工具来回答用户的问题。"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# 使用最新兼容性最强的函数构建 Agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
