import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools.tavily_search import TavilySearchResults

# ==========================================
# 1. 页面配置与初始化
# ==========================================
st.set_page_config(page_title="全能 AI 助手", page_icon="🤖", layout="wide")
st.title("🤖 满血版 AI 助手 (已接入 Tavily 搜索)")

# ==========================================
# 2. 环境变量与 API Key 配置 (从 Streamlit Secrets 获取)
# ==========================================
# 注意：务必在 Streamlit Cloud 的 Settings -> Secrets 中配置 TAVILY_API_KEY
try:
    os.environ["OPENAI_API_KEY"] = st.secrets["ZHIPU_API_KEY"]
    os.environ["OPENAI_API_BASE"] = st.secrets.get("OPENAI_API_BASE", "https://open.bigmodel.cn/api/paas/v4/")
    os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]
except KeyError as e:
    st.error(f"⚠️ 缺少必要的 API Key: {e}。请在 Streamlit 的 Secrets 中配置！")
    st.stop()

# ==========================================
# 3. 初始化大模型 (LLM)
# ==========================================
# 这里默认使用 glm-4，如果你用的是其他模型，请自行修改 model 名称
llm = ChatOpenAI(model="glm-4", temperature=0.5)

# ==========================================
# 4. 定义工具库 (Tools)
# ==========================================
# 这是我们新换上的“超级引擎” Tavily，绝不罢工！
search_tool = TavilySearchResults(
    max_results=3, 
    description="用于搜索互联网上的实时信息，如天气、新闻、最新数据等。当用户询问最新情况时必须使用此工具。"
)

# 这里可以将你之前的“文档检索工具(RetrieverTool)”加进来
tools = [search_tool] 

# ==========================================
# 5. 定义 Prompt (大脑指令)
# ==========================================
# 注意这里新增了 chat_history，让 AI 拥有记忆！
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个强大且热心的人工智能助手。你可以使用工具来搜索网络和查阅信息。遇到不知道的实时信息，必须调用搜索工具。"),
    MessagesPlaceholder(variable_name="chat_history", optional=True), # 开启记忆功能
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"), # AI 思考过程的草稿本
])

# ==========================================
# 6. 构建 Agent 和执行器
# ==========================================
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ==========================================
# 7. 会话状态管理 (Session State)
# ==========================================
# 初始化聊天记录列表
if "messages" not in st.session_state:
    st.session_state.messages = []

# 在侧边栏加一个清空记忆的按钮
with st.sidebar:
    st.header("⚙️ 助手设置")
    if st.button("🗑️ 清空大脑记忆", use_container_width=True):
        st.session_state.messages = []
        st.success("记忆已清空！咱们重新开始聊吧。")

# 将历史聊天记录渲染到页面上
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==========================================
# 8. 聊天交互逻辑
# ==========================================
if user_input := st.chat_input("来问我今天的天气吧，或者和我随便聊聊..."):
    # 1. 把用户的话显示出来并存入记忆
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. 提取聊天历史，格式化成 LangChain 需要的样子
    # 我们只取前几次的对话，避免 token 超出限制
    chat_history = []
    for msg in st.session_state.messages[:-1]: # 不包含刚刚输入的那一句
        if msg["role"] == "user":
            chat_history.append(("human", msg["content"]))
        else:
            chat_history.append(("ai", msg["content"]))

    # 3. 让 Agent 思考并回答
    with st.chat_message("assistant"):
        with st.spinner("🚀 AI 正在极速冲浪/思考中..."):
            try:
                # 把问题和历史记录一起喂给它
                response = agent_executor.invoke({
                    "input": user_input,
                    "chat_history": chat_history
                })
                
                answer = response["output"]
                st.markdown(answer)
                
                # 将 AI 的回答存入记忆
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"哎呀，脑子短路了，报错信息: {e}")
