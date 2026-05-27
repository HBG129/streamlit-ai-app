import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun

# 页面配置
st.set_page_config(page_title="AI 智能体对话", page_icon="🤖")
st.title("🤖 具备自主路由能力的 AI Agent")

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 设置")
    api_key = st.text_input("请输入你的 API Key", type="password")
    base_url = st.text_input("API Base URL (可选)", value="https://api.openai.com/v1")
    model_name = st.text_input("输入模型名称", value="gpt-3.5-turbo")
    web_search_enabled = st.toggle("🌐 开启全网搜索工具", value=True)
    st.markdown("---")
    st.markdown("💡 **当前高阶能力**：Function Calling 工具调用。")

# 初始化对话历史
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "你好！我现在是一个具备判断力的 Agent。你可以问我常识，或者让我去网上搜索最新资讯！"}]

# 显示历史对话
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 接收用户输入
prompt = st.chat_input("问我任何问题...")

if prompt:
    # 记录并显示用户输入
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # 检查 API Key
    if not api_key:
        st.info("请先在左侧输入 API Key！")
        st.stop()

    # 初始化大模型（必须是支持 Tool Calling 的模型）
    llm = ChatOpenAI(
        api_key=api_key,
        base_url=base_url if base_url else None,
        model=model_name,
        temperature=0.7
    )

    # 准备工具箱
    search_tool = DuckDuckGoSearchRun(
        name="web_search",
        description="当用户询问最新的新闻、当前事件、天气或你不知道的实时数据时，必须使用这个工具去全网搜索。"
    )
    # 根据用户的开关决定是否把搜索工具交给模型
    tools = [search_tool] if web_search_enabled else []

    # 设置 Agent 大脑模板
    agent_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个极其聪明的 AI 助手。你可以自主决定是否使用工具来回答用户的问题。如果使用工具，请结合工具返回的结果给用户一个完整的回答。"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"), # 关键：这是 Agent 记录思考和工具调用过程的草稿本
    ])

    # 创建 Agent 和执行器
    agent = create_tool_calling_agent(llm, tools, agent_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # 运行 Agent 并展示结果
    with st.chat_message("assistant"):
        with st.spinner("🧠 正在思考是否需要调用工具..."):
            try:
                response = agent_executor.invoke({"input": prompt})
                final_answer = response["output"]
                st.write(final_answer)
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
            except Exception as e:
                st.error(f"调用出错啦，请检查模型是否支持 Function Calling 或网络是否正常：{e}")
