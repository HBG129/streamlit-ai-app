import streamlit as st
import os
import tempfile
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
# 【新增】专门用于拦截流式输出并展示在 Streamlit 上的神器组件
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

st.set_page_config(page_title="全能 AI 助手", page_icon="🤖", layout="wide")
st.title("🤖 满血版 AI 助手 (网搜 + 文档 + 导出 + 流式打字)")

try:
    os.environ["OPENAI_API_KEY"] = st.secrets["ZHIPU_API_KEY"]
    os.environ["OPENAI_API_BASE"] = "https://open.bigmodel.cn/api/paas/v4/"
    os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]
except KeyError as e:
    st.error(f"⚠️ 缺少 API Key: {e}。请检查 Secrets 配置！")
    st.stop()

# 【关键改动1】在这里加上 streaming=True，告诉大模型：别憋着，有字就赶紧吐出来！
llm = ChatOpenAI(model="glm-4-flash", temperature=0.5, streaming=True)

with st.sidebar:
    st.header("📂 喂给 AI 本地知识")
    uploaded_file = st.file_uploader("上传 PDF / TXT / CSV 文件", type=["pdf", "txt", "csv"])
    
    st.divider()
    st.header("⚙️ 助手设置")
    if st.button("🗑️ 清空大脑记忆", use_container_width=True):
        st.session_state.messages = []
        st.success("记忆已清空！")
        
    if "messages" in st.session_state and len(st.session_state.messages) > 0:
        chat_text = "\n\n".join([f"{msg['role'].upper()}:\n{msg['content']}" for msg in st.session_state.messages])
        st.download_button(
            label="💾 一键导出聊天记录",
            data=chat_text,
            file_name="AI助手聊天记录.txt",
            mime="text/plain",
            use_container_width=True
        )

tools = []

search_tool = TavilySearchResults(
    max_results=3, 
    description="用于搜索互联网上的实时信息，如天气、新闻。如果问题涉及实时数据，必须使用此工具。"
)
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
        
        retriever_tool = create_retriever_tool(
            retriever,
            "document_search",
            "当你需要回答关于用户上传的文档或表格里的内容时，必须使用此工具搜索文档。"
        )
        tools.append(retriever_tool)
        st.success(f"✅ 文件 {uploaded_file.name} 已加载，可以提问了！")

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个全能助手。遇到不知道的实时信息用 search_tool。遇到关于用户上传文档的问题用 document_search。"),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("问我天气，或者传个文档/表格问我里面的内容..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    chat_history = [
        ("human", msg["content"]) if msg["role"] == "user" else ("ai", msg["content"])
        for msg in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant"):
        # 【关键改动2】去掉了干巴巴的 spinner，换上了高大上的回调拦截器
        st_callback = StreamlitCallbackHandler(st.container())
        try:
            # 【关键改动3】在 invoke 的时候，把拦截器塞进去，接管页面的实时渲染
            response = agent_executor.invoke(
                {"input": user_input, "chat_history": chat_history},
                {"callbacks": [st_callback]}
            )
            answer = response["output"]
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun() 
        except Exception as e:
            st.error(f"发生错误: {e}")
