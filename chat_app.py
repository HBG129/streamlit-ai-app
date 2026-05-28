import os
import tempfile
import streamlit as st

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import create_retriever_tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

st.set_page_config(page_title="我的全能 AI", page_icon="🤖")
st.title("🤖 满血版手搓 AI 智能体 (Agent 模式)")

# 1. 配置云端词向量模型
@st.cache_resource
def get_embeddings_model():
    return OpenAIEmbeddings(
        openai_api_base="https://open.bigmodel.cn/api/paas/v4",
        openai_api_key=st.secrets["ZHIPU_API_KEY"],
        model="embedding-3"
    )

embeddings = get_embeddings_model()

# 2. 配置大语言模型 (确保模型支持 Tool Calling)
llm = ChatOpenAI(
    temperature=0.5,
    openai_api_base="https://open.bigmodel.cn/api/paas/v4",
    openai_api_key=st.secrets["ZHIPU_API_KEY"],
    model_name="glm-4-flash",
    streaming=True
)

# 初始化状态
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# 侧边栏：多文件上传 & 联网开关
with st.sidebar:
    st.header("⚙️ 技能面板")
    web_search_enabled = st.toggle("🌐 开启全网冲浪", value=False)
    st.caption("开启后，遇到不懂的问题我会自己去网上搜！")
    
    st.divider()
    
    st.header("📁 文件大师")
    uploaded_file = st.file_uploader(
        "支持 PDF, Word, TXT, CSV", 
        type=["pdf", "docx", "txt", "csv"]
    )
    
    if uploaded_file is not None and st.session_state.vectorstore is None:
        with st.spinner("正在努力解析文档中..."):
            file_extension = uploaded_file.name.split(".")[-1].lower()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            if file_extension == "pdf":
                loader = PyPDFLoader(tmp_file_path)
            elif file_extension == "docx":
                loader = Docx2txtLoader(tmp_file_path)
            elif file_extension == "txt":
                loader = TextLoader(tmp_file_path, encoding="utf-8")
            elif file_extension == "csv":
                loader = CSVLoader(tmp_file_path, encoding="utf-8")
                
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            splits = text_splitter.split_documents(docs)
            
            st.session_state.vectorstore = FAISS.from_documents(splits, embeddings)
            st.success(f"《{uploaded_file.name}》阅读完毕！我已经掌握了它的内容。")
            os.remove(tmp_file_path)

# 历史对话展示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 聊天输入与逻辑处理
if prompt := st.chat_input("发号施令吧！"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 动态组装工具箱 (Tools)
        tools = []
        
        # 技能 1：把本地文档变成一个 Tool
        if st.session_state.vectorstore is not None:
            retriever = st.session_state.vectorstore.as_retriever()
            doc_tool = create_retriever_tool(
                retriever,
                "document_search",
                "搜索并读取用户刚刚上传的文档（PDF/Word/TXT/CSV）中的内容。如果用户问关于资料、文档里的问题，优先用这个工具。"
            )
            tools.append(doc_tool)
            
        # 技能 2：把全网搜索变成一个 Tool
        if web_search_enabled:
            search_tool = DuckDuckGoSearchRun(
                name="web_search",
                description="当你不知道某些实时信息、最新新闻或需要查阅互联网资料时，使用这个工具去全网搜索。"
            )
            tools.append(search_tool)

        # 判断是走 Agent 模式还是普通聊天模式
        if tools:
            # 准备一个高大上的 UI 容器，用来展示 AI 调用工具的思考过程
            st_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
            
            # Agent 大脑模板
            agent_prompt = ChatPromptTemplate.from_messages([
                ("system", "你是一个极度聪明、全能的数字员工。请合理利用手头的工具来解答老板的问题。"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
            
            # 创建智能体
            agent = create_tool_calling_agent(llm, tools, agent_prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
            
            # 运行 Agent 并把页面回调传进去
            response = agent_executor.invoke(
                {"input": prompt},
                {"callbacks": [st_callback]}
            )
            
            final_answer = response["output"]
            st.markdown(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            
        else:
            # 啥工具都没开，纯靠大模型脑力的流式输出
            chain = llm | StrOutputParser()
            response = st.write_stream(chain.stream(prompt))
            st.session_state.messages.append({"role": "assistant", "content": response})
