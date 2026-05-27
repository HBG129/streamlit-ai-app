import os
import tempfile
import streamlit as st

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun

st.set_page_config(page_title="我的全能 AI", page_icon="🤖")
st.title("🤖 满血版手搓 AI 智能体")

# 1. 配置云端词向量模型
@st.cache_resource
def get_embeddings_model():
    return OpenAIEmbeddings(
        openai_api_base="https://open.bigmodel.cn/api/paas/v4",
        openai_api_key=st.secrets["ZHIPU_API_KEY"],
        model="embedding-3"
    )

embeddings = get_embeddings_model()

# 2. 配置大语言模型 (加入 streaming 参数支持流式)
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
            st.success(f"《{uploaded_file.name}》阅读完毕！")
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
        context_text = ""
        
        # 技能 1：从文档里找答案
        if st.session_state.vectorstore is not None:
            retriever = st.session_state.vectorstore.as_retriever()
            docs = retriever.invoke(prompt)
            context_text += "【来自本地文档的参考资料】：\n" + "\n".join(doc.page_content for doc in docs) + "\n\n"
            
        # 技能 2：去全网找答案
        if web_search_enabled:
            with st.spinner("🌍 正在全网搜集情报..."):
                search = DuckDuckGoSearchRun()
                search_results = search.invoke(prompt)
                context_text += "【来自全网搜索的最新情报】：\n" + search_results + "\n\n"

        # 脑力整合与流式输出
        if context_text != "":
            template = """你是一个全能数字员工。请参考我为你提供的资料来回答问题。
            如果资料里没有提到，你可以结合你的常识回答。
            
            参考资料：
            {context}
            
            老板的问题：{question}
            """
            final_prompt = ChatPromptTemplate.from_template(template)
            
            # 使用 LangChain 的管道语法 (LCEL) 和 StrOutputParser
            chain = final_prompt | llm | StrOutputParser()
            
            # st.write_stream 完美接收数据流，实现打字机效果！
            response = st.write_stream(chain.stream({"context": context_text, "question": prompt}))
        else:
            # 啥都没开，直接裸聊的流式输出
            chain = llm | StrOutputParser()
            response = st.write_stream(chain.stream(prompt))
            
        st.session_state.messages.append({"role": "assistant", "content": response})
