import os
import tempfile
import streamlit as st

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="我的全能 AI", page_icon="🤖")
st.title("🤖 我的第一个手搓全能 AI 智能体")

@st.cache_resource
def get_embeddings_model():
    return OpenAIEmbeddings(
        openai_api_base="https://open.bigmodel.cn/api/paas/v4",
        openai_api_key=st.secrets["ZHIPU_API_KEY"],
        model="embedding-3"
    )

embeddings = get_embeddings_model()

llm = ChatOpenAI(
    temperature=0.5,
    openai_api_base="https://open.bigmodel.cn/api/paas/v4",
    openai_api_key=st.secrets["ZHIPU_API_KEY"],
    model_name="glm-4-flash"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

with st.sidebar:
    st.header("📁 上传知识库")
    uploaded_file = st.file_uploader("上传一个 PDF 文档", type="pdf")
    
    if uploaded_file is not None and st.session_state.vectorstore is None:
        with st.spinner("正在拼命阅读文档中..."):

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            splits = text_splitter.split_documents(docs)
            
            st.session_state.vectorstore = FAISS.from_documents(splits, embeddings)
            st.success("文档阅读完毕！现在你可以向我提问了。")
            os.remove(tmp_file_path)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("你想问我点什么？"):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if st.session_state.vectorstore is not None:

            retriever = st.session_state.vectorstore.as_retriever()
            
            template = """你是一个乐于助人的 AI 助手。请根据下面提供的上下文来回答用户的问题。
            如果你不知道答案，就说你不知道，不要试图编造答案。
            
            上下文：
            {context}
            
            用户问题：{question}
            """
            prompt_template = ChatPromptTemplate.from_template(template)
            
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
                
            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt_template
                | llm
                | StrOutputParser()
            )
            
            response = rag_chain.invoke(prompt)
        else:

            response = llm.invoke(prompt).content
            
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
