import os
import tempfile
import datetime

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

st.set_page_config(page_title="手搓版 AI Agent", page_icon="🤖", layout="wide")
st.title("🤖 终极版：纯手搓智能体 + 原生思考可视化")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

@st.cache_resource
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")

embeddings = get_embeddings_model()

if "vectorstore" not in st.session_state:
    try:
        st.session_state.vectorstore = FAISS.load_local("advanced_db", embeddings, allow_dangerous_deserialization=True)
    except Exception:
        st.session_state.vectorstore = None

with st.sidebar:
    st.header("📁 知识库管理")
    uploaded_file = st.file_uploader("上传新文档 (支持 PDF / TXT)", type=['pdf', 'txt'])
    
    if uploaded_file and uploaded_file.name not in st.session_state.processed_files:
        with st.spinner(f"正在给 AI 喂食 {uploaded_file.name}，请稍等..."):
            file_extension = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            if file_extension.lower() == '.pdf':
                loader = PyPDFLoader(tmp_path)
            else:
                loader = TextLoader(tmp_path, encoding='utf-8')
                
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            splits = splitter.split_documents(docs)
            
            if st.session_state.vectorstore is None:
                st.session_state.vectorstore = FAISS.from_documents(splits, embeddings)
            else:
                st.session_state.vectorstore.add_documents(splits)
                
            st.session_state.vectorstore.save_local("advanced_db")
            st.session_state.processed_files.append(uploaded_file.name)
            os.remove(tmp_path) 
            st.success(f"✅ {uploaded_file.name} 学习完成！")


tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取系统当前的准确时间。当你被问到时间、日期时，必须调用这个工具。",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "从本地知识库中检索信息。当用户询问有关上传的文档、本地资料等内容时调用。",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "要搜索的关键词"}},
                "required": ["query"]
            }
        }
    }
]

def execute_tool(tool_name, tool_args):
    if tool_name == "get_current_time":
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    elif tool_name == "search_knowledge_base":
        query = tool_args.get("query", "")
        if st.session_state.vectorstore is None:
            return "本地知识库为空，请提醒用户上传文档。"
        docs = st.session_state.vectorstore.similarity_search(query, k=3)
        if not docs:
            return "知识库中未找到相关内容。"
        return "\n\n".join([d.page_content for d in docs])
        
    return "找不到该工具"

llm = ChatOpenAI(
    temperature=0.5,
    openai_api_base="https://open.bigmodel.cn/api/paas/v4",
    openai_api_key=st.secrets["ZHIPU_API_KEY"], # 👈 关键修改
    model_name="glm-4-flash"
)
llm_with_tools = llm.bind_tools(tools_schema)


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("和智能体聊聊，或查阅知识库..."):
    with st.chat_message("user"):
        st.markdown(user_input)
            
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    chat_history = [SystemMessage(content="你是一个超级智能体。遇到不懂的问题、需要查阅文档或询问时间时，请自主调用工具。回答要生动活泼。")]
    for msg in st.session_state.messages[-5:]:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            chat_history.append(AIMessage(content=msg["content"]))
    
    with st.chat_message("assistant"):

        with st.status("🧠 AI 正在思考...", expanded=True) as status:
            
            response = llm_with_tools.invoke(chat_history)
            
            if hasattr(response, 'tool_calls') and len(response.tool_calls) > 0:
                chat_history.append(response) # 把 AI 想要用工具的请求记录下来
                
                for tool_call in response.tool_calls:
                    t_name = tool_call["name"]
                    t_args = tool_call["args"]
                    t_id = tool_call["id"]
                    
                    status.write(f"🛠️ **决定使用工具**：`{t_name}`")
                    status.write(f"📥 **提取参数为**：`{t_args}`")
                    
                    t_result = execute_tool(t_name, t_args)
                    status.write(f"📄 **工具返回结果**：成功获取 {len(str(t_result))} 个字符的情报！")

                    chat_history.append(ToolMessage(content=str(t_result), tool_call_id=t_id))

                status.write("💡 情报收集完毕，正在组织语言...")
                final_response = llm_with_tools.invoke(chat_history)
                final_text = final_response.content
                
            else:
                status.write("💡 这个问题很简单，直接回答。")
                final_text = response.content
                
            status.update(label="✅ 思考完成！", state="complete", expanded=False)
            
        st.markdown(final_text)
        st.session_state.messages.append({"role": "assistant", "content": final_text})