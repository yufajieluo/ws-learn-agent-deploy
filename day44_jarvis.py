import os
import streamlit as st
import tempfile
import uuid
import shutil
from dotenv import load_dotenv

# --- LangChain 组件 ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# --- ⚙️ 1. 页面配置 ---
st.set_page_config(page_title="Jarvis: Time Traveler", layout="wide", page_icon="🕰️")
st.title("🕰️ Jarvis: 支持'时间旅行'的 RAG 助手")

# --- 🔐 2. Session ID 管理 (核心修改) ---

# 初始化一个默认的随机 ID (仅在第一次运行时生成)
if "init_id" not in st.session_state:
    st.session_state.init_id = str(uuid.uuid4())[:8] # 取前8位方便输入

# --- 🎨 3. 侧边栏控制中心 ---
with st.sidebar:
    st.header("🎮 控制台")
    
    # API Key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        api_key = st.text_input("Google API Key", type="password")

    st.divider()

    # 🔥 A. 手动 Session ID 输入框
    st.subheader("🆔 会话管理 (Session)")
    
    # 这里是关键：输入框的值决定了当前的 session_id
    # 默认值是那个随机生成的，但你可以手动改成 "test", "demo", "user1" 等
    custom_session_id = st.text_input(
        "当前 Session ID", 
        value=st.session_state.init_id,
        help="修改此 ID 可切换不同的对话历史。输入旧的 ID 可以找回记忆。"
    )

    # 简单的状态监测
    if "last_session_id" not in st.session_state:
        st.session_state.last_session_id = custom_session_id
    
    # 💡 监测 ID 变化：如果用户切了 ID，我们要清理 RAG 上下文，防止数据串味
    if custom_session_id != st.session_state.last_session_id:
        st.toast(f"🔄 切换会话: {st.session_state.last_session_id} -> {custom_session_id}")
        st.session_state.last_session_id = custom_session_id
        # 清除旧的 Retriever (RAG)
        if "retriever" in st.session_state:
            del st.session_state.retriever
        st.rerun()

    st.divider()

    # 🔥 B. 自定义人设
    system_persona = st.text_area(
        "🎭 系统人设 (System Prompt)",
        value="你是一个乐于助人的 AI 助手。请用中文回答。",
        height=100
    )

    # 🔥 C. RAG 文档
    st.subheader("📚 知识库 (RAG)")
    uploaded_file = st.file_uploader("上传当前会话的文档 (PDF/TXT)", type=["pdf", "txt"])
    
    # 清空历史按钮
    if st.button("🗑️ 清空当前 ID 的历史"):
        # 连接数据库并清空指定 Session 的记录
        history_db = SQLChatMessageHistory(
            session_id=custom_session_id,
            connection="sqlite:///chat_history.db"
        )
        history_db.clear()
        st.toast("历史记录已抹除！")
        st.rerun()

# --- 🧠 4. 核心逻辑 ---

if not api_key:
    st.info("👈 请先在左侧配置 API Key")
    st.stop()

# 定义获取历史记录的函数 (LangChain 需要这个)
def get_session_history(session_id):
    """根据 session_id 从 SQLite 读取历史"""
    return SQLChatMessageHistory(
        session_id=session_id,
        connection="sqlite:///chat_history.db"
    )

# 处理文件上传
def process_file(uploaded_file, session_id):
    """处理文件并存入隔离的向量库"""
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path) if uploaded_file.name.endswith('.pdf') else TextLoader(tmp_path)
    docs = loader.load()
    os.remove(tmp_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
    
    # 使用 session_id 作为 collection name，实现 RAG 物理隔离
    # 注意：Chroma collection name 只能包含字母数字和下划线，且不能太长
    safe_collection_name = f"rag_{session_id}".replace("-", "_")
    
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings,
        collection_name=safe_collection_name, 
        persist_directory="./chroma_db"
    )
    return vectorstore.as_retriever()

# --- 🔗 5. 构建 Chain ---

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0.7
)

# 逻辑分流：有文件 vs 无文件
if uploaded_file:
    # 只有当 retriever 不存在时才处理，避免每次刷新都重新 Embedding
    if "retriever" not in st.session_state:
        with st.spinner("正在向量化文档..."):
            st.session_state.retriever = process_file(uploaded_file, custom_session_id)
    
    retriever = st.session_state.retriever
    
    # RAG Prompt
    rag_system_prompt = (
        f"{system_persona}\n\n"
        "【指令】：请基于以下上下文回答问题。如果不知道，就说不知道。\n"
        "【上下文】:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", rag_system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])
    
    # 辅助函数：格式化文档
    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    # RAG Chain
    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"]))
        )
        | prompt
        | llm
        | StrOutputParser()
    )
else:
    # 普通对话 Chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_persona),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])
    chain = prompt | llm | StrOutputParser()

# 注入历史记录能力
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

# --- 💬 6. 聊天界面 ---

st.caption(f"当前正在对话的 Session ID: **{custom_session_id}**")

# A. 渲染历史记录 (从 SQLite 读取)
# 我们直接调用 get_session_history 来获取当前 ID 的历史
current_history = get_session_history(custom_session_id)
if not current_history.messages:
    st.info("👋 这是一个新的会话（或者历史记录为空）。")

for msg in current_history.messages:
    # 简单的样式映射
    role = "user" if msg.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# B. 处理输入
if user_input := st.chat_input("说点什么..."):
    # 1. 显示用户输入
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # 2. 调用 AI (流式)
    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""
        
        # RunnableWithMessageHistory 需要 configurable 参数来指定 session_id
        config = {"configurable": {"session_id": custom_session_id}}
        
        stream = chain_with_history.stream(
            {"question": user_input},
            config=config
        )
        
        for chunk in stream:
            full_response += chunk
            response_container.markdown(full_response + "▌")
        
        response_container.markdown(full_response)