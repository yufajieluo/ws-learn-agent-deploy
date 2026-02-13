import os
import operator
import sqlite3
import requests
import urllib3
import streamlit as st
from typing import Annotated, Literal, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import Tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.tools.retriever import create_retriever_tool

os.environ['GOOGLE_API_KEY'] = st.secrets['GOOGLE_API_KEY']
os.environ['TAVILY_API_KEY'] = st.secrets['TAVILY_API_KEY']

PERSIST_DIR = 'day30_chroma_db_data'
DB_PATH = 'agent_memory.sqlite'

@st.cache_resource
def init_agent():
    embedding_function = GoogleGenerativeAIEmbeddings(
        model = 'models/gemini-embedding-001',
        task_type = 'retrieval_query'
    )

    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding_function
    )
    retriever = vectorstore.as_retriever(
        search_kwargs = {'k': 3}
    )

    tool_internal = create_retriever_tool(
        retriever= retriever,
        name = 'search_internal_knowledge',
        description='【绝密】仅用于查询公司内部 AI 助手 "arvis" 的配置、IP地址、开发者或紧急联系人。'
    )
    tool_external = TavilySearchResults(max_results=2)
    tool_external.name = 'search_external_knowledge'
    tools = [
        tool_internal,
        tool_external
    ]

    llm = ChatGoogleGenerativeAI(
        model = 'models/gemini-2.5-flash',
        temperature=0
    )
    llm_with_tools = llm.bind_tools(tools)

    class AgentState(TypedDict):
        messages: Annotated[list[BaseMessage], operator.add]

    def node_chatbot(state: AgentState):
        return {
            'messages': [
                llm_with_tools.invoke(state['messages'])
            ]
        }
    
    node_tool = ToolNode(tools=tools)

    def should_continue(state: AgentState) -> Literal['tools', '__end__']:
        if state['messages'][-1].tool_calls:
            return 'tools'
        else:
            return '__end__'
        
    workflow = StateGraph(AgentState)
    workflow.add_node('agent', node_chatbot)
    workflow.add_node('tools', node_tool)
    workflow.set_entry_point('agent')
    workflow.add_conditional_edges('agent', should_continue)
    workflow.add_edge('tools', 'agent')

    conn = sqlite3.connect(DB_PATH, check_same_thread = False)
    memory = SqliteSaver(conn)

    return workflow.compile(checkpointer=memory)

app = init_agent()

st.set_page_config(
    page_title = 'Jarvis Super Agent',
    page_icon = '🤖'
)
st.title('🤖 Jarvis - 企业级全能助手')

with st.sidebar:
    st.header("⚙️ 会话管理")
    thread_id = st.text_input('当前会话 ID (Thread ID)', value = 'user_1')
    st.caption('切换不同的 ID 即可开启全新的记忆线。')

config = {
    'configurable': {
        'thread_id': thread_id
    }
}

state = app.get_state(config)
if 'messages' in state.values:
    for msg in state.values['messages']:
        if isinstance(msg, HumanMessage):
            st.chat_message('user').write(msg.content)
        elif isinstance(msg, AIMessage):
            if msg.content and msg.content.strip():
                st.chat_message('assistant').write(msg.content)

if prompt := st.chat_input('问我关于 Jarvis 内部机密，或者今天的新闻...'):
    st.chat_message('user').write(prompt)

    with st.chat_message('assistant'):
        status_placeholder = st.empty()
        text_placeholder = st.empty()

        events = app.stream(
            {
                'messages': [HumanMessage(content = prompt)]
            },
            config = config,
            stream_mode = 'updates'
        )

        for event in events:
            if 'agent' in event:
                ai_msg = event['agent']['messages'][0]

                if hasattr(ai_msg, 'tool_calls') and ai_msg.tool_calls:
                    for tool in ai_msg.tool_calls:
                        with status_placeholder.status(f'🛠️ 正在调用工具: `{tool["name"]}`...', expanded=True):
                            st.write(f'携带参数: {tool["args"]}')
                
                if ai_msg.content:
                    text_placeholder.markdown(ai_msg.content)

            elif 'tools' in event:
                status_placeholder.status('✅ 工具调用完成！', state = 'complete', expanded = False)
            