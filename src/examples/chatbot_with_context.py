import os
import getpass
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, trim_messages

# types
# from langchain_core.messages import BaseMessage
# from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableConfig
# from typing import cast, List

MODEL_NAME = "gpt-4o-mini"
EXAMPLE_CONFIG: RunnableConfig = {"configurable": {"thread_id": "abc123"}}
EXAMPLE_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="You should always responde user with Chinese"),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


# 初始化语言模型
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass(
        "Please enter you API key for OpenAI: "
    )
llm = ChatOpenAI(model=MODEL_NAME, temperature=0)

trimmer = trim_messages(
    # 最大消息长度
    # 此处是 65 个 token，非常短，模型会很快丢失之前的对话历史
    # 这么做只是为了能够尽快看到 timmer 生效的效果
    max_tokens=65,
    
    # 裁剪策略
    # last：表示当需要裁剪消息历史时，从最早的消息开始删除，优先保留最近的消息
    strategy="last",
    
    # token 计算器
    token_counter=llm,
    
    # 是否保留系统消息
    # 表示即使裁剪发生，也不会裁剪掉 SystemMessage
    # 我们通常通过 SystemMessage 定义模型的角色、表现、回答风格等，因此不希望删除它
    include_system=True,
    
    # 是否允许部分消息被裁剪
    # 表示如果要裁剪，那就裁剪掉整个完整的消息
    # 而不是严格根据设定的最大 token，而不是严格按照 max_token 设定的数值来
    # 避免出现一条消息只裁剪了一半的情况
    allow_partial=False,
    
    # 指定保留消息时从哪种类型的消息开始
    # human：确保保留的消息序列始终从用户消息开始
    start_on="human",
)


# 将模型的调用封装到一个函数中，供 LangGraph 中的节点调用
def call_llm(state: MessagesState):
    trimmed_messages = trimmer.invoke(state["messages"])
    prompt = EXAMPLE_PROMPT_TEMPLATE.invoke({"messages": trimmed_messages})
    response = llm.invoke(prompt)
    return {"messages": response}


# 初始化 graph 并定义 graph 中的节点
workflow = StateGraph(state_schema=MessagesState)
workflow.add_edge(START, "llm")
workflow.add_node("llm", call_llm)
# 加入记忆持久化
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


def chatbot_with_context_example():
    print(f"""开始与 {MODEL_NAME} 的对话：""")

    while True:
        user_input = input("\ninput：").strip()
        user_message = [HumanMessage(content=user_input)]
        
        # 这是普通的输出模型，模型在将回答生成完成后再返回答案
        # output = app.invoke({"messages": user_message}, EXAMPLE_CONFIG)
        # typed_response = cast(List[BaseMessage], outputh["messages"])
        # print(typed_response[-1].content)
        
        # 使用流式传输
        app_on_streaming = app.stream(
            {"messages": user_message}, 
            EXAMPLE_CONFIG, 
            stream_mode="messages"
        )
        for chunk, metadata in app_on_streaming:
            if isinstance(chunk, AIMessage):
                print(chunk.content, end="", flush=True)
