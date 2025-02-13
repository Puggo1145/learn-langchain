import os
import getpass
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

# from langchain.globals import set_debug
# set_debug(True)

MODEL_NAME = "gpt-4o-mini"
EXAMPLE_CONFIG: RunnableConfig = {"configurable": {"thread_id": "abc123"}}

# 初始化 API Keys
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]
if not TAVILY_API_KEY:
    TAVILY_API_KEY = getpass.getpass("TAVILY_API_KEY required: ")

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
if not TAVILY_API_KEY:
    TAVILY_API_KEY = getpass.getpass("TAVILY_API_KEY required: ")

# tools
test_tool = {
    "type": "function",
    "function": {
        "name": "get_hihi_weather",
        "description": "Get current temperature for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and country e.g. Bogotá, Colombia",
                }
            },
            "required": ["location"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}
search = TavilySearchResults(max_results=2)
tools = [
    search,
    # test_tool
]

# model
model = ChatOpenAI(temperature=0, model=MODEL_NAME)
model_with_tools = model.bind_tools(tools)

# agent
memory = MemorySaver()
agent_executor = create_react_agent(model, tools, checkpointer=memory)


def agent_example():
    messages = [HumanMessage(content="What is the weather in Chengdu")]
    response_on_streaming = agent_executor.stream(
        {"messages": messages}, 
        config=EXAMPLE_CONFIG
    )
    for chunk in response_on_streaming:
        print(chunk)
        print("---")
