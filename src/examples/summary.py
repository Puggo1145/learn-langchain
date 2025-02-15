import os
import getpass
import operator
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.combine_documents.reduce import (
    acollapse_docs,
    split_list_of_docs,
)
from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

# types
from typing import Annotated, List, Literal, TypedDict, cast
from langgraph.types import Send


MODEL_NAME = "gpt-4o-mini"

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

llm = init_chat_model(MODEL_NAME, model_provider="openai")

# load and chunk source info
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()
text_spliter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
split_docs = text_spliter.split_documents(docs)

MAP_PROMPT_TEMPLATE = """
Write a concise summary of the following:
{context}
"""
map_prompt = ChatPromptTemplate([("system", MAP_PROMPT_TEMPLATE)])

REDUCE_PROMPT_TEMPLATE = """
The following is a set of summaries:
{docs}
Take these and distill it into a final, consolidated summary
of the main themes.
"""
reduce_prompt = ChatPromptTemplate([("human", REDUCE_PROMPT_TEMPLATE)])


MAX_TOKEN = 1000


def length_function(documents: List[Document]) -> int:
    """Get number of tokens for input contents"""
    return sum(llm.get_num_tokens(doc.page_content) for doc in documents)


# overall state of the main graph including original contents and its corresponding summary, a copllapsed_summaries and a final summary
class OverallState(TypedDict):
    contents: List[str]
    # summaries 注解了 add 操作，表示我们希望将不同 node 的总结结果和并到一个数组中
    summaries: Annotated[list, operator.add]
    collapsed_summaries: List[Document]
    final_summary: str


# 基于某个 document 生成 summary
class SummaryState(TypedDict):
    content: str


async def generate_summary(state: SummaryState):
    prompt = map_prompt.invoke({"context": state["content"]})
    response = await llm.ainvoke(prompt)
    return {"summaries": [response.content]}


# Here we define the logic to map out over the documents
# We will use this an edge in the graph
def map_summaries(state: OverallState):
    # We will return a list of `Send` objects
    # Each `Send` object consists of the name of a node in the graph
    # as well as the state to send to that node
    return [
        Send("generate_summary", {"content": content}) for content in state["contents"]
    ]


def collect_summaries(state: OverallState):
    return {
        "collapsed_summaries": [Document(summary) for summary in state["summaries"]]
    }


# summary function for multiple documents summary or final summary
async def _reduce(docs: List[Document], **kwargs) -> str:
    prompt = reduce_prompt.invoke({"docs": [doc.page_content for doc in docs]})
    response = await llm.ainvoke(prompt)
    return cast(str, response.content)


# generate summaries for each original document
async def collapse_summaries(state: OverallState):
    doc_lists = split_list_of_docs(
        state["collapsed_summaries"], length_function, MAX_TOKEN
    )
    results = []
    for doc_list in doc_lists:
        results.append(await acollapse_docs(doc_list, _reduce))

    return {"collapsed_summaries": results}


# conditional edge determing whether we should collapse the summaries or not
def should_collapse(
    state: OverallState,
) -> Literal["collapse_summaries", "generate_final_summary"]:
    num_tokens = length_function(state["collapsed_summaries"])
    if num_tokens > MAX_TOKEN:
        return "collapse_summaries"
    else:
        return "generate_final_summary"


# Here we will generate the final summary
async def generate_final_summary(state: OverallState):
    response = await _reduce(state["collapsed_summaries"])
    return {"final_summary": response}


# Construct the graph
# Nodes:
graph = StateGraph(OverallState)
graph.add_node("generate_summary", generate_summary)  # same as before
graph.add_node("collect_summaries", collect_summaries)
graph.add_node("collapse_summaries", collapse_summaries)
graph.add_node("generate_final_summary", generate_final_summary)

# Edges:
graph.add_conditional_edges(START, map_summaries, ["generate_summary"])  # type: ignore
graph.add_edge("generate_summary", "collect_summaries")
graph.add_conditional_edges("collect_summaries", should_collapse)
graph.add_conditional_edges("collapse_summaries", should_collapse)
graph.add_edge("generate_final_summary", END)

app = graph.compile()


async def summary_example():
    app_on_stream = app.astream(
        {"contents": [doc.page_content for doc in split_docs]}, {"recursion_limit": 10}
    )
    async for step in app_on_stream:
        print(list(step.keys()))
        print(step)

