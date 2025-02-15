import os
import getpass
from langchain.chat_models import init_chat_model
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from bs4.filter import SoupStrainer
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, MessagesState
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, AIMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import END

# constants
API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    API_KEY = getpass.getpass("Please enter your OPENAI_API_KEY")
MODEL_NAME = "gpt-4o-mini"

# init models, vector_store, embeddings
model = init_chat_model(model=MODEL_NAME, model_provider="openai")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = InMemoryVectorStore(embedding=embeddings)

# load source info
filter = SoupStrainer(class_=("post-content", "post-title", "post-header"))
loader = WebBaseLoader(
    web_path=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": filter},
)
docs = loader.load()
# text spliting
text_spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_spliter.split_documents(docs)
# do doc embedding
vector_store.add_documents(documents=splits)

# graph
graph_builder = StateGraph(MessagesState)


# Node 1: Decide whether do retrieve or respond
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond"""
    model_with_tools = model.bind_tools([retrieve])
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# Node 2: Retrieve Tool Node
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query"""
    retrieved_docs = vector_store.similarity_search(query=query, k=2)
    serialized = "\n\n".join(
        f"Source: {doc.metadata}\n" f"Content: {doc.page_content}"
        for doc in retrieved_docs
    )

    return serialized, retrieved_docs
tools = ToolNode([retrieve]) # transform this tool to a graph node


# Node 3: Generate a response using the retrieved content
def generate(state: MessagesState):
    """Generate answer"""
    # filter generated ToolMessage
    recent_tool_messages = []
    # reverse all messages to make it faster to get the most recent tool messags
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    response = model.invoke(prompt)
    return {"messages": [response]}


graph_builder.add_sequence([
    query_or_respond,
    tools,
    generate
])
graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"}
)
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile()


def rag_two_example():
    print("Press 'q' to end the conversation")
    while True:
        user_input = input("\nYour question: ").strip()
        
        if user_input == 'q':
            print("User ended the conversation")
            break
        
        for step in graph.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            stream_mode="values",
        ):
            step["messages"][-1].pretty_print()