import os
import getpass
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from bs4.filter import SoupStrainer
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langgraph.graph.state import StateGraph
from langgraph.graph import START

# types
from langchain_core.documents import Document
from typing import List, TypedDict


OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
if not OPENAI_API_KEY:
    OPENAI_API_KEY = getpass.getpass("require OPENAI_API_KEY: ")

# 初始化
model = init_chat_model(model="gpt-4o-mini", model_provider="openai")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = InMemoryVectorStore(embeddings)

# 1. 索引：
# 1.1 加载文档：从网站上获取数据并加载为一个 document
bs4_strainer = SoupStrainer(
    class_=("post-title", "post-header", "post-content")
)  # 一个过滤器，只会选择网页中带有特定 CSS 类名的元素
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()
# 1.2 分割文档
text_spliter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_spliter.split_documents(docs)
# 1.3 嵌入
embedded_splits = vector_store.add_documents(documents=all_splits)


# 2. 检索
prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.
Context: {context}"""
        ),
        ("human", "{question}"),
    ]
)


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    prompt = prompt_template.invoke(
        {"question": state["question"], "context": docs_content}
    )
    response = model.invoke(prompt)
    return {"answer": response.content}


# graph
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()


def rag_one_example():
    print("This is a RAG system example. Ask please. (input 'q' to quit)")

    while True:
        question = input("question: ").strip()

        if question == "q":
            break

        result = graph.invoke({"question": question})
        print(f"Context: \n{result['context']}\n")
        print(f"Answer: \n{result['answer']}\n")
