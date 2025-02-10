from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import getpass
import os
# types
from langchain_core.documents import Document
from langchain_openai.embeddings import OpenAIEmbeddings
from typing import List


# 初始化 embedding 和 vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectors_store = InMemoryVectorStore(embeddings)
methods = [
    "1",
    "2",
    "3",
]

def load_document(file_path: str) -> List[Document]:
    print("Step 1/3: 读取文本")
    loader = PyPDFLoader(file_path)
    doc = loader.load()

    return doc


def split_document_text(doc: List[Document]):
    print("Step 2/3: 分割文本")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    splits = text_splitter.split_documents(doc)

    return splits


# def text_embedding(splits: List[Document]):
#     print("Step 3/3: 文本嵌入")

#     api_key = os.environ.get("OPENAI_API_KEY")
#     if not api_key:
#         api_key = getpass.getpass("Enter your API key for OpenAI: ")
    
#     for index, split in enumerate(splits):
#         embeddings.embed_query(split.page_content)
#         print(f"完成第 {index + 1} 个文本嵌入")


def search(query: str, search_method: str) -> None:
    if search_method == methods[0]:
        results = vectors_store.similarity_search(query)
        print(results[0])
    
    elif search_method == methods[1]:
        results = vectors_store.similarity_search_with_score(query)
        doc, score = results[0]
        print(f"Score: {score}\n")
        print(doc)
        
    elif search_method == methods[2]:
        query_embedding = embeddings.embed_query(query)
        results = vectors_store.similarity_search_by_vector(query_embedding)
        print(results[0])
        
    print("\n")
    

def add_documents_into_vector_store(splits: List[Document]):
    print("Step 3/3 嵌入文本并存储")
    vectors_store.add_documents(documents=splits) # add documents 这里会自动帮我们完成嵌入工作


def semantic_search_example():
    print("正在进行查询前的准备，请稍后...")
    
    # 读取文档并分割
    doc = load_document(file_path="./data/sample-file.pdf")
    splits = split_document_text(doc=doc)
    
    # 文本嵌入并存储
    test_splits = splits[:10] # 这里我们不嵌入整个文档
    add_documents_into_vector_store(splits=test_splits)

    # 检索
    print("\n准备完成，请选择你的检索方式：")
    print("[1] similarity_search (默认)")
    print("[2] similarity_search_with_score")
    print("[3] similarity_search_by_vector")
    print("输入 'q' 退出程序")
    
    while True:
        search_method = input("请输入查询方式前的数字来选择您的检索方式（不输入默认使用 similarity_search）：")
        if search_method.lower() == 'q':
            break
        
        if not search_method:  # 如果用户直接回车，使用默认方式
            search_method = "1"
        
        if search_method not in methods:
            print("您的查询方式名称有误，请重试\n")
            continue
        
        query = input("请输入问题：").strip()
        if query.lower() == 'q':
            break
        
        search(query=query, search_method=search_method)
