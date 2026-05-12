import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("SILICON_API_KEY"),
    base_url="https://api.siliconflow.cn/v1"
)

# 打开已有数据库，注意是 PersistentClient，路径要和 ingest.py 一致
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("knowledge")

def retrieve(query, top_k=3):
    # 第一步：问题向量化
    response = client.embeddings.create(
        model="BAAI/bge-large-zh-v1.5",
        input=query
    )
    query_embedding = response.data[0].embedding

    # 第二步：检索最相近的 top_k 个块
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    # 第三步：整理结果，返回文本列表
    chunks = results["documents"][0]  # 检索到的原文
    sources = results["metadatas"][0]  # 来源文件名

    return [
        {"text": chunk, "source": meta["filename"]}
        for chunk, meta in zip(chunks, sources)
    ]