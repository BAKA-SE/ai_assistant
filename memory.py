import json
import os
from datetime import datetime
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

HISTORY_FILE = "history.json"

client = OpenAI(
    api_key=os.getenv("SILICON_API_KEY"),
    base_url="https://api.siliconflow.cn/v1"
)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
def load_history():
    """从文件中加载历史"""
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:  # 文件存在但是空的
            return []
        return json.loads(content)

def save_message(role, content):
    timestamp = datetime.now().isoformat()
    history = load_history()
    history.append({
        "role": role,
        "content": content,
        "timestamp": timestamp
    })
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    return timestamp

def get_recent(n=20):
    """取最近n条，去掉timestamp，格式符合API要求"""
    history = load_history()
    recent = history[-n:]
    return[{"role":m["role"],"content":m["content"]} for m in recent]

def _get_or_create_collection():
    try:
        return chroma_client.get_collection("memory")
    except:
        return chroma_client.create_collection("memory")

def save_memory_vector(role,content,timestamp):
    collection = _get_or_create_collection()
    response = client.embeddings.create(
        model="BAAI/bge-large-zh-v1.5",
        input=content
    )
    embedding = response.data[0].embedding
    collection.add(
        ids=[timestamp],
        embeddings=[embedding],
        documents=[content],
        metadatas=[{"role": role, "timestamp": timestamp}]
    )

def retrieve_memory(query, top_k=3):
    collection = _get_or_create_collection()
    response = client.embeddings.create(
        model="BAAI/bge-large-zh-v1.5",
        input=query
    )
    query_embedding = response.data[0].embedding
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    chunks = results["documents"][0]
    metas = results["metadatas"][0]
    return [
        {"role": meta["role"], "content": chunk, "timestamp": meta["timestamp"]}
        for chunk, meta in zip(chunks, metas)
    ]