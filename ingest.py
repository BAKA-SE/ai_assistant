import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("SILICON_API_KEY"),
    base_url="https://api.siliconflow.cn/v1"
)

def load_documents(docs_dir):
    documents = []
    for filename in os.listdir(docs_dir):
        if filename.endswith("txt"):
            filepath = os.path.join(docs_dir,filename)
            with open(filepath,"r",encoding="utf-8") as f:
                text = f.read()
                documents.append({
                    "filename":filename,
                    "text":text
                })
    return documents

def split_into_chunks(text,chunk_size=500,overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return  chunks


def get_embeddings_batch(texts):
    response = client.embeddings.create(
        model="BAAI/bge-large-zh-v1.5",
        input=texts
    )
    return [item.embedding for item in response.data]


def build_knowledge_base(docs_dir):
    chroma_client = chromadb.PersistentClient(path="./chroma_db")

    try:
        chroma_client.delete_collection("knowledge")
    except:
        pass
    collection = chroma_client.create_collection("knowledge")

    documents = load_documents(docs_dir)

    all_chunks = []
    all_metadata = []

    for doc in documents:
        chunks = split_into_chunks(doc["text"])
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadata.append({"filename": doc["filename"]})

    # 批量向量化
    embeddings = get_embeddings_batch(all_chunks)

    # 批量存入数据库
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(all_chunks))],
        embeddings=embeddings,
        documents=all_chunks,
        metadatas=all_metadata
    )

    print(f"知识库构建完成，共 {len(all_chunks)} 个块")


if __name__ == "__main__":
    build_knowledge_base("./docs")