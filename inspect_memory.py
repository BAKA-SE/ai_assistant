# 新建 inspect_memory.py，跑完就可以删掉
import chromadb

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("memory")

# 查看所有记录
results = collection.get()

print(f"共有 {len(results['ids'])} 条记忆\n")

for i, (doc, meta) in enumerate(zip(results['documents'], results['metadatas'])):
    print(f"[{i+1}] {meta['role']} | {meta['timestamp']}")
    print(f"    {doc[:50]}...")  # 只显示前50个字
    print()