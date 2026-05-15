# 新建一个 clean_memory.py，跑一次就删掉
import chromadb

chroma_client = chromadb.PersistentClient(path="./chroma_db")

try:
    chroma_client.delete_collection("memory")
    print("memory 集合已清空")
except:
    print("集合不存在，无需清理")