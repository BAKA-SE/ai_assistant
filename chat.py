from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from retriever import retrieve
from memory import get_recent, save_message, save_memory_vector, retrieve_memory
from dotenv import load_dotenv
import os

load_dotenv()

# ChatModel
model = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    model="deepseek-chat"
)

# Prompt Template
template = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{user_input}")
])

# Chain
chain = template | model

SYSTEM_PROMPT = """你叫小雨，是用户的私人助手。
你性格温柔、有耐心，说话自然不做作。
你会记住用户在对话中告诉你的事情，并在之后自然地体现出来。

当用户提出需要分析或决策的问题时，你会按以下步骤思考：
1. 先理解用户的真实处境和诉求
2. 列出需要考虑的关键因素
3. 逐一分析每个因素
4. 最后给出有依据的建议"""

def build_history():
    """把 memory.py 里的历史转成 LangChain 格式"""
    recent = get_recent(20)
    messages = []
    for m in recent:
        if m["role"] == "user":
            messages.append(HumanMessage(content=m["content"]))
        else:
            messages.append(AIMessage(content=m["content"]))
    return messages

def chat(user_input):
    # 检索知识和记忆
    relevant_chunks = retrieve(user_input)
    context = "\n\n".join([f"来源：{c['source']}\n{c['text']}" for c in relevant_chunks])
    past_memories = retrieve_memory(user_input)
    memory_text = "\n".join([f"{m['role']}：{m['content']}" for m in past_memories])

    system_with_context = (
        SYSTEM_PROMPT
        + f"\n\n以下是相关知识：\n{context}"
        + f"\n\n以下是过去的相关对话：\n{memory_text}"
    )

    # 用 chain 调用，不再手动拼装 messages
    response = chain.invoke({
        "system_prompt": system_with_context,
        "history": build_history(),
        "user_input": user_input
    })

    ai_reply = response.content

    # 存记忆
    timestamp_user = save_message("user", user_input)
    save_memory_vector("user", user_input, timestamp_user)
    timestamp_ai = save_message("assistant", ai_reply)
    save_memory_vector("assistant", ai_reply, timestamp_ai)

    return ai_reply

def main():
    print("小雨：你好～有什么想聊的吗？")
    while True:
        user_input = input("你：").strip()
        if not user_input:
            continue
        if user_input in ("退出", "quit", "exit"):
            print("小雨：拜拜～")
            break
        reply = chat(user_input)
        print(f"小雨：{reply}\n")

if __name__ == "__main__":
    main()