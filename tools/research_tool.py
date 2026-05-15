from langchain_core.tools import tool
from ddgs import DDGS
from datetime import datetime, timedelta


@tool
def search_research(topic: str, days: int = 7, max_results: int = 5) -> str:
    """搜索指定计算机方向的最新前沿进展，包括新模型、新论文、新架构。
    topic 是研究方向，比如 LLM、多模态、Agent、图神经网络。
    days 是往前查几天，默认7天。
    max_results 是返回几条结果，默认5条。
    当用户询问某个技术方向的最新进展、新论文、新模型时使用，把摘要返回为中文格式。"""

    # 构造搜索关键词，加上时间限制
    since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    query = f"{topic} research paper OR new model OR breakthrough {since}"

    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(
                f"标题：{r['title']}\n"
                f"摘要：{r['body']}\n"
                f"链接：{r['href']}\n"
            )

    if not results:
        return f"未找到关于 {topic} 的最新进展"

    return f"找到 {len(results)} 条关于 {topic} 的最新进展：\n\n" + "\n---\n".join(results)