from langchain_core.tools import tool
from datetime import datetime

@tool
def get_current_time() -> str:
    """获取当前的实时时间。每次被问到时间时必须调用此工具，
    不能使用历史记录中的时间，因为时间一直在变化。"""
    now = datetime.now()
    return f"当前时间是 {now.strftime('%Y年%m月%d日 %H:%M')}"