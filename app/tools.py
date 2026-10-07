"""研究助手的业务工具；Deep Agents 内置文件工具由后续 Backend 配置提供。"""

import os
from datetime import datetime
from zoneinfo import ZoneInfo

from tavily import TavilyClient


def get_current_time() -> dict:
    """读取当前北京时间（Asia/Shanghai），返回日期、时间和星期。

    用户询问今天、现在、最新资料或其他相对日期时，用它确认时间。
    返回的是电脑时钟的当前时间，不是网页发布时间或行情更新时间。
    """
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    weekdays = ("星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日")
    result = {
        "datetime": now.isoformat(timespec="seconds"),
        "date": now.date().isoformat(),
        "weekday": weekdays[now.weekday()],
        "timezone": "Asia/Shanghai",
    }
    return result


def internet_search(query: str) -> dict:
    """搜索研究资料。

    Args:
        query: 要搜索的关键词或问题。

    Returns:
        搜索资料和来源 URL；searched_at 表示查询时间，不是资料发布时间。
    """
    # 每次搜索都重新读取时间，程序跨天运行也不会沿用启动时的日期。
    searched_at = get_current_time()
    client = TavilyClient(
        api_key=os.environ["TAVILY_API_KEY"],
    )

    result = client.search(
        query=query,
        max_results=5,
    )
    result["searched_at"] = searched_at

    return result
