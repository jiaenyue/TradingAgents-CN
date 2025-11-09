import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time
import random
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    retry_if_result,
)

# 导入日志模块
from tradingagents.utils.logging_manager import get_logger
logger = get_logger('agents')


def is_rate_limited(response):
    """检查HTTP响应是否表示受到了速率限制。

    Args:
        response: requests的响应对象。

    Returns:
        bool: 如果响应状态码为 429 (Too Many Requests)，则返回 True，
            否则返回 False。
    """
    return response.status_code == 429


@retry(
    retry=(retry_if_result(is_rate_limited) | retry_if_exception_type(requests.exceptions.ConnectionError) | retry_if_exception_type(requests.exceptions.Timeout)),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
)
def make_request(url, headers):
    """发起一个带有健壮重试逻辑的HTTP GET请求。

    该函数使用 `tenacity` 库来自动处理以下情况：
    - 速率限制 (HTTP 429)。
    - 连接错误 (`requests.exceptions.ConnectionError`)。
    - 超时 (`requests.exceptions.Timeout`)。

    重试策略采用指数退避，最多尝试5次。同时，在每次请求前会加入一个
    随机延迟，以模拟更自然的用户行为，减少被服务器检测到的风险。

    Args:
        url (str): 要请求的URL。
        headers (dict): 请求头。

    Returns:
        requests.Response: HTTP响应对象。
    """
    # Random delay before each request to avoid detection
    time.sleep(random.uniform(2, 6))
    # 添加超时参数，设置连接超时和读取超时
    response = requests.get(url, headers=headers, timeout=(10, 30))  # 连接超时10秒，读取超时30秒
    return response


def getNewsData(query, start_date, end_date):
    """从Google News抓取指定查询和日期范围的新闻数据。

    该函数会模拟浏览器行为，通过分页方式抓取Google News搜索结果页面，
    并使用BeautifulSoup解析HTML，提取每条新闻的标题、链接、摘要、
    日期和来源。

    Args:
        query (str): 搜索关键词。
        start_date (str): 开始日期，支持 'yyyy-mm-dd' 或 'mm/dd/yyyy' 格式。
        end_date (str): 结束日期，支持 'yyyy-mm-dd' 或 'mm/dd/yyyy' 格式。

    Returns:
        list: 一个包含新闻数据的字典列表，每个字典代表一条新闻。
    """
    if "-" in start_date:
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        start_date = start_date.strftime("%m/%d/%Y")
    if "-" in end_date:
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        end_date = end_date.strftime("%m/%d/%Y")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/101.0.4951.54 Safari/537.36"
        )
    }

    news_results = []
    page = 0
    while True:
        offset = page * 10
        url = (
            f"https://www.google.com/search?q={query}"
            f"&tbs=cdr:1,cd_min:{start_date},cd_max:{end_date}"
            f"&tbm=nws&start={offset}"
        )

        try:
            response = make_request(url, headers)
            soup = BeautifulSoup(response.content, "html.parser")
            results_on_page = soup.select("div.SoaBEf")

            if not results_on_page:
                break  # No more results found

            for el in results_on_page:
                try:
                    link = el.find("a")["href"]
                    title = el.select_one("div.MBeuO").get_text()
                    snippet = el.select_one(".GI74Re").get_text()
                    date = el.select_one(".LfVVr").get_text()
                    source = el.select_one(".NUnG9d span").get_text()
                    news_results.append(
                        {
                            "link": link,
                            "title": title,
                            "snippet": snippet,
                            "date": date,
                            "source": source,
                        }
                    )
                except Exception as e:
                    logger.error(f"Error processing result: {e}")
                    # If one of the fields is not found, skip this result
                    continue

            # Update the progress bar with the current count of results scraped

            # Check for the "Next" link (pagination)
            next_link = soup.find("a", id="pnnext")
            if not next_link:
                break

            page += 1

        except requests.exceptions.Timeout as e:
            logger.error(f"连接超时: {e}")
            # 不立即中断，记录错误后继续尝试下一页
            page += 1
            if page > 3:  # 如果连续多页都超时，则退出循环
                logger.error("多次连接超时，停止获取Google新闻")
                break
            continue
        except requests.exceptions.ConnectionError as e:
            logger.error(f"连接错误: {e}")
            # 不立即中断，记录错误后继续尝试下一页
            page += 1
            if page > 3:  # 如果连续多页都连接错误，则退出循环
                logger.error("多次连接错误，停止获取Google新闻")
                break
            continue
        except Exception as e:
            logger.error(f"获取Google新闻失败: {e}")
            break

    return news_results
