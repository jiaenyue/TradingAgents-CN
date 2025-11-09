"""
Reddit 数据获取工具

该模块提供从预先下载的 Reddit 数据文件中提取热门帖子的功能。
数据源是按类别 (category) 和 subreddit 组织的 JSONL 文件。

主要功能:
- 从指定的类别和日期中提取帖子。
- 支持按关键词查询过滤公司相关新闻。
- 根据帖子的点赞数 (upvotes) 进行排序, 并返回最热门的帖子。

模块依赖:
- re: 用于正则表达式匹配, 以在帖子内容中搜索公司名称。
- json: 用于解析 JSONL 文件中的每一行。
- datetime: 用于处理帖子的发布日期。
"""

import requests
import time
import json
from datetime import datetime, timedelta
from contextlib import contextmanager
from typing import Annotated
import os
import re

# 将股票代码映射到公司名称或相关搜索词。
# 用于在 "company_news" 类别中进行更精确的内容匹配。
# "OR" 用于分隔多个可能的搜索词。
ticker_to_company = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Google",
    "AMZN": "Amazon",
    "TSLA": "Tesla",
    "NVDA": "Nvidia",
    "TSM": "Taiwan Semiconductor Manufacturing Company OR TSMC",
    "JPM": "JPMorgan Chase OR JP Morgan",
    "JNJ": "Johnson & Johnson OR JNJ",
    "V": "Visa",
    "WMT": "Walmart",
    "META": "Meta OR Facebook",
    "AMD": "AMD",
    "INTC": "Intel",
    "QCOM": "Qualcomm",
    "BABA": "Alibaba",
    "ADBE": "Adobe",
    "NFLX": "Netflix",
    "CRM": "Salesforce",
    "PYPL": "PayPal",
    "PLTR": "Palantir",
    "MU": "Micron",
    "SQ": "Block OR Square",
    "ZM": "Zoom",
    "CSCO": "Cisco",
    "SHOP": "Shopify",
    "ORCL": "Oracle",
    "X": "Twitter OR X",
    "SPOT": "Spotify",
    "AVGO": "Broadcom",
    "ASML": "ASML ",
    "TWLO": "Twilio",
    "SNAP": "Snap Inc.",
    "TEAM": "Atlassian",
    "SQSP": "Squarespace",
    "UBER": "Uber",
    "ROKU": "Roku",
    "PINS": "Pinterest",
}


def fetch_top_from_category(
    category: Annotated[
        str, "Category to fetch top post from. Collection of subreddits."
    ],
    date: Annotated[str, "Date to fetch top posts from."],
    max_limit: Annotated[int, "Maximum number of posts to fetch."],
    query: Annotated[str, "Optional query to search for in the subreddit."] = None,
    data_path: Annotated[
        str,
        "Path to the data folder. Default is 'reddit_data'.",
    ] = "reddit_data",
):
    """
    从指定类别和日期的本地数据文件中获取热门 Reddit 帖子。

    该函数会遍历指定类别目录下的所有 subreddit 数据文件 (JSONL 格式),
    筛选出符合日期和查询条件的帖子, 并根据点赞数排序, 最终返回
    一个综合的热门帖子列表。

    Args:
        category (str): 要获取帖子的类别, 对应于 `data_path` 下的一个目录名。
                        该目录包含多个 subreddit 的数据文件。
        date (str): 要筛选的帖子日期, 格式为 "YYYY-MM-DD"。
        max_limit (int): 要返回的最大帖子数量。这个数量会被平均分配到
                         该类别下的每个 subreddit 文件中。
        query (str, optional): 一个查询字符串, 通常是股票代码。如果 `category`
                               包含 "company", 则会使用此查询在帖子标题和
                               内容中搜索相关的公司名称。默认为 None。
        data_path (str, optional): 存放 Reddit 数据的根目录路径。
                                   默认为 "reddit_data"。

    Returns:
        list: 一个包含帖子信息的字典列表。每个字典代表一个帖子, 包含
              'title', 'content', 'url', 'upvotes' 和 'posted_date' 键。

    Raises:
        ValueError: 如果 `max_limit` 小于类别下的文件数量, 导致无法为
                    每个文件分配至少一个帖子的名额, 则会引发此异常。
    """
    base_path = data_path

    all_content = []

    if max_limit < len(os.listdir(os.path.join(base_path, category))):
        raise ValueError(
            "REDDIT FETCHING ERROR: max limit is less than the number of files in the category. Will not be able to fetch any posts"
        )

    limit_per_subreddit = max_limit // len(
        os.listdir(os.path.join(base_path, category))
    )

    for data_file in os.listdir(os.path.join(base_path, category)):
        # check if data_file is a .jsonl file
        if not data_file.endswith(".jsonl"):
            continue

        all_content_curr_subreddit = []

        with open(os.path.join(base_path, category, data_file), "rb") as f:
            for i, line in enumerate(f):
                # skip empty lines
                if not line.strip():
                    continue

                parsed_line = json.loads(line)

                # select only lines that are from the date
                post_date = datetime.utcfromtimestamp(
                    parsed_line["created_utc"]
                ).strftime("%Y-%m-%d")
                if post_date != date:
                    continue

                # if is company_news, check that the title or the content has the company's name (query) mentioned
                if "company" in category and query:
                    search_terms = []
                    if "OR" in ticker_to_company[query]:
                        search_terms = ticker_to_company[query].split(" OR ")
                    else:
                        search_terms = [ticker_to_company[query]]

                    search_terms.append(query)

                    found = False
                    for term in search_terms:
                        if re.search(
                            term, parsed_line["title"], re.IGNORECASE
                        ) or re.search(term, parsed_line["selftext"], re.IGNORECASE):
                            found = True
                            break

                    if not found:
                        continue

                post = {
                    "title": parsed_line["title"],
                    "content": parsed_line["selftext"],
                    "url": parsed_line["url"],
                    "upvotes": parsed_line["ups"],
                    "posted_date": post_date,
                }

                all_content_curr_subreddit.append(post)

        # sort all_content_curr_subreddit by upvote_ratio in descending order
        all_content_curr_subreddit.sort(key=lambda x: x["upvotes"], reverse=True)

        all_content.extend(all_content_curr_subreddit[:limit_per_subreddit])

    return all_content
