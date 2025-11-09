import json
import os

# 导入日志模块
from tradingagents.utils.logging_manager import get_logger
logger = get_logger('agents')



def get_data_in_range(ticker, start_date, end_date, data_type, data_dir, period=None):
    """从本地磁盘加载已经下载并预处理过的Finnhub数据。

    该函数用于读取特定股票在指定日期范围内的本地Finnhub数据文件。
    数据文件应为JSON格式，其中键是日期字符串（'YYYY-MM-DD'），值是
    该日期对应的数据列表。

    Args:
        ticker (str): 股票代码。
        start_date (str): 开始日期，格式为 'YYYY-MM-DD'。
        end_date (str): 结束日期，格式为 'YYYY-MM-DD'。
        data_type (str): 要获取的Finnhub数据类型。
            可选值包括: 'insider_trans', 'SEC_filings', 'news_data',
            'insider_senti', 'fin_as_reported'。
        data_dir (str): 存储数据的根目录。
        period (str, optional): 数据的报告周期，仅在某些数据类型
            （如财务报告）下使用。可选值为 'annual' 或 'quarterly'。
            默认为 None。

    Returns:
        dict: 一个字典，其中键是位于指定日期范围内的日期字符串，值是
            对应的数据。如果文件不存在或发生错误，则返回一个空字典。
    """

    if period:
        data_path = os.path.join(
            data_dir,
            "finnhub_data",
            data_type,
            f"{ticker}_{period}_data_formatted.json",
        )
    else:
        data_path = os.path.join(
            data_dir, "finnhub_data", data_type, f"{ticker}_data_formatted.json"
        )

    try:
        if not os.path.exists(data_path):
            logger.warning(f"⚠️ [DEBUG] 数据文件不存在: {data_path}")
            logger.warning(f"⚠️ [DEBUG] 请确保已下载相关数据或检查数据目录配置")
            return {}
        
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error(f"❌ [ERROR] 文件未找到: {data_path}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"❌ [ERROR] JSON解析错误: {e}")
        return {}
    except Exception as e:
        logger.error(f"❌ [ERROR] 读取数据文件时发生错误: {e}")
        return {}

    # filter keys (date, str in format YYYY-MM-DD) by the date range (str, str in format YYYY-MM-DD)
    filtered_data = {}
    for key, value in data.items():
        if start_date <= key <= end_date and len(value) > 0:
            filtered_data[key] = value
    return filtered_data
