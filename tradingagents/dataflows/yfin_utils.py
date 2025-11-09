"""
Yahoo Finance 数据工具

该模块提供了一个 `YFinanceUtils` 类, 封装了 `yfinance` 库的常用功能,
用于获取股票的各种数据, 包括历史行情、公司信息、财务报表和分析师建议等。

通过使用装饰器, 该模块简化了 `yf.Ticker` 对象的初始化过程, 使得所有
方法都可以直接接收股票代码作为参数。
"""

import yfinance as yf
from typing import Annotated, Callable, Any, Optional
from pandas import DataFrame
import pandas as pd
from functools import wraps

from .utils import save_output, SavePathType, decorate_all_methods

# 导入日志模块
from tradingagents.utils.logging_manager import get_logger
logger = get_logger('agents')

# 导入缓存管理器
try:
    from .cache_manager import get_cache

    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    logger.warning(f"⚠️ 缓存管理器不可用，将直接从API获取数据")


def init_ticker(func: Callable) -> Callable:
    """
    一个装饰器, 用于自动初始化 yf.Ticker 对象并将其作为第一个参数
    传递给被装饰的函数。

    这样, 类中的方法在定义时可以省略 `yf.Ticker(symbol)` 的重复代码。

    Args:
        func (Callable): 需要被装饰的函数。该函数的第一个参数应为股票代码 `symbol`。

    Returns:
        Callable: 装饰后的新函数。
    """

    @wraps(func)
    def wrapper(symbol: Annotated[str, "ticker symbol"], *args, **kwargs) -> Any:
        ticker = yf.Ticker(symbol)
        return func(ticker, *args, **kwargs)

    return wrapper


@decorate_all_methods(init_ticker)
class YFinanceUtils:
    """
    一个工具类, 包含一系列静态方法, 用于从 Yahoo Finance 获取股票数据。
    所有方法都被 `init_ticker` 装饰器处理, 因此可以直接传入股票代码字符串。
    """

    def get_stock_data(
        symbol: Annotated[str, "ticker symbol"],
        start_date: Annotated[
            str, "start date for retrieving stock price data, YYYY-mm-dd"
        ],
        end_date: Annotated[
            str, "end date for retrieving stock price data, YYYY-mm-dd"
        ],
        save_path: SavePathType = None,
    ) -> DataFrame:
        """
        检索指定股票代码在给定日期范围内的历史行情数据。

        Args:
            symbol (str): 公司的股票代码 (ticker symbol)。
            start_date (str): 数据检索的开始日期, 格式为 "YYYY-MM-DD"。
            end_date (str): 数据检索的结束日期, 格式为 "YYYY-MM-DD"。
            save_path (SavePathType, optional): 如果提供路径, 会将数据保存为 CSV 文件。
                                                默认为 None。

        Returns:
            DataFrame: 包含历史行情数据的 Pandas DataFrame。
        """
        ticker = symbol
        # 将结束日期增加一天, 以确保数据范围是闭合的
        end_date = pd.to_datetime(end_date) + pd.DateOffset(days=1)
        end_date = end_date.strftime("%Y-%m-%d")
        stock_data = ticker.history(start=start_date, end=end_date)
        # save_output(stock_data, f"Stock data for {ticker.ticker}", save_path)
        return stock_data

    def get_stock_info(
        symbol: Annotated[str, "ticker symbol"],
    ) -> dict:
        """
        获取并返回最新的股票摘要信息。

        Args:
            symbol (str): 公司的股票代码。

        Returns:
            dict: 包含股票各种信息的字典, 如市值、行业、市盈率等。
        """
        ticker = symbol
        stock_info = ticker.info
        return stock_info

    def get_company_info(
        symbol: Annotated[str, "ticker symbol"],
        save_path: Optional[str] = None,
    ) -> DataFrame:
        """
        获取并以 DataFrame 格式返回公司的基本信息。

        Args:
            symbol (str): 公司的股票代码。
            save_path (Optional[str], optional): 如果提供路径, 会将数据保存为 CSV 文件。
                                                 默认为 None。

        Returns:
            DataFrame: 包含公司名称、行业、国家等信息的 DataFrame。
        """
        ticker = symbol
        info = ticker.info
        company_info = {
            "Company Name": info.get("shortName", "N/A"),
            "Industry": info.get("industry", "N/A"),
            "Sector": info.get("sector", "N/A"),
            "Country": info.get("country", "N/A"),
            "Website": info.get("website", "N/A"),
        }
        company_info_df = DataFrame([company_info])
        if save_path:
            company_info_df.to_csv(save_path)
            logger.info(f"Company info for {ticker.ticker} saved to {save_path}")
        return company_info_df

    def get_stock_dividends(
        symbol: Annotated[str, "ticker symbol"],
        save_path: Optional[str] = None,
    ) -> DataFrame:
        """
        获取并返回最新的股息数据。

        Args:
            symbol (str): 公司的股票代码。
            save_path (Optional[str], optional): 如果提供路径, 会将数据保存为 CSV 文件。
                                                 默认为 None。

        Returns:
            DataFrame: 包含历史股息派发记录的 DataFrame。
        """
        ticker = symbol
        dividends = ticker.dividends
        if save_path:
            dividends.to_csv(save_path)
            logger.info(f"Dividends for {ticker.ticker} saved to {save_path}")
        return dividends

    def get_income_stmt(symbol: Annotated[str, "ticker symbol"]) -> DataFrame:
        """
        获取并返回公司最新的损益表 (Income Statement)。

        Args:
            symbol (str): 公司的股票代码。

        Returns:
            DataFrame: 包含损益表数据的 DataFrame。
        """
        ticker = symbol
        income_stmt = ticker.financials
        return income_stmt

    def get_balance_sheet(symbol: Annotated[str, "ticker symbol"]) -> DataFrame:
        """
        获取并返回公司最新的资产负债表 (Balance Sheet)。

        Args:
            symbol (str): 公司的股票代码。

        Returns:
            DataFrame: 包含资产负债表数据的 DataFrame。
        """
        ticker = symbol
        balance_sheet = ticker.balance_sheet
        return balance_sheet

    def get_cash_flow(symbol: Annotated[str, "ticker symbol"]) -> DataFrame:
        """
        获取并返回公司最新的现金流量表 (Cash Flow Statement)。

        Args:
            symbol (str): 公司的股票代码。

        Returns:
            DataFrame: 包含现金流量表数据的 DataFrame。
        """
        ticker = symbol
        cash_flow = ticker.cashflow
        return cash_flow

    def get_analyst_recommendations(symbol: Annotated[str, "ticker symbol"]) -> tuple:
        """
        获取最新的分析师评级, 并返回最常见的评级及其计数。

        Args:
            symbol (str): 公司的股票代码。

        Returns:
            tuple: 一个元组, 第一个元素是最常见的评级 (str), 第二个元素是
                   该评级的计数 (int)。如果没有评级数据, 返回 (None, 0)。
        """
        ticker = symbol
        recommendations = ticker.recommendations
        if recommendations.empty:
            return None, 0  # No recommendations available

        # 假设 'period' 列存在且需要被排除
        row_0 = recommendations.iloc[0, 1:]  # Exclude 'period' column if necessary

        # 找到票数最多的结果
        max_votes = row_0.max()
        majority_voting_result = row_0[row_0 == max_votes].index.tolist()

        return majority_voting_result[0], max_votes
