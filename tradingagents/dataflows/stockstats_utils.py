"""
股票技术指标计算工具

该模块提供了一个 `StockstatsUtils` 类, 用于利用 `stockstats` 库
计算股票的各种技术分析指标。

它支持两种数据加载模式:
- **离线模式**: 从本地预存的 CSV 文件加载历史数据。
- **在线模式**: 通过 `yfinance` 实时下载最新的历史数据, 并进行缓存。
"""

import pandas as pd
import yfinance as yf
from stockstats import wrap
from typing import Annotated
import os
from .config import get_config


class StockstatsUtils:
    """
    一个工具类, 封装了使用 stockstats 库计算股票技术指标的静态方法。
    """
    @staticmethod
    def get_stock_stats(
        symbol: Annotated[str, "ticker symbol for the company"],
        indicator: Annotated[
            str, "quantitative indicators based off of the stock data for the company"
        ],
        curr_date: Annotated[
            str, "curr date for retrieving stock price data, YYYY-mm-dd"
        ],
        data_dir: Annotated[
            str,
            "directory where the stock data is stored.",
        ],
        online: Annotated[
            bool,
            "whether to use online tools to fetch data or offline tools. If True, will use online tools.",
        ] = False,
    ):
        """
        为指定的股票、在特定日期计算并返回一个技术指标的值。

        该方法可以工作在两种模式下:
        - 离线 (`online=False`): 从 `data_dir` 指定的目录中读取预先下载好的
          CSV 文件。文件名应遵循特定格式。
        - 在线 (`online=True`): 使用 `yfinance` 下载长达15年的历史数据,
          并将其缓存在由 `config` 指定的目录中以备后用。

        计算完成后, 它会查找指定日期 (`curr_date`) 的指标值并返回。

        Args:
            symbol (str): 公司的股票代码 (ticker symbol)。
            indicator (str): 要计算的技术指标名称 (例如, 'rsi_14', 'macd')。
                             这是 `stockstats` 库支持的指标。
            curr_date (str): 要检索指标数据的目标日期, 格式为 "YYYY-MM-DD"。
            data_dir (str): 在离线模式下, 存放股票数据CSV文件的目录。
            online (bool, optional): 是否启用在线模式。如果为 True, 将会
                                     通过网络下载数据; 否则, 尝试从本地
                                     文件读取。默认为 False。

        Returns:
            float or str: 如果找到指定日期的指标, 返回其浮点数值。
                          如果当天不是交易日 (例如周末或假日), 则返回
                          字符串 "N/A: Not a trading day (weekend or holiday)"。

        Raises:
            Exception: 在离线模式下, 如果找不到所需的CSV数据文件, 则会
                       引发此异常。
        """
        df = None
        data = None

        if not online:
            try:
                data = pd.read_csv(
                    os.path.join(
                        data_dir,
                        f"{symbol}-YFin-data-2015-01-01-2025-03-25.csv",
                    )
                )
                df = wrap(data)
            except FileNotFoundError:
                raise Exception("Stockstats fail: Yahoo Finance data not fetched yet!")
        else:
            # Get today's date as YYYY-mm-dd to add to cache
            today_date = pd.Timestamp.today()
            curr_date = pd.to_datetime(curr_date)

            end_date = today_date
            start_date = today_date - pd.DateOffset(years=15)
            start_date = start_date.strftime("%Y-%m-%d")
            end_date = end_date.strftime("%Y-%m-%d")

            # Get config and ensure cache directory exists
            config = get_config()
            os.makedirs(config["data_cache_dir"], exist_ok=True)

            data_file = os.path.join(
                config["data_cache_dir"],
                f"{symbol}-YFin-data-{start_date}-{end_date}.csv",
            )

            if os.path.exists(data_file):
                data = pd.read_csv(data_file)
                data["Date"] = pd.to_datetime(data["Date"])
            else:
                data = yf.download(
                    symbol,
                    start=start_date,
                    end=end_date,
                    multi_level_index=False,
                    progress=False,
                    auto_adjust=True,
                )
                data = data.reset_index()
                data.to_csv(data_file, index=False)

            df = wrap(data)
            df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
            curr_date = curr_date.strftime("%Y-%m-%d")

        df[indicator]  # trigger stockstats to calculate the indicator
        matching_rows = df[df["Date"].str.startswith(curr_date)]

        if not matching_rows.empty:
            indicator_value = matching_rows[indicator].values[0]
            return indicator_value
        else:
            return "N/A: Not a trading day (weekend or holiday)"
