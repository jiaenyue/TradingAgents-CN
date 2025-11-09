"""
数据流通用工具函数

该模块包含一系列在数据流处理过程中可能用到的辅助函数,
例如保存数据、获取当前日期以及处理工作日等。
"""

import os
import json
import pandas as pd
from datetime import date, timedelta, datetime
from typing import Annotated

# 导入日志模块
from tradingagents.utils.logging_manager import get_logger
logger = get_logger('agents')


SavePathType = Annotated[str, "File path to save data. If None, data is not saved."]

def save_output(data: pd.DataFrame, tag: str, save_path: SavePathType = None) -> None:
    """
    将 DataFrame 保存到 CSV 文件。

    如果提供了 `save_path`, 此函数会将给定的 DataFrame 写入该路径,
    并记录一条日志信息。

    Args:
        data (pd.DataFrame): 需要保存的 Pandas DataFrame。
        tag (str): 用于日志记录的标签, 以便识别是哪个部分的数据被保存。
        save_path (SavePathType, optional): 目标文件的完整路径。如果为 None,
                                           则不执行任何操作。默认为 None。
    """
    if save_path:
        data.to_csv(save_path)
        logger.info(f"{tag} saved to {save_path}")


def get_current_date() -> str:
    """
    获取当前日期并格式化为 "YYYY-MM-DD" 字符串。

    Returns:
        str: 格式化后的当前日期字符串。
    """
    return date.today().strftime("%Y-%m-%d")


def decorate_all_methods(decorator):
    """
    一个类装饰器工厂, 用于将一个装饰器应用于一个类的所有可调用方法。

    Args:
        decorator: 要应用于类中所有方法的装饰器函数。

    Returns:
        function: 一个类装饰器, 当应用于一个类时, 会将其所有方法替换为
                  被 `decorator` 装饰后的版本。
    """
    def class_decorator(cls):
        for attr_name, attr_value in cls.__dict__.items():
            if callable(attr_value):
                setattr(cls, attr_name, decorator(attr_value))
        return cls

    return class_decorator


def get_next_weekday(date_input: Union[str, datetime]) -> datetime:
    """
    计算并返回下一个工作日。

    如果输入的日期本身就是一个工作日 (周一至周五), 则直接返回该日期。
    如果输入的是周末 (周六或周日), 则返回接下来的周一。

    Args:
        date_input (Union[str, datetime]): 输入的日期。可以是一个 `datetime` 对象,
                                          或一个 "YYYY-MM-DD" 格式的字符串。

    Returns:
        datetime: 代表下一个工作日的 `datetime` 对象。
    """

    if not isinstance(date_input, datetime):
        date_obj = datetime.strptime(date_input, "%Y-%m-%d")
    else:
        date_obj = date_input

    if date_obj.weekday() >= 5:  # 5是周六, 6是周日
        days_to_add = 7 - date_obj.weekday()
        next_weekday = date_obj + timedelta(days=days_to_add)
        return next_weekday
    else:
        return date_obj
