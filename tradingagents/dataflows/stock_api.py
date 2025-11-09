#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票数据高级API接口

该模块提供了一组简单易用的高级函数, 用于访问 `StockDataService` 提供的
核心功能, 如获取股票基础信息、历史行情数据等。

它封装了服务实例化和方法调用的细节, 并内置了完整的降级机制, 确保在
不同环境配置下 (例如, 是否有MongoDB支持) 都能提供稳健的数据服务。
"""

from typing import Dict, List, Optional, Any
from .stock_data_service import get_stock_data_service

# 导入日志模块
from tradingagents.utils.logging_manager import get_logger
logger = get_logger('agents')

def get_stock_info(stock_code: str) -> Optional[Dict[str, Any]]:
    """
    获取单个股票的基础信息。

    通过底层的 `StockDataService` 来查询指定股票代码的基本资料。

    Args:
        stock_code (str): 股票代码 (例如 '000001')。
    
    Returns:
        Optional[Dict[str, Any]]: 包含股票信息的字典, 其中包括 'code',
                                  'name', 'market', 'category' 等字段。
                                  如果获取失败, 则返回一个包含 'error'
                                  字段的字典或 None。
    
    Example:
        >>> info = get_stock_info('000001')
        >>> if info and 'error' not in info:
        ...     print(info.get('name'))
        平安银行
    """
    service = get_stock_data_service()
    return service.get_stock_basic_info(stock_code)

def get_all_stocks() -> List[Dict[str, Any]]:
    """
    获取市场上所有股票的基础信息列表。

    Returns:
        List[Dict[str, Any]]: 一个包含多个股票信息字典的列表。如果获取失败,
                              列表的第一个元素将是一个包含 'error' 字段的字典。
    
    Example:
        >>> stocks = get_all_stocks()
        >>> if stocks and 'error' not in stocks[0]:
        ...     logger.info(f"共获取到 {len(stocks)} 只股票")
    """
    service = get_stock_data_service()
    result = service.get_stock_basic_info()
    
    if isinstance(result, list):
        return result
    elif isinstance(result, dict) and 'error' in result:
        return [result]  # 将错误信息包装在列表中返回
    else:
        return []

def get_stock_data(stock_code: str, start_date: str, end_date: str) -> str:
    """
    获取指定股票在特定时间范围内的历史行情数据。

    该函数调用了带有降级机制的底层服务, 确保在主数据源不可用时,
    能自动切换到备用数据源, 尽可能保证数据的可用性。

    Args:
        stock_code (str): 股票代码。
        start_date (str): 开始日期, 格式为 'YYYY-MM-DD'。
        end_date (str): 结束日期, 格式为 'YYYY-MM-DD'。
    
    Returns:
        str: 经过格式化的、人类可读的股票数据分析报告字符串。
    
    Example:
        >>> data_report = get_stock_data('000001', '2024-01-01', '2024-01-31')
        >>> print(data_report)
    """
    service = get_stock_data_service()
    return service.get_stock_data_with_fallback(stock_code, start_date, end_date)

def search_stocks_by_name(name: str) -> List[Dict[str, Any]]:
    """
    根据股票名称或关键词进行模糊搜索。

    注意: 此功能依赖于 MongoDB 的配置。如果未配置 MongoDB,
    该函数将返回一个错误信息。

    Args:
        name (str): 用于搜索的股票名称关键词。
    
    Returns:
        List[Dict[str, Any]]: 匹配到的股票列表, 每个元素是一个包含股票
                              基础信息的字典。如果功能不可用, 返回包含
                              'error' 信息的列表。
    
    Example:
        >>> results = search_stocks_by_name('银行')
        >>> if results and 'error' not in results[0]:
        ...     for stock in results:
        ...         logger.info(f"{stock['code']}: {stock['name']}")
    """
    # 这个功能需要MongoDB支持，暂时通过原有方式实现
    try:
        from ..examples.stock_query_examples import EnhancedStockQueryService

        service = EnhancedStockQueryService()
        return service.query_stocks_by_name(name)
    except Exception as e:
        return [{'error': f'名称搜索功能不可用: {str(e)}'}]

def check_data_sources() -> Dict[str, Any]:
    """
    检查当前系统配置下各个数据源的可用状态。

    这有助于诊断数据获取问题, 并了解系统当前是在最佳性能模式还是在
    降级模式下运行。

    Returns:
        Dict[str, Any]: 一个包含各数据源状态信息的字典, 包括:
                        - 'mongodb_available': MongoDB 是否可用 (bool)。
                        - 'unified_api_available': 统一数据接口是否可用 (bool)。
                        - 'fallback_mode': 是否处于降级模式 (bool)。
                        - 'recommendation': 对当前配置的建议 (str)。
    
    Example:
        >>> status = check_data_sources()
        >>> logger.info(f"MongoDB可用: {status['mongodb_available']}")
        >>> logger.info(f"推荐配置: {status['recommendation']}")
    """
    service = get_stock_data_service()
    
    return {
        'mongodb_available': service.db_manager is not None and service.db_manager.mongodb_db is not None,
        'unified_api_available': True,  # 统一接口总是可用
        'enhanced_fetcher_available': True,  # 这个通常都可用
        'fallback_mode': service.db_manager is None or service.db_manager.mongodb_db is None,
        'recommendation': (
            "所有数据源正常" if service.db_manager and service.db_manager.mongodb_db 
            else "建议配置MongoDB以获得最佳性能，当前使用统一数据接口降级模式"
        )
    }