#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票数据API接口
提供便捷的股票数据获取接口，支持完整的降级机制
"""

import sys
import os
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

# 导入日志模块
from tradingagents.utils.logging_manager import get_logger
logger = get_logger('agents')

# 添加dataflows目录到路径
dataflows_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataflows')
if dataflows_path not in sys.path:
    sys.path.append(dataflows_path)

# 导入统一日志系统
from tradingagents.utils.logging_init import get_logger

try:
    from stock_data_service import get_stock_data_service

    SERVICE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ 股票数据服务不可用: {e}")
    SERVICE_AVAILABLE = False

def get_stock_info(stock_code: str) -> Dict[str, Any]:
    """获取单个股票的基础信息。

    本函数通过后端的股票数据服务查询指定股票代码的基础信息，
    包括股票名称、所属市场、类别等。如果数据服务不可用或未找到
    对应股票，将返回包含错误信息的字典。

    Args:
        stock_code: 要查询的股票代码，例如 '000001'。

    Returns:
        一个包含股票基础信息的字典。如果成功，字典将包含 'code',
        'name', 'market' 等键。如果失败，则包含 'error' 键及
        错误描述。
    
    Example:
        >>> info = get_stock_info('000001')
        >>> if 'error' not in info:
        ...     print(info.get('name'))
        平安银行
    """
    if not SERVICE_AVAILABLE:
        return {
            'error': '股票数据服务不可用',
            'code': stock_code,
            'suggestion': '请检查服务配置'
        }
    
    service = get_stock_data_service()
    result = service.get_stock_basic_info(stock_code)
    
    if result is None:
        return {
            'error': f'未找到股票{stock_code}的信息',
            'code': stock_code,
            'suggestion': '请检查股票代码是否正确'
        }
    
    return result

def get_all_stocks() -> List[Dict[str, Any]]:
    """获取市场中所有股票的基础信息列表。

    通过后端的股票数据服务获取一个包含所有上市股票基础信息的列表。
    每个股票的信息以字典形式表示。如果服务不可用或无法获取数据，
    将返回一个包含错误信息的单元素列表。

    Returns:
        一个包含多个股票信息字典的列表。如果获取失败，返回一个
        形如 [{'error': '...', 'suggestion': '...'}] 的列表。

    Example:
        >>> stocks = get_all_stocks()
        >>> if 'error' not in stocks[0]:
        ...     print(f"市场共有 {len(stocks)} 只股票。")
    """
    if not SERVICE_AVAILABLE:
        return [{
            'error': '股票数据服务不可用',
            'suggestion': '请检查服务配置'
        }]
    
    service = get_stock_data_service()
    result = service.get_stock_basic_info()
    
    if result is None or (isinstance(result, dict) and 'error' in result):
        return [{
            'error': '无法获取股票列表',
            'suggestion': '请检查网络连接和数据库配置'
        }]
    
    return result if isinstance(result, list) else [result]

def get_stock_data(stock_code: str, start_date: str = None, end_date: str = None) -> str:
    """获取指定股票在特定时间范围内的历史市场数据。

    此函数调用后端服务，该服务内置了降级机制（例如，从Tushare失败
    后尝试Akshare），以确保数据获取的稳定性。返回的数据是经过格式化
    的字符串，通常用于直接展示或传递给大型语言模型进行分析。

    Args:
        stock_code: 要查询的股票代码。
        start_date: 数据查询的开始日期，格式为 'YYYY-MM-DD'。
                      如果为 None，则默认为当前日期之前的30天。
        end_date: 数据查询的结束日期，格式为 'YYYY-MM-DD'。
                    如果为 None，则默认为当前日期。

    Returns:
        包含股票历史数据的格式化字符串。如果数据获取失败，则返回
        相应的错误信息字符串。

    Example:
        >>> data = get_stock_data('000001', '2024-01-01', '2024-01-31')
        >>> print(data[:100])
    """
    if not SERVICE_AVAILABLE:
        return "❌ 股票数据服务不可用，请检查服务配置"
    
    # 设置默认日期
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    service = get_stock_data_service()
    return service.get_stock_data_with_fallback(stock_code, start_date, end_date)

def search_stocks(keyword: str) -> List[Dict[str, Any]]:
    """根据关键词（股票代码或名称）在本地缓存的股票列表中进行搜索。

    此函数首先获取完整的股票列表，然后遍历该列表，匹配所有
    代码或名称中包含指定关键词的股票。搜索过程不区分大小写。

    Args:
        keyword: 用于搜索的关键词，可以是股票代码的一部分或
                 公司名称的一部分。

    Returns:
        一个列表，包含所有匹配搜索条件的股票信息字典。如果
        无法获取股票列表，将返回一个包含错误信息的列表。

    Example:
        >>> results = search_stocks('科技')
        >>> for stock in results:
        ...     print(f"{stock.get('code')}: {stock.get('name')}")
    """
    all_stocks = get_all_stocks()
    
    if not all_stocks or (len(all_stocks) == 1 and 'error' in all_stocks[0]):
        return all_stocks
    
    # 搜索匹配的股票
    matches = []
    keyword_lower = keyword.lower()
    
    for stock in all_stocks:
        if 'error' in stock:
            continue
            
        code = stock.get('code', '').lower()
        name = stock.get('name', '').lower()
        
        if keyword_lower in code or keyword_lower in name:
            matches.append(stock)
    
    return matches

def get_market_summary() -> Dict[str, Any]:
    """获取整个市场的概览统计信息。

    该函数通过分析完整的股票列表，提供关于市场的宏观数据，
    包括总股票数量、沪市和深市的股票数量、按类别划分的统计
    以及数据源和更新时间等信息。

    Returns:
        一个包含市场概览信息的字典。如果无法获取股票列表，
        则返回包含错误信息的字典。

    Example:
        >>> summary = get_market_summary()
        >>> if 'error' not in summary:
        ...     print(f"总股票数: {summary.get('total_count')}")
    """
    all_stocks = get_all_stocks()
    
    if not all_stocks or (len(all_stocks) == 1 and 'error' in all_stocks[0]):
        return {
            'error': '无法获取市场数据',
            'suggestion': '请检查网络连接和数据库配置'
        }
    
    # 统计市场信息
    shanghai_count = 0
    shenzhen_count = 0
    category_stats = {}
    
    for stock in all_stocks:
        if 'error' in stock:
            continue
            
        market = stock.get('market', '')
        category = stock.get('category', '未知')
        
        if market == '上海':
            shanghai_count += 1
        elif market == '深圳':
            shenzhen_count += 1
        
        category_stats[category] = category_stats.get(category, 0) + 1
    
    return {
        'total_count': len([s for s in all_stocks if 'error' not in s]),
        'shanghai_count': shanghai_count,
        'shenzhen_count': shenzhen_count,
        'category_stats': category_stats,
        'data_source': all_stocks[0].get('source', 'unknown') if all_stocks else 'unknown',
        'updated_at': datetime.now().isoformat()
    }

def check_service_status() -> Dict[str, Any]:
    """检查后端数据服务的整体健康状况。

    此函数提供一个全面的健康检查端点，用于监控服务及其依赖项
    （如MongoDB、统一数据接口）的状态。它可以帮助快速诊断系统
    中可能存在的问题。

    Returns:
        一个包含服务状态信息的字典，其中包括服务是否可用、
        数据库连接状态、API可用性等关键指标。

    Example:
        >>> status = check_service_status()
        >>> print(f"服务是否可用: {status.get('service_available')}")
    """
    if not SERVICE_AVAILABLE:
        return {
            'service_available': False,
            'error': '股票数据服务不可用',
            'suggestion': '请检查服务配置和依赖'
        }
    
    service = get_stock_data_service()
    
    # 检查MongoDB状态
    mongodb_status = 'disconnected'
    if service.db_manager:
        try:
            # 尝试检查数据库管理器的连接状态
            if hasattr(service.db_manager, 'is_mongodb_available') and service.db_manager.is_mongodb_available():
                mongodb_status = 'connected'
            elif hasattr(service.db_manager, 'mongodb_client') and service.db_manager.mongodb_client:
                # 尝试执行一个简单的查询来测试连接
                service.db_manager.mongodb_client.admin.command('ping')
                mongodb_status = 'connected'
            else:
                mongodb_status = 'unavailable'
        except Exception:
            mongodb_status = 'error'
    
    # 检查统一数据接口状态
    unified_api_status = 'unavailable'
    try:
        # 尝试获取一个股票信息来测试统一接口
        test_result = service.get_stock_basic_info('000001')
        if test_result and 'error' not in test_result:
            unified_api_status = 'available'
        else:
            unified_api_status = 'limited'
    except Exception:
        unified_api_status = 'error'
    
    return {
        'service_available': True,
        'mongodb_status': mongodb_status,
        'unified_api_status': unified_api_status,
        'data_sources_available': ['tushare', 'akshare', 'baostock'],
        'fallback_available': True,
        'checked_at': datetime.now().isoformat()
    }

# 便捷的别名函数
get_stock = get_stock_info  # 别名
get_stocks = get_all_stocks  # 别名
search = search_stocks  # 别名
status = check_service_status  # 别名

if __name__ == '__main__':
    # 简单的命令行测试
    logger.debug(f"🔍 股票数据API测试")
    logger.info(f"=" * 50)
    
    # 检查服务状态
    logger.info(f"\n📊 服务状态检查:")
    status_info = check_service_status()
    for key, value in status_info.items():
        logger.info(f"  {key}: {value}")
    
    # 测试获取单个股票信息
    logger.info(f"\n🏢 获取平安银行信息:")
    stock_info = get_stock_info('000001')
    if 'error' not in stock_info:
        logger.info(f"  代码: {stock_info.get('code')}")
        logger.info(f"  名称: {stock_info.get('name')}")
        logger.info(f"  市场: {stock_info.get('market')}")
        logger.info(f"  类别: {stock_info.get('category')}")
        logger.info(f"  数据源: {stock_info.get('source')}")
    else:
        logger.error(f"  错误: {stock_info.get('error')}")
    
    # 测试搜索功能
    logger.debug(f"\n🔍 搜索'平安'相关股票:")
    search_results = search_stocks('平安')
    for i, stock in enumerate(search_results[:3]):  # 只显示前3个结果
        if 'error' not in stock:
            logger.info(f"  {i+1}. {stock.get('code')}")

    # 测试市场概览
    logger.info(f"\n📈 市场概览:")
    summary = get_market_summary()
    if 'error' not in summary:
        logger.info(f"  总股票数: {summary.get('total_count')}")
        logger.info(f"  沪市股票: {summary.get('shanghai_count')}")
        logger.info(f"  深市股票: {summary.get('shenzhen_count')}")
        logger.info(f"  数据源: {summary.get('data_source')}")
    else:
        logger.error(f"  错误: {summary.get('error')}")