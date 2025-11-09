#!/usr/bin/env python3
"""
股票数据缓存管理器
支持本地缓存股票数据，减少API调用，提高响应速度
"""

import os
import json
import pickle
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
import hashlib

# 导入日志模块
from tradingagents.utils.logging_manager import get_logger
logger = get_logger('agents')


class StockDataCache:
    """股票数据文件缓存管理器。

    该类负责将股票相关的各类数据（历史行情、新闻、基本面分析）高效地
    存储在本地文件系统中，以减少对外部API的重复调用。它具有以下特性：

    - **市场分类存储**: 自动根据股票代码区分A股和美股，并将数据存放在
      不同的子目录中，便于管理。
    - **智能TTL（生存时间）**: 为不同类型和市场的数据配置了不同的默认
      缓存有效期，例如A股数据更新更频繁，缓存时间更短。
    - **元数据驱动**: 每个缓存文件都关联一个JSON格式的元数据文件，记录
      了数据的来源、时间戳、参数等信息，用于精确查找和验证缓存。
    - **内容长度检查**: 可选功能，用于防止因内容过长（例如超长的新闻或
      分析报告）而缓存过多数据，特别是当没有配置长文本处理的大语言模型
      (LLM)时，可以自动跳过缓存。
    - **格式支持**: 支持将`pandas.DataFrame`保存为CSV，或将普通字符串
      保存为TXT文件。
    """

    def __init__(self, cache_dir: str = None):
        """初始化StockDataCache。

        Args:
            cache_dir (str, optional): 缓存文件的根目录。如果为None，
                则默认为 `tradingagents/dataflows/data_cache`。
        """
        if cache_dir is None:
            # 获取当前文件所在目录
            current_dir = Path(__file__).parent
            cache_dir = current_dir / "data_cache"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # 创建子目录 - 按市场分类
        self.us_stock_dir = self.cache_dir / "us_stocks"
        self.china_stock_dir = self.cache_dir / "china_stocks"
        self.us_news_dir = self.cache_dir / "us_news"
        self.china_news_dir = self.cache_dir / "china_news"
        self.us_fundamentals_dir = self.cache_dir / "us_fundamentals"
        self.china_fundamentals_dir = self.cache_dir / "china_fundamentals"
        self.metadata_dir = self.cache_dir / "metadata"

        # 创建所有目录
        for dir_path in [self.us_stock_dir, self.china_stock_dir, self.us_news_dir,
                        self.china_news_dir, self.us_fundamentals_dir,
                        self.china_fundamentals_dir, self.metadata_dir]:
            dir_path.mkdir(exist_ok=True)

        # 缓存配置 - 针对不同市场设置不同的TTL
        self.cache_config = {
            'us_stock_data': {
                'ttl_hours': 2,  # 美股数据缓存2小时（考虑到API限制）
                'max_files': 1000,
                'description': '美股历史数据'
            },
            'china_stock_data': {
                'ttl_hours': 1,  # A股数据缓存1小时（实时性要求高）
                'max_files': 1000,
                'description': 'A股历史数据'
            },
            'us_news': {
                'ttl_hours': 6,  # 美股新闻缓存6小时
                'max_files': 500,
                'description': '美股新闻数据'
            },
            'china_news': {
                'ttl_hours': 4,  # A股新闻缓存4小时
                'max_files': 500,
                'description': 'A股新闻数据'
            },
            'us_fundamentals': {
                'ttl_hours': 24,  # 美股基本面数据缓存24小时
                'max_files': 200,
                'description': '美股基本面数据'
            },
            'china_fundamentals': {
                'ttl_hours': 12,  # A股基本面数据缓存12小时
                'max_files': 200,
                'description': 'A股基本面数据'
            }
        }

        # 内容长度限制配置（文件缓存默认不限制）
        self.content_length_config = {
            'max_content_length': int(os.getenv('MAX_CACHE_CONTENT_LENGTH', '50000')),  # 50K字符
            'long_text_providers': ['dashscope', 'openai', 'google'],  # 支持长文本的提供商
            'enable_length_check': os.getenv('ENABLE_CACHE_LENGTH_CHECK', 'false').lower() == 'true'  # 文件缓存默认不限制
        }

        logger.info(f"📁 缓存管理器初始化完成，缓存目录: {self.cache_dir}")
        logger.info(f"🗄️ 数据库缓存管理器初始化完成")
        logger.info(f"   美股数据: ✅ 已配置")
        logger.info(f"   A股数据: ✅ 已配置")

    def _determine_market_type(self, symbol: str) -> str:
        """根据股票代码的格式粗略判断其所属市场（中国或美国）。

        Args:
            symbol (str): 股票代码。

        Returns:
            str: 'china' 或 'us'。
        """
        import re

        # 判断是否为中国A股（6位数字）
        if re.match(r'^\d{6}$', str(symbol)):
            return 'china'
        else:
            return 'us'

    def _check_provider_availability(self) -> List[str]:
        """检查当前环境中配置了哪些大语言模型（LLM）提供商的API密钥。

        Returns:
            List[str]: 可用的提供商名称列表 (例如, ['dashscope', 'openai'])。
        """
        available_providers = []
        
        # 检查DashScope
        dashscope_key = os.getenv("DASHSCOPE_API_KEY")
        if dashscope_key and dashscope_key.strip():
            available_providers.append('dashscope')
        
        # 检查OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key and openai_key.strip():
            # 简单的格式检查
            if openai_key.startswith('sk-') and len(openai_key) >= 40:
                available_providers.append('openai')
        
        # 检查Google AI
        google_key = os.getenv("GOOGLE_API_KEY")
        if google_key and google_key.strip():
            available_providers.append('google')
        
        # 检查Anthropic
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key and anthropic_key.strip():
            available_providers.append('anthropic')
        
        return available_providers

    def should_skip_cache_for_content(self, content: str, data_type: str = "unknown") -> bool:
        """根据内容长度和可用的LLM提供商，决定是否应跳过缓存。

        如果启用了长度检查、内容超过了最大长度，并且没有配置任何支持
        长文本处理的LLM提供商，则此方法建议跳过缓存。

        Args:
            content (str): 准备要缓存的文本内容。
            data_type (str, optional): 数据类型描述，用于日志记录。

        Returns:
            bool: 如果应跳过缓存，则为 True，否则为 False。
        """
        # 如果未启用长度检查，直接返回False
        if not self.content_length_config['enable_length_check']:
            return False
        
        # 检查内容长度
        content_length = len(content)
        max_length = self.content_length_config['max_content_length']
        
        if content_length <= max_length:
            return False
        
        # 内容超长，检查是否有可用的长文本处理提供商
        available_providers = self._check_provider_availability()
        long_text_providers = self.content_length_config['long_text_providers']
        
        # 找到可用的长文本提供商
        available_long_providers = [p for p in available_providers if p in long_text_providers]
        
        if not available_long_providers:
            logger.warning(f"⚠️ 内容过长({content_length:,}字符 > {max_length:,}字符)且无可用长文本提供商，跳过{data_type}缓存")
            logger.info(f"💡 可用提供商: {available_providers}")
            logger.info(f"💡 长文本提供商: {long_text_providers}")
            return True
        else:
            logger.info(f"✅ 内容较长({content_length:,}字符)但有可用长文本提供商({available_long_providers})，继续缓存")
            return False
    
    def _generate_cache_key(self, data_type: str, symbol: str, **kwargs) -> str:
        """基于数据类型、股票代码及其他参数生成一个确定性的缓存键。

        Args:
            data_type (str): 数据类型 (例如, 'stock_data')。
            symbol (str): 股票代码。
            **kwargs: 其他用于生成键的参数 (例如, start_date, data_source)。

        Returns:
            str: 生成的唯一缓存键。
        """
        # 创建一个包含所有参数的字符串
        params_str = f"{data_type}_{symbol}"
        for key, value in sorted(kwargs.items()):
            params_str += f"_{key}_{value}"
        
        # 使用MD5生成短的唯一标识
        cache_key = hashlib.md5(params_str.encode()).hexdigest()[:12]
        return f"{symbol}_{data_type}_{cache_key}"
    
    def _get_cache_path(self, data_type: str, cache_key: str, file_format: str = "json", symbol: str = None) -> Path:
        """根据数据类型、市场和文件名构建缓存文件的完整路径。

        Args:
            data_type (str): 数据类型。
            cache_key (str): 缓存键。
            file_format (str, optional): 文件扩展名。
            symbol (str, optional): 股票代码，用于确定市场子目录。

        Returns:
            Path: 指向缓存文件的 `pathlib.Path` 对象。
        """
        if symbol:
            market_type = self._determine_market_type(symbol)
        else:
            # 从缓存键中尝试提取市场类型
            market_type = 'us' if not cache_key.startswith(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')) else 'china'

        # 根据数据类型和市场类型选择目录
        if data_type == "stock_data":
            base_dir = self.china_stock_dir if market_type == 'china' else self.us_stock_dir
        elif data_type == "news":
            base_dir = self.china_news_dir if market_type == 'china' else self.us_news_dir
        elif data_type == "fundamentals":
            base_dir = self.china_fundamentals_dir if market_type == 'china' else self.us_fundamentals_dir
        else:
            base_dir = self.cache_dir

        return base_dir / f"{cache_key}.{file_format}"
    
    def _get_metadata_path(self, cache_key: str) -> Path:
        """获取指定缓存键的元数据文件路径。

        Args:
            cache_key (str): 缓存键。

        Returns:
            Path: 指向元数据文件的路径。
        """
        return self.metadata_dir / f"{cache_key}_meta.json"
    
    def _save_metadata(self, cache_key: str, metadata: Dict[str, Any]):
        """将元数据以JSON格式保存到文件。

        Args:
            cache_key (str): 缓存键。
            metadata (Dict[str, Any]): 要保存的元数据字典。
        """
        metadata_path = self._get_metadata_path(cache_key)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在
        metadata['cached_at'] = datetime.now().isoformat()
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    def _load_metadata(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """从文件加载并解析元数据。

        Args:
            cache_key (str): 缓存键。

        Returns:
            Optional[Dict[str, Any]]: 如果成功，返回元数据字典；
                否则返回 None。
        """
        metadata_path = self._get_metadata_path(cache_key)
        if not metadata_path.exists():
            return None
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"⚠️ 加载元数据失败: {e}")
            return None
    
    def is_cache_valid(self, cache_key: str, max_age_hours: int = None, symbol: str = None, data_type: str = None) -> bool:
        """检查指定的缓存键是否仍然有效。

        它会根据元数据中的时间戳和智能TTL配置来判断缓存是否过期。

        Args:
            cache_key (str): 要检查的缓存键。
            max_age_hours (int, optional): 手动指定的最大缓存小时数。
                如果为None，则使用 `cache_config` 中的智能配置。
            symbol (str, optional): 股票代码，用于帮助确定TTL。
            data_type (str, optional): 数据类型，用于帮助确定TTL。

        Returns:
            bool: 如果缓存有效，返回 True；否则返回 False。
        """
        metadata = self._load_metadata(cache_key)
        if not metadata:
            return False

        # 如果没有指定TTL，根据数据类型和市场自动确定
        if max_age_hours is None:
            if symbol and data_type:
                market_type = self._determine_market_type(symbol)
                cache_type = f"{market_type}_{data_type}"
                max_age_hours = self.cache_config.get(cache_type, {}).get('ttl_hours', 24)
            else:
                # 从元数据中获取信息
                symbol = metadata.get('symbol', '')
                data_type = metadata.get('data_type', 'stock_data')
                market_type = self._determine_market_type(symbol)
                cache_type = f"{market_type}_{data_type}"
                max_age_hours = self.cache_config.get(cache_type, {}).get('ttl_hours', 24)

        cached_at = datetime.fromisoformat(metadata['cached_at'])
        age = datetime.now() - cached_at

        is_valid = age.total_seconds() < max_age_hours * 3600

        if is_valid:
            market_type = self._determine_market_type(metadata.get('symbol', ''))
            cache_type = f"{market_type}_{metadata.get('data_type', 'stock_data')}"
            desc = self.cache_config.get(cache_type, {}).get('description', '数据')
            logger.info(f"✅ 缓存有效: {desc} - {metadata.get('symbol')} (剩余 {max_age_hours - age.total_seconds()/3600:.1f}h)")

        return is_valid
    
    def save_stock_data(self, symbol: str, data: Union[pd.DataFrame, str],
                       start_date: str = None, end_date: str = None,
                       data_source: str = "unknown") -> str:
        """将股票历史行情数据保存到缓存。

        根据数据是DataFrame还是字符串，分别保存为CSV或TXT文件。

        Args:
            symbol (str): 股票代码。
            data (Union[pd.DataFrame, str]): 股票数据。
            start_date (str, optional): 数据开始日期。
            end_date (str, optional): 数据结束日期。
            data_source (str, optional): 数据来源 (例如, "yfinance")。

        Returns:
            str: 生成的缓存键。如果因内容过长而跳过，返回的是一个
                虚拟的缓存键。
        """
        # 检查内容长度是否需要跳过缓存
        content_to_check = str(data)
        if self.should_skip_cache_for_content(content_to_check, "股票数据"):
            # 生成一个虚拟的缓存键，但不实际保存
            market_type = self._determine_market_type(symbol)
            cache_key = self._generate_cache_key("stock_data", symbol,
                                               start_date=start_date,
                                               end_date=end_date,
                                               source=data_source,
                                               market=market_type,
                                               skipped=True)
            logger.info(f"🚫 股票数据因内容过长被跳过缓存: {symbol} -> {cache_key}")
            return cache_key

        market_type = self._determine_market_type(symbol)
        cache_key = self._generate_cache_key("stock_data", symbol,
                                           start_date=start_date,
                                           end_date=end_date,
                                           source=data_source,
                                           market=market_type)

        # 保存数据
        if isinstance(data, pd.DataFrame):
            cache_path = self._get_cache_path("stock_data", cache_key, "csv", symbol)
            cache_path.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在
            data.to_csv(cache_path, index=True)
        else:
            cache_path = self._get_cache_path("stock_data", cache_key, "txt", symbol)
            cache_path.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在
            with open(cache_path, 'w', encoding='utf-8') as f:
                f.write(str(data))

        # 保存元数据
        metadata = {
            'symbol': symbol,
            'data_type': 'stock_data',
            'market_type': market_type,
            'start_date': start_date,
            'end_date': end_date,
            'data_source': data_source,
            'file_path': str(cache_path),
            'file_format': 'csv' if isinstance(data, pd.DataFrame) else 'txt',
            'content_length': len(content_to_check)
        }
        self._save_metadata(cache_key, metadata)

        # 获取描述信息
        cache_type = f"{market_type}_stock_data"
        desc = self.cache_config.get(cache_type, {}).get('description', '股票数据')
        logger.info(f"💾 {desc}已缓存: {symbol} ({data_source}) -> {cache_key}")
        return cache_key
    
    def load_stock_data(self, cache_key: str) -> Optional[Union[pd.DataFrame, str]]:
        """根据缓存键从文件加载股票数据。

        Args:
            cache_key (str): 缓存键。

        Returns:
            Optional[Union[pd.DataFrame, str]]: 返回加载的数据，如果
                文件不存在或加载失败，则返回 None。
        """
        metadata = self._load_metadata(cache_key)
        if not metadata:
            return None
        
        cache_path = Path(metadata['file_path'])
        if not cache_path.exists():
            return None
        
        try:
            if metadata['file_format'] == 'csv':
                return pd.read_csv(cache_path, index_col=0)
            else:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            logger.error(f"⚠️ 加载缓存数据失败: {e}")
            return None
    
    def find_cached_stock_data(self, symbol: str, start_date: str = None,
                              end_date: str = None, data_source: str = None,
                              max_age_hours: int = None) -> Optional[str]:
        """查找与给定参数匹配的有效股票数据缓存。

        首先尝试精确匹配所有参数，如果找不到，则会放宽条件，查找该股票
        代码下的任何其他有效缓存。

        Args:
            symbol (str): 股票代码。
            start_date (str, optional): 数据开始日期。
            end_date (str, optional): 数据结束日期。
            data_source (str, optional): 数据来源。
            max_age_hours (int, optional): 手动指定最大缓存时间。

        Returns:
            Optional[str]: 如果找到，返回缓存键；否则返回 None。
        """
        market_type = self._determine_market_type(symbol)

        # 如果没有指定TTL，使用智能配置
        if max_age_hours is None:
            cache_type = f"{market_type}_stock_data"
            max_age_hours = self.cache_config.get(cache_type, {}).get('ttl_hours', 24)

        # 生成查找键
        search_key = self._generate_cache_key("stock_data", symbol,
                                            start_date=start_date,
                                            end_date=end_date,
                                            source=data_source,
                                            market=market_type)

        # 检查精确匹配
        if self.is_cache_valid(search_key, max_age_hours, symbol, 'stock_data'):
            desc = self.cache_config.get(f"{market_type}_stock_data", {}).get('description', '数据')
            logger.info(f"🎯 找到精确匹配的{desc}: {symbol} -> {search_key}")
            return search_key

        # 如果没有精确匹配，查找部分匹配（相同股票代码的其他缓存）
        for metadata_file in self.metadata_dir.glob(f"*_meta.json"):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                if (metadata.get('symbol') == symbol and
                    metadata.get('data_type') == 'stock_data' and
                    metadata.get('market_type') == market_type and
                    (data_source is None or metadata.get('data_source') == data_source)):

                    cache_key = metadata_file.stem.replace('_meta', '')
                    if self.is_cache_valid(cache_key, max_age_hours, symbol, 'stock_data'):
                        desc = self.cache_config.get(f"{market_type}_stock_data", {}).get('description', '数据')
                        logger.info(f"📋 找到部分匹配的{desc}: {symbol} -> {cache_key}")
                        return cache_key
            except Exception:
                continue

        desc = self.cache_config.get(f"{market_type}_stock_data", {}).get('description', '数据')
        logger.error(f"❌ 未找到有效的{desc}缓存: {symbol}")
        return None
    
    def save_news_data(self, symbol: str, news_data: str, 
                      start_date: str = None, end_date: str = None,
                      data_source: str = "unknown") -> str:
        """将新闻数据（字符串）保存到缓存。

        Args:
            symbol (str): 相关股票代码。
            news_data (str): 新聞文本内容。
            start_date (str, optional): 新闻时间范围的开始日期。
            end_date (str, optional): 新闻时间范围的结束日期。
            data_source (str, optional): 新闻来源。

        Returns:
            str: 生成的缓存键。
        """
        # 检查内容长度是否需要跳过缓存
        if self.should_skip_cache_for_content(news_data, "新闻数据"):
            # 生成一个虚拟的缓存键，但不实际保存
            cache_key = self._generate_cache_key("news", symbol,
                                               start_date=start_date,
                                               end_date=end_date,
                                               source=data_source,
                                               skipped=True)
            logger.info(f"🚫 新闻数据因内容过长被跳过缓存: {symbol} -> {cache_key}")
            return cache_key

        cache_key = self._generate_cache_key("news", symbol,
                                           start_date=start_date,
                                           end_date=end_date,
                                           source=data_source)
        
        cache_path = self._get_cache_path("news", cache_key, "txt")
        cache_path.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在
        with open(cache_path, 'w', encoding='utf-8') as f:
            f.write(news_data)
        
        metadata = {
            'symbol': symbol,
            'data_type': 'news',
            'start_date': start_date,
            'end_date': end_date,
            'data_source': data_source,
            'file_path': str(cache_path),
            'file_format': 'txt',
            'content_length': len(news_data)
        }
        self._save_metadata(cache_key, metadata)
        
        logger.info(f"📰 新闻数据已缓存: {symbol} ({data_source}) -> {cache_key}")
        return cache_key
    
    def save_fundamentals_data(self, symbol: str, fundamentals_data: str,
                              data_source: str = "unknown") -> str:
        """将基本面分析数据（字符串）保存到缓存。

        Args:
            symbol (str): 相关股票代码。
            fundamentals_data (str): 基本面分析的文本内容。
            data_source (str, optional): 数据来源 (例如, "openai")。

        Returns:
            str: 生成的缓存键。
        """
        # 检查内容长度是否需要跳过缓存
        if self.should_skip_cache_for_content(fundamentals_data, "基本面数据"):
            # 生成一个虚拟的缓存键，但不实际保存
            market_type = self._determine_market_type(symbol)
            cache_key = self._generate_cache_key("fundamentals", symbol,
                                               source=data_source,
                                               market=market_type,
                                               date=datetime.now().strftime("%Y-%m-%d"),
                                               skipped=True)
            logger.info(f"🚫 基本面数据因内容过长被跳过缓存: {symbol} -> {cache_key}")
            return cache_key

        market_type = self._determine_market_type(symbol)
        cache_key = self._generate_cache_key("fundamentals", symbol,
                                           source=data_source,
                                           market=market_type,
                                           date=datetime.now().strftime("%Y-%m-%d"))
        
        cache_path = self._get_cache_path("fundamentals", cache_key, "txt", symbol)
        cache_path.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在
        with open(cache_path, 'w', encoding='utf-8') as f:
            f.write(fundamentals_data)
        
        metadata = {
            'symbol': symbol,
            'data_type': 'fundamentals',
            'data_source': data_source,
            'market_type': market_type,
            'file_path': str(cache_path),
            'file_format': 'txt',
            'content_length': len(fundamentals_data)
        }
        self._save_metadata(cache_key, metadata)
        
        desc = self.cache_config.get(f"{market_type}_fundamentals", {}).get('description', '基本面数据')
        logger.info(f"💼 {desc}已缓存: {symbol} ({data_source}) -> {cache_key}")
        return cache_key
    
    def load_fundamentals_data(self, cache_key: str) -> Optional[str]:
        """根据缓存键从文件加载基本面数据。

        Args:
            cache_key (str): 缓存键。

        Returns:
            Optional[str]: 返回加载的文本数据，如果文件不存在或加载
                失败，则返回 None。
        """
        metadata = self._load_metadata(cache_key)
        if not metadata:
            return None
        
        cache_path = Path(metadata['file_path'])
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"⚠️ 加载基本面缓存数据失败: {e}")
            return None
    
    def find_cached_fundamentals_data(self, symbol: str, data_source: str = None,
                                    max_age_hours: int = None) -> Optional[str]:
        """查找与给定参数匹配的有效基本面数据缓存。

        Args:
            symbol (str): 股票代码。
            data_source (str, optional): 数据来源。
            max_age_hours (int, optional): 手动指定最大缓存时间。

        Returns:
            Optional[str]: 如果找到，返回缓存键；否则返回 None。
        """
        market_type = self._determine_market_type(symbol)
        
        # 如果没有指定TTL，使用智能配置
        if max_age_hours is None:
            cache_type = f"{market_type}_fundamentals"
            max_age_hours = self.cache_config.get(cache_type, {}).get('ttl_hours', 24)
        
        # 查找匹配的缓存
        for metadata_file in self.metadata_dir.glob(f"*_meta.json"):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                if (metadata.get('symbol') == symbol and
                    metadata.get('data_type') == 'fundamentals' and
                    metadata.get('market_type') == market_type and
                    (data_source is None or metadata.get('data_source') == data_source)):
                    
                    cache_key = metadata_file.stem.replace('_meta', '')
                    if self.is_cache_valid(cache_key, max_age_hours, symbol, 'fundamentals'):
                        desc = self.cache_config.get(f"{market_type}_fundamentals", {}).get('description', '基本面数据')
                        logger.info(f"🎯 找到匹配的{desc}缓存: {symbol} ({data_source}) -> {cache_key}")
                        return cache_key
            except Exception:
                continue
        
        desc = self.cache_config.get(f"{market_type}_fundamentals", {}).get('description', '基本面数据')
        logger.error(f"❌ 未找到有效的{desc}缓存: {symbol} ({data_source})")
        return None
    
    def clear_old_cache(self, max_age_days: int = 7):
        """清理所有超过指定天数的旧缓存文件及其元数据。

        Args:
            max_age_days (int, optional): 清理的阈值天数。默认7天。
        """
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        cleared_count = 0
        
        for metadata_file in self.metadata_dir.glob("*_meta.json"):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                cached_at = datetime.fromisoformat(metadata['cached_at'])
                if cached_at < cutoff_time:
                    # 删除数据文件
                    data_file = Path(metadata['file_path'])
                    if data_file.exists():
                        data_file.unlink()
                    
                    # 删除元数据文件
                    metadata_file.unlink()
                    cleared_count += 1
                    
            except Exception as e:
                logger.warning(f"⚠️ 清理缓存时出错: {e}")
        
        logger.info(f"🧹 已清理 {cleared_count} 个过期缓存文件")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取并返回当前缓存系统的统计信息。

        包括各类数据的数量、总文件大小以及因内容过长而跳过的缓存数量。

        Returns:
            Dict[str, Any]: 包含统计信息的字典。
        """
        stats = {
            'total_files': 0,
            'stock_data_count': 0,
            'news_count': 0,
            'fundamentals_count': 0,
            'total_size_mb': 0,
            'skipped_count': 0  # 新增：跳过的缓存数量
        }
        
        for metadata_file in self.metadata_dir.glob("*_meta.json"):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                data_type = metadata.get('data_type', 'unknown')
                if data_type == 'stock_data':
                    stats['stock_data_count'] += 1
                elif data_type == 'news':
                    stats['news_count'] += 1
                elif data_type == 'fundamentals':
                    stats['fundamentals_count'] += 1
                
                # 检查是否为跳过的缓存（没有实际文件）
                data_file = Path(metadata.get('file_path', ''))
                if not data_file.exists():
                    stats['skipped_count'] += 1
                else:
                    # 计算文件大小
                    stats['total_size_mb'] += data_file.stat().st_size / (1024 * 1024)
                
                stats['total_files'] += 1
                
            except Exception:
                continue
        
        stats['total_size_mb'] = round(stats['total_size_mb'], 2)
        return stats

    def get_content_length_config_status(self) -> Dict[str, Any]:
        """获取关于内容长度检查配置的当前状态和诊断信息。

        Returns:
            Dict[str, Any]: 包含配置状态的字典，例如是否启用、
                最大长度、可用的长文本LLM提供商等。
        """
        available_providers = self._check_provider_availability()
        long_text_providers = self.content_length_config['long_text_providers']
        available_long_providers = [p for p in available_providers if p in long_text_providers]
        
        return {
            'enabled': self.content_length_config['enable_length_check'],
            'max_content_length': self.content_length_config['max_content_length'],
            'max_content_length_formatted': f"{self.content_length_config['max_content_length']:,}字符",
            'long_text_providers': long_text_providers,
            'available_providers': available_providers,
            'available_long_providers': available_long_providers,
            'has_long_text_support': len(available_long_providers) > 0,
            'will_skip_long_content': self.content_length_config['enable_length_check'] and len(available_long_providers) == 0
        }


# 全局缓存实例
_cache_instance = None

def get_cache() -> StockDataCache:
    """获取StockDataCache的全局单例。

    Returns:
        StockDataCache: 缓存管理器的单例实例。
    """
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = StockDataCache()
    return _cache_instance
