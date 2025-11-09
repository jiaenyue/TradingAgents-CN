#!/usr/bin/env python3
"""
集成缓存管理器模块。

该模块提供了一个 `IntegratedCacheManager` 类, 旨在统一和智能地管理
系统中的数据缓存。它结合了传统的基于文件的缓存 (`StockDataCache`) 和
一个更现代的、支持数据库后端的自适应缓存系统 (`AdaptiveCache`)。

主要目标:
- **智能切换**: 根据数据库 (MongoDB, Redis) 的可用性, 自动选择
  最高效的缓存策略。
- **向后兼容**: 提供与旧 `cache_manager` 模块兼容的接口, 确保
  系统平滑升级。
- **统一接口**: 为不同类型的数据 (股票行情、新闻、基本面) 提供
  一致的 `save`, `load`, `find` 方法。
- **可配置性**: 允许系统在不同环境中 (例如, 只有文件系统、有完整
  数据库支持) 都能高效运行。
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
import pandas as pd

# 导入统一日志系统
from tradingagents.utils.logging_init import setup_dataflow_logging

# 导入原有缓存系统
from .cache_manager import StockDataCache

# 导入自适应缓存系统
try:
    from .adaptive_cache import get_cache_system
    from ..config.database_manager import get_database_manager
    ADAPTIVE_CACHE_AVAILABLE = True
except ImportError:
    ADAPTIVE_CACHE_AVAILABLE = False

class IntegratedCacheManager:
    """集成缓存管理器, 用于智能选择和管理缓存策略。

    该类在初始化时会检测数据库服务的可用性。如果检测到可用的数据库
    (如 MongoDB 或 Redis), 它会优先使用高性能的自适应缓存系统。
    如果数据库不可用, 或者 `adaptive_cache` 模块导入失败, 它会自动
    降级, 使用传统的、基于文件的缓存系统 (`StockDataCache`) 作为备用,
    从而保证系统的健壮性。

    所有缓存操作 (如保存、加载、查找) 都通过该类提供的统一接口进行,
    上层应用无需关心底层的缓存实现细节。
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        初始化集成缓存管理器。

        Args:
            cache_dir (Optional[str], optional):
                用于传统文件缓存的根目录。如果为 None, 则使用默认路径。
                此参数主要用于向后兼容。
        """
        self.logger = setup_dataflow_logging()
        
        # 初始化原有缓存系统（作为备用）
        self.legacy_cache = StockDataCache(cache_dir)
        
        # 尝试初始化自适应缓存系统
        self.adaptive_cache = None
        self.use_adaptive = False
        
        if ADAPTIVE_CACHE_AVAILABLE:
            try:
                self.adaptive_cache = get_cache_system()
                self.db_manager = get_database_manager()
                self.use_adaptive = True
                self.logger.info("✅ 自适应缓存系统已启用")
            except Exception as e:
                self.logger.warning(f"自适应缓存系统初始化失败，使用传统缓存: {e}")
                self.use_adaptive = False
        else:
            self.logger.info("自适应缓存系统不可用，使用传统文件缓存")
        
        # 显示当前配置
        self._log_cache_status()
    
    def _log_cache_status(self):
        """记录并打印当前的缓存系统状态和配置。"""
        if self.use_adaptive:
            backend = self.adaptive_cache.primary_backend
            mongodb_available = self.db_manager.is_mongodb_available()
            redis_available = self.db_manager.is_redis_available()
            
            self.logger.info(f"📊 缓存配置:")
            self.logger.info(f"  主要后端: {backend}")
            self.logger.info(f"  MongoDB: {'✅ 可用' if mongodb_available else '❌ 不可用'}")
            self.logger.info(f"  Redis: {'✅ 可用' if redis_available else '❌ 不可用'}")
            self.logger.info(f"  降级支持: {'✅ 启用' if self.adaptive_cache.fallback_enabled else '❌ 禁用'}")
        else:
            self.logger.info("📁 使用传统文件缓存系统")
    
    def save_stock_data(self, symbol: str, data: Any, start_date: Optional[str] = None,
                       end_date: Optional[str] = None, data_source: str = "default") -> str:
        """
        将股票行情数据保存到缓存。

        根据当前激活的缓存系统 (自适应或传统), 将数据进行存储。

        Args:
            symbol (str): 股票代码。
            data (Any): 要缓存的股票数据 (例如, pd.DataFrame)。
            start_date (Optional[str], optional): 数据的开始日期。
            end_date (Optional[str], optional): 数据的结束日期。
            data_source (str, optional): 数据来源的标识符。

        Returns:
            str: 用于将来检索数据的缓存键。
        """
        if self.use_adaptive:
            # 使用自适应缓存系统
            return self.adaptive_cache.save_data(
                symbol=symbol,
                data=data,
                start_date=start_date or "",
                end_date=end_date or "",
                data_source=data_source,
                data_type="stock_data"
            )
        else:
            # 使用传统缓存系统
            return self.legacy_cache.save_stock_data(
                symbol=symbol,
                data=data,
                start_date=start_date,
                end_date=end_date,
                data_source=data_source
            )
    
    def load_stock_data(self, cache_key: str) -> Optional[Any]:
        """
        根据缓存键从缓存中加载股票行情数据。

        Args:
            cache_key (str): 由 `save_stock_data` 返回的缓存键。

        Returns:
            Optional[Any]: 缓存的股票数据。如果未找到, 则返回 None。
        """
        if self.use_adaptive:
            # 使用自适应缓存系统
            return self.adaptive_cache.load_data(cache_key)
        else:
            # 使用传统缓存系统
            return self.legacy_cache.load_stock_data(cache_key)
    
    def find_cached_stock_data(self, symbol: str, start_date: Optional[str] = None,
                              end_date: Optional[str] = None, data_source: str = "default") -> Optional[str]:
        """
        根据查询参数查找已缓存的股票行情数据。

        Args:
            symbol (str): 股票代码。
            start_date (Optional[str], optional): 数据的开始日期。
            end_date (Optional[str], optional): 数据的结束日期。
            data_source (str, optional): 数据来源的标识符。

        Returns:
            Optional[str]: 如果找到匹配的缓存, 则返回缓存键; 否则返回 None。
        """
        if self.use_adaptive:
            # 使用自适应缓存系统
            return self.adaptive_cache.find_cached_data(
                symbol=symbol,
                start_date=start_date or "",
                end_date=end_date or "",
                data_source=data_source,
                data_type="stock_data"
            )
        else:
            # 使用传统缓存系统
            return self.legacy_cache.find_cached_stock_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                data_source=data_source
            )
    
    def save_news_data(self, symbol: str, data: Any, data_source: str = "default") -> str:
        """将新闻数据保存到缓存。

        Args:
            symbol (str): 股票代码或相关主题。
            data (Any): 要缓存的新闻数据。
            data_source (str, optional): 数据来源。

        Returns:
            str: 数据的缓存键。
        """
        if self.use_adaptive:
            return self.adaptive_cache.save_data(
                symbol=symbol,
                data=data,
                data_source=data_source,
                data_type="news_data"
            )
        else:
            return self.legacy_cache.save_news_data(symbol, data, data_source)
    
    def load_news_data(self, cache_key: str) -> Optional[Any]:
        """根据缓存键加载新闻数据。

        Args:
            cache_key (str): 缓存键。

        Returns:
            Optional[Any]: 缓存的新闻数据, 或 None。
        """
        if self.use_adaptive:
            return self.adaptive_cache.load_data(cache_key)
        else:
            return self.legacy_cache.load_news_data(cache_key)
    
    def save_fundamentals_data(self, symbol: str, data: Any, data_source: str = "default") -> str:
        """将基本面数据保存到缓存。

        Args:
            symbol (str): 股票代码。
            data (Any): 要缓存的基本面数据。
            data_source (str, optional): 数据来源。

        Returns:
            str: 数据的缓存键。
        """
        if self.use_adaptive:
            return self.adaptive_cache.save_data(
                symbol=symbol,
                data=data,
                data_source=data_source,
                data_type="fundamentals_data"
            )
        else:
            return self.legacy_cache.save_fundamentals_data(symbol, data, data_source)
    
    def load_fundamentals_data(self, cache_key: str) -> Optional[Any]:
        """根据缓存键加载基本面数据。

        Args:
            cache_key (str): 缓存键。

        Returns:
            Optional[Any]: 缓存的基本面数据, 或 None。
        """
        if self.use_adaptive:
            return self.adaptive_cache.load_data(cache_key)
        else:
            return self.legacy_cache.load_fundamentals_data(cache_key)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取关于缓存系统的统计信息。

        如果自适应缓存系统被激活, 将同时返回自适应和传统缓存的统计数据。
        否则, 只返回传统缓存的统计数据。

        Returns:
            Dict[str, Any]: 包含缓存系统类型、统计数据和数据库状态的字典。
        """
        if self.use_adaptive:
            # 获取自适应缓存统计
            adaptive_stats = self.adaptive_cache.get_cache_stats()
            
            # 添加传统缓存统计
            legacy_stats = self.legacy_cache.get_cache_stats()
            
            return {
                "cache_system": "adaptive",
                "adaptive_cache": adaptive_stats,
                "legacy_cache": legacy_stats,
                "database_available": self.db_manager.is_database_available(),
                "mongodb_available": self.db_manager.is_mongodb_available(),
                "redis_available": self.db_manager.is_redis_available()
            }
        else:
            # 只返回传统缓存统计
            legacy_stats = self.legacy_cache.get_cache_stats()
            return {
                "cache_system": "legacy",
                "legacy_cache": legacy_stats,
                "database_available": False,
                "mongodb_available": False,
                "redis_available": False
            }
    
    def clear_expired_cache(self):
        """清理所有缓存系统中的过期条目。"""
        if self.use_adaptive:
            self.adaptive_cache.clear_expired_cache()
        
        # 总是清理传统缓存
        self.legacy_cache.clear_expired_cache()
    
    def get_cache_backend_info(self) -> Dict[str, Any]:
        """获取当前缓存后端的配置信息。

        Returns:
            Dict[str, Any]: 包含当前缓存系统、主后端、备用策略和
                            数据库状态的字典。
        """
        if self.use_adaptive:
            return {
                "system": "adaptive",
                "primary_backend": self.adaptive_cache.primary_backend,
                "fallback_enabled": self.adaptive_cache.fallback_enabled,
                "mongodb_available": self.db_manager.is_mongodb_available(),
                "redis_available": self.db_manager.is_redis_available()
            }
        else:
            return {
                "system": "legacy",
                "primary_backend": "file",
                "fallback_enabled": False,
                "mongodb_available": False,
                "redis_available": False
            }
    
    def is_database_available(self) -> bool:
        """检查是否有任何数据库后端 (MongoDB 或 Redis) 可用。

        Returns:
            bool: 如果至少有一个数据库可用, 则为 True, 否则为 False。
        """
        if self.use_adaptive:
            return self.db_manager.is_database_available()
        return False
    
    def get_performance_mode(self) -> str:
        """根据当前可用的后端服务, 返回缓存系统的性能模式描述。

        Returns:
            str: 描述当前性能模式的字符串。
        """
        if not self.use_adaptive:
            return "基础模式 (文件缓存)"
        
        mongodb_available = self.db_manager.is_mongodb_available()
        redis_available = self.db_manager.is_redis_available()
        
        if redis_available and mongodb_available:
            return "高性能模式 (Redis + MongoDB + 文件)"
        elif redis_available:
            return "快速模式 (Redis + 文件)"
        elif mongodb_available:
            return "持久化模式 (MongoDB + 文件)"
        else:
            return "标准模式 (智能文件缓存)"


# 全局集成缓存管理器实例
_integrated_cache = None

def get_cache() -> IntegratedCacheManager:
    """获取 `IntegratedCacheManager` 的全局单例。

    Returns:
        IntegratedCacheManager: 全局缓存管理器实例。
    """
    global _integrated_cache
    if _integrated_cache is None:
        _integrated_cache = IntegratedCacheManager()
    return _integrated_cache

# 向后兼容的函数
def get_stock_cache():
    """向后兼容的函数, 用于获取全局缓存管理器。

    Returns:
        IntegratedCacheManager: 全局缓存管理器实例。
    """
    return get_cache()

def create_cache_manager(cache_dir: Optional[str] = None) -> IntegratedCacheManager:
    """向后兼容的函数, 用于创建一个新的 `IntegratedCacheManager` 实例。

    Args:
        cache_dir (Optional[str], optional): 文件缓存的根目录。

    Returns:
        IntegratedCacheManager: 一个新的缓存管理器实例。
    """
    return IntegratedCacheManager(cache_dir)
