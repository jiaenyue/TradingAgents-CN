#!/usr/bin/env python3
"""
自适应缓存系统
根据数据库可用性自动选择最佳缓存策略
"""

import os
import json
import pickle
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Union
import pandas as pd

from ..config.database_manager import get_database_manager

class AdaptiveCacheSystem:
    """自适应缓存系统。

    根据数据库（Redis, MongoDB）的可用性动态选择最佳的缓存策略。
    如果配置的主要后端（如Redis）不可用，它可以自动降级到备用后端
    （如文件缓存），从而确保系统的健壮性和高性能。

    Attributes:
        logger: 日志记录器实例。
        db_manager: 数据库管理器实例，用于访问Redis和MongoDB。
        cache_dir (Path): 文件缓存的存储目录。
        config (dict): 从数据库管理器加载的全局配置。
        cache_config (dict): 缓存相关的特定配置。
        primary_backend (str): 首选的主要缓存后端 ('redis', 'mongodb', 'file')。
        fallback_enabled (bool): 是否启用降级到文件缓存的机制。
    """
    
    def __init__(self, cache_dir: str = "data/cache"):
        """初始化自适应缓存系统。

        Args:
            cache_dir (str, optional): 用于文件缓存的根目录。
                默认为 "data/cache"。
        """
        self.logger = logging.getLogger(__name__)
        
        # 获取数据库管理器
        self.db_manager = get_database_manager()
        
        # 设置缓存目录
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取配置
        self.config = self.db_manager.get_config()
        self.cache_config = self.config["cache"]
        
        # 初始化缓存后端
        self.primary_backend = self.cache_config["primary_backend"]
        self.fallback_enabled = self.cache_config["fallback_enabled"]
        
        self.logger.info(f"自适应缓存系统初始化 - 主要后端: {self.primary_backend}")
    
    def _get_cache_key(self, symbol: str, start_date: str = "", end_date: str = "", 
                      data_source: str = "default", data_type: str = "stock_data") -> str:
        """根据输入参数生成一个确定性的缓存键。

        使用MD5哈希算法确保键的唯一性和固定长度。

        Args:
            symbol (str): 股票代码。
            start_date (str, optional): 数据开始日期。默认为 ""。
            end_date (str, optional): 数据结束日期。默认为 ""。
            data_source (str, optional): 数据来源。默认为 "default"。
            data_type (str, optional): 数据类型 (例如, 'stock_data', 'news')。
                默认为 "stock_data"。

        Returns:
            str: 生成的MD5缓存键。
        """
        key_data = f"{symbol}_{start_date}_{end_date}_{data_source}_{data_type}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_ttl_seconds(self, symbol: str, data_type: str = "stock_data") -> int:
        """根据市场和数据类型获取缓存的生存时间（TTL）。

        Args:
            symbol (str): 股票代码，用于判断市场（中国或美国）。
            data_type (str, optional): 数据类型。默认为 "stock_data"。

        Returns:
            int: 缓存的有效时间（秒）。
        """
        # 判断市场类型
        if len(symbol) == 6 and symbol.isdigit():
            market = "china"
        else:
            market = "us"
        
        # 获取TTL配置
        ttl_key = f"{market}_{data_type}"
        ttl_seconds = self.cache_config["ttl_settings"].get(ttl_key, 7200)
        return ttl_seconds
    
    def _is_cache_valid(self, cache_time: datetime, ttl_seconds: int) -> bool:
        """检查文件缓存是否在有效期内。

        Args:
            cache_time (datetime): 缓存的创建时间。
            ttl_seconds (int): 缓存的有效秒数。

        Returns:
            bool: 如果缓存仍然有效，则返回 True，否则返回 False。
        """
        if cache_time is None:
            return False
        
        expiry_time = cache_time + timedelta(seconds=ttl_seconds)
        return datetime.now() < expiry_time
    
    def _save_to_file(self, cache_key: str, data: Any, metadata: Dict) -> bool:
        """将数据和元数据以pickle格式保存到文件缓存。

        Args:
            cache_key (str): 缓存键。
            data (Any): 要缓存的数据。
            metadata (Dict): 数据的元信息。

        Returns:
            bool: 如果保存成功，返回 True，否则返回 False。
        """
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            cache_data = {
                'data': data,
                'metadata': metadata,
                'timestamp': datetime.now(),
                'backend': 'file'
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            self.logger.debug(f"文件缓存保存成功: {cache_key}")
            return True
            
        except Exception as e:
            self.logger.error(f"文件缓存保存失败: {e}")
            return False
    
    def _load_from_file(self, cache_key: str) -> Optional[Dict]:
        """从文件缓存中加载pickle格式的数据。

        Args:
            cache_key (str): 缓存键。

        Returns:
            Optional[Dict]: 如果找到缓存文件，返回包含数据和元信息的字典，
                否则返回 None。
        """
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if not cache_file.exists():
                return None
            
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.logger.debug(f"文件缓存加载成功: {cache_key}")
            return cache_data
            
        except Exception as e:
            self.logger.error(f"文件缓存加载失败: {e}")
            return None
    
    def _save_to_redis(self, cache_key: str, data: Any, metadata: Dict, ttl_seconds: int) -> bool:
        """将数据序列化后保存到Redis，并设置TTL。

        Args:
            cache_key (str): 缓存键。
            data (Any): 要缓存的数据。
            metadata (Dict): 数据的元信息。
            ttl_seconds (int): 在Redis中的生存时间（秒）。

        Returns:
            bool: 如果保存成功，返回 True，否则返回 False。
        """
        redis_client = self.db_manager.get_redis_client()
        if not redis_client:
            return False
        
        try:
            cache_data = {
                'data': data,
                'metadata': metadata,
                'timestamp': datetime.now().isoformat(),
                'backend': 'redis'
            }
            
            serialized_data = pickle.dumps(cache_data)
            redis_client.setex(cache_key, ttl_seconds, serialized_data)
            
            self.logger.debug(f"Redis缓存保存成功: {cache_key}")
            return True
            
        except Exception as e:
            self.logger.error(f"Redis缓存保存失败: {e}")
            return False
    
    def _load_from_redis(self, cache_key: str) -> Optional[Dict]:
        """从Redis加载数据并反序列化。

        Args:
            cache_key (str): 缓存键。

        Returns:
            Optional[Dict]: 如果在Redis中找到数据，返回反序列化后的字典，
                否则返回 None。
        """
        redis_client = self.db_manager.get_redis_client()
        if not redis_client:
            return None
        
        try:
            serialized_data = redis_client.get(cache_key)
            if not serialized_data:
                return None
            
            cache_data = pickle.loads(serialized_data)
            
            # 转换时间戳
            if isinstance(cache_data['timestamp'], str):
                cache_data['timestamp'] = datetime.fromisoformat(cache_data['timestamp'])
            
            self.logger.debug(f"Redis缓存加载成功: {cache_key}")
            return cache_data
            
        except Exception as e:
            self.logger.error(f"Redis缓存加载失败: {e}")
            return None
    
    def _save_to_mongodb(self, cache_key: str, data: Any, metadata: Dict, ttl_seconds: int) -> bool:
        """将数据（特别是DataFrame）高效地保存到MongoDB。

        对 pandas DataFrame 进行特殊处理，以JSON格式存储，其他类型
        的数据则使用pickle序列化。

        Args:
            cache_key (str): 缓存键，用作MongoDB文档的 `_id`。
            data (Any): 要缓存的数据。
            metadata (Dict): 数据的元信息。
            ttl_seconds (int): 用于计算 `expires_at` 字段，以支持TTL索引。

        Returns:
            bool: 如果保存成功，返回 True，否则返回 False。
        """
        mongodb_client = self.db_manager.get_mongodb_client()
        if not mongodb_client:
            return False
        
        try:
            db = mongodb_client.tradingagents
            collection = db.cache
            
            # 序列化数据
            if isinstance(data, pd.DataFrame):
                serialized_data = data.to_json()
                data_type = 'dataframe'
            else:
                serialized_data = pickle.dumps(data).hex()
                data_type = 'pickle'
            
            cache_doc = {
                '_id': cache_key,
                'data': serialized_data,
                'data_type': data_type,
                'metadata': metadata,
                'timestamp': datetime.now(),
                'expires_at': datetime.now() + timedelta(seconds=ttl_seconds),
                'backend': 'mongodb'
            }
            
            collection.replace_one({'_id': cache_key}, cache_doc, upsert=True)
            
            self.logger.debug(f"MongoDB缓存保存成功: {cache_key}")
            return True
            
        except Exception as e:
            self.logger.error(f"MongoDB缓存保存失败: {e}")
            return False
    
    def _load_from_mongodb(self, cache_key: str) -> Optional[Dict]:
        """从MongoDB加载数据，并根据存储类型进行反序列化。

        Args:
            cache_key (str): 缓存键 (文档的 `_id`)。

        Returns:
            Optional[Dict]: 如果找到并且文档未过期，返回包含数据和元信息
                的字典，否则返回 None。
        """
        mongodb_client = self.db_manager.get_mongodb_client()
        if not mongodb_client:
            return None
        
        try:
            db = mongodb_client.tradingagents
            collection = db.cache
            
            doc = collection.find_one({'_id': cache_key})
            if not doc:
                return None
            
            # 检查是否过期
            if doc.get('expires_at') and doc['expires_at'] < datetime.now():
                collection.delete_one({'_id': cache_key})
                return None
            
            # 反序列化数据
            if doc['data_type'] == 'dataframe':
                data = pd.read_json(doc['data'])
            else:
                data = pickle.loads(bytes.fromhex(doc['data']))
            
            cache_data = {
                'data': data,
                'metadata': doc['metadata'],
                'timestamp': doc['timestamp'],
                'backend': 'mongodb'
            }
            
            self.logger.debug(f"MongoDB缓存加载成功: {cache_key}")
            return cache_data
            
        except Exception as e:
            self.logger.error(f"MongoDB缓存加载失败: {e}")
            return None
    
    def save_data(self, symbol: str, data: Any, start_date: str = "", end_date: str = "", 
                  data_source: str = "default", data_type: str = "stock_data") -> str:
        """将数据保存到配置的缓存后端。

        如果主要后端保存失败且启用了降级，会自动尝试保存到文件缓存。

        Args:
            symbol (str): 股票代码。
            data (Any): 要缓存的数据。
            start_date (str, optional): 数据开始日期。默认为 ""。
            end_date (str, optional): 数据结束日期。默认为 ""。
            data_source (str, optional): 数据来源。默认为 "default"。
            data_type (str, optional): 数据类型。默认为 "stock_data"。

        Returns:
            str: 生成并用于保存数据的缓存键。
        """
        # 生成缓存键
        cache_key = self._get_cache_key(symbol, start_date, end_date, data_source, data_type)
        
        # 准备元数据
        metadata = {
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date,
            'data_source': data_source,
            'data_type': data_type
        }
        
        # 获取TTL
        ttl_seconds = self._get_ttl_seconds(symbol, data_type)
        
        # 根据主要后端保存
        success = False
        
        if self.primary_backend == "redis":
            success = self._save_to_redis(cache_key, data, metadata, ttl_seconds)
        elif self.primary_backend == "mongodb":
            success = self._save_to_mongodb(cache_key, data, metadata, ttl_seconds)
        elif self.primary_backend == "file":
            success = self._save_to_file(cache_key, data, metadata)
        
        # 如果主要后端失败，使用降级策略
        if not success and self.fallback_enabled:
            self.logger.warning(f"主要后端({self.primary_backend})保存失败，使用文件缓存降级")
            success = self._save_to_file(cache_key, data, metadata)
        
        if success:
            self.logger.info(f"数据缓存成功: {symbol} -> {cache_key} (后端: {self.primary_backend})")
        else:
            self.logger.error(f"数据缓存失败: {symbol}")
        
        return cache_key
    
    def load_data(self, cache_key: str) -> Optional[Any]:
        """根据配置的策略从缓存中加载数据。

        如果从主要后端加载失败且启用了降级，会自动尝试从文件缓存加载。

        Args:
            cache_key (str): 要加载的数据的缓存键。

        Returns:
            Optional[Any]: 如果找到有效的缓存数据，则返回数据本身，
                否则返回 None。
        """
        cache_data = None
        
        # 根据主要后端加载
        if self.primary_backend == "redis":
            cache_data = self._load_from_redis(cache_key)
        elif self.primary_backend == "mongodb":
            cache_data = self._load_from_mongodb(cache_key)
        elif self.primary_backend == "file":
            cache_data = self._load_from_file(cache_key)
        
        # 如果主要后端失败，尝试降级
        if not cache_data and self.fallback_enabled:
            self.logger.debug(f"主要后端({self.primary_backend})加载失败，尝试文件缓存")
            cache_data = self._load_from_file(cache_key)
        
        if not cache_data:
            return None
        
        # 检查缓存是否有效（仅对文件缓存，数据库缓存有自己的TTL机制）
        if cache_data.get('backend') == 'file':
            symbol = cache_data['metadata'].get('symbol', '')
            data_type = cache_data['metadata'].get('data_type', 'stock_data')
            ttl_seconds = self._get_ttl_seconds(symbol, data_type)
            
            if not self._is_cache_valid(cache_data['timestamp'], ttl_seconds):
                self.logger.debug(f"文件缓存已过期: {cache_key}")
                return None
        
        return cache_data['data']
    
    def find_cached_data(self, symbol: str, start_date: str = "", end_date: str = "", 
                        data_source: str = "default", data_type: str = "stock_data") -> Optional[str]:
        """查找是否存在给定参数的有效缓存数据。

        Args:
            symbol (str): 股票代码。
            start_date (str, optional): 数据开始日期。默认为 ""。
            end_date (str, optional): 数据结束日期。默认为 ""。
            data_source (str, optional): 数据来源。默认为 "default"。
            data_type (str, optional): 数据类型。默认为 "stock_data"。

        Returns:
            Optional[str]: 如果找到有效的缓存数据，返回其缓存键，
                否则返回 None。
        """
        cache_key = self._get_cache_key(symbol, start_date, end_date, data_source, data_type)
        
        # 检查缓存是否存在且有效
        if self.load_data(cache_key) is not None:
            return cache_key
        
        return None
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取并返回缓存系统的综合统计信息。

        包括后端配置、数据库可用性、以及各个缓存后端的具体统计
        （如键数量、内存使用等）。

        Returns:
            Dict[str, Any]: 包含缓存统计信息的字典。
        """
        stats = {
            'primary_backend': self.primary_backend,
            'fallback_enabled': self.fallback_enabled,
            'database_available': self.db_manager.is_database_available(),
            'mongodb_available': self.db_manager.is_mongodb_available(),
            'redis_available': self.db_manager.is_redis_available(),
            'file_cache_directory': str(self.cache_dir),
            'file_cache_count': len(list(self.cache_dir.glob("*.pkl"))),
        }
        
        # Redis统计
        redis_client = self.db_manager.get_redis_client()
        if redis_client:
            try:
                redis_info = redis_client.info()
                stats['redis_memory_used'] = redis_info.get('used_memory_human', 'N/A')
                stats['redis_keys'] = redis_client.dbsize()
            except:
                stats['redis_status'] = 'Error'
        
        # MongoDB统计
        mongodb_client = self.db_manager.get_mongodb_client()
        if mongodb_client:
            try:
                db = mongodb_client.tradingagents
                stats['mongodb_cache_count'] = db.cache.count_documents({})
            except:
                stats['mongodb_status'] = 'Error'
        
        return stats
    
    def clear_expired_cache(self):
        """手动清理文件缓存中的过期数据。

        注意：Redis和MongoDB有自己的自动过期机制，此方法主要
        用于维护文件缓存。
        """
        self.logger.info("开始清理过期缓存...")
        
        # 清理文件缓存
        cleared_files = 0
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                symbol = cache_data['metadata'].get('symbol', '')
                data_type = cache_data['metadata'].get('data_type', 'stock_data')
                ttl_seconds = self._get_ttl_seconds(symbol, data_type)
                
                if not self._is_cache_valid(cache_data['timestamp'], ttl_seconds):
                    cache_file.unlink()
                    cleared_files += 1
                    
            except Exception as e:
                self.logger.error(f"清理缓存文件失败 {cache_file}: {e}")
        
        self.logger.info(f"文件缓存清理完成，删除 {cleared_files} 个过期文件")
        
        # MongoDB会自动清理过期文档（通过expires_at字段）
        # Redis会自动清理过期键


# 全局缓存系统实例
_cache_system = None

def get_cache_system() -> AdaptiveCacheSystem:
    """获取全局唯一的自适应缓存系统实例（Singleton模式）。

    Returns:
        AdaptiveCacheSystem: 缓存系统的单例。
    """
    global _cache_system
    if _cache_system is None:
        _cache_system = AdaptiveCacheSystem()
    return _cache_system
