#!/usr/bin/env python3
"""
MongoDB存储适配器
用于将token使用记录存储到MongoDB数据库
"""

import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import asdict
from .config_manager import UsageRecord

# 导入日志模块
from tradingagents.utils.logging_manager import get_logger
logger = get_logger('agents')

try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    MongoClient = None


class MongoDBStorage:
    """一个用于将 `UsageRecord` 对象持久化到 MongoDB 数据库的适配器。

    该类封装了与 MongoDB 的所有交互，包括连接管理、数据插入、查询、
    聚合统计以及数据清理。它会自动处理连接失败的情况，并能在连接成功时
    自动创建索引以优化查询性能。

    Attributes:
        connection_string (str): 用于连接 MongoDB 的 URI。
        database_name (str): 数据库名称。
        collection_name (str): 集合（表）名称，默认为 'token_usage'。
        client: `pymongo.MongoClient` 实例。
        db: 数据库对象。
        collection: 集合对象。
    """
    
    def __init__(self, connection_string: str = None, database_name: str = "tradingagents"):
        """初始化 MongoDBStorage 适配器。

        在初始化时，它会检查 `pymongo` 库是否已安装，并尝试使用提供
        的或从环境变量中获取的连接字符串来建立与数据库的连接。

        Args:
            connection_string: MongoDB 连接字符串。如果为 None，将尝试从
                               `MONGODB_CONNECTION_STRING` 环境变量中获取。
            database_name: 要使用的数据库名称，默认为 'tradingagents'。

        Raises:
            ImportError: 如果 `pymongo` 未安装。
            ValueError: 如果连接字符串既未通过参数提供，也未在环境变量中设置。
        """
        if not MONGODB_AVAILABLE:
            raise ImportError("pymongo is not installed. Please install it with: pip install pymongo")
        
        # 修复硬编码问题 - 如果没有提供连接字符串且环境变量也未设置，则抛出错误
        self.connection_string = connection_string or os.getenv("MONGODB_CONNECTION_STRING")
        if not self.connection_string:
            raise ValueError(
                "MongoDB连接字符串未配置。请通过以下方式之一进行配置：\n"
                "1. 设置环境变量 MONGODB_CONNECTION_STRING\n"
                "2. 在初始化时传入 connection_string 参数\n"
                "例如: MONGODB_CONNECTION_STRING=mongodb://localhost:27017/"
            )
        
        self.database_name = database_name
        self.collection_name = "token_usage"
        
        self.client = None
        self.db = None
        self.collection = None
        self._connected = False
        
        # 尝试连接
        self._connect()
    
    def _connect(self):
        """连接到MongoDB"""
        try:
            self.client = MongoClient(
                self.connection_string,
                serverSelectionTimeoutMS=5000  # 5秒超时
            )
            # 测试连接
            self.client.admin.command('ping')
            
            self.db = self.client[self.database_name]
            self.collection = self.db[self.collection_name]
            
            # 创建索引以提高查询性能
            self._create_indexes()
            
            self._connected = True
            logger.info(f"✅ MongoDB连接成功: {self.database_name}.{self.collection_name}")
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"❌ MongoDB连接失败: {e}")
            logger.info(f"将使用本地JSON文件存储")
            self._connected = False
        except Exception as e:
            logger.error(f"❌ MongoDB初始化失败: {e}")
            self._connected = False
    
    def _create_indexes(self):
        """创建数据库索引"""
        try:
            # 创建复合索引
            self.collection.create_index([
                ("timestamp", -1),  # 按时间倒序
                ("provider", 1),
                ("model_name", 1)
            ])
            
            # 创建会话ID索引
            self.collection.create_index("session_id")
            
            # 创建分析类型索引
            self.collection.create_index("analysis_type")
            
        except Exception as e:
            logger.error(f"创建MongoDB索引失败: {e}")
    
    def is_connected(self) -> bool:
        """检查当前是否成功连接到 MongoDB。

        Returns:
            如果连接处于活动状态，则返回 True；否则返回 False。
        """
        return self._connected
    
    def save_usage_record(self, record: UsageRecord) -> bool:
        """将一个 `UsageRecord` 对象保存到 MongoDB 集合中。

        在插入前，记录会先被转换为字典，并添加一个 `_created_at` 时间戳。

        Args:
            record: 需要保存的 `UsageRecord` 对象。

        Returns:
            如果记录成功插入，则返回 True；否则返回 False。
        """
        if not self._connected:
            return False
        
        try:
            # 转换为字典格式
            record_dict = asdict(record)
            
            # 添加MongoDB特有的字段
            record_dict['_created_at'] = datetime.now()
            
            # 插入记录
            result = self.collection.insert_one(record_dict)
            
            if result.inserted_id:
                return True
            else:
                logger.error(f"MongoDB插入失败：未返回插入ID")
                return False
                
        except Exception as e:
            logger.error(f"保存记录到MongoDB失败: {e}")
            return False
    
    def load_usage_records(self, limit: int = 10000, days: int = None) -> List[UsageRecord]:
        """从 MongoDB 加载使用记录。

        可以按时间范围（最近 N 天）和数量限制进行查询。

        Args:
            limit: 返回记录的最大数量。
            days: 可选参数，用于限定只查询最近指定天数内的记录。

        Returns:
            一个 `UsageRecord` 对象的列表。如果查询失败或无连接，返回空列表。
        """
        if not self._connected:
            return []
        
        try:
            # 构建查询条件
            query = {}
            if days:
                from datetime import timedelta
                cutoff_date = datetime.now() - timedelta(days=days)
                query['timestamp'] = {'$gte': cutoff_date.isoformat()}
            
            # 查询记录，按时间倒序
            cursor = self.collection.find(query).sort('timestamp', -1).limit(limit)
            
            records = []
            for doc in cursor:
                # 移除MongoDB特有的字段
                doc.pop('_id', None)
                doc.pop('_created_at', None)
                
                # 转换为UsageRecord对象
                try:
                    record = UsageRecord(**doc)
                    records.append(record)
                except Exception as e:
                    logger.error(f"解析记录失败: {e}, 记录: {doc}")
                    continue
            
            return records
            
        except Exception as e:
            logger.error(f"从MongoDB加载记录失败: {e}")
            return []
    
    def get_usage_statistics(self, days: int = 30) -> Dict[str, Any]:
        """使用 MongoDB 聚合管道计算指定时间范围内的总体使用统计。

        Args:
            days: 要统计的天数。

        Returns:
            一个包含总成本、总 token 数和总请求数的字典。如果查询失败，
            返回一个包含零值的字典。
        """
        if not self._connected:
            return {}
        
        try:
            from datetime import timedelta
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # 聚合查询
            pipeline = [
                {
                    '$match': {
                        'timestamp': {'$gte': cutoff_date.isoformat()}
                    }
                },
                {
                    '$group': {
                        '_id': None,
                        'total_cost': {'$sum': '$cost'},
                        'total_input_tokens': {'$sum': '$input_tokens'},
                        'total_output_tokens': {'$sum': '$output_tokens'},
                        'total_requests': {'$sum': 1}
                    }
                }
            ]
            
            result = list(self.collection.aggregate(pipeline))
            
            if result:
                stats = result[0]
                return {
                    'period_days': days,
                    'total_cost': round(stats.get('total_cost', 0), 4),
                    'total_input_tokens': stats.get('total_input_tokens', 0),
                    'total_output_tokens': stats.get('total_output_tokens', 0),
                    'total_requests': stats.get('total_requests', 0)
                }
            else:
                return {
                    'period_days': days,
                    'total_cost': 0,
                    'total_input_tokens': 0,
                    'total_output_tokens': 0,
                    'total_requests': 0
                }
                
        except Exception as e:
            logger.error(f"获取MongoDB统计失败: {e}")
            return {}
    
    def get_provider_statistics(self, days: int = 30) -> Dict[str, Dict[str, Any]]:
        """按 LLM 供应商分组，统计各自的使用情况。

        Args:
            days: 要统计的天数。

        Returns:
            一个字典，其键为供应商名称，值为包含该供应商成本、token 数
            和请求数的统计字典。
        """
        if not self._connected:
            return {}
        
        try:
            from datetime import timedelta
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # 按供应商聚合
            pipeline = [
                {
                    '$match': {
                        'timestamp': {'$gte': cutoff_date.isoformat()}
                    }
                },
                {
                    '$group': {
                        '_id': '$provider',
                        'cost': {'$sum': '$cost'},
                        'input_tokens': {'$sum': '$input_tokens'},
                        'output_tokens': {'$sum': '$output_tokens'},
                        'requests': {'$sum': 1}
                    }
                }
            ]
            
            results = list(self.collection.aggregate(pipeline))
            
            provider_stats = {}
            for result in results:
                provider = result['_id']
                provider_stats[provider] = {
                    'cost': round(result.get('cost', 0), 4),
                    'input_tokens': result.get('input_tokens', 0),
                    'output_tokens': result.get('output_tokens', 0),
                    'requests': result.get('requests', 0)
                }
            
            return provider_stats
            
        except Exception as e:
            logger.error(f"获取供应商统计失败: {e}")
            return {}
    
    def cleanup_old_records(self, days: int = 90) -> int:
        """从数据库中删除早于指定天数的旧记录。

        Args:
            days: 记录保留的最大天数。早于这个天数的记录将被删除。

        Returns:
            被删除的记录数量。
        """
        if not self._connected:
            return 0
        
        try:
            from datetime import timedelta

            cutoff_date = datetime.now() - timedelta(days=days)
            
            result = self.collection.delete_many({
                'timestamp': {'$lt': cutoff_date.isoformat()}
            })
            
            deleted_count = result.deleted_count
            if deleted_count > 0:
                logger.info(f"清理了 {deleted_count} 条超过 {days} 天的记录")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"清理旧记录失败: {e}")
            return 0
    
    def close(self):
        """安全地关闭与 MongoDB 的连接。"""
        if self.client:
            self.client.close()
            self._connected = False
            logger.info(f"MongoDB连接已关闭")