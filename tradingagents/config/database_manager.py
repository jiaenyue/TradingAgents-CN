#!/usr/bin/env python3
"""
智能数据库管理器
自动检测MongoDB和Redis可用性，提供降级方案
使用项目现有的.env配置
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

class DatabaseManager:
    """一个智能数据库管理器，用于自动检测和管理数据库连接。

    该类的核心功能是在初始化时自动检测 MongoDB 和 Redis 服务的可用性，
    并根据检测结果智能地选择最合适的缓存后端（优先使用 Redis，然后是
    MongoDB，最后降级到文件系统）。它从环境变量中加载配置，并提供
    了获取数据库客户端、检查服务状态和获取配置信息的接口。

    Attributes:
        mongodb_available (bool): 如果 MongoDB 连接可用，则为 True。
        redis_available (bool): 如果 Redis 连接可用，则为 True。
        mongodb_client: 一个 `pymongo.MongoClient` 实例，如果连接成功。
        redis_client: 一个 `redis.Redis` 实例，如果连接成功。
        primary_backend (str): 自动选择的主要缓存后端 ('redis', 'mongodb', or 'file')。
    """

    def __init__(self):
        """初始化 DatabaseManager。

        在初始化过程中，它会执行以下操作：
        1. 加载 .env 文件中的数据库配置。
        2. 检测 MongoDB 和 Redis 的可用性。
        3. 根据检测结果初始化数据库连接。
        4. 设置主要的缓存后端。
        """
        self.logger = logging.getLogger(__name__)

        # 加载.env配置
        self._load_env_config()

        # 数据库连接状态
        self.mongodb_available = False
        self.redis_available = False
        self.mongodb_client = None
        self.redis_client = None

        # 检测数据库可用性
        self._detect_databases()

        # 初始化连接
        self._initialize_connections()

        self.logger.info(f"数据库管理器初始化完成 - MongoDB: {self.mongodb_available}, Redis: {self.redis_available}")
    
    def _load_env_config(self):
        """从.env文件加载配置"""
        # 尝试加载python-dotenv
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            self.logger.info("python-dotenv未安装，直接读取环境变量")

        # 使用强健的布尔值解析（兼容Python 3.13+）
        from .env_utils import parse_bool_env
        self.mongodb_enabled = parse_bool_env("MONGODB_ENABLED", False)
        self.redis_enabled = parse_bool_env("REDIS_ENABLED", False)

        # 从环境变量读取MongoDB配置
        self.mongodb_config = {
            "enabled": self.mongodb_enabled,
            "host": os.getenv("MONGODB_HOST", "localhost"),
            "port": int(os.getenv("MONGODB_PORT", "27017")),
            "username": os.getenv("MONGODB_USERNAME"),
            "password": os.getenv("MONGODB_PASSWORD"),
            "database": os.getenv("MONGODB_DATABASE", "tradingagents"),
            "auth_source": os.getenv("MONGODB_AUTH_SOURCE", "admin"),
            "timeout": 2000
        }

        # 从环境变量读取Redis配置
        self.redis_config = {
            "enabled": self.redis_enabled,
            "host": os.getenv("REDIS_HOST", "localhost"),
            "port": int(os.getenv("REDIS_PORT", "6379")),
            "password": os.getenv("REDIS_PASSWORD"),
            "db": int(os.getenv("REDIS_DB", "0")),
            "timeout": 2
        }

        self.logger.info(f"MongoDB启用: {self.mongodb_enabled}")
        self.logger.info(f"Redis启用: {self.redis_enabled}")
        if self.mongodb_enabled:
            self.logger.info(f"MongoDB配置: {self.mongodb_config['host']}:{self.mongodb_config['port']}")
        if self.redis_enabled:
            self.logger.info(f"Redis配置: {self.redis_config['host']}:{self.redis_config['port']}")
    

    
    def _detect_mongodb(self) -> Tuple[bool, str]:
        """检测MongoDB是否可用"""
        # 首先检查是否启用
        if not self.mongodb_enabled:
            return False, "MongoDB未启用 (MONGODB_ENABLED=false)"

        try:
            import pymongo
            from pymongo import MongoClient

            # 构建连接参数
            connect_kwargs = {
                "host": self.mongodb_config["host"],
                "port": self.mongodb_config["port"],
                "serverSelectionTimeoutMS": self.mongodb_config["timeout"],
                "connectTimeoutMS": self.mongodb_config["timeout"]
            }

            # 如果有用户名和密码，添加认证
            if self.mongodb_config["username"] and self.mongodb_config["password"]:
                connect_kwargs.update({
                    "username": self.mongodb_config["username"],
                    "password": self.mongodb_config["password"],
                    "authSource": self.mongodb_config["auth_source"]
                })

            client = MongoClient(**connect_kwargs)

            # 测试连接
            client.server_info()
            client.close()

            return True, "MongoDB连接成功"

        except ImportError:
            return False, "pymongo未安装"
        except Exception as e:
            return False, f"MongoDB连接失败: {str(e)}"
    
    def _detect_redis(self) -> Tuple[bool, str]:
        """检测Redis是否可用"""
        # 首先检查是否启用
        if not self.redis_enabled:
            return False, "Redis未启用 (REDIS_ENABLED=false)"

        try:
            import redis

            # 构建连接参数
            connect_kwargs = {
                "host": self.redis_config["host"],
                "port": self.redis_config["port"],
                "db": self.redis_config["db"],
                "socket_timeout": self.redis_config["timeout"],
                "socket_connect_timeout": self.redis_config["timeout"]
            }

            # 如果有密码，添加密码
            if self.redis_config["password"]:
                connect_kwargs["password"] = self.redis_config["password"]

            client = redis.Redis(**connect_kwargs)

            # 测试连接
            client.ping()

            return True, "Redis连接成功"

        except ImportError:
            return False, "redis未安装"
        except Exception as e:
            return False, f"Redis连接失败: {str(e)}"
    
    def _detect_databases(self):
        """检测所有数据库"""
        self.logger.info("开始检测数据库可用性...")
        
        # 检测MongoDB
        mongodb_available, mongodb_msg = self._detect_mongodb()
        self.mongodb_available = mongodb_available
        
        if mongodb_available:
            self.logger.info(f"✅ MongoDB: {mongodb_msg}")
        else:
            self.logger.info(f"❌ MongoDB: {mongodb_msg}")
        
        # 检测Redis
        redis_available, redis_msg = self._detect_redis()
        self.redis_available = redis_available
        
        if redis_available:
            self.logger.info(f"✅ Redis: {redis_msg}")
        else:
            self.logger.info(f"❌ Redis: {redis_msg}")
        
        # 更新配置
        self._update_config_based_on_detection()
    
    def _update_config_based_on_detection(self):
        """根据检测结果更新配置"""
        # 确定缓存后端
        if self.redis_available:
            self.primary_backend = "redis"
        elif self.mongodb_available:
            self.primary_backend = "mongodb"
        else:
            self.primary_backend = "file"

        self.logger.info(f"主要缓存后端: {self.primary_backend}")
    
    def _initialize_connections(self):
        """初始化数据库连接"""
        # 初始化MongoDB连接
        if self.mongodb_available:
            try:
                import pymongo

                # 构建连接参数
                connect_kwargs = {
                    "host": self.mongodb_config["host"],
                    "port": self.mongodb_config["port"],
                    "serverSelectionTimeoutMS": self.mongodb_config["timeout"]
                }

                # 如果有用户名和密码，添加认证
                if self.mongodb_config["username"] and self.mongodb_config["password"]:
                    connect_kwargs.update({
                        "username": self.mongodb_config["username"],
                        "password": self.mongodb_config["password"],
                        "authSource": self.mongodb_config["auth_source"]
                    })

                self.mongodb_client = pymongo.MongoClient(**connect_kwargs)
                self.logger.info("MongoDB客户端初始化成功")
            except Exception as e:
                self.logger.error(f"MongoDB客户端初始化失败: {e}")
                self.mongodb_available = False

        # 初始化Redis连接
        if self.redis_available:
            try:
                import redis

                # 构建连接参数
                connect_kwargs = {
                    "host": self.redis_config["host"],
                    "port": self.redis_config["port"],
                    "db": self.redis_config["db"],
                    "socket_timeout": self.redis_config["timeout"]
                }

                # 如果有密码，添加密码
                if self.redis_config["password"]:
                    connect_kwargs["password"] = self.redis_config["password"]

                self.redis_client = redis.Redis(**connect_kwargs)
                self.logger.info("Redis客户端初始化成功")
            except Exception as e:
                self.logger.error(f"Redis客户端初始化失败: {e}")
                self.redis_available = False
    
    def get_mongodb_client(self):
        """返回已初始化的 MongoDB 客户端实例。

        Returns:
            如果 MongoDB 可用且客户端已成功初始化，则返回 `pymongo.MongoClient`
            实例；否则返回 None。
        """
        if self.mongodb_available and self.mongodb_client:
            return self.mongodb_client
        return None
    
    def get_redis_client(self):
        """返回已初始化的 Redis 客户端实例。

        Returns:
            如果 Redis 可用且客户端已成功初始化，则返回 `redis.Redis` 实例；
            否则返回 None。
        """
        if self.redis_available and self.redis_client:
            return self.redis_client
        return None
    
    def is_mongodb_available(self) -> bool:
        """检查 MongoDB 是否被检测为可用。

        Returns:
            如果 MongoDB 可用，则为 True；否则为 False。
        """
        return self.mongodb_available
    
    def is_redis_available(self) -> bool:
        """检查 Redis 是否被检测为可用。

        Returns:
            如果 Redis 可用，则为 True；否则为 False。
        """
        return self.redis_available
    
    def is_database_available(self) -> bool:
        """检查是否有任何数据库（MongoDB 或 Redis）可用。

        Returns:
            如果至少有一个数据库可用，则为 True；否则为 False。
        """
        return self.mongodb_available or self.redis_available
    
    def get_cache_backend(self) -> str:
        """获取当前被选为主要缓存后端的系统名称。

        Returns:
            一个字符串，值为 'redis'、'mongodb' 或 'file'。
        """
        return self.primary_backend

    def get_config(self) -> Dict[str, Any]:
        """返回包含当前数据库配置和状态的字典。

        Returns:
            一个包含 MongoDB 和 Redis 配置、可用性状态以及
            主要后端信息的字典。
        """
        return {
            "mongodb": self.mongodb_config,
            "redis": self.redis_config,
            "primary_backend": self.primary_backend,
            "mongodb_available": self.mongodb_available,
            "redis_available": self.redis_available
        }

    def get_status_report(self) -> Dict[str, Any]:
        """生成并返回一份关于数据库连接状态的详细报告。

        Returns:
            一个包含各数据库可用性、连接信息和当前缓存后端等
            信息的字典。
        """
        return {
            "database_available": self.is_database_available(),
            "mongodb": {
                "available": self.mongodb_available,
                "host": self.mongodb_config["host"],
                "port": self.mongodb_config["port"]
            },
            "redis": {
                "available": self.redis_available,
                "host": self.redis_config["host"],
                "port": self.redis_config["port"]
            },
            "cache_backend": self.get_cache_backend(),
            "fallback_enabled": True  # 总是启用降级
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存系统的统计信息，主要针对 Redis。

        如果 Redis 可用，此方法将尝试获取其数据库中的键数量和内存使用情况。

        Returns:
            一个包含缓存统计信息的字典。
        """
        stats = {
            "mongodb_available": self.mongodb_available,
            "redis_available": self.redis_available,
            "redis_keys": 0,
            "redis_memory": "N/A"
        }

        # Redis统计
        if self.redis_available and self.redis_client:
            try:
                info = self.redis_client.info()
                stats["redis_keys"] = self.redis_client.dbsize()
                stats["redis_memory"] = info.get("used_memory_human", "N/A")
            except Exception as e:
                self.logger.error(f"获取Redis统计失败: {e}")

        return stats

    def cache_clear_pattern(self, pattern: str) -> int:
        """根据指定的模式字符串清理 Redis 缓存中的键。

        此功能仅在 Redis 可用时生效。

        Args:
            pattern: 用于匹配键的模式，例如 'cache:prefix:*'。

        Returns:
            被成功删除的键的数量。
        """
        cleared_count = 0

        if self.redis_available and self.redis_client:
            try:
                keys = self.redis_client.keys(pattern)
                if keys:
                    cleared_count += self.redis_client.delete(*keys)
            except Exception as e:
                self.logger.error(f"Redis缓存清理失败: {e}")

        return cleared_count


# 全局数据库管理器实例
_database_manager = None

def get_database_manager() -> DatabaseManager:
    """获取全局数据库管理器实例"""
    global _database_manager
    if _database_manager is None:
        _database_manager = DatabaseManager()
    return _database_manager

def is_mongodb_available() -> bool:
    """检查MongoDB是否可用"""
    return get_database_manager().is_mongodb_available()

def is_redis_available() -> bool:
    """检查Redis是否可用"""
    return get_database_manager().is_redis_available()

def get_cache_backend() -> str:
    """获取当前缓存后端"""
    return get_database_manager().get_cache_backend()

def get_mongodb_client():
    """获取MongoDB客户端"""
    return get_database_manager().get_mongodb_client()

def get_redis_client():
    """获取Redis客户端"""
    return get_database_manager().get_redis_client()
