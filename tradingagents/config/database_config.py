#!/usr/bin/env python3
"""
数据库配置管理模块
统一管理MongoDB和Redis的连接配置
"""

import os
from typing import Dict, Any, Optional


class DatabaseConfig:
    """提供一个集中管理和访问数据库连接配置的静态类。

    该类从环境变量中读取 MongoDB 和 Redis 的配置信息，确保了配置与
    代码的分离，提高了安全性和灵活性。所有方法均为静态方法，无需
    实例化即可直接调用。
    """
    
    @staticmethod
    def get_mongodb_config() -> Dict[str, Any]:
        """从环境变量中获取 MongoDB 的连接配置。

        它主要依赖 `MONGODB_CONNECTION_STRING` 环境变量。如果该变量
        未设置，将抛出 `ValueError` 异常，强制要求进行配置。

        Returns:
            一个包含 MongoDB 连接参数的字典，包括 'connection_string',
            'database', 和 'auth_source'。

        Raises:
            ValueError: 如果 `MONGODB_CONNECTION_STRING` 环境变量未设置。
        """
        connection_string = os.getenv('MONGODB_CONNECTION_STRING')
        if not connection_string:
            raise ValueError(
                "MongoDB连接字符串未配置。请设置环境变量 MONGODB_CONNECTION_STRING\n"
                "例如: MONGODB_CONNECTION_STRING=mongodb://localhost:27017/"
            )
        
        return {
            'connection_string': connection_string,
            'database': os.getenv('MONGODB_DATABASE', 'tradingagents'),
            'auth_source': os.getenv('MONGODB_AUTH_SOURCE', 'admin')
        }
    
    @staticmethod
    def get_redis_config() -> Dict[str, Any]:
        """从环境变量中获取 Redis 的连接配置。

        此方法支持两种配置方式，并优先使用连接字符串：
        1. `REDIS_CONNECTION_STRING`: 一个完整的 Redis 连接 URL。
        2. `REDIS_HOST` 和 `REDIS_PORT`: 分别指定主机和端口。

        如果两种方式均未提供完整的配置信息，将抛出 `ValueError`。

        Returns:
            一个包含 Redis 连接参数的字典。

        Raises:
            ValueError: 如果必要的 Redis 连接环境变量均未设置。
        """
        # 优先使用连接字符串
        connection_string = os.getenv('REDIS_CONNECTION_STRING')
        if connection_string:
            return {
                'connection_string': connection_string,
                'database': int(os.getenv('REDIS_DATABASE', 0))
            }
        
        # 使用分离的配置参数
        host = os.getenv('REDIS_HOST')
        port = os.getenv('REDIS_PORT')
        
        if not host or not port:
            raise ValueError(
                "Redis连接配置未完整设置。请设置以下环境变量之一：\n"
                "1. REDIS_CONNECTION_STRING=redis://localhost:6379/0\n"
                "2. REDIS_HOST + REDIS_PORT (例如: REDIS_HOST=localhost, REDIS_PORT=6379)"
            )
        
        return {
            'host': host,
            'port': int(port),
            'password': os.getenv('REDIS_PASSWORD'),
            'database': int(os.getenv('REDIS_DATABASE', 0))
        }
    
    @staticmethod
    def validate_config() -> Dict[str, bool]:
        """检查 MongoDB 和 Redis 的配置是否都已正确设置。

        通过尝试调用各自的 `get_*_config` 方法并捕获可能出现的
        `ValueError` 来判断配置的有效性。

        Returns:
            一个字典，其中 'mongodb_valid' 和 'redis_valid' 键
            对应的值（True/False）表示相应数据库的配置是否有效。
        """
        result = {
            'mongodb_valid': False,
            'redis_valid': False
        }
        
        try:
            DatabaseConfig.get_mongodb_config()
            result['mongodb_valid'] = True
        except ValueError:
            pass
        
        try:
            DatabaseConfig.get_redis_config()
            result['redis_valid'] = True
        except ValueError:
            pass
        
        return result
    
    @staticmethod
    def get_config_status() -> str:
        """提供一个用户友好的字符串，描述当前数据库配置的总体状态。

        Returns:
            一个描述配置状态的字符串，例如 "✅ 所有数据库配置正常" 或
            "❌ 数据库配置缺失，请检查环境变量"。
        """
        validation = DatabaseConfig.validate_config()
        
        if validation['mongodb_valid'] and validation['redis_valid']:
            return "✅ 所有数据库配置正常"
        elif validation['mongodb_valid']:
            return "⚠️ MongoDB配置正常，Redis配置缺失"
        elif validation['redis_valid']:
            return "⚠️ Redis配置正常，MongoDB配置缺失"
        else:
            return "❌ 数据库配置缺失，请检查环境变量"