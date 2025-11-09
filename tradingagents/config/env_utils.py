#!/usr/bin/env python3
"""
环境变量解析工具
提供兼容Python 3.13+的强健环境变量解析功能
"""

import os
from typing import Any, Union, Optional


def parse_bool_env(env_var: str, default: bool = False) -> bool:
    """健壮地将环境变量解析为布尔值。

    此函数能够识别多种表示“真”或“假”的字符串，例如 'true', '1',
    'yes', 'on' 等，不区分大小写。如果环境变量未设置，或者其值
    无法被明确解析为布尔值，将返回指定的默认值。

    Args:
        env_var: 要解析的环境变量的名称。
        default: 当环境变量未设置或无法解析时返回的默认布尔值。

    Returns:
        解析出的布尔值，或在无法解析时返回默认值。
    """
    value = os.getenv(env_var)
    
    if value is None:
        return default
    
    # 转换为字符串并去除空白
    value_str = str(value).strip()
    
    if not value_str:
        return default
    
    # 转换为小写进行比较
    value_lower = value_str.lower()
    
    # 真值列表
    true_values = {
        'true', '1', 'yes', 'on', 'enable', 'enabled', 
        't', 'y', 'ok', 'okay'
    }
    
    # 假值列表
    false_values = {
        'false', '0', 'no', 'off', 'disable', 'disabled',
        'f', 'n', 'none', 'null', 'nil'
    }
    
    if value_lower in true_values:
        return True
    elif value_lower in false_values:
        return False
    else:
        # 如果无法识别，记录警告并返回默认值
        print(f"⚠️ 无法解析环境变量 {env_var}='{value}'，使用默认值 {default}")
        return default


def parse_int_env(env_var: str, default: int = 0) -> int:
    """将环境变量解析为整数。

    如果环境变量未设置或其值不是有效的整数格式，则返回默认值。

    Args:
        env_var: 要解析的环境变量的名称。
        default: 当无法解析时返回的默认整数值。

    Returns:
        解析出的整数，或默认值。
    """
    value = os.getenv(env_var)
    
    if value is None:
        return default
    
    try:
        return int(value.strip())
    except (ValueError, AttributeError):
        print(f"⚠️ 无法解析环境变量 {env_var}='{value}' 为整数，使用默认值 {default}")
        return default


def parse_float_env(env_var: str, default: float = 0.0) -> float:
    """将环境变量解析为浮点数。

    如果环境变量未设置或其值不是有效的浮点数格式，则返回默认值。

    Args:
        env_var: 要解析的环境变量的名称。
        default: 当无法解析时返回的默认浮点数值。

    Returns:
        解析出的浮点数，或默认值。
    """
    value = os.getenv(env_var)
    
    if value is None:
        return default
    
    try:
        return float(value.strip())
    except (ValueError, AttributeError):
        print(f"⚠️ 无法解析环境变量 {env_var}='{value}' 为浮点数，使用默认值 {default}")
        return default


def parse_str_env(env_var: str, default: str = "") -> str:
    """将环境变量解析为字符串，并去除首尾空白。

    如果环境变量未设置，则返回默认值。

    Args:
        env_var: 要解析的环境变量的名称。
        default: 当环境变量未设置时返回的默认字符串。

    Returns:
        解析出的字符串，或默认值。
    """
    value = os.getenv(env_var)
    
    if value is None:
        return default
    
    return str(value).strip()


def parse_list_env(env_var: str, separator: str = ",", default: Optional[list] = None) -> list:
    """将环境变量解析为字符串列表。

    该函数使用指定的分隔符将环境变量的值分割成一个列表，并清除
    每个元素的首尾空白及过滤掉空字符串。

    Args:
        env_var: 要解析的环境变量的名称。
        separator: 用于分割字符串的分隔符，默认为逗号。
        default: 当环境变量未设置时返回的默认列表。如果为 None，则默认为空列表。

    Returns:
        解析出的字符串列表，或默认值。
    """
    if default is None:
        default = []
    
    value = os.getenv(env_var)
    
    if value is None:
        return default
    
    try:
        # 分割并去除空白
        items = [item.strip() for item in value.split(separator)]
        # 过滤空字符串
        return [item for item in items if item]
    except AttributeError:
        print(f"⚠️ 无法解析环境变量 {env_var}='{value}' 为列表，使用默认值 {default}")
        return default


def get_env_info(env_var: str) -> dict:
    """获取关于指定环境变量的详细元数据。

    Args:
        env_var: 要查询的环境变量的名称。

    Returns:
        一个字典，包含环境变量的名称、值、是否存在、是否为空、
        值的类型以及长度等信息。
    """
    value = os.getenv(env_var)
    
    return {
        'name': env_var,
        'value': value,
        'exists': value is not None,
        'empty': value is None or str(value).strip() == '',
        'type': type(value).__name__ if value is not None else 'None',
        'length': len(str(value)) if value is not None else 0
    }


def validate_required_env_vars(required_vars: list) -> dict:
    """检查一组必需的环境变量是否都已设置且不为空。

    Args:
        required_vars: 一个包含必需环境变量名称的字符串列表。

    Returns:
        一个包含验证结果的字典，其中包括一个总体的 `all_set` 标志，
        以及 `missing`（未设置）、`empty`（值为空）和 `valid`（有效）
        的变量列表。
    """
    results = {
        'all_set': True,
        'missing': [],
        'empty': [],
        'valid': []
    }
    
    for var in required_vars:
        info = get_env_info(var)
        
        if not info['exists']:
            results['missing'].append(var)
            results['all_set'] = False
        elif info['empty']:
            results['empty'].append(var)
            results['all_set'] = False
        else:
            results['valid'].append(var)
    
    return results


# 兼容性函数：保持向后兼容
def get_bool_env(env_var: str, default: bool = False) -> bool:
    """向后兼容的布尔值解析函数"""
    return parse_bool_env(env_var, default)


def get_int_env(env_var: str, default: int = 0) -> int:
    """向后兼容的整数解析函数"""
    return parse_int_env(env_var, default)


def get_str_env(env_var: str, default: str = "") -> str:
    """向后兼容的字符串解析函数"""
    return parse_str_env(env_var, default)


# 导出主要函数
__all__ = [
    'parse_bool_env',
    'parse_int_env', 
    'parse_float_env',
    'parse_str_env',
    'parse_list_env',
    'get_env_info',
    'validate_required_env_vars',
    'get_bool_env',  # 向后兼容
    'get_int_env',   # 向后兼容
    'get_str_env'    # 向后兼容
]
