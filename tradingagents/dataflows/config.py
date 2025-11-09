import tradingagents.default_config as default_config
from typing import Dict, Optional
from tradingagents.config.config_manager import config_manager

# Use default config but allow it to be overridden
_config: Optional[Dict] = None
DATA_DIR: Optional[str] = None


def initialize_config():
    """初始化模块级配置。

    此函数负责加载默认配置，并尝试从 `config_manager` 中获取持久化
    的设置（特别是 `data_dir`）来覆盖默认值。它确保了 `_config` 和
    `DATA_DIR` 两个全局变量被正确初始化，并调用 `config_manager`
    来确保所需的数据目录物理上存在。
    """
    global _config, DATA_DIR
    if _config is None:
        # 优先使用配置管理器的设置
        settings = config_manager.load_settings()
        _config = default_config.DEFAULT_CONFIG.copy()
        
        # 如果配置管理器中有数据目录设置，使用它
        if settings.get("data_dir"):
            _config["data_dir"] = settings["data_dir"]
        
        DATA_DIR = _config["data_dir"]
        
        # 确保目录存在
        config_manager.ensure_directories_exist()


def set_config(config: Dict):
    """动态更新模块的配置。

    允许外部调用者传入一个字典来覆盖当前的配置项。如果 `data_dir`
    被更新，此函数还会同步调用 `config_manager` 来持久化这个改动。

    Args:
        config (Dict): 一个包含要更新的配置项的字典。
    """
    global _config, DATA_DIR
    if _config is None:
        _config = default_config.DEFAULT_CONFIG.copy()
    
    _config.update(config)
    DATA_DIR = _config["data_dir"]
    
    # 如果设置了数据目录，同时更新配置管理器
    if "data_dir" in config:
        config_manager.set_data_dir(config["data_dir"])


def get_config() -> Dict:
    """获取当前模块的完整配置信息。

    在返回配置前，它会与 `config_manager` 同步 `data_dir`，以确保
    获取到的是最新的数据目录路径。

    注意：此函数返回的配置副本中不再包含数据库相关的设置，因为这部分
    已移交 `database_manager` 统一管理，以避免配置冲突。

    Returns:
        Dict: 当前配置的副本。
    """
    if _config is None:
        initialize_config()

    # 动态获取最新的数据目录配置
    current_data_dir = config_manager.get_data_dir()
    if _config["data_dir"] != current_data_dir:
        _config["data_dir"] = current_data_dir
        global DATA_DIR
        DATA_DIR = current_data_dir

    # 注意：数据库配置现在由 tradingagents.config.database_manager 管理
    # 这里不再包含数据库配置，避免配置冲突
    config_copy = _config.copy()

    return config_copy


def get_data_dir() -> str:
    """获取数据存储的根目录路径。

    这是一个便捷函数，直接从 `config_manager` 获取最新的数据目录路径，
    确保了路径的一致性和实时性。

    Returns:
        str: 数据目录的绝对路径。
    """
    return config_manager.get_data_dir()


def set_data_dir(data_dir: str):
    """设置并持久化数据存储的根目录路径。

    此函数会将新的数据目录路径更新到 `config_manager`（持久化），
    并同步更新本模块内的全局变量（`_config` 和 `DATA_DIR`）。

    Args:
        data_dir (str): 新的数据目录路径。
    """
    config_manager.set_data_dir(data_dir)
    # 更新全局变量
    global _config, DATA_DIR
    if _config is None:
        initialize_config()
    _config["data_dir"] = data_dir
    DATA_DIR = data_dir


# Initialize with default config
initialize_config()
