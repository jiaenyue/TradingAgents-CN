#!/usr/bin/env python3
"""
配置管理器
管理API密钥、模型配置、费率设置等
"""

import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from dotenv import load_dotenv

# 导入统一日志系统
from tradingagents.utils.logging_init import get_logger

# 导入日志模块
from tradingagents.utils.logging_manager import get_logger
logger = get_logger('agents')

try:
    from .mongodb_storage import MongoDBStorage
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    MongoDBStorage = None


@dataclass
class ModelConfig:
    """模型配置"""
    provider: str  # 供应商：dashscope, openai, google, etc.
    model_name: str  # 模型名称
    api_key: str  # API密钥
    base_url: Optional[str] = None  # 自定义API地址
    max_tokens: int = 4000  # 最大token数
    temperature: float = 0.7  # 温度参数
    enabled: bool = True  # 是否启用


@dataclass
class PricingConfig:
    """定价配置"""
    provider: str  # 供应商
    model_name: str  # 模型名称
    input_price_per_1k: float  # 输入token价格（每1000个token）
    output_price_per_1k: float  # 输出token价格（每1000个token）
    currency: str = "CNY"  # 货币单位


@dataclass
class UsageRecord:
    """使用记录"""
    timestamp: str  # 时间戳
    provider: str  # 供应商
    model_name: str  # 模型名称
    input_tokens: int  # 输入token数
    output_tokens: int  # 输出token数
    cost: float  # 成本
    session_id: str  # 会话ID
    analysis_type: str  # 分析类型


class ConfigManager:
    """管理应用程序的所有配置，包括模型、定价、使用记录和常规设置。

    该类负责处理配置文件的加载、保存和初始化。它支持从环境变量、
    JSON 文件以及可选的 MongoDB 数据库中读取和写入配置。通过集中管理，
    确保了配置的一致性和可维护性。

    Attributes:
        config_dir (Path): 存放 JSON 配置文件的目录路径。
        models_file (Path): 模型配置文件（models.json）的路径。
        pricing_file (Path): 定价配置文件（pricing.json）的路径。
        usage_file (Path): 使用记录文件（usage.json）的路径。
        settings_file (Path): 常规设置文件（settings.json）的路径。
        mongodb_storage (MongoDBStorage, optional): MongoDB 存储后端实例，
                                                     如果可用且已配置。
    """
    
    def __init__(self, config_dir: str = "config"):
        """初始化 ConfigManager。

        Args:
            config_dir: 存放配置文件的目录路径，默认为 "config"。
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)

        self.models_file = self.config_dir / "models.json"
        self.pricing_file = self.config_dir / "pricing.json"
        self.usage_file = self.config_dir / "usage.json"
        self.settings_file = self.config_dir / "settings.json"

        # 加载.env文件（保持向后兼容）
        self._load_env_file()

        # 初始化MongoDB存储（如果可用）
        self.mongodb_storage = None
        self._init_mongodb_storage()

        self._init_default_configs()

    def _load_env_file(self):
        """加载.env文件（保持向后兼容）"""
        # 尝试从项目根目录加载.env文件
        project_root = Path(__file__).parent.parent.parent
        env_file = project_root / ".env"

        if env_file.exists():
            load_dotenv(env_file, override=True)

    def _get_env_api_key(self, provider: str) -> str:
        """从环境变量获取API密钥"""
        env_key_map = {
            "dashscope": "DASHSCOPE_API_KEY",
            "openai": "OPENAI_API_KEY",
            "google": "GOOGLE_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY"
        }

        env_key = env_key_map.get(provider.lower())
        if env_key:
            api_key = os.getenv(env_key, "")
            # 对OpenAI密钥进行格式验证（始终启用）
            if provider.lower() == "openai" and api_key:
                if not self.validate_openai_api_key_format(api_key):
                    logger.warning(f"⚠️ OpenAI API密钥格式不正确，将被忽略: {api_key[:10]}...")
                    return ""
            return api_key
        return ""
    
    def validate_openai_api_key_format(self, api_key: str) -> bool:
        """验证提供的字符串是否符合 OpenAI API 密钥的典型格式。

        该方法执行以下检查：
        1. 密钥必须以 "sk-" 开头。
        2. 密钥的总长度必须为 51 个字符。
        3. "sk-" 之后的部分必须由 48 个字母数字字符组成。

        Args:
            api_key: 需要验证的 OpenAI API 密钥字符串。

        Returns:
            如果密钥格式有效，则返回 True；否则返回 False。
        """
        if not api_key or not isinstance(api_key, str):
            return False
        
        # 检查是否以 'sk-' 开头
        if not api_key.startswith('sk-'):
            return False
        
        # 检查长度（OpenAI密钥通常为51个字符）
        if len(api_key) != 51:
            return False
        
        # 检查格式：sk- 后面应该是48个字符的字母数字组合
        pattern = r'^sk-[A-Za-z0-9]{48}$'
        if not re.match(pattern, api_key):
            return False
        
        return True
    
    def _init_mongodb_storage(self):
        """初始化MongoDB存储"""
        if not MONGODB_AVAILABLE:
            return
        
        # 检查是否启用MongoDB存储
        use_mongodb = os.getenv("USE_MONGODB_STORAGE", "false").lower() == "true"
        if not use_mongodb:
            return
        
        try:
            connection_string = os.getenv("MONGODB_CONNECTION_STRING")
            database_name = os.getenv("MONGODB_DATABASE_NAME", "tradingagents")
            
            self.mongodb_storage = MongoDBStorage(
                connection_string=connection_string,
                database_name=database_name
            )
            
            if self.mongodb_storage.is_connected():
                logger.info("✅ MongoDB存储已启用")
            else:
                self.mongodb_storage = None
                logger.warning("⚠️ MongoDB连接失败，将使用JSON文件存储")

        except Exception as e:
            logger.error(f"❌ MongoDB初始化失败: {e}", exc_info=True)
            self.mongodb_storage = None

    def _init_default_configs(self):
        """初始化默认配置"""
        # 默认模型配置
        if not self.models_file.exists():
            default_models = [
                ModelConfig(
                    provider="dashscope",
                    model_name="qwen-turbo",
                    api_key="",
                    max_tokens=4000,
                    temperature=0.7
                ),
                ModelConfig(
                    provider="dashscope",
                    model_name="qwen-plus-latest",
                    api_key="",
                    max_tokens=8000,
                    temperature=0.7
                ),
                ModelConfig(
                    provider="openai",
                    model_name="gpt-3.5-turbo",
                    api_key="",
                    max_tokens=4000,
                    temperature=0.7,
                    enabled=False
                ),
                ModelConfig(
                    provider="openai",
                    model_name="gpt-4",
                    api_key="",
                    max_tokens=8000,
                    temperature=0.7,
                    enabled=False
                ),
                ModelConfig(
                    provider="google",
                    model_name="gemini-2.5-pro",
                    api_key="",
                    max_tokens=4000,
                    temperature=0.7,
                    enabled=False
                ),
                ModelConfig(
                    provider="deepseek",
                    model_name="deepseek-chat",
                    api_key="",
                    max_tokens=8000,
                    temperature=0.7,
                    enabled=False
                )
            ]
            self.save_models(default_models)
        
        # 默认定价配置
        if not self.pricing_file.exists():
            default_pricing = [
                # 阿里百炼定价 (人民币)
                PricingConfig("dashscope", "qwen-turbo", 0.002, 0.006, "CNY"),
                PricingConfig("dashscope", "qwen-plus-latest", 0.004, 0.012, "CNY"),
                PricingConfig("dashscope", "qwen-max", 0.02, 0.06, "CNY"),

                # DeepSeek定价 (人民币) - 2025年最新价格
                PricingConfig("deepseek", "deepseek-chat", 0.0014, 0.0028, "CNY"),
                PricingConfig("deepseek", "deepseek-coder", 0.0014, 0.0028, "CNY"),

                # OpenAI定价 (美元)
                PricingConfig("openai", "gpt-3.5-turbo", 0.0015, 0.002, "USD"),
                PricingConfig("openai", "gpt-4", 0.03, 0.06, "USD"),
                PricingConfig("openai", "gpt-4-turbo", 0.01, 0.03, "USD"),

                # Google定价 (美元)
                PricingConfig("google", "gemini-2.5-pro", 0.00025, 0.0005, "USD"),
                PricingConfig("google", "gemini-2.5-flash", 0.00025, 0.0005, "USD"),
                PricingConfig("google", "gemini-2.0-flash", 0.00025, 0.0005, "USD"),
                PricingConfig("google", "gemini-1.5-pro", 0.00025, 0.0005, "USD"),
                PricingConfig("google", "gemini-1.5-flash", 0.00025, 0.0005, "USD"),
                PricingConfig("google", "gemini-2.5-flash-lite-preview-06-17", 0.00025, 0.0005, "USD"),
                PricingConfig("google", "gemini-pro", 0.00025, 0.0005, "USD"),
                PricingConfig("google", "gemini-pro-vision", 0.00025, 0.0005, "USD"),
            ]
            self.save_pricing(default_pricing)
        
        # 默认设置
        if not self.settings_file.exists():
            # 导入默认数据目录配置
            import os
            default_data_dir = os.path.join(os.path.expanduser("~"), "Documents", "TradingAgents", "data")
            
            default_settings = {
                "default_provider": "dashscope",
                "default_model": "qwen-turbo",
                "enable_cost_tracking": True,
                "cost_alert_threshold": 100.0,  # 成本警告阈值
                "currency_preference": "CNY",
                "auto_save_usage": True,
                "max_usage_records": 10000,
                "data_dir": default_data_dir,  # 数据目录配置
                "cache_dir": os.path.join(default_data_dir, "cache"),  # 缓存目录
                "results_dir": os.path.join(os.path.expanduser("~"), "Documents", "TradingAgents", "results"),  # 结果目录
                "auto_create_dirs": True,  # 自动创建目录
                "openai_enabled": False,  # OpenAI模型是否启用
            }
            self.save_settings(default_settings)
    
    def load_models(self) -> List[ModelConfig]:
        """从 models.json 文件加载模型配置列表。

        此方法会执行以下操作：
        1. 读取 JSON 文件并将其内容解析为 `ModelConfig` 对象列表。
        2. 对于每个模型配置，检查相应的环境变量（如 `DASHSCOPE_API_KEY`）
           是否存在 API 密钥。如果存在，环境变量中的密钥将覆盖 JSON
           文件中的设置，并自动启用该模型。
        3. 根据全局设置（`openai_enabled`）和 API 密钥的格式验证，
           特殊处理 OpenAI 模型的启用状态。

        Returns:
            一个包含 `ModelConfig` 对象的列表。如果加载失败，返回空列表。
        """
        try:
            with open(self.models_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                models = [ModelConfig(**item) for item in data]

                # 获取设置
                settings = self.load_settings()
                openai_enabled = settings.get("openai_enabled", False)

                # 合并.env中的API密钥（优先级更高）
                for model in models:
                    env_api_key = self._get_env_api_key(model.provider)
                    if env_api_key:
                        model.api_key = env_api_key
                        # 如果.env中有API密钥，自动启用该模型
                        if not model.enabled:
                            model.enabled = True
                    
                    # 特殊处理OpenAI模型
                    if model.provider.lower() == "openai":
                        # 检查OpenAI是否在配置中启用
                        if not openai_enabled:
                            model.enabled = False
                            logger.info(f"🔒 OpenAI模型已禁用: {model.model_name}")
                        # 如果有API密钥但格式不正确，禁用模型（验证始终启用）
                        elif model.api_key and not self.validate_openai_api_key_format(model.api_key):
                            model.enabled = False
                            logger.warning(f"⚠️ OpenAI模型因密钥格式不正确而禁用: {model.model_name}")

                return models
        except Exception as e:
            logger.error(f"加载模型配置失败: {e}")
            return []
    
    def save_models(self, models: List[ModelConfig]):
        """将模型配置列表保存到 models.json 文件。

        Args:
            models: 一个包含 `ModelConfig` 对象的列表，将被序列化并写入文件。
        """
        try:
            data = [asdict(model) for model in models]
            with open(self.models_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存模型配置失败: {e}")
    
    def load_pricing(self) -> List[PricingConfig]:
        """从 pricing.json 文件加载定价配置列表。

        Returns:
            一个包含 `PricingConfig` 对象的列表。如果加载失败，返回空列表。
        """
        try:
            with open(self.pricing_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return [PricingConfig(**item) for item in data]
        except Exception as e:
            logger.error(f"加载定价配置失败: {e}")
            return []
    
    def save_pricing(self, pricing: List[PricingConfig]):
        """将定价配置列表保存到 pricing.json 文件。

        Args:
            pricing: 一个包含 `PricingConfig` 对象的列表，将被序列化并写入文件。
        """
        try:
            data = [asdict(price) for price in pricing]
            with open(self.pricing_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存定价配置失败: {e}")
    
    def load_usage_records(self) -> List[UsageRecord]:
        """从 usage.json 文件加载 token 使用记录。

        Returns:
            一个包含 `UsageRecord` 对象的列表。如果文件不存在或加载失败，
            返回空列表。
        """
        try:
            if not self.usage_file.exists():
                return []
            with open(self.usage_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return [UsageRecord(**item) for item in data]
        except Exception as e:
            logger.error(f"加载使用记录失败: {e}")
            return []
    
    def save_usage_records(self, records: List[UsageRecord]):
        """将 token 使用记录列表保存到 usage.json 文件。

        Args:
            records: 一个包含 `UsageRecord` 对象的列表，将被序列化并写入文件。
        """
        try:
            data = [asdict(record) for record in records]
            with open(self.usage_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存使用记录失败: {e}")
    
    def add_usage_record(self, provider: str, model_name: str, input_tokens: int,
                        output_tokens: int, session_id: str, analysis_type: str = "stock_analysis"):
        """创建一个新的使用记录，并将其持久化。

        该方法首先根据定价配置计算本次调用的成本，然后创建一个 `UsageRecord`
        对象。记录会优先尝试保存到 MongoDB（如果已配置），如果失败则回退到
        本地的 usage.json 文件。同时，它还会根据设置对本地记录数量进行限制，
        以防止文件过大。

        Args:
            provider: LLM 供应商的名称（如 'dashscope'）。
            model_name: 所用模型的名称。
            input_tokens: 输入的 token 数量。
            output_tokens: 输出的 token 数量。
            session_id: 标识当前会话的唯一字符串。
            analysis_type: 本次分析的类型，默认为 'stock_analysis'。

        Returns:
            创建并保存的 `UsageRecord` 对象。
        """
        # 计算成本
        cost = self.calculate_cost(provider, model_name, input_tokens, output_tokens)
        
        record = UsageRecord(
            timestamp=datetime.now().isoformat(),
            provider=provider,
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            session_id=session_id,
            analysis_type=analysis_type
        )
        
        # 优先使用MongoDB存储
        if self.mongodb_storage and self.mongodb_storage.is_connected():
            success = self.mongodb_storage.save_usage_record(record)
            if success:
                return record
            else:
                logger.error(f"⚠️ MongoDB保存失败，回退到JSON文件存储")
        
        # 回退到JSON文件存储
        records = self.load_usage_records()
        records.append(record)
        
        # 限制记录数量
        settings = self.load_settings()
        max_records = settings.get("max_usage_records", 10000)
        if len(records) > max_records:
            records = records[-max_records:]
        
        self.save_usage_records(records)
        return record
    
    def calculate_cost(self, provider: str, model_name: str, input_tokens: int, output_tokens: int) -> float:
        """根据指定的供应商、模型和 token 数量计算调用成本。

        该方法从定价配置中查找匹配的费率，并基于每 1000 个 token 的
        价格计算输入和输出的总成本。

        Args:
            provider: LLM 供应商的名称。
            model_name: 所用模型的名称。
            input_tokens: 输入的 token 数量。
            output_tokens: 输出的 token 数量。

        Returns:
            计算得出的总成本，浮点数，保留六位小数。如果找不到对应的
            定价配置，则返回 0.0。
        """
        pricing_configs = self.load_pricing()

        for pricing in pricing_configs:
            if pricing.provider == provider and pricing.model_name == model_name:
                input_cost = (input_tokens / 1000) * pricing.input_price_per_1k
                output_cost = (output_tokens / 1000) * pricing.output_price_per_1k
                total_cost = input_cost + output_cost
                return round(total_cost, 6)

        # 只在找不到配置时输出调试信息
        logger.warning(f"⚠️ [calculate_cost] 未找到匹配的定价配置: {provider}/{model_name}")
        logger.debug(f"⚠️ [calculate_cost] 可用的配置:")
        for pricing in pricing_configs:
            logger.debug(f"⚠️ [calculate_cost]   - {pricing.provider}/{pricing.model_name}")

        return 0.0
    
    def load_settings(self) -> Dict[str, Any]:
        """加载应用程序的常规设置。

        此方法从 `settings.json` 文件加载基础设置，并使用 `.env` 文件中
        定义的相应环境变量来覆盖这些设置。这种机制允许通过环境变量
        灵活地修改配置，而无需直接编辑 JSON 文件。

        Returns:
            一个包含所有合并后设置项的字典。
        """
        try:
            if self.settings_file.exists():
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
            else:
                # 如果设置文件不存在，创建默认设置
                settings = {
                    "default_provider": "dashscope",
                    "default_model": "qwen-turbo",
                    "enable_cost_tracking": True,
                    "cost_alert_threshold": 100.0,
                    "currency_preference": "CNY",
                    "auto_save_usage": True,
                    "max_usage_records": 10000,
                    "data_dir": os.path.join(os.path.expanduser("~"), "Documents", "TradingAgents", "data"),
                    "cache_dir": os.path.join(os.path.expanduser("~"), "Documents", "TradingAgents", "data", "cache"),
                    "results_dir": os.path.join(os.path.expanduser("~"), "Documents", "TradingAgents", "results"),
                    "auto_create_dirs": True,
                    "openai_enabled": False,
                }
                self.save_settings(settings)
        except Exception as e:
            logger.error(f"加载设置失败: {e}")
            settings = {}

        # 合并.env中的其他配置
        env_settings = {
            "finnhub_api_key": os.getenv("FINNHUB_API_KEY", ""),
            "reddit_client_id": os.getenv("REDDIT_CLIENT_ID", ""),
            "reddit_client_secret": os.getenv("REDDIT_CLIENT_SECRET", ""),
            "reddit_user_agent": os.getenv("REDDIT_USER_AGENT", ""),
            "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", ""),
            "log_level": os.getenv("TRADINGAGENTS_LOG_LEVEL", "INFO"),
            "data_dir": os.getenv("TRADINGAGENTS_DATA_DIR", ""),  # 数据目录环境变量
            "cache_dir": os.getenv("TRADINGAGENTS_CACHE_DIR", ""),  # 缓存目录环境变量
        }

        # 添加OpenAI相关配置
        openai_enabled_env = os.getenv("OPENAI_ENABLED", "").lower()
        if openai_enabled_env in ["true", "false"]:
            env_settings["openai_enabled"] = openai_enabled_env == "true"

        # 只有当环境变量存在且不为空时才覆盖
        for key, value in env_settings.items():
            # 对于布尔值，直接使用
            if isinstance(value, bool):
                settings[key] = value
            # 对于字符串，只有非空时才覆盖
            elif value != "" and value is not None:
                settings[key] = value

        return settings

    def get_env_config_status(self) -> Dict[str, Any]:
        """获取并返回当前 .env 文件配置的状态摘要。

        此方法用于快速检查 .env 文件是否存在，以及其中关键的 API 密钥
        和其他配置是否已设置。

        Returns:
            一个字典，包含了 .env 文件的存在状态、各类 API 密钥的配置情况
            以及其他关键配置项的值。
        """
        return {
            "env_file_exists": (Path(__file__).parent.parent.parent / ".env").exists(),
            "api_keys": {
                "dashscope": bool(os.getenv("DASHSCOPE_API_KEY")),
                "openai": bool(os.getenv("OPENAI_API_KEY")),
                "google": bool(os.getenv("GOOGLE_API_KEY")),
                "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
                "finnhub": bool(os.getenv("FINNHUB_API_KEY")),
            },
            "other_configs": {
                "reddit_configured": bool(os.getenv("REDDIT_CLIENT_ID") and os.getenv("REDDIT_CLIENT_SECRET")),
                "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", "./results"),
                "log_level": os.getenv("TRADINGAGENTS_LOG_LEVEL", "INFO"),
            }
        }

    def save_settings(self, settings: Dict[str, Any]):
        """将常规设置字典保存到 settings.json 文件。

        Args:
            settings: 包含常规设置的字典。
        """
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存设置失败: {e}")
    
    def get_enabled_models(self) -> List[ModelConfig]:
        """获取所有当前已启用且 API 密钥已配置的模型列表。

        Returns:
            一个 `ModelConfig` 对象的列表，其中每个模型都满足
            `enabled` 为 True 且 `api_key` 不为空的条件。
        """
        models = self.load_models()
        return [model for model in models if model.enabled and model.api_key]
    
    def get_model_by_name(self, provider: str, model_name: str) -> Optional[ModelConfig]:
        """根据供应商和模型名称查找并返回具体的模型配置。

        Args:
            provider: LLM 供应商的名称。
            model_name: 模型的名称。

        Returns:
            如果找到匹配的模型，则返回 `ModelConfig` 对象；否则返回 None。
        """
        models = self.load_models()
        for model in models:
            if model.provider == provider and model.model_name == model_name:
                return model
        return None
    
    def get_usage_statistics(self, days: int = 30) -> Dict[str, Any]:
        """获取指定时间范围内的 LLM 使用情况统计数据。

        此方法优先从 MongoDB（如果可用）获取统计数据，以获得更好的性能
        和可扩展性。如果 MongoDB 不可用，它将回退到从本地的 usage.json
        文件加载记录并进行计算。

        Args:
            days: 要统计的天数，默认为 30 天。

        Returns:
            一个包含统计信息的字典，包括总成本、总 token 数、总请求数
            以及按供应商分类的详细统计。
        """
        # 优先使用MongoDB获取统计
        if self.mongodb_storage and self.mongodb_storage.is_connected():
            try:
                # 从MongoDB获取基础统计
                stats = self.mongodb_storage.get_usage_statistics(days)
                # 获取供应商统计
                provider_stats = self.mongodb_storage.get_provider_statistics(days)
                
                if stats:
                    stats["provider_stats"] = provider_stats
                    stats["records_count"] = stats.get("total_requests", 0)
                    return stats
            except Exception as e:
                logger.error(f"⚠️ MongoDB统计获取失败，回退到JSON文件: {e}")
        
        # 回退到JSON文件统计
        records = self.load_usage_records()
        
        # 过滤最近N天的记录
        from datetime import datetime, timedelta

        cutoff_date = datetime.now() - timedelta(days=days)
        
        recent_records = []
        for record in records:
            try:
                record_date = datetime.fromisoformat(record.timestamp)
                if record_date >= cutoff_date:
                    recent_records.append(record)
            except:
                continue
        
        # 统计数据
        total_cost = sum(record.cost for record in recent_records)
        total_input_tokens = sum(record.input_tokens for record in recent_records)
        total_output_tokens = sum(record.output_tokens for record in recent_records)
        
        # 按供应商统计
        provider_stats = {}
        for record in recent_records:
            if record.provider not in provider_stats:
                provider_stats[record.provider] = {
                    "cost": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "requests": 0
                }
            provider_stats[record.provider]["cost"] += record.cost
            provider_stats[record.provider]["input_tokens"] += record.input_tokens
            provider_stats[record.provider]["output_tokens"] += record.output_tokens
            provider_stats[record.provider]["requests"] += 1
        
        return {
            "period_days": days,
            "total_cost": round(total_cost, 4),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_requests": len(recent_records),
            "provider_stats": provider_stats,
            "records_count": len(recent_records)
        }
    
    def get_data_dir(self) -> str:
        """获取应用程序的数据存储根目录路径。

        Returns:
            数据目录的字符串路径。
        """
        settings = self.load_settings()
        data_dir = settings.get("data_dir")
        if not data_dir:
            # 如果没有配置，使用默认路径
            data_dir = os.path.join(os.path.expanduser("~"), "Documents", "TradingAgents", "data")
        return data_dir

    def set_data_dir(self, data_dir: str):
        """设置应用程序的数据存储根目录，并自动更新缓存目录路径。

        如果设置中启用了 `auto_create_dirs`，此方法还会确保目录存在。

        Args:
            data_dir: 新的数据目录路径。
        """
        settings = self.load_settings()
        settings["data_dir"] = data_dir
        # 同时更新缓存目录
        settings["cache_dir"] = os.path.join(data_dir, "cache")
        self.save_settings(settings)
        
        # 如果启用自动创建目录，则创建目录
        if settings.get("auto_create_dirs", True):
            self.ensure_directories_exist()

    def ensure_directories_exist(self):
        """检查并创建所有在设置中定义的必要目录。

        这包括数据目录、缓存目录、结果目录以及其他特定数据的子目录，
        确保应用程序在写入文件之前路径是可用的。
        """
        settings = self.load_settings()
        
        directories = [
            settings.get("data_dir"),
            settings.get("cache_dir"),
            settings.get("results_dir"),
            os.path.join(settings.get("data_dir", ""), "finnhub_data"),
            os.path.join(settings.get("data_dir", ""), "finnhub_data", "news_data"),
            os.path.join(settings.get("data_dir", ""), "finnhub_data", "insider_sentiment"),
            os.path.join(settings.get("data_dir", ""), "finnhub_data", "insider_transactions")
        ]
        
        for directory in directories:
            if directory and not os.path.exists(directory):
                try:
                    os.makedirs(directory, exist_ok=True)
                    logger.info(f"✅ 创建目录: {directory}")
                except Exception as e:
                    logger.error(f"❌ 创建目录失败 {directory}: {e}")
    
    def set_openai_enabled(self, enabled: bool):
        """在设置文件中更新 OpenAI 模型的全局启用状态。

        Args:
            enabled: True 表示启用，False 表示禁用。
        """
        settings = self.load_settings()
        settings["openai_enabled"] = enabled
        self.save_settings(settings)
        logger.info(f"🔧 OpenAI模型启用状态已设置为: {enabled}")
    
    def is_openai_enabled(self) -> bool:
        """检查 OpenAI 模型是否在全局设置中被启用。

        Returns:
            如果已启用，则返回 True；否则返回 False。
        """
        settings = self.load_settings()
        return settings.get("openai_enabled", False)
    
    def get_openai_config_status(self) -> Dict[str, Any]:
        """获取 OpenAI 配置的详细状态。

        此方法提供一个摘要，说明 OpenAI API 密钥是否存在、格式是否正确、
        模型是否在全局设置中启用，以及最终是否可用。

        Returns:
            一个包含 OpenAI 配置状态的字典。
        """
        openai_key = os.getenv("OPENAI_API_KEY", "")
        key_valid = self.validate_openai_api_key_format(openai_key) if openai_key else False
        
        return {
            "api_key_present": bool(openai_key),
            "api_key_valid_format": key_valid,
            "enabled": self.is_openai_enabled(),
            "models_available": self.is_openai_enabled() and key_valid,
            "api_key_preview": f"{openai_key[:10]}..." if openai_key else "未配置"
        }


class TokenTracker:
    """负责跟踪和管理大型语言模型（LLM）的 token 使用情况和相关成本。

    该类与 `ConfigManager` 协同工作，记录每次 LLM 调用的 token 消耗，
    计算成本，并根据配置的阈值提供成本警告。

    Attributes:
        config_manager (ConfigManager): 用于访问配置和保存使用记录的
                                        `ConfigManager` 实例。
    """

    def __init__(self, config_manager: ConfigManager):
        """初始化 TokenTracker。

        Args:
            config_manager: 一个 `ConfigManager` 的实例。
        """
        self.config_manager = config_manager

    def track_usage(self, provider: str, model_name: str, input_tokens: int,
                   output_tokens: int, session_id: str = None, analysis_type: str = "stock_analysis"):
        """记录一次 LLM 调䂝的 token 使用情况。

        如果成本跟踪功能被禁用，则此方法不执行任何操作。否则，它会调用
        `ConfigManager` 来添加一条新的使用记录，并检查是否触发了成本警告。

        Args:
            provider: LLM 供应商的名称。
            model_name: 所用模型的名称。
            input_tokens: 输入的 token 数量。
            output_tokens: 输出的 token 数量。
            session_id: 标识当前会话的唯一字符串。如果为 None，则自动生成。
            analysis_type: 本次分析的类型。

        Returns:
            如果跟踪成功，返回创建的 `UsageRecord` 对象；如果成本跟踪被禁用，
            则返回 None。
        """
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 检查是否启用成本跟踪
        settings = self.config_manager.load_settings()
        cost_tracking_enabled = settings.get("enable_cost_tracking", True)

        if not cost_tracking_enabled:
            return None

        # 添加使用记录
        record = self.config_manager.add_usage_record(
            provider=provider,
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            session_id=session_id,
            analysis_type=analysis_type
        )

        # 检查成本警告
        if record:
            self._check_cost_alert(record.cost)

        return record

    def _check_cost_alert(self, current_cost: float):
        """检查当日总成本是否超过预设的警告阈值。"""
        settings = self.config_manager.load_settings()
        threshold = settings.get("cost_alert_threshold", 100.0)

        # 获取今日总成本
        today_stats = self.config_manager.get_usage_statistics(1)
        total_today = today_stats["total_cost"]

        if total_today >= threshold:
            logger.warning(f"⚠️ 成本警告: 今日成本已达到 ¥{total_today:.4f}，超过阈值 ¥{threshold}",
                          extra={'cost': total_today, 'threshold': threshold, 'event_type': 'cost_alert'})

    def get_session_cost(self, session_id: str) -> float:
        """计算并返回指定会话的总成本。

        Args:
            session_id: 要查询的会话 ID。

        Returns:
            该会话累计的总成本。
        """
        records = self.config_manager.load_usage_records()
        session_cost = sum(record.cost for record in records if record.session_id == session_id)
        return session_cost

    def estimate_cost(self, provider: str, model_name: str, estimated_input_tokens: int,
                     estimated_output_tokens: int) -> float:
        """根据预估的 token 数量估算一次 LLM 调用的成本。

        Args:
            provider: LLM 供应商的名称。
            model_name: 模型的名称。
            estimated_input_tokens: 预估的输入 token 数。
            estimated_output_tokens: 预估的输出 token 数。

        Returns:
            预估的调用成本。
        """
        return self.config_manager.calculate_cost(
            provider, model_name, estimated_input_tokens, estimated_output_tokens
        )




# 全局配置管理器实例 - 使用项目根目录的配置
def _get_project_config_dir():
    """获取项目根目录的配置目录"""
    # 从当前文件位置推断项目根目录
    current_file = Path(__file__)  # tradingagents/config/config_manager.py
    project_root = current_file.parent.parent.parent  # 向上三级到项目根目录
    return str(project_root / "config")

config_manager = ConfigManager(_get_project_config_dir())
token_tracker = TokenTracker(config_manager)
