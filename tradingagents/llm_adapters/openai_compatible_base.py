"""
OpenAI 兼容适配器基类模块。

该模块提供了一个核心基类 `OpenAICompatibleBase`，旨在为所有提供 OpenAI 兼容
API 接口的 LLM 服务（如 DeepSeek, 阿里百炼, 百度千帆等）提供一个统一、可扩展的
LangChain 适配器框架。

**核心设计理念:**
- **继承与复用:** 通过继承 `langchain_openai.ChatOpenAI`，直接复用其成熟的
  API 调用、工具绑定和消息处理逻辑。
- **标准化:** 为不同提供商的适配器提供统一的初始化流程和 token 追踪机制。
- **可扩展性:** 子类只需简单地提供提供商特定的元数据（如 `provider_name`,
  `base_url` 等），即可快速创建一个功能完备的适配器。
- **工厂模式:** 提供 `create_openai_compatible_llm` 工厂函数，用于根据
  提供商名称动态创建相应的 LLM 实例。
"""

import os
import time
from typing import Any, Dict, List, Optional, Union
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import CallbackManagerForLLMRun

# 导入统一日志系统
from tradingagents.utils.logging_init import setup_llm_logging

# 导入日志模块
from tradingagents.utils.logging_manager import get_logger, get_logger_manager
logger = get_logger('agents')
logger = setup_llm_logging()

# 导入token跟踪器
try:
    from tradingagents.config.config_manager import token_tracker
    TOKEN_TRACKING_ENABLED = True
    logger.info("✅ Token跟踪功能已启用")
except ImportError:
    TOKEN_TRACKING_ENABLED = False
    logger.warning("⚠️ Token跟踪功能未启用")


class OpenAICompatibleBase(ChatOpenAI):
    """
    一个为所有支持 OpenAI 兼容接口的 LLM 提供商设计的统一基类。

    该类通过封装通用的初始化逻辑、API 密钥管理和 token 使用量追踪功能，
    极大地简化了为新 LLM 提供商创建 LangChain 适配器的过程。

    子类化此类时，通常只需要在 `__init__` 方法中调用 `super().__init__(...)`
    并传入提供商特定的配置信息即可。
    """
    
    def __init__(
        self,
        provider_name: str,
        model: str,
        api_key_env_var: str,
        base_url: str,
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        初始化 OpenAI 兼容适配器基类。

        Args:
            provider_name (str): LLM 提供商的名称 (例如, "deepseek", "dashscope")。
            model (str): 要使用的具体模型名称。
            api_key_env_var (str): 用于查找 API 密钥的环境变量的名称。
            base_url (str): 该提供商的 OpenAI 兼容 API 端点 URL。
            api_key (Optional[str], optional): API 密钥。如果为 None, 将从 `api_key_env_var` 指定的环境变量中读取。
            temperature (float, optional): 控制生成文本的随机性。默认为 0.1。
            max_tokens (Optional[int], optional): 生成的最大 token 数量。默认为 None。
            **kwargs: 其他传递给 `ChatOpenAI` 父类构造函数的关键字参数。

        Raises:
            ValueError: 如果 API 密钥既没有在参数中提供，也没有在指定的环境变量中设置。
        """
        
        # 在父类初始化前先缓存元信息到私有属性（避免Pydantic字段限制）
        object.__setattr__(self, "_provider_name", provider_name)
        object.__setattr__(self, "_model_name_alias", model)
        
        # 获取API密钥
        if api_key is None:
            api_key = os.getenv(api_key_env_var)
            if not api_key:
                raise ValueError(
                    f"{provider_name} API密钥未找到。"
                    f"请设置{api_key_env_var}环境变量或传入api_key参数。"
                )
        
        # 设置OpenAI兼容参数
        # 注意：model参数会被Pydantic映射到model_name字段
        openai_kwargs = {
            "model": model,  # 这会被映射到model_name字段
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        # 根据LangChain版本使用不同的参数名
        try:
            # 新版本LangChain
            openai_kwargs.update({
                "api_key": api_key,
                "base_url": base_url
            })
        except:
            # 旧版本LangChain
            openai_kwargs.update({
                "openai_api_key": api_key,
                "openai_api_base": base_url
            })
        
        # 初始化父类
        super().__init__(**openai_kwargs)

        # 再次确保元信息存在（有些实现会在super()中重置__dict__）
        object.__setattr__(self, "_provider_name", provider_name)
        object.__setattr__(self, "_model_name_alias", model)

        logger.info(f"✅ {provider_name} OpenAI兼容适配器初始化成功")
        logger.info(f"   模型: {model}")
        logger.info(f"   API Base: {base_url}")

    @property
    def provider_name(self) -> Optional[str]:
        return getattr(self, "_provider_name", None)

    # 移除model_name property定义，使用Pydantic字段
    # model_name字段由ChatOpenAI基类的Pydantic字段提供
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        重写 `_generate` 方法以集成统一的 token 使用量追踪。

        此方法首先调用父类 (`ChatOpenAI`) 的 `_generate` 方法来执行实际的 API 调用，
        然后调用 `_track_token_usage` 方法来处理和记录返回结果中的 token 使用信息。

        Args:
            messages (List[BaseMessage]): 用于生成回复的聊天消息列表。
            stop (Optional[List[str]], optional): 停止生成的字符串列表。
            run_manager (Optional[CallbackManagerForLLMRun], optional): LangChain 的回调管理器。
            **kwargs (Any): 其他传递给 API 的参数。

        Returns:
            ChatResult: 从父类 `_generate` 方法返回的原始 `ChatResult` 对象。
        """
        
        # 记录开始时间
        start_time = time.time()
        
        # 调用父类生成方法
        result = super()._generate(messages, stop, run_manager, **kwargs)
        
        # 记录token使用
        self._track_token_usage(result, kwargs, start_time)
        
        return result

    def _track_token_usage(self, result: ChatResult, kwargs: Dict, start_time: float):
        """
        一个通用的 token 追踪方法。

        该方法从 `ChatResult` 对象中提取 `usage_metadata`，并记录
        总 tokens、输入 tokens、输出 tokens 以及调用耗时。

        Args:
            result (ChatResult): `_generate` 方法返回的结果对象。
            kwargs (Dict): 传递给 `_generate` 的关键字参数字典。
            start_time (float): API 调用的起始时间戳。
        """
        if not TOKEN_TRACKING_ENABLED:
            return
        try:
            # 统计token信息
            usage = getattr(result, "usage_metadata", None)
            total_tokens = usage.get("total_tokens") if usage else None
            prompt_tokens = usage.get("input_tokens") if usage else None
            completion_tokens = usage.get("output_tokens") if usage else None

            elapsed = time.time() - start_time
            logger.info(
                f"📊 Token使用 - Provider: {getattr(self, 'provider_name', 'unknown')}, Model: {getattr(self, 'model_name', 'unknown')}, "
                f"总tokens: {total_tokens}, 提示: {prompt_tokens}, 补全: {completion_tokens}, 用时: {elapsed:.2f}s"
            )
        except Exception as e:
            logger.warning(f"⚠️ Token跟踪记录失败: {e}")


class ChatDeepSeekOpenAI(OpenAICompatibleBase):
    """
    用于 DeepSeek 模型的 OpenAI 兼容适配器。

    该类继承自 `OpenAICompatibleBase`，并为其构造函数提供了
    DeepSeek 特定的默认值，如 `provider_name`, `api_key_env_var` 和 `base_url`。
    """
    
    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            provider_name="deepseek",
            model=model,
            api_key_env_var="DEEPSEEK_API_KEY",
            base_url="https://api.deepseek.com",
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )


class ChatDashScopeOpenAIUnified(OpenAICompatibleBase):
    """
    用于阿里百炼 (DashScope) 模型的 OpenAI 兼容适配器。

    该类继承自 `OpenAICompatibleBase`，并为其构造函数提供了
    DashScope 特定的默认值。
    """
    
    def __init__(
        self,
        model: str = "qwen-turbo",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            provider_name="dashscope",
            model=model,
            api_key_env_var="DASHSCOPE_API_KEY",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )


class ChatQianfanOpenAI(OpenAICompatibleBase):
    """
    用于百度文心一言 (Qianfan) 平台的 OpenAI 兼容适配器。

    该类继承自 `OpenAICompatibleBase`，并为其构造函数提供了
    Qianfan 特定的默认值。此外，它还增加了对 API 密钥格式的校验，
    以及一个消息截断功能 `_truncate_messages`，以适应某些 Qianfan
    模型较短的上下文窗口限制。
    """
    
    def __init__(
        self,
        model: str = "ernie-3.5-8k",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        # 千帆新一代API使用单一API Key认证
        # 格式: bce-v3/ALTAK-xxx/xxx
        
        qianfan_api_key = api_key or os.getenv('QIANFAN_API_KEY')
        
        if not qianfan_api_key:
            raise ValueError(
                "千帆模型需要设置QIANFAN_API_KEY环境变量，格式为: bce-v3/ALTAK-xxx/xxx"
            )
        
        if not qianfan_api_key.startswith('bce-v3/'):
            raise ValueError(
                "QIANFAN_API_KEY格式错误，应为: bce-v3/ALTAK-xxx/xxx"
            )
        
        super().__init__(
            provider_name="qianfan",
            model=model,
            api_key_env_var="QIANFAN_API_KEY",
            base_url="https://qianfan.baidubce.com/v2",
            api_key=qianfan_api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    
    def _estimate_tokens(self, text: str) -> int:
        """
        估算给定文本在 Qianfan 模型中的 token 数量。

        Args:
            text (str): 需要估算的文本。

        Returns:
            int: 估算出的 token 数量。
        """
        # 千帆模型的token估算：中文约1.5字符/token，英文约4字符/token
        # 保守估算：2字符/token
        return max(1, len(text) // 2)
    
    def _truncate_messages(self, messages: List[BaseMessage], max_tokens: int = 4500) -> List[BaseMessage]:
        """
        截断消息列表以适应 Qianfan 模型的上下文长度限制。

        该方法从消息列表的末尾开始向前保留消息，直到达到 `max_tokens` 的
        token 限制。如果第一条消息本身就超长，则会对其内容进行截断。

        Args:
            messages (List[BaseMessage]): 原始消息列表。
            max_tokens (int, optional): 允许的最大 token 数量。默认为 4500。

        Returns:
            List[BaseMessage]: 截断后的消息列表。
        """
        # 为千帆模型预留一些token空间，使用4500而不是5120
        truncated_messages = []
        total_tokens = 0
        
        # 从最后一条消息开始，向前保留消息
        for message in reversed(messages):
            content = str(message.content) if hasattr(message, 'content') else str(message)
            message_tokens = self._estimate_tokens(content)
            
            if total_tokens + message_tokens <= max_tokens:
                truncated_messages.insert(0, message)
                total_tokens += message_tokens
            else:
                # 如果是第一条消息且超长，进行内容截断
                if not truncated_messages:
                    remaining_tokens = max_tokens - 100  # 预留100个token
                    max_chars = remaining_tokens * 2  # 2字符/token
                    truncated_content = content[:max_chars] + "...(内容已截断)"
                    
                    # 创建截断后的消息
                    if hasattr(message, 'content'):
                        message.content = truncated_content
                    truncated_messages.insert(0, message)
                break
        
        if len(truncated_messages) < len(messages):
            logger.warning(f"⚠️ 千帆模型输入过长，已截断 {len(messages) - len(truncated_messages)} 条消息")
        
        return truncated_messages
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        重写 `_generate` 方法，在调用父类方法之前先执行消息截断。
        """
        
        # 对千帆模型进行输入token截断
        truncated_messages = self._truncate_messages(messages)
        
        # 调用父类的_generate方法
        return super()._generate(truncated_messages, stop, run_manager, **kwargs)


class ChatCustomOpenAI(OpenAICompatibleBase):
    """
    用于连接自定义 OpenAI 兼容端点（如代理或聚合平台）的适配器。

    该类允许用户灵活地配置 `base_url`，以连接到非官方的、但遵循
    OpenAI API 规范的任何服务。
    """
    
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        if base_url is None:
            base_url = os.getenv("CUSTOM_OPENAI_BASE_URL", "https://api.openai.com/v1")
        
        super().__init__(
            provider_name="custom_openai",
            model=model,
            api_key_env_var="CUSTOM_OPENAI_API_KEY",
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )


# 支持的OpenAI兼容模型配置
OPENAI_COMPATIBLE_PROVIDERS = {
    "deepseek": {
        "adapter_class": ChatDeepSeekOpenAI,
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "models": {
            "deepseek-chat": {"context_length": 32768, "supports_function_calling": True},
            "deepseek-coder": {"context_length": 16384, "supports_function_calling": True}
        }
    },
    "dashscope": {
        "adapter_class": ChatDashScopeOpenAIUnified,
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
        "models": {
            "qwen-turbo": {"context_length": 8192, "supports_function_calling": True},
            "qwen-plus": {"context_length": 32768, "supports_function_calling": True},
            "qwen-plus-latest": {"context_length": 32768, "supports_function_calling": True},
            "qwen-max": {"context_length": 32768, "supports_function_calling": True},
            "qwen-max-latest": {"context_length": 32768, "supports_function_calling": True}
        }
    },
    "qianfan": {
        "adapter_class": ChatQianfanOpenAI,
        "base_url": "https://qianfan.baidubce.com/v2",
        "api_key_env": "QIANFAN_API_KEY",
        "models": {
            "ernie-3.5-8k": {"context_length": 5120, "supports_function_calling": True},
            "ernie-4.0-turbo-8k": {"context_length": 5120, "supports_function_calling": True},
            "ERNIE-Speed-8K": {"context_length": 5120, "supports_function_calling": True},
            "ERNIE-Lite-8K": {"context_length": 5120, "supports_function_calling": True}
        }
    },
    "custom_openai": {
        "adapter_class": ChatCustomOpenAI,
        "base_url": None,  # 将由用户配置
        "api_key_env": "CUSTOM_OPENAI_API_KEY",
        "models": {
            "gpt-3.5-turbo": {"context_length": 16384, "supports_function_calling": True},
            "gpt-4": {"context_length": 8192, "supports_function_calling": True},
            "gpt-4-turbo": {"context_length": 128000, "supports_function_calling": True},
            "gpt-4o": {"context_length": 128000, "supports_function_calling": True},
            "gpt-4o-mini": {"context_length": 128000, "supports_function_calling": True},
            "claude-3-haiku": {"context_length": 200000, "supports_function_calling": True},
            "claude-3-sonnet": {"context_length": 200000, "supports_function_calling": True},
            "claude-3-opus": {"context_length": 200000, "supports_function_calling": True},
            "claude-3.5-sonnet": {"context_length": 200000, "supports_function_calling": True},
            "gemini-pro": {"context_length": 32768, "supports_function_calling": True},
            "gemini-1.5-pro": {"context_length": 1000000, "supports_function_calling": True},
            "llama-3.1-8b": {"context_length": 128000, "supports_function_calling": True},
            "llama-3.1-70b": {"context_length": 128000, "supports_function_calling": True},
            "llama-3.1-405b": {"context_length": 128000, "supports_function_calling": True},
            "custom-model": {"context_length": 32768, "supports_function_calling": True}
        }
    }
}


def create_openai_compatible_llm(
    provider: str,
    model: str,
    api_key: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: Optional[int] = None,
    base_url: Optional[str] = None,
    **kwargs
) -> OpenAICompatibleBase:
    """
    一个统一的工厂函数，用于根据提供商名称创建相应的 LLM 适配器实例。

    该函数充当了所有 OpenAI 兼容适配器的单一入口点。它通过查找
    `OPENAI_COMPATIBLE_PROVIDERS` 字典来获取所需提供商的配置
    （如适配器类、默认 `base_url` 等），然后实例化并返回相应的适配器对象。

    Args:
        provider (str): LLM 提供商的名称 (例如, "deepseek", "qianfan")。
        model (str): 要使用的具体模型名称。
        api_key (Optional[str], optional): API 密钥。
        temperature (float, optional): 温度参数。
        max_tokens (Optional[int], optional): 最大 token 数。
        base_url (Optional[str], optional): 可选，用于覆盖提供商的默认 `base_url`。
        **kwargs: 其他传递给适配器构造函数的关键字参数。

    Returns:
        OpenAICompatibleBase: 一个所请求的提供商的适配器实例。

    Raises:
        ValueError: 如果指定的 `provider` 不被支持。
    """
    provider_info = OPENAI_COMPATIBLE_PROVIDERS.get(provider)
    if not provider_info:
        raise ValueError(f"不支持的OpenAI兼容提供商: {provider}")

    adapter_class = provider_info["adapter_class"]

    # 如果调用未提供 base_url，则采用 provider 的默认值（可能为 None）
    if base_url is None:
        base_url = provider_info.get("base_url")

    # 仅当 provider 未内置 base_url（如 custom_openai）时，才将 base_url 传递给适配器，
    # 避免与适配器内部的 super().__init__(..., base_url=...) 冲突导致 "multiple values" 错误。
    init_kwargs = dict(
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
    if provider_info.get("base_url") is None and base_url:
        init_kwargs["base_url"] = base_url

    return adapter_class(**init_kwargs)


def test_openai_compatible_adapters():
    """
    快速测试所有已注册的 OpenAI 兼容适配器是否能够被成功实例化。

    这个函数会遍历 `OPENAI_COMPATIBLE_PROVIDERS` 中的所有提供商，
    并尝试使用测试/虚拟参数来创建每个适配器类的实例。

    注意：此测试不发起任何真实的 API 请求，仅用于验证构造函数和
    依赖关系是否配置正确。
    """
    for provider, info in OPENAI_COMPATIBLE_PROVIDERS.items():
        cls = info["adapter_class"]
        try:
            if provider == "custom_openai":
                cls(model="gpt-3.5-turbo", api_key="test", base_url="https://api.openai.com/v1")
            elif provider == "qianfan":
                # 千帆新一代API仅需QIANFAN_API_KEY，格式: bce-v3/ALTAK-xxx/xxx
                cls(model="ernie-3.5-8k", api_key="bce-v3/test-key/test-secret")
            else:
                cls(model=list(info["models"].keys())[0], api_key="test")
            logger.info(f"✅ 适配器实例化成功: {provider}")
        except Exception as e:
            logger.warning(f"⚠️ 适配器实例化失败（预期或可忽略）: {provider} - {e}")


if __name__ == "__main__":
    test_openai_compatible_adapters()
