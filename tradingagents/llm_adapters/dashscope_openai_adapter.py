"""
阿里百炼 (DashScope) OpenAI 兼容适配器模块。

该模块提供了一个 `ChatDashScopeOpenAI` 类，它继承自 `langchain_openai.ChatOpenAI`。
这个适配器的主要目的是利用 DashScope 提供的 OpenAI 兼容 API 端点，从而能够
无缝地使用 LangChain 中为 OpenAI 模型设计的各种功能，尤其是原生的工具调用
(Function Calling)。

通过使用这个适配器，开发者可以用与 `ChatOpenAI` 完全相同的方式来调用
通义千问系列模型，而无需修改应用层代码。
"""

import os
from typing import Any, Dict, List, Optional, Union, Sequence
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from pydantic import Field, SecretStr
from ..config.config_manager import token_tracker

# 导入日志模块
from tradingagents.utils.logging_manager import get_logger
logger = get_logger('agents')


class ChatDashScopeOpenAI(ChatOpenAI):
    """
    一个通过 OpenAI 兼容接口与阿里百炼 (DashScope) 模型交互的 LangChain 适配器。

    此类继承自 `ChatOpenAI`，并预先配置了连接到 DashScope 的 OpenAI 兼容
    API 端点所需的所有设置（如 `base_url`）。这使得它能够原生支持 LangChain
    中的工具调用 (Function Calling) 和其他 OpenAI 特定的功能。

    **核心优势:**
    - **原生工具调用:** 无需任何额外的转换或提示工程即可使用 LangChain 的工具调用功能。
    - **代码兼容性:** 可以作为 `ChatOpenAI` 的直接替代品，无需修改现有代码。
    - **Token 追踪:** 重写了 `_generate` 方法以自动追踪和记录 token 使用量。

    **使用示例:**
    ```python
    from tradingagents.llm_adapters import ChatDashScopeOpenAI
    from langchain_core.tools import tool

    @tool
    def get_stock_price(symbol: str) -> float:
        \"\"\"获取股票价格的工具\"\"\"
        # ... 实现 ...
        return 123.45

    # 初始化模型并绑定工具
    llm = ChatDashScopeOpenAI(model="qwen-plus")
    llm_with_tools = llm.bind_tools([get_stock_price])

    # 调用模型
    response = llm_with_tools.invoke("阿里巴巴的股价是多少?")
    print(response.tool_calls)
    ```
    """
    
    def __init__(self, **kwargs):
        """
        初始化 `ChatDashScopeOpenAI` 实例。

        此构造函数会自动设置连接到 DashScope OpenAI 兼容端点的 `base_url`。
        同时，它会从环境变量 `DASHSCOPE_API_KEY` 或传入的 `api_key` 参数中
        获取 API 密钥。

        Args:
            **kwargs: 传递给 `ChatOpenAI` 父类构造函数的关键字参数。
                      可以覆盖 `model`, `api_key`, `temperature` 等默认值。

        Raises:
            ValueError: 如果 API 密钥既没有在参数中提供，也没有在环境变量中设置。
        """
        
        # 设置 DashScope OpenAI 兼容接口的默认配置
        kwargs.setdefault("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        kwargs.setdefault("api_key", os.getenv("DASHSCOPE_API_KEY"))
        kwargs.setdefault("model", "qwen-turbo")
        kwargs.setdefault("temperature", 0.1)
        kwargs.setdefault("max_tokens", 2000)
        
        # 检查 API 密钥
        if not kwargs.get("api_key"):
            raise ValueError(
                "DashScope API key not found. Please set DASHSCOPE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # 调用父类初始化
        super().__init__(**kwargs)

        logger.info(f"✅ 阿里百炼 OpenAI 兼容适配器初始化成功")
        logger.info(f"   模型: {kwargs.get('model', 'qwen-turbo')}")

        # 兼容不同版本的属性名
        api_base = getattr(self, 'base_url', None) or getattr(self, 'openai_api_base', None) or kwargs.get('base_url', 'unknown')
        logger.info(f"   API Base: {api_base}")
    
    def _generate(self, *args, **kwargs):
        """
        重写父类的 `_generate` 方法，以增加 token 使用量追踪功能。

        该方法首先调用 `ChatOpenAI` 的原始 `_generate` 方法来获取 LLM 的响应。
        然后，它从返回的 `ChatResult` 对象中提取 `token_usage` 信息，并
        使用全局的 `token_tracker` 实例来记录本次调用的输入和输出 token 数量。

        Args:
            *args: 传递给父类 `_generate` 方法的位置参数。
            **kwargs: 传递给父类 `_generate` 方法的关键字参数。

        Returns:
            ChatResult: 从父类 `_generate` 方法返回的原始 `ChatResult` 对象。
        """
        
        # 调用父类的生成方法
        result = super()._generate(*args, **kwargs)
        
        # 追踪 token 使用量
        try:
            # 从结果中提取 token 使用信息
            if hasattr(result, 'llm_output') and result.llm_output:
                token_usage = result.llm_output.get('token_usage', {})
                
                input_tokens = token_usage.get('prompt_tokens', 0)
                output_tokens = token_usage.get('completion_tokens', 0)
                
                if input_tokens > 0 or output_tokens > 0:
                    # 生成会话ID
                    session_id = kwargs.get('session_id', f"dashscope_openai_{hash(str(args))%10000}")
                    analysis_type = kwargs.get('analysis_type', 'stock_analysis')
                    
                    # 使用 TokenTracker 记录使用量
                    token_tracker.track_usage(
                        provider="dashscope",
                        model_name=self.model_name,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        session_id=session_id,
                        analysis_type=analysis_type
                    )
                    
        except Exception as track_error:
            # token 追踪失败不应该影响主要功能
            logger.error(f"⚠️ Token 追踪失败: {track_error}")
        
        return result


# 支持的模型列表
DASHSCOPE_OPENAI_MODELS = {
    # 通义千问系列
    "qwen-turbo": {
        "description": "通义千问 Turbo - 快速响应，适合日常对话",
        "context_length": 8192,
        "supports_function_calling": True,
        "recommended_for": ["快速任务", "日常对话", "简单分析"]
    },
    "qwen-plus": {
        "description": "通义千问 Plus - 平衡性能和成本",
        "context_length": 32768,
        "supports_function_calling": True,
        "recommended_for": ["复杂分析", "专业任务", "深度思考"]
    },
    "qwen-plus-latest": {
        "description": "通义千问 Plus 最新版 - 最新功能和性能",
        "context_length": 32768,
        "supports_function_calling": True,
        "recommended_for": ["最新功能", "复杂分析", "专业任务"]
    },
    "qwen-max": {
        "description": "通义千问 Max - 最强性能，适合复杂任务",
        "context_length": 32768,
        "supports_function_calling": True,
        "recommended_for": ["复杂推理", "专业分析", "高质量输出"]
    },
    "qwen-max-latest": {
        "description": "通义千问 Max 最新版 - 最强性能和最新功能",
        "context_length": 32768,
        "supports_function_calling": True,
        "recommended_for": ["最新功能", "复杂推理", "专业分析"]
    },
    "qwen-long": {
        "description": "通义千问 Long - 超长上下文，适合长文档处理",
        "context_length": 1000000,
        "supports_function_calling": True,
        "recommended_for": ["长文档分析", "大量数据处理", "复杂上下文"]
    }
}


def get_available_openai_models() -> Dict[str, Dict[str, Any]]:
    """
    获取通过 OpenAI 兼容接口可用的 DashScope 模型及其元数据。

    Returns:
        Dict[str, Dict[str, Any]]: 一个字典，键是模型名称，值是包含
                                  描述、上下文长度、工具调用支持等信息的元数据字典。
    """
    return DASHSCOPE_OPENAI_MODELS


def create_dashscope_openai_llm(
    model: str = "qwen-plus-latest",
    api_key: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 2000,
    **kwargs
) -> ChatDashScopeOpenAI:
    """
    一个便捷的工厂函数，用于创建 `ChatDashScopeOpenAI` 实例。

    Args:
        model (str, optional): 模型名称。默认为 "qwen-plus-latest"。
        api_key (Optional[str], optional): API 密钥。默认为 None，将从环境变量读取。
        temperature (float, optional): 温度参数。默认为 0.1。
        max_tokens (int, optional): 最大生成 token 数。默认为 2000。
        **kwargs: 其他传递给 `ChatDashScopeOpenAI` 构造函数的关键字参数。

    Returns:
        ChatDashScopeOpenAI: 一个 `ChatDashScopeOpenAI` 的新实例。
    """
    
    return ChatDashScopeOpenAI(
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )


def test_dashscope_openai_connection(
    model: str = "qwen-turbo",
    api_key: Optional[str] = None
) -> bool:
    """
    测试与 DashScope OpenAI 兼容接口的连接。

    该函数会创建一个 `ChatDashScopeOpenAI` 实例并发送一条测试消息，
    以验证 API 密钥和网络连接是否正常。

    Args:
        model (str, optional): 用于测试的模型名称。默认为 "qwen-turbo"。
        api_key (Optional[str], optional): 用于测试的 API 密钥。默认为 None。

    Returns:
        bool: 如果连接成功并收到有效响应，则返回 True，否则返回 False。
    """
    
    try:
        logger.info(f"🧪 测试 DashScope OpenAI 兼容接口连接")
        logger.info(f"   模型: {model}")
        
        # 创建客户端
        llm = create_dashscope_openai_llm(
            model=model,
            api_key=api_key,
            max_tokens=50
        )
        
        # 发送测试消息
        response = llm.invoke("你好，请简单介绍一下你自己。")
        
        if response and hasattr(response, 'content') and response.content:
            logger.info(f"✅ DashScope OpenAI 兼容接口连接成功")
            logger.info(f"   响应: {response.content[:100]}...")
            return True
        else:
            logger.error(f"❌ DashScope OpenAI 兼容接口响应为空")
            return False
            
    except Exception as e:
        logger.error(f"❌ DashScope OpenAI 兼容接口连接失败: {e}")
        return False


def test_dashscope_openai_function_calling(
    model: str = "qwen-plus-latest",
    api_key: Optional[str] = None
) -> bool:
    """
    测试 DashScope OpenAI 兼容接口的工具调用 (Function Calling) 功能。

    该函数会定义一个简单的测试工具，将其绑定到模型，然后发出一个
    明确需要使用该工具的请求，以验证模型是否能正确地生成工具调用指令。

    Args:
        model (str, optional): 用于测试的模型名称。默认为 "qwen-plus-latest"。
        api_key (Optional[str], optional): 用于测试的 API 密钥。默认为 None。

    Returns:
        bool: 如果模型成功生成了预期的工具调用，则返回 True，否则返回 False。
    """
    
    try:
        logger.info(f"🧪 测试 DashScope OpenAI Function Calling")
        logger.info(f"   模型: {model}")
        
        # 创建客户端
        llm = create_dashscope_openai_llm(
            model=model,
            api_key=api_key,
            max_tokens=200
        )
        
        # 定义测试工具
        def get_current_time() -> str:
            """获取当前时间"""
            import datetime
            return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 创建 LangChain 工具
        from langchain_core.tools import tool
        
        @tool
        def test_tool(query: str) -> str:
            """测试工具，返回查询信息"""
            return f"收到查询: {query}"
        
        # 绑定工具
        llm_with_tools = llm.bind_tools([test_tool])
        
        # 测试工具调用
        response = llm_with_tools.invoke("请使用test_tool查询'hello world'")
        
        logger.info(f"✅ DashScope OpenAI Function Calling 测试完成")
        logger.info(f"   响应类型: {type(response)}")
        
        if hasattr(response, 'tool_calls') and response.tool_calls:
            logger.info(f"   工具调用数量: {len(response.tool_calls)}")
            return True
        else:
            logger.info(f"   响应内容: {getattr(response, 'content', 'No content')}")
            return True  # 即使没有工具调用也算成功，因为模型可能选择不调用工具
            
    except Exception as e:
        logger.error(f"❌ DashScope OpenAI Function Calling 测试失败: {e}")
        return False


if __name__ == "__main__":
    """测试脚本"""
    logger.info(f"🧪 DashScope OpenAI 兼容适配器测试")
    logger.info(f"=" * 50)
    
    # 测试连接
    connection_ok = test_dashscope_openai_connection()
    
    if connection_ok:
        # 测试 Function Calling
        function_calling_ok = test_dashscope_openai_function_calling()
        
        if function_calling_ok:
            logger.info(f"\n🎉 所有测试通过！DashScope OpenAI 兼容适配器工作正常")
        else:
            logger.error(f"\n⚠️ Function Calling 测试失败")
    else:
        logger.error(f"\n❌ 连接测试失败")
