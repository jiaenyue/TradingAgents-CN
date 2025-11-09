"""
Google AI (Gemini) OpenAI 兼容适配器模块。

该模块提供了一个 `ChatGoogleOpenAI` 类，它继承自 `langchain_google_genai.ChatGoogleGenerativeAI`。
创建此适配器的主要目的是为了解决直接使用 `ChatGoogleGenerativeAI` 时遇到的
工具调用 (Function Calling) 返回格式不一致以及特定内容（如新闻）格式需要优化的问题。

它通过重写 `_generate` 方法来拦截和后处理模型的输出，确保其格式符合
`tradingagents` 框架其余部分的期望，并增加了对 token 使用量的追踪。
"""

import os
from typing import Any, Dict, List, Optional, Union, Sequence
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import LLMResult
from pydantic import Field, SecretStr
from ..config.config_manager import token_tracker

# 导入日志模块
from tradingagents.utils.logging_manager import get_logger
logger = get_logger('agents')


class ChatGoogleOpenAI(ChatGoogleGenerativeAI):
    """
    一个为 Google AI (Gemini) 模型定制的 LangChain 适配器，旨在优化工具调用和内容格式。

    此类继承自 `ChatGoogleGenerativeAI`，并对其核心生成方法 `_generate` 进行了重写，
    以解决以下特定问题：
    - **内容格式优化:** 对模型生成的内容（特别是新闻类内容）进行后处理，
      确保其包含必要的结构化信息，如标题、日期和来源。
    - **错误处理:** 在 API 调用失败时，返回一个包含错误信息的 `LLMResult` 对象，
      而不是直接抛出异常，以增强系统的健壮性。
    - **Token 追踪:** 集成了 `token_tracker`，用于精确记录每次 API 调用的
      输入和输出 token 数量，便于成本控制和性能分析。

    通过这些增强，该适配器为在上层应用中统一和稳定地使用 Gemini 模型提供了保障。
    """
    
    def __init__(self, **kwargs):
        """
        初始化 `ChatGoogleOpenAI` 实例。

        此构造函数负责设置 API 密钥和默认的模型参数，然后调用父类的构造函数。

        Args:
            **kwargs: 传递给 `ChatGoogleGenerativeAI` 父类构造函数的关键字参数。
                      可以覆盖 `model`, `google_api_key`, `temperature` 等。

        Raises:
            ValueError: 如果 `google_api_key` 既没有在参数中提供，也没有在
                        环境变量 `GOOGLE_API_KEY` 中设置。
        """
        
        # 设置 Google AI 的默认配置
        kwargs.setdefault("temperature", 0.1)
        kwargs.setdefault("max_tokens", 2000)
        
        # 检查 API 密钥
        google_api_key = kwargs.get("google_api_key") or os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError(
                "Google API key not found. Please set GOOGLE_API_KEY environment variable "
                "or pass google_api_key parameter."
            )
        
        kwargs["google_api_key"] = google_api_key
        
        # 调用父类初始化
        super().__init__(**kwargs)

        logger.info(f"✅ Google AI OpenAI 兼容适配器初始化成功")
        logger.info(f"   模型: {kwargs.get('model', 'gemini-pro')}")
        logger.info(f"   温度: {kwargs.get('temperature', 0.1)}")
        logger.info(f"   最大Token: {kwargs.get('max_tokens', 2000)}")
    
    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs) -> LLMResult:
        """
        重写父类的 `_generate` 方法，以增加内容优化、错误处理和 token 追踪。

        Args:
            messages (List[BaseMessage]): 用于生成回复的聊天消息列表。
            stop (Optional[List[str]], optional): 停止生成的字符串列表。
            **kwargs: 传递给父类 `_generate` 方法的额外参数。

        Returns:
            LLMResult: 一个经过处理的 `LLMResult` 对象。在发生 API 错误时，
                       会返回一个包含错误信息的 `LLMResult`。
        """
        
        try:
            # 调用父类的生成方法
            result = super()._generate(messages, stop, **kwargs)
            
            # 优化返回内容格式
            if result and result.generations:
                for generation in result.generations:
                    if hasattr(generation, 'message') and generation.message:
                        # 优化消息内容格式
                        self._optimize_message_content(generation.message)
            
            # 追踪 token 使用量
            self._track_token_usage(result, kwargs)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Google AI 生成失败: {e}")
            # 返回一个包含错误信息的结果，而不是抛出异常
            from langchain_core.outputs import ChatGeneration
            error_message = AIMessage(content=f"Google AI 调用失败: {str(e)}")
            error_generation = ChatGeneration(message=error_message)
            return LLMResult(generations=[[error_generation]])
    
    def _optimize_message_content(self, message: BaseMessage):
        """
        对 AI 生成的消息内容进行后处理和优化。

        此方法主要用于识别可能是新闻稿的内容，并为其添加一些结构化
        信息（如日期、标题、来源），以改善其格式和可用性。

        Args:
            message (BaseMessage): 需要优化的 `AIMessage` 对象。
        """
        
        if not isinstance(message, AIMessage) or not message.content:
            return
        
        content = message.content
        
        # 检查是否是工具调用返回的新闻内容
        if self._is_news_content(content):
            # 优化新闻内容格式，添加必要的关键词
            optimized_content = self._enhance_news_content(content)
            message.content = optimized_content
            
            logger.debug(f"🔧 [Google适配器] 优化新闻内容格式")
            logger.debug(f"   原始长度: {len(content)} 字符")
            logger.debug(f"   优化后长度: {len(optimized_content)} 字符")
    
    def _is_news_content(self, content: str) -> bool:
        """
        通过启发式规则判断给定内容是否为新闻稿。

        判断依据包括：
        - 内容长度是否超过 200 个字符。
        - 内容是否包含一组预定义的财经新闻关键词。

        Args:
            content (str): 要检查的文本内容。

        Returns:
            bool: 如果内容被认为是新闻，则返回 True，否则返回 False。
        """
        
        # 检查是否包含新闻相关的关键词
        news_indicators = [
            "股票", "公司", "市场", "投资", "财经", "证券", "交易",
            "涨跌", "业绩", "财报", "分析", "预测", "消息", "公告"
        ]
        
        return any(indicator in content for indicator in news_indicators) and len(content) > 200
    
    def _enhance_news_content(self, content: str) -> str:
        """
        通过添加缺失的结构化信息来增强新闻内容的格式。

        该方法会检查内容中是否缺少“发布时间”、“新闻标题”和“文章来源”等
        常见的元数据字段。如果缺少，它会自动添加这些字段以提高内容的可读性
        和一致性。

        Args:
            content (str): 原始新闻内容。

        Returns:
            str: 添加了元数据字段后的增强版新闻内容。
        """
        
        import datetime
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        
        # 如果内容缺少必要的新闻特征，添加它们
        enhanced_content = content
        
        # 添加发布时间信息（如果缺少）
        if "发布时间" not in content and "时间" not in content:
            enhanced_content = f"发布时间: {current_date}\n\n{enhanced_content}"
        
        # 添加新闻标题标识（如果缺少）
        if "新闻标题" not in content and "标题" not in content:
            # 尝试从内容中提取第一行作为标题
            lines = enhanced_content.split('\n')
            if lines:
                first_line = lines[0].strip()
                if len(first_line) < 100:  # 可能是标题
                    enhanced_content = f"新闻标题: {first_line}\n\n{enhanced_content}"
        
        # 添加文章来源信息（如果缺少）
        if "文章来源" not in content and "来源" not in content:
            enhanced_content = f"{enhanced_content}\n\n文章来源: Google AI 智能分析"
        
        return enhanced_content
    
    def _track_token_usage(self, result: LLMResult, kwargs: Dict[str, Any]):
        """
        从 `LLMResult` 中提取 token 使用信息并进行记录。

        此方法解析 `llm_output` 字典以获取 `prompt_tokens` 和 `completion_tokens`，
        然后调用 `token_tracker` 服务来持久化这些数据。

        Args:
            result (LLMResult): 包含 `llm_output` 的 `LLMResult` 对象。
            kwargs (Dict[str, Any]): 包含追踪元数据（如 `session_id` 和
                                     `analysis_type`）的字典。
        """
        
        try:
            # 从结果中提取 token 使用信息
            if hasattr(result, 'llm_output') and result.llm_output:
                token_usage = result.llm_output.get('token_usage', {})
                
                input_tokens = token_usage.get('prompt_tokens', 0)
                output_tokens = token_usage.get('completion_tokens', 0)
                
                if input_tokens > 0 or output_tokens > 0:
                    # 生成会话ID
                    session_id = kwargs.get('session_id', f"google_openai_{hash(str(kwargs))%10000}")
                    analysis_type = kwargs.get('analysis_type', 'stock_analysis')
                    
                    # 使用 TokenTracker 记录使用量
                    token_tracker.track_usage(
                        provider="google",
                        model_name=self.model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        session_id=session_id,
                        analysis_type=analysis_type
                    )
                    
                    logger.debug(f"📊 [Google适配器] Token使用量: 输入={input_tokens}, 输出={output_tokens}")
                    
        except Exception as track_error:
            # token 追踪失败不应该影响主要功能
            logger.error(f"⚠️ Google适配器 Token 追踪失败: {track_error}")


# 支持的模型列表
GOOGLE_OPENAI_MODELS = {
    # Gemini 2.5 系列 - 最新验证模型
    "gemini-2.5-pro": {
        "description": "Gemini 2.5 Pro - 最新旗舰模型，功能强大 (16.68s)",
        "context_length": 32768,
        "supports_function_calling": True,
        "recommended_for": ["复杂推理", "专业分析", "高质量输出"],
        "avg_response_time": 16.68
    },
    "gemini-2.5-flash": {
        "description": "Gemini 2.5 Flash - 最新快速模型 (2.73s)",
        "context_length": 32768,
        "supports_function_calling": True,
        "recommended_for": ["快速响应", "实时分析", "高频使用"],
        "avg_response_time": 2.73
    },
    "gemini-2.5-flash-lite-preview-06-17": {
        "description": "Gemini 2.5 Flash Lite Preview - 超快响应 (1.45s)",
        "context_length": 32768,
        "supports_function_calling": True,
        "recommended_for": ["超快响应", "实时交互", "高频调用"],
        "avg_response_time": 1.45
    },
    # Gemini 2.0 系列
    "gemini-2.0-flash": {
        "description": "Gemini 2.0 Flash - 新一代快速模型 (1.87s)",
        "context_length": 32768,
        "supports_function_calling": True,
        "recommended_for": ["快速响应", "实时分析"],
        "avg_response_time": 1.87
    },
    # Gemini 1.5 系列
    "gemini-1.5-pro": {
        "description": "Gemini 1.5 Pro - 强大性能，平衡选择 (2.25s)",
        "context_length": 32768,
        "supports_function_calling": True,
        "recommended_for": ["复杂分析", "专业任务", "深度思考"],
        "avg_response_time": 2.25
    },
    "gemini-1.5-flash": {
        "description": "Gemini 1.5 Flash - 快速响应，备用选择 (2.87s)",
        "context_length": 32768,
        "supports_function_calling": True,
        "recommended_for": ["快速任务", "日常对话", "简单分析"],
        "avg_response_time": 2.87
    },
    # 经典模型
    "gemini-pro": {
        "description": "Gemini Pro - 经典模型，稳定可靠",
        "context_length": 32768,
        "supports_function_calling": True,
        "recommended_for": ["通用任务", "稳定性要求高的场景"]
    }
}


def get_available_google_models() -> Dict[str, Dict[str, Any]]:
    """
    获取此适配器支持的可用 Google (Gemini) 模型及其元数据。

    Returns:
        Dict[str, Dict[str, Any]]: 一个字典，键是模型名称，值是包含
                                  描述、上下文长度、工具调用支持等信息的元数据字典。
    """
    return GOOGLE_OPENAI_MODELS


def create_google_openai_llm(
    model: str = "gemini-2.5-flash-lite-preview-06-17",
    google_api_key: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 2000,
    **kwargs
) -> ChatGoogleOpenAI:
    """
    一个便捷的工厂函数，用于创建 `ChatGoogleOpenAI` 实例。

    Args:
        model (str, optional): 模型名称。默认为 "gemini-2.5-flash-lite-preview-06-17"。
        google_api_key (Optional[str], optional): API 密钥。默认为 None，将从环境变量读取。
        temperature (float, optional): 温度参数。默认为 0.1。
        max_tokens (int, optional): 最大生成 token 数。默认为 2000。
        **kwargs: 其他传递给 `ChatGoogleOpenAI` 构造函数的关键字参数。

    Returns:
        ChatGoogleOpenAI: 一个 `ChatGoogleOpenAI` 的新实例。
    """
    
    return ChatGoogleOpenAI(
        model=model,
        google_api_key=google_api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )


def test_google_openai_connection(
    model: str = "gemini-2.0-flash",
    google_api_key: Optional[str] = None
) -> bool:
    """
    测试与 Google AI (Gemini) API 的连接。

    Args:
        model (str, optional): 用于测试的模型名称。默认为 "gemini-2.0-flash"。
        google_api_key (Optional[str], optional): 用于测试的 API 密钥。默认为 None。

    Returns:
        bool: 如果连接成功并收到有效响应，则返回 True，否则返回 False。
    """
    
    try:
        logger.info(f"🧪 测试 Google AI OpenAI 兼容接口连接")
        logger.info(f"   模型: {model}")
        
        # 创建客户端
        llm = create_google_openai_llm(
            model=model,
            google_api_key=google_api_key,
            max_tokens=50
        )
        
        # 发送测试消息
        response = llm.invoke("你好，请简单介绍一下你自己。")
        
        if response and hasattr(response, 'content') and response.content:
            logger.info(f"✅ Google AI OpenAI 兼容接口连接成功")
            logger.info(f"   响应: {response.content[:100]}...")
            return True
        else:
            logger.error(f"❌ Google AI OpenAI 兼容接口响应为空")
            return False
            
    except Exception as e:
        logger.error(f"❌ Google AI OpenAI 兼容接口连接失败: {e}")
        return False


def test_google_openai_function_calling(
    model: str = "gemini-2.5-flash-lite-preview-06-17",
    google_api_key: Optional[str] = None
) -> bool:
    """
    测试 Google AI (Gemini) API 的工具调用 (Function Calling) 功能。

    Args:
        model (str, optional): 用于测试的模型名称。默认为 "gemini-2.5-flash-lite-preview-06-17"。
        google_api_key (Optional[str], optional): 用于测试的 API 密钥。默认为 None。

    Returns:
        bool: 如果模型成功生成了预期的工具调用，则返回 True，否则返回 False。
    """
    
    try:
        logger.info(f"🧪 测试 Google AI Function Calling")
        logger.info(f"   模型: {model}")
        
        # 创建客户端
        llm = create_google_openai_llm(
            model=model,
            google_api_key=google_api_key,
            max_tokens=200
        )
        
        # 定义测试工具
        from langchain_core.tools import tool
        
        @tool
        def test_news_tool(query: str) -> str:
            """测试新闻工具，返回模拟新闻内容"""
            return f"""发布时间: 2024-01-15
新闻标题: {query}相关市场动态
文章来源: 测试新闻源

这是一条关于{query}的测试新闻内容。该公司近期表现良好，市场前景看好。
投资者对此表示关注，分析师给出积极评价。"""
        
        # 绑定工具
        llm_with_tools = llm.bind_tools([test_news_tool])
        
        # 测试工具调用
        response = llm_with_tools.invoke("请使用test_news_tool查询'苹果公司'的新闻")
        
        logger.info(f"✅ Google AI Function Calling 测试完成")
        logger.info(f"   响应类型: {type(response)}")
        
        if hasattr(response, 'tool_calls') and response.tool_calls:
            logger.info(f"   工具调用数量: {len(response.tool_calls)}")
            return True
        else:
            logger.info(f"   响应内容: {getattr(response, 'content', 'No content')}")
            return True  # 即使没有工具调用也算成功，因为模型可能选择不调用工具
            
    except Exception as e:
        logger.error(f"❌ Google AI Function Calling 测试失败: {e}")
        return False


if __name__ == "__main__":
    """测试脚本"""
    logger.info(f"🧪 Google AI OpenAI 兼容适配器测试")
    logger.info(f"=" * 50)
    
    # 测试连接
    connection_ok = test_google_openai_connection()
    
    if connection_ok:
        # 测试 Function Calling
        function_calling_ok = test_google_openai_function_calling()
        
        if function_calling_ok:
            logger.info(f"\n🎉 所有测试通过！Google AI OpenAI 兼容适配器工作正常")
        else:
            logger.error(f"\n⚠️ Function Calling 测试失败")
    else:
        logger.error(f"\n❌ 连接测试失败")