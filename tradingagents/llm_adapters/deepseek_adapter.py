"""
DeepSeek LLM 适配器模块，增加了对 Token 使用量的精确统计和成本计算功能。

该模块提供了一个 `ChatDeepSeek` 类，它继承自 `langchain_openai.ChatOpenAI`。
这个适配器的核心特性是重写了 `_generate` 方法，以便在每次调用 DeepSeek API 后，
能够捕获并记录详细的 token 使用信息（包括输入和输出 token 数），并根据预设的
价格计算该次调用的成本。

如果 API 响应中没有提供 token 使用量，该适配器还会回退到基于字符数的估算方法。
"""

import os
import time
from typing import Any, Dict, List, Optional, Union
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
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


class ChatDeepSeek(ChatOpenAI):
    """
    一个为 DeepSeek 聊天模型定制的 LangChain 适配器，增加了详细的 Token 使用统计功能。

    该类通过继承 `ChatOpenAI` 来利用其与 OpenAI 兼容 API 的交互能力。
    其主要增强之处在于覆盖了 `_generate` 方法，以实现以下功能：
    - **精确 Token 追踪:** 从 DeepSeek API 的响应中直接提取 `prompt_tokens` 和 `completion_tokens`。
    - **成本计算:** 利用 `token_tracker` 服务，根据使用的模型和 token 数量计算 API 调用成本。
    - **Token 估算:** 在 API 未返回 token 使用量时，提供一个基于字符数的备用估算方法。
    - **详细日志:** 记录每一次调用的 token 使用量和计算出的成本。
    """
    
    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        初始化 `ChatDeepSeek` 适配器。

        Args:
            model (str, optional): 要使用的 DeepSeek 模型名称。默认为 "deepseek-chat"。
            api_key (Optional[str], optional): DeepSeek API 密钥。如果为 None，将从环境变量 `DEEPSEEK_API_KEY` 读取。
            base_url (str, optional): DeepSeek API 的基础 URL。默认为 "https://api.deepseek.com"。
            temperature (float, optional): 控制生成文本的随机性。默认为 0.1。
            max_tokens (Optional[int], optional): 生成的最大 token 数量。默认为 None (由模型决定)。
            **kwargs: 其他传递给 `ChatOpenAI` 父类构造函数的关键字参数。

        Raises:
            ValueError: 如果 API 密钥既没有在参数中提供，也没有在环境变量中设置。
        """
        
        # 获取API密钥
        if api_key is None:
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError("DeepSeek API密钥未找到。请设置DEEPSEEK_API_KEY环境变量或传入api_key参数。")
        
        # 初始化父类
        super().__init__(
            model=model,
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        self.model_name = model
        
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        重写父类的 `_generate` 方法，以增加 token 使用量追踪和成本计算功能。

        该方法的核心流程如下：
        1. 调用父类 (`ChatOpenAI`) 的 `_generate` 方法来获取 LLM 的原始响应。
        2. 尝试从 `llm_output` 中提取精确的 `token_usage` 信息。
        3. 如果无法获取精确信息，则调用 `_estimate_input_tokens` 和 `_estimate_output_tokens`
           方法进行估算。
        4. 如果 `TOKEN_TRACKING_ENABLED` 为 True，则调用 `token_tracker.track_usage`
           来记录使用量并计算成本。
        5. 将成本信息和其他元数据通过日志系统记录下来。

        Args:
            messages (List[BaseMessage]): 用于生成回复的聊天消息列表。
            stop (Optional[List[str]], optional): 停止生成的字符串列表。
            run_manager (Optional[CallbackManagerForLLMRun], optional): LangChain 的回调管理器。
            **kwargs (Any): 其他传递给 API 的参数。支持自定义的 `session_id` 和 `analysis_type`
                             用于更精细的追踪。

        Returns:
            ChatResult: 从父类 `_generate` 方法返回的原始 `ChatResult` 对象。

        Raises:
            Exception: 如果 API 调用失败。
        """

        # 记录开始时间
        start_time = time.time()

        # 提取并移除自定义参数，避免传递给父类
        session_id = kwargs.pop('session_id', None)
        analysis_type = kwargs.pop('analysis_type', None)

        try:
            # 调用父类方法生成响应
            result = super()._generate(messages, stop, run_manager, **kwargs)
            
            # 提取token使用量
            input_tokens = 0
            output_tokens = 0
            
            # 尝试从响应中提取token使用量
            if hasattr(result, 'llm_output') and result.llm_output:
                token_usage = result.llm_output.get('token_usage', {})
                if token_usage:
                    input_tokens = token_usage.get('prompt_tokens', 0)
                    output_tokens = token_usage.get('completion_tokens', 0)
            
            # 如果没有获取到token使用量，进行估算
            if input_tokens == 0 and output_tokens == 0:
                input_tokens = self._estimate_input_tokens(messages)
                output_tokens = self._estimate_output_tokens(result)
                logger.debug(f"🔍 [DeepSeek] 使用估算token: 输入={input_tokens}, 输出={output_tokens}")
            else:
                logger.info(f"📊 [DeepSeek] 实际token使用: 输入={input_tokens}, 输出={output_tokens}")
            
            # 记录token使用量
            if TOKEN_TRACKING_ENABLED and (input_tokens > 0 or output_tokens > 0):
                try:
                    # 使用提取的参数或生成默认值
                    if session_id is None:
                        session_id = f"deepseek_{hash(str(messages))%10000}"
                    if analysis_type is None:
                        analysis_type = 'stock_analysis'

                    # 记录使用量
                    usage_record = token_tracker.track_usage(
                        provider="deepseek",
                        model_name=self.model_name,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        session_id=session_id,
                        analysis_type=analysis_type
                    )

                    if usage_record:
                        if usage_record.cost == 0.0:
                            logger.warning(f"⚠️ [DeepSeek] 成本计算为0，可能配置有问题")
                        else:
                            logger.info(f"💰 [DeepSeek] 本次调用成本: ¥{usage_record.cost:.6f}")

                        # 使用统一日志管理器的Token记录方法
                        logger_manager = get_logger_manager()
                        logger_manager.log_token_usage(
                            logger, "deepseek", self.model_name,
                            input_tokens, output_tokens, usage_record.cost,
                            session_id
                        )
                    else:
                        logger.warning(f"⚠️ [DeepSeek] 未创建使用记录")

                except Exception as track_error:
                    logger.error(f"⚠️ [DeepSeek] Token统计失败: {track_error}", exc_info=True)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ [DeepSeek] 调用失败: {e}", exc_info=True)
            raise
    
    def _estimate_input_tokens(self, messages: List[BaseMessage]) -> int:
        """
        估算输入消息列表的 token 数量。

        这是一个备用方法，仅在无法从 API 响应中获取精确 token 数时使用。
        它基于一个简单的启发式规则：计算所有消息内容的总字符数，然后
        除以一个估算的系数（此处为 2 字符/token）。

        Args:
            messages (List[BaseMessage]): 需要估算 token 数的输入消息列表。

        Returns:
            int: 估算出的输入 token 数量。
        """
        total_chars = 0
        for message in messages:
            if hasattr(message, 'content'):
                total_chars += len(str(message.content))
        
        # 粗略估算：中文约1.5字符/token，英文约4字符/token
        # 这里使用保守估算：2字符/token
        estimated_tokens = max(1, total_chars // 2)
        return estimated_tokens
    
    def _estimate_output_tokens(self, result: ChatResult) -> int:
        """
        估算输出结果的 token 数量。

        这是一个备用方法，仅在无法从 API 响应中获取精确 token 数时使用。
        它基于一个简单的启发式规则：计算 `ChatResult` 中所有生成内容的总字符数，
        然后除以一个估算的系数（此处为 2 字符/token）。

        Args:
            result (ChatResult): 包含了模型生成内容的 `ChatResult` 对象。

        Returns:
            int: 估算出的输出 token 数量。
        """
        total_chars = 0
        for generation in result.generations:
            if hasattr(generation, 'message') and hasattr(generation.message, 'content'):
                total_chars += len(str(generation.message.content))
        
        # 粗略估算：2字符/token
        estimated_tokens = max(1, total_chars // 2)
        return estimated_tokens
    
    def invoke(
        self,
        input: Union[str, List[BaseMessage]],
        config: Optional[Dict] = None,
        **kwargs: Any,
    ) -> AIMessage:
        """
        调用模型以生成响应，并确保 token 追踪参数得以传递。

        此方法重写了 LangChain 的标准 `invoke` 方法，以便在调用 `_generate` 之前
        正确处理输入，并确保 `session_id` 和 `analysis_type` 等自定义的
        追踪参数能够被传递下去。

        Args:
            input (Union[str, List[BaseMessage]]): 输入的提示字符串或消息列表。
            config (Optional[Dict], optional): LangChain 的运行时配置字典。
            **kwargs (Any): 额外的关键字参数，可包含 `session_id` 和 `analysis_type`。

        Returns:
            AIMessage: 模型生成的 AI 消息响应。
        """
        
        # 处理输入
        if isinstance(input, str):
            messages = [HumanMessage(content=input)]
        else:
            messages = input
        
        # 调用生成方法
        result = self._generate(messages, **kwargs)
        
        # 返回第一个生成结果的消息
        if result.generations:
            return result.generations[0].message
        else:
            return AIMessage(content="")


def create_deepseek_llm(
    model: str = "deepseek-chat",
    temperature: float = 0.1,
    max_tokens: Optional[int] = None,
    **kwargs
) -> ChatDeepSeek:
    """
    一个便捷的工厂函数，用于创建 `ChatDeepSeek` 实例。

    Args:
        model (str, optional): 模型名称。默认为 "deepseek-chat"。
        temperature (float, optional): 温度参数。默认为 0.1。
        max_tokens (Optional[int], optional): 最大生成 token 数。默认为 None。
        **kwargs: 其他传递给 `ChatDeepSeek` 构造函数的关键字参数。

    Returns:
        ChatDeepSeek: 一个 `ChatDeepSeek` 的新实例。
    """
    return ChatDeepSeek(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )


# 为了向后兼容，提供别名
DeepSeekLLM = ChatDeepSeek
