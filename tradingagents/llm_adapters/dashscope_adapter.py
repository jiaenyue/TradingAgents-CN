"""
阿里百炼大模型 (DashScope) 适配器模块。

该模块为阿里巴巴的 DashScope (通义千问) 系列模型提供了一个与 LangChain 兼容的
`BaseChatModel` 接口。它处理与 DashScope API 的通信、消息格式转换、
API 密钥管理以及 token 使用量的跟踪。

注意：这个基础适配器不直接支持 DashScope 的原生工具调用 (Function Calling)，
工具的绑定和处理需要在应用层面实现。如需原生工具调用支持，请使用
`dashscope_openai_adapter.py` 中提供的 OpenAI 兼容适配器。
"""

import os
import json
from typing import Any, Dict, List, Optional, Union, Iterator, AsyncIterator, Sequence
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.callbacks.manager import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field, SecretStr
import dashscope
from dashscope import Generation
from ..config.config_manager import token_tracker

# 导入日志模块
from tradingagents.utils.logging_manager import get_logger
logger = get_logger('agents')



class ChatDashScope(BaseChatModel):
    """
    一个与 LangChain 兼容的阿里百炼 (DashScope) 大模型聊天适配器。

    该类继承自 `BaseChatModel`，实现了与 DashScope API 交互所需的核心方法。
    它负责将 LangChain 的消息格式转换为 DashScope API 所需的格式，并解析
    API 的响应。

    **主要功能:**
    - 通过 `_generate` 方法与 DashScope API 进行同步通信。
    - 通过 `_convert_messages_to_dashscope_format` 处理消息格式转换。
    - 从环境变量或构造函数参数中自动管理 `DASHSCOPE_API_KEY`。
    - 使用 `token_tracker` 记录每次调用的 token 使用情况。
    - 提供一个 `bind_tools` 方法的存根，但实际的工具调用需要应用层逻辑支持。

    **使用示例:**
    ```python
    from tradingagents.llm_adapters.dashscope_adapter import ChatDashScope
    from langchain_core.messages import HumanMessage

    # 初始化模型
    llm = ChatDashScope(model="qwen-plus")

    # 发送消息
    response = llm.invoke([HumanMessage(content="你好，通义千问！")])
    print(response.content)
    ```
    """
    
    # 模型配置
    model: str = Field(default="qwen-turbo", description="DashScope 模型名称")
    api_key: Optional[SecretStr] = Field(default=None, description="DashScope API 密钥")
    temperature: float = Field(default=0.1, description="生成温度")
    max_tokens: int = Field(default=2000, description="最大生成token数")
    top_p: float = Field(default=0.9, description="核采样参数")
    
    # 内部属性
    _client: Any = None
    
    def __init__(self, **kwargs):
        """
        初始化 `ChatDashScope` 实例。

        此构造函数会设置 DashScope 的 API 密钥。密钥的来源优先级为：
        1. 构造函数中传入的 `api_key` 参数。
        2. 环境变量 `DASHSCOPE_API_KEY`。

        Args:
            **kwargs: 传递给 `pydantic.BaseModel` 的关键字参数。

        Raises:
            ValueError: 如果 API 密钥既没有在参数中提供，也没有在环境变量中设置。
        """
        super().__init__(**kwargs)
        
        # 设置API密钥
        api_key = self.api_key
        if api_key is None:
            api_key = os.getenv("DASHSCOPE_API_KEY")
        
        if api_key is None:
            raise ValueError(
                "DashScope API key not found. Please set DASHSCOPE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # 配置 DashScope
        if isinstance(api_key, SecretStr):
            dashscope.api_key = api_key.get_secret_value()
        else:
            dashscope.api_key = api_key
    
    @property
    def _llm_type(self) -> str:
        """返回一个标识 LLM 类型的字符串。"""
        return "dashscope"
    
    def _convert_messages_to_dashscope_format(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """
        将 LangChain 的 `BaseMessage` 对象列表转换为 DashScope API 所需的字典列表格式。

        该方法会处理不同类型的消息（System, Human, AI），并将它们映射到
        DashScope API 对应的角色（"system", "user", "assistant"）。

        Args:
            messages (List[BaseMessage]): 要转换的 LangChain 消息列表。

        Returns:
            List[Dict[str, str]]: 一个符合 DashScope API 格式的字典列表。
        """
        dashscope_messages = []
        
        for message in messages:
            if isinstance(message, SystemMessage):
                role = "system"
            elif isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            else:
                # 默认作为用户消息处理
                role = "user"
            
            content = message.content
            if isinstance(content, list):
                # 处理多模态内容，目前只提取文本
                text_content = ""
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_content += item.get("text", "")
                content = text_content
            
            dashscope_messages.append({
                "role": role,
                "content": str(content)
            })
        
        return dashscope_messages
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        通过调用 DashScope API 生成聊天回复（同步模式）。

        这是实现 `BaseChatModel` 所必需的核心方法。它执行以下步骤：
        1. 将 LangChain 消息转换为 DashScope 格式。
        2. 构建 API 请求参数，包括模型名称、温度等。
        3. 调用 `dashscope.Generation.call` 方法发送请求。
        4. 解析 API 响应，提取生成的内容和 token 使用量。
        5. 使用 `token_tracker` 记录 token 消耗。
        6. 将结果封装在 `ChatResult` 对象中返回。

        Args:
            messages (List[BaseMessage]): 用于生成回复的聊天消息列表。
            stop (Optional[List[str]], optional): 停止生成的字符串列表。
            run_manager (Optional[CallbackManagerForLLMRun], optional): LangChain 的回调管理器。
            **kwargs (Any): 其他传递给 DashScope API 的参数。

        Returns:
            ChatResult: 包含生成结果的 `ChatResult` 对象。

        Raises:
            Exception: 如果 API 调用失败或返回非 200 状态码。
        """
        
        # 转换消息格式
        dashscope_messages = self._convert_messages_to_dashscope_format(messages)
        
        # 准备请求参数
        request_params = {
            "model": self.model,
            "messages": dashscope_messages,
            "result_format": "message",
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }
        
        # 添加停止词
        if stop:
            request_params["stop"] = stop
        
        # 合并额外参数
        request_params.update(kwargs)
        
        try:
            # 调用 DashScope API
            response = Generation.call(**request_params)
            
            if response.status_code == 200:
                # 解析响应
                output = response.output
                message_content = output.choices[0].message.content
                
                # 提取token使用量信息
                input_tokens = 0
                output_tokens = 0
                
                # DashScope API响应中包含usage信息
                if hasattr(response, 'usage') and response.usage:
                    usage = response.usage
                    # 根据API文档，usage可能包含input_tokens和output_tokens
                    if hasattr(usage, 'input_tokens'):
                        input_tokens = usage.input_tokens
                    if hasattr(usage, 'output_tokens'):
                        output_tokens = usage.output_tokens
                    # 有些情况下可能是total_tokens
                    elif hasattr(usage, 'total_tokens'):
                        # 估算输入和输出token（如果没有分别提供）
                        total_tokens = usage.total_tokens
                        # 简单估算：假设输入占30%，输出占70%
                        input_tokens = int(total_tokens * 0.3)
                        output_tokens = int(total_tokens * 0.7)
                
                # 记录token使用量
                if input_tokens > 0 or output_tokens > 0:
                    try:
                        # 生成会话ID（如果没有提供）
                        session_id = kwargs.get('session_id', f"dashscope_{hash(str(messages))%10000}")
                        analysis_type = kwargs.get('analysis_type', 'stock_analysis')
                        
                        # 使用TokenTracker记录使用量
                        token_tracker.track_usage(
                            provider="dashscope",
                            model_name=self.model,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            session_id=session_id,
                            analysis_type=analysis_type
                        )
                    except Exception as track_error:
                        # 记录失败不应该影响主要功能
                        logger.info(f"Token tracking failed: {track_error}")
                
                # 创建 AI 消息
                ai_message = AIMessage(content=message_content)
                
                # 创建生成结果
                generation = ChatGeneration(message=ai_message)
                
                return ChatResult(generations=[generation])
            else:
                raise Exception(f"DashScope API error: {response.code} - {response.message}")
                
        except Exception as e:
            raise Exception(f"Error calling DashScope API: {str(e)}")
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        通过调用 DashScope API 生成聊天回复（异步模式）。

        注意：当前实现为了简单起见，直接调用了同步的 `_generate` 方法。
        在需要高并发 I/O 的场景下，应将其替换为真正的异步实现（例如，使用 `aiohttp`）。

        Args:
            messages (List[BaseMessage]): 用于生成回复的聊天消息列表。
            stop (Optional[List[str]], optional): 停止生成的字符串列表。
            run_manager (Optional[AsyncCallbackManagerForLLMRun], optional): LangChain 的异步回调管理器。
            **kwargs (Any): 其他传递给 DashScope API 的参数。

        Returns:
            ChatResult: 包含生成结果的 `ChatResult` 对象。
        """
        # 目前使用同步方法，后续可以实现真正的异步
        return self._generate(messages, stop, None, **kwargs)
    
    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], type, BaseTool]],
        **kwargs: Any,
    ) -> "ChatDashScope":
        """
        将工具“绑定”到模型。

        重要提示：此版本的 `ChatDashScope` 适配器不直接支持 DashScope API 的原生
        工具调用 (Function Calling)。此方法仅将工具信息存储在模型实例上，
        但不会在 API 调用中实际使用它们。工具调用的逻辑需要由应用层代码
        （例如，通过特定的提示工程）来处理。

        如需原生工具调用支持，请使用 `ChatDashScopeOpenAI` 适配器。

        Args:
            tools (Sequence[Union[Dict[str, Any], type, BaseTool]]): 要绑定的工具序列。
            **kwargs (Any): 额外的关键字参数。

        Returns:
            ChatDashScope: 一个新的 `ChatDashScope` 实例，其中包含了格式化的工具信息。
        """
        # 注意：DashScope 目前不直接支持工具调用
        # 这里我们返回一个新的实例，但实际上工具调用需要在应用层处理
        formatted_tools = []
        for tool in tools:
            if hasattr(tool, "name") and hasattr(tool, "description"):
                # 这是一个 BaseTool 实例
                formatted_tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": getattr(tool, "args_schema", {})
                })
            elif isinstance(tool, dict):
                formatted_tools.append(tool)
            else:
                # 尝试转换为 OpenAI 工具格式
                try:
                    formatted_tools.append(convert_to_openai_tool(tool))
                except Exception:
                    pass

        # 创建新实例，保存工具信息
        new_instance = self.__class__(
            model=self.model,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            **kwargs
        )
        new_instance._tools = formatted_tools
        return new_instance

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """
        返回用于标识此 LLM 实例的唯一参数字典。

        这些参数用于 LangChain 内部的缓存和识别机制。

        Returns:
            Dict[str, Any]: 包含模型名称和关键生成参数的字典。
        """
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }


# 支持的模型列表
DASHSCOPE_MODELS = {
    # 通义千问系列
    "qwen-turbo": {
        "description": "通义千问 Turbo - 快速响应，适合日常对话",
        "context_length": 8192,
        "recommended_for": ["快速任务", "日常对话", "简单分析"]
    },
    "qwen-plus": {
        "description": "通义千问 Plus - 平衡性能和成本",
        "context_length": 32768,
        "recommended_for": ["复杂分析", "专业任务", "深度思考"]
    },
    "qwen-max": {
        "description": "通义千问 Max - 最强性能",
        "context_length": 32768,
        "recommended_for": ["最复杂任务", "专业分析", "高质量输出"]
    },
    "qwen-max-longcontext": {
        "description": "通义千问 Max 长文本版 - 支持超长上下文",
        "context_length": 1000000,
        "recommended_for": ["长文档分析", "大量数据处理", "复杂推理"]
    },
}


def get_available_models() -> Dict[str, Dict[str, Any]]:
    """
    获取此适配器支持的可用 DashScope 模型及其元数据。

    Returns:
        Dict[str, Dict[str, Any]]: 一个字典，键是模型名称，值是包含
                                  描述、上下文长度和推荐用途的元数据字典。
    """
    return DASHSCOPE_MODELS


def create_dashscope_llm(
    model: str = "qwen-plus",
    api_key: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 2000,
    **kwargs
) -> ChatDashScope:
    """
    一个便捷的工厂函数，用于创建 `ChatDashScope` 实例。

    Args:
        model (str, optional): 模型名称。默认为 "qwen-plus"。
        api_key (Optional[str], optional): DashScope API 密钥。默认为 None，将从环境变量读取。
        temperature (float, optional): 温度参数。默认为 0.1。
        max_tokens (int, optional): 最大生成 token 数。默认为 2000。
        **kwargs: 其他传递给 `ChatDashScope` 构造函数的关键字参数。

    Returns:
        ChatDashScope: 一个 `ChatDashScope` 的新实例。
    """
    
    return ChatDashScope(
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )
