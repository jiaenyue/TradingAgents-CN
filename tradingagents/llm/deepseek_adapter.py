"""
DeepSeek V3 LLM 适配器模块。

该模块提供了一个 `DeepSeekAdapter` 类，用于与 DeepSeek V3 系列的语言模型进行交互。
它封装了模型初始化、智能体创建、工具绑定和聊天功能，使其易于在
`tradingagents` 框架中使用。适配器通过 `langchain_openai` 的 `ChatOpenAI`
类与 DeepSeek 的 OpenAI 兼容 API 端点进行通信。
"""

import os
import logging
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.schema import BaseMessage
from langchain.tools import BaseTool
from langchain.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

class DeepSeekAdapter:
    """
    DeepSeek V3 适配器类。

    该类封装了与 DeepSeek V3 模型的交互逻辑，提供了以下功能：
    - 初始化 LangChain 的 `ChatOpenAI` 以连接到 DeepSeek API。
    - 创建支持工具调用的 `AgentExecutor`。
    - 将工具绑定到 LLM 实例。
    - 提供一个简单的聊天接口。
    - 检查 API 可用性和测试连接。
    """
    
    # 支持的模型列表（专注于最适合股票分析的模型）
    SUPPORTED_MODELS = {
        "deepseek-chat": "deepseek-chat",      # 通用对话模型，最适合股票投资分析
        # 注意：deepseek-coder 虽然支持工具调用，但专注于代码任务，不如通用模型适合投资分析
        # 注意：deepseek-reasoner 不支持工具调用，因此不包含在此列表中
    }
    
    # DeepSeek API基础URL
    BASE_URL = "https://api.deepseek.com"
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model: str = "deepseek-chat",
        temperature: float = 0.1,
        max_tokens: int = 2000,
        base_url: Optional[str] = None
    ):
        """
        初始化 DeepSeek V3 适配器。

        Args:
            api_key (Optional[str], optional): DeepSeek API 密钥。如果未提供，将从环境变量 `DEEPSEEK_API_KEY` 读取。
            model (str, optional): 要使用的模型名称。默认为 "deepseek-chat"。
            temperature (float, optional): 控制生成文本的随机性。默认为 0.1。
            max_tokens (int, optional): 生成的最大 token 数量。默认为 2000。
            base_url (Optional[str], optional): API 的基础 URL。如果未提供，将从环境变量 `DEEPSEEK_BASE_URL` 读取，
                                              否则使用默认的 "https://api.deepseek.com"。

        Raises:
            ValueError: 如果 API 密钥未提供也未在环境变量中设置。
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url or os.getenv("DEEPSEEK_BASE_URL", self.BASE_URL)
        
        if not self.api_key:
            raise ValueError("需要提供DEEPSEEK_API_KEY")
        
        # 获取实际模型名称
        self.model = self.SUPPORTED_MODELS.get(model, "deepseek-chat")
        
        # 初始化LangChain模型
        self._init_llm()
        
        logger.info(f"DeepSeek V3适配器初始化完成，模型: {self.model}")
    
    def _init_llm(self):
        """
        初始化 LangChain LLM 实例。

        该方法尝试使用新旧不同版本的 `ChatOpenAI` 参数来实例化 LLM，
        以确保与不同 `langchain-openai` 版本的兼容性。

        Raises:
            Exception: 如果使用新旧两种参数格式都无法成功初始化 `ChatOpenAI`。
        """
        try:
            # 使用最新的LangChain OpenAI接口
            self.llm = ChatOpenAI(
                model=self.model,
                api_key=self.api_key,  # 新版本使用api_key而不是openai_api_key
                base_url=self.base_url,  # 新版本使用base_url而不是openai_api_base
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                streaming=False
            )
            logger.info("LangChain ChatOpenAI (DeepSeek)初始化成功")
        except Exception as e:
            # 尝试使用旧版本的参数名
            try:
                self.llm = ChatOpenAI(
                    model=self.model,
                    openai_api_key=self.api_key,
                    openai_api_base=self.base_url,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    streaming=False
                )
                logger.info("LangChain ChatOpenAI (DeepSeek)初始化成功 - 使用兼容模式")
            except Exception as e2:
                logger.error(f"初始化DeepSeek模型失败: {e}")
                logger.error(f"兼容模式也失败: {e2}")
                raise e
    
    def create_agent(
        self, 
        tools: List[BaseTool], 
        system_prompt: str,
        max_iterations: int = 10,
        verbose: bool = False
    ) -> AgentExecutor:
        """
        创建支持工具调用的智能体。

        该方法使用 `langchain` 的 `create_openai_functions_agent` 函数
        来构建一个能够利用所提供工具的代理。

        Args:
            tools (List[BaseTool]): 供智能体使用的工具列表。
            system_prompt (str): 定义智能体行为和目标的系统提示。
            max_iterations (int, optional): 智能体在得出结论前的最大迭代次数。默认为 10。
            verbose (bool, optional): 是否打印智能体的详细执行日志。默认为 False。

        Returns:
            AgentExecutor: 一个可执行的 LangChain 智能体实例。

        Raises:
            Exception: 如果在创建过程中发生任何错误。
        """
        try:
            # 创建提示词模板
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}")
            ])
            
            # 创建智能体
            agent = create_openai_functions_agent(
                llm=self.llm,
                tools=tools,
                prompt=prompt
            )
            
            # 创建智能体执行器
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                max_iterations=max_iterations,
                verbose=verbose,
                return_intermediate_steps=True,
                handle_parsing_errors=True
            )
            
            logger.info(f"智能体创建成功，工具数量: {len(tools)}")
            return agent_executor
            
        except Exception as e:
            logger.error(f"创建智能体失败: {e}")
            raise
    
    def bind_tools(self, tools: List[BaseTool]):
        """
        将工具绑定到 LLM 实例。

        这允许 LLM 在其响应中引用这些工具，通常用于工具调用。

        Args:
            tools (List[BaseTool]): 要绑定的工具列表。

        Returns:
            一个配置了工具的新 LLM 实例。
        """
        return self.llm.bind_tools(tools)
    
    def chat(
        self, 
        messages: List[BaseMessage], 
        **kwargs
    ) -> str:
        """
        提供一个直接的聊天接口。

        Args:
            messages (List[BaseMessage]): `langchain` 格式的消息列表。
            **kwargs: 传递给 `llm.invoke` 的其他关键字参数。

        Returns:
            str: 模型生成的回复内容。

        Raises:
            Exception: 如果 LLM 调用失败。
        """
        try:
            response = self.llm.invoke(messages, **kwargs)
            return response.content
        except Exception as e:
            logger.error(f"聊天调用失败: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取当前适配器配置的模型信息。

        Returns:
            Dict[str, Any]: 包含提供商、模型名称、温度等信息的字典。
        """
        return {
            "provider": "DeepSeek",
            "model": self.model,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "base_url": self.base_url,
            "supports_tools": True,
            "supports_streaming": False,
            "context_length": "128K" if "chat" in self.model else "64K"
        }
    
    @classmethod
    def get_available_models(cls) -> Dict[str, str]:
        """
        获取此适配器支持的可用模型列表。

        Returns:
            Dict[str, str]: 一个包含支持的模型名称的字典。
        """
        return cls.SUPPORTED_MODELS.copy()
    
    @staticmethod
    def is_available() -> bool:
        """
        检查 DeepSeek API 是否可用。

        可用性取决于 `DEEPSEEK_API_KEY` 和 `DEEPSEEK_ENABLED` (设置为 'true')
        这两个环境变量是否都已设置。

        Returns:
            bool: 如果 DeepSeek API 已配置并启用，则返回 True，否则返回 False。
        """
        api_key = os.getenv("DEEPSEEK_API_KEY")
        enabled = os.getenv("DEEPSEEK_ENABLED", "false").lower() == "true"
        
        return bool(api_key and enabled)
    
    def test_connection(self) -> bool:
        """
        测试与 DeepSeek API 的连接。

        该方法发送一条简单的测试消息，并检查是否能收到有效的回复。

        Returns:
            bool: 如果连接成功并收到回复，则返回 True，否则返回 False。
        """
        try:
            from langchain.schema import HumanMessage
            test_message = [HumanMessage(content="Hello, this is a test.")]
            response = self.chat(test_message)
            return bool(response)
        except Exception as e:
            logger.error(f"连接测试失败: {e}")
            return False


def create_deepseek_adapter(
    model: str = "deepseek-chat",
    temperature: float = 0.1,
    **kwargs
) -> DeepSeekAdapter:
    """
    一个便捷的工厂函数，用于创建 `DeepSeekAdapter` 实例。

    Args:
        model (str, optional): 模型名称。默认为 "deepseek-chat"。
        temperature (float, optional): 温度参数。默认为 0.1。
        **kwargs: 其他传递给 `DeepSeekAdapter` 构造函数的关键字参数。

    Returns:
        DeepSeekAdapter: 一个 `DeepSeekAdapter` 的新实例。
    """
    return DeepSeekAdapter(
        model=model,
        temperature=temperature,
        **kwargs
    )


# 导出主要类和函数
__all__ = [
    "DeepSeekAdapter",
    "create_deepseek_adapter"
]
