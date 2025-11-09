#!/usr/bin/env python3
"""
DeepSeek 直接适配器模块。

该模块提供了一个 `DeepSeekDirectAdapter` 类，它不依赖于 `langchain` 或
`langchain_openai`，而是直接使用 `openai` 官方 Python 库来与 DeepSeek 的
OpenAI 兼容 API 端点进行交互。

创建此适配器的主要动机是避免潜在的 `DefaultHttpxClient` 兼容性问题，
并提供一个更轻量级、依赖更少的解决方案来调用 DeepSeek 模型。
"""

import os
import json
from typing import Any, Dict, List, Optional, Union
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

import logging
logger = logging.getLogger(__name__)

class DeepSeekDirectAdapter:
    """
    一个直接使用 `openai` 库与 DeepSeek API 进行交互的适配器。

    此类封装了对 DeepSeek API 的直接调用，处理 API 密钥管理、
    客户端初始化以及发送聊天请求。它提供了一个简单的 `invoke` 方法
    来发送请求并获取响应。

    **主要特点:**
    - **轻量级:** 仅依赖 `openai` 和 `python-dotenv` 库。
    - **兼容性:** 避免了 `langchain` 中可能出现的 HTTP 客户端兼容性问题。
    - **简单易用:** 提供了类似于 LangChain 的 `invoke` 和 `chat` 方法。
    """
    
    def __init__(
        self,
        model: str = "deepseek-chat",
        temperature: float = 0.1,
        max_tokens: int = 1000,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com"
    ):
        """
        初始化 `DeepSeekDirectAdapter`。

        Args:
            model (str, optional): 要使用的 DeepSeek 模型名称。默认为 "deepseek-chat"。
            temperature (float, optional): 控制生成文本的随机性。默认为 0.1。
            max_tokens (int, optional): 生成的最大 token 数量。默认为 1000。
            api_key (Optional[str], optional): DeepSeek API 密钥。如果为 None, 将从环境变量 `DEEPSEEK_API_KEY` 读取。
            base_url (str, optional): DeepSeek API 的基础 URL。默认为 "https://api.deepseek.com"。

        Raises:
            ValueError: 如果 API 密钥既没有在参数中提供，也没有在环境变量中设置。
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # 获取API密钥
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("未找到DEEPSEEK_API_KEY，请在.env文件中配置或通过参数传入")
        
        # 创建OpenAI客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url
        )
        
        logger.info(f"✅ DeepSeek直接适配器初始化成功，模型: {model}")
    
    def invoke(self, messages: Union[str, List[Dict[str, str]]]) -> str:
        """
        调用 DeepSeek API 以获取聊天响应。

        此方法接受字符串或 OpenAI 格式的消息列表作为输入，然后调用
        `openai` 客户端的 `chat.completions.create` 方法。

        Args:
            messages (Union[str, List[Dict[str, str]]]): 输入的提示。
                可以是一个简单的字符串（将被视为用户消息），或者是一个
                符合 OpenAI API 格式的字典列表 (e.g., `[{"role": "user", "content": "..."}]`)。

        Returns:
            str: 模型生成的文本响应内容。

        Raises:
            ValueError: 如果输入的 `messages` 不是支持的格式。
            Exception: 如果 API 调用失败。
        """
        try:
            # 处理输入消息格式
            if isinstance(messages, str):
                formatted_messages = [{"role": "user", "content": messages}]
            elif isinstance(messages, list):
                formatted_messages = messages
            else:
                raise ValueError(f"不支持的消息格式: {type(messages)}")
            
            # 调用API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=formatted_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            result = response.choices[0].message.content
            logger.debug(f"DeepSeek API调用成功，响应长度: {len(result)}")
            return result
            
        except Exception as e:
            logger.error(f"DeepSeek API调用失败: {e}")
            raise
    
    def chat(self, message: str) -> str:
        """
        一个简单的聊天接口，是对 `invoke` 方法的封装。

        Args:
            message (str): 用户的输入消息字符串。

        Returns:
            str: 模型的文本响应。
        """
        return self.invoke(message)
    
    def analyze_with_tools(self, query: str, tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        使用提供的工具信息来引导模型进行分析。

        **注意:** 此方法不执行实际的工具调用。它通过在提示中描述可用工具
        来模拟工具的使用，让模型在其分析和推理过程中“考虑”这些工具。
        这是一种基于提示工程的工具使用模拟，而不是真正的函数调用。

        Args:
            query (str): 需要分析的用户查询。
            tools (List[Dict[str, Any]]): 一个描述可用工具的字典列表。
                                         每个字典应包含 'name' 和 'description'。

        Returns:
            Dict[str, Any]: 一个包含分析结果、查询、所用工具列表（模拟）和状态的字典。
        """
        try:
            # 构建包含工具信息的提示
            tools_description = "\n".join([
                f"- {tool.get('name', 'Unknown')}: {tool.get('description', 'No description')}"
                for tool in tools
            ])
            
            prompt = f"""
你是一个专业的股票分析师。请根据以下查询进行分析：

查询：{query}

可用工具：
{tools_description}

请提供详细的分析结果，包括：
1. 分析思路
2. 关键发现
3. 投资建议
4. 风险提示

请用中文回答。
"""
            
            response = self.invoke(prompt)
            
            return {
                "query": query,
                "analysis": response,
                "tools_used": [tool.get('name') for tool in tools],
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"工具分析失败: {e}")
            return {
                "query": query,
                "analysis": f"分析失败: {str(e)}",
                "tools_used": [],
                "status": "error"
            }

def create_deepseek_direct_adapter(
    model: str = "deepseek-chat",
    temperature: float = 0.1,
    max_tokens: int = 1000,
    **kwargs
) -> DeepSeekDirectAdapter:
    """
    一个便捷的工厂函数，用于创建 `DeepSeekDirectAdapter` 实例。

    Args:
        model (str, optional): 模型名称。默认为 "deepseek-chat"。
        temperature (float, optional): 温度参数。默认为 0.1。
        max_tokens (int, optional): 最大生成 token 数。默认为 1000。
        **kwargs: 其他传递给 `DeepSeekDirectAdapter` 构造函数的关键字参数。

    Returns:
        DeepSeekDirectAdapter: 一个 `DeepSeekDirectAdapter` 的新实例。
    """
    return DeepSeekDirectAdapter(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )