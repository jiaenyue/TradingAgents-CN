"""
LLM Adapters for TradingAgents.

This package contains various adapters to provide a consistent interface
for different Large Language Models (LLMs) used within the TradingAgents framework.
Each adapter handles the specific API and data formats of an LLM provider,
allowing the core application logic to remain agnostic to the underlying model.
"""
from .dashscope_adapter import ChatDashScope
from .dashscope_openai_adapter import ChatDashScopeOpenAI
from .google_openai_adapter import ChatGoogleOpenAI

__all__ = ["ChatDashScope", "ChatDashScopeOpenAI", "ChatGoogleOpenAI"]
