"""
TradingAgents Graph 模块

该模块是 `TradingAgents` 项目的核心, 负责构建、管理和执行基于图的
智能代理工作流。它定义了构成交易决策流程的各个组件, 如状态的传播、
条件的判断、信号的处理和流程的自我反思/修正。

通过将复杂的决策过程分解为图中的节点和边, 该模块实现了高度模块化、
可扩展和可维护的代理架构。

主要组件:
- `TradingAgentsGraph`: 图的核心结构, 负责管理整个工作流程的执行。
- `ConditionalLogic`: 实现图中的条件分支逻辑。
- `GraphSetup`: 负责图的初始化和配置。
- `Propagator`: 管理信息和状态在图节点之间的传播。
- `Reflector`: 实现工作流的自我反思和动态调整机制。
- `SignalProcessor`: 处理和转换在图中流动的各种信号和数据。
"""
# TradingAgents/graph/__init__.py

from .trading_graph import TradingAgentsGraph
from .conditional_logic import ConditionalLogic
from .setup import GraphSetup
from .propagation import Propagator
from .reflection import Reflector
from .signal_processing import SignalProcessor

# 导入统一日志系统
from tradingagents.utils.logging_init import get_logger
logger = get_logger("default")

__all__ = [
    "TradingAgentsGraph",
    "ConditionalLogic",
    "GraphSetup",
    "Propagator",
    "Reflector",
    "SignalProcessor",
]
