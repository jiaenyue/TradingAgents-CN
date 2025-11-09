"""
图状态的初始化与传播

该模块定义了 `Propagator` 类, 负责为 `TradingAgents` 图工作流创建
初始状态, 并提供图执行所需的配置参数。
"""
# TradingAgents/graph/propagation.py

from typing import Dict, Any

# 导入统一日志系统
from tradingagents.utils.logging_init import get_logger
logger = get_logger("default")
from tradingagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)


class Propagator:
    """
    处理图状态的初始化和在图中的传播。

    这个类的方法为图的启动提供了必要的初始数据结构和配置。
    """

    def __init__(self, max_recur_limit=100):
        """
        用配置参数进行初始化。

        Args:
            max_recur_limit (int): 图执行的最大递归深度限制, 用于防止无限循环。
        """
        self.max_recur_limit = max_recur_limit

    def create_initial_state(
        self, company_name: str, trade_date: str
    ) -> Dict[str, Any]:
        """
        为代理图创建一个初始状态字典。

        这个状态字典是 `AgentState` 的实例,包含了所有后续节点将要读取或
        修改的数据字段, 如公司名称、交易日期以及各种报告和辩论的初始空状态。

        Args:
            company_name (str): 初始要分析的公司名称。
            trade_date (str): 交易日期, 格式为 "YYYY-MM-DD"。

        Returns:
            Dict[str, Any]: 符合 `AgentState` 结构的初始状态字典。
        """
        return {
            "messages": [("human", company_name)],
            "company_of_interest": company_name,
            "trade_date": str(trade_date),
            "investment_debate_state": InvestDebateState(
                {"history": "", "current_response": "", "count": 0}
            ),
            "risk_debate_state": RiskDebateState(
                {
                    "history": "",
                    "current_risky_response": "",
                    "current_safe_response": "",
                    "current_neutral_response": "",
                    "count": 0,
                    "latest_speaker": "", # 初始化 latest_speaker
                }
            ),
            "market_report": "",
            "fundamentals_report": "",
            "sentiment_report": "",
            "news_report": "",
        }

    def get_graph_args(self) -> Dict[str, Any]:
        """
        获取用于调用图执行的参数。

        这些参数配置了图的执行模式 (例如, 流式输出) 和递归限制。

        Returns:
            Dict[str, Any]: 一个包含图调用配置的字典。
        """
        return {
            "stream_mode": "values",
            "config": {"recursion_limit": self.max_recur_limit},
        }
