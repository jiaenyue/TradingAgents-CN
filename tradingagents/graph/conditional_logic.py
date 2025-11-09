"""
图流程的条件逻辑

该模块定义了 `ConditionalLogic` 类, 它包含了决定 `TradingAgents` 图中
工作流向的所有条件判断函数。这些函数根据当前的 `AgentState` 来判断
下一步应该执行哪个节点, 从而实现了图的动态路由功能。
"""
# TradingAgents/graph/conditional_logic.py

from tradingagents.agents.utils.agent_states import AgentState

# 导入统一日志系统
from tradingagents.utils.logging_init import get_logger
logger = get_logger("default")


class ConditionalLogic:
    """
    处理用于确定图工作流向的条件逻辑。

    这个类的方法被用作图中的条件边 (conditional edges), 根据当前的状态
    决定下一个要调用的节点。
    """

    def __init__(self, max_debate_rounds=1, max_risk_discuss_rounds=1):
        """
        用配置参数进行初始化。

        Args:
            max_debate_rounds (int): 投资辩论的最大轮数。
            max_risk_discuss_rounds (int): 风险讨论的最大轮数。
        """
        self.max_debate_rounds = max_debate_rounds
        self.max_risk_discuss_rounds = max_risk_discuss_rounds

    def should_continue_market(self, state: AgentState) -> str:
        """
        判断市场分析流程是否应该继续。

        如果最后一条消息包含工具调用, 意味着代理需要执行工具,
        则流程转向工具执行节点。否则, 分析结束, 流程转向清理节点。

        Args:
            state (AgentState): 当前的代理状态。

        Returns:
            str: 下一个节点的名称 ("tools_market" 或 "Msg Clear Market")。
        """
        messages = state["messages"]
        last_message = messages[-1]

        # 只有AIMessage才有tool_calls属性
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools_market"
        return "Msg Clear Market"

    def should_continue_social(self, state: AgentState) -> str:
        """
        判断社交媒体分析流程是否应该继续。

        逻辑与 `should_continue_market` 类似。

        Args:
            state (AgentState): 当前的代理状态。

        Returns:
            str: 下一个节点的名称 ("tools_social" 或 "Msg Clear Social")。
        """
        messages = state["messages"]
        last_message = messages[-1]

        # 只有AIMessage才有tool_calls属性
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools_social"
        return "Msg Clear Social"

    def should_continue_news(self, state: AgentState) -> str:
        """
        判断新闻分析流程是否应该继续。

        逻辑与 `should_continue_market` 类似。

        Args:
            state (AgentState): 当前的代理状态。

        Returns:
            str: 下一个节点的名称 ("tools_news" 或 "Msg Clear News")。
        """
        messages = state["messages"]
        last_message = messages[-1]

        # 只有AIMessage才有tool_calls属性
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools_news"
        return "Msg Clear News"

    def should_continue_fundamentals(self, state: AgentState) -> str:
        """
        判断基本面分析流程是否应该继续。

        逻辑与 `should_continue_market` 类似。

        Args:
            state (AgentState): 当前的代理状态。

        Returns:
            str: 下一个节点的名称 ("tools_fundamentals" 或 "Msg Clear Fundamentals")。
        """
        messages = state["messages"]
        last_message = messages[-1]

        # 只有AIMessage才有tool_calls属性
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools_fundamentals"
        return "Msg Clear Fundamentals"

    def should_continue_debate(self, state: AgentState) -> str:
        """
        判断投资辩论是否应该继续, 并决定下一个发言者。

        如果辩论达到最大轮数, 则结束辩论并转向研究经理。
        否则, 根据上一位发言者的立场 (看涨或看跌), 决定下一位
        应由对立方发言。

        Args:
            state (AgentState): 当前的代理状态。

        Returns:
            str: 下一个节点的名称 ("Research Manager", "Bear Researcher",
                 或 "Bull Researcher")。
        """

        if (
            state["investment_debate_state"]["count"] >= 2 * self.max_debate_rounds
        ):  # 2个代理之间的多轮来回
            return "Research Manager"
        if state["investment_debate_state"]["current_response"].startswith("Bull"):
            return "Bear Researcher"
        return "Bull Researcher"

    def should_continue_risk_analysis(self, state: AgentState) -> str:
        """
        判断风险分析讨论是否应该继续, 并决定下一个发言者。

        如果讨论达到最大轮数, 则结束讨论并转向风险裁判。
        否则, 根据上一位发言者的立场 (激进、稳健、中立), 轮流
        让下一位发言。

        Args:
            state (AgentState): 当前的代理状态。

        Returns:
            str: 下一个节点的名称 ("Risk Judge", "Safe Analyst",
                 "Neutral Analyst", 或 "Risky Analyst")。
        """
        if (
            state["risk_debate_state"]["count"] >= 3 * self.max_risk_discuss_rounds
        ):  # 3个代理之间的多轮来回
            return "Risk Judge"
        if state["risk_debate_state"]["latest_speaker"].startswith("Risky"):
            return "Safe Analyst"
        if state["risk_debate_state"]["latest_speaker"].startswith("Safe"):
            return "Neutral Analyst"
        return "Risky Analyst"
