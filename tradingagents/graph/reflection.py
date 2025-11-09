# TradingAgents/graph/reflection.py

from typing import Dict, Any
from langchain_openai import ChatOpenAI

# 导入统一日志系统
from tradingagents.utils.logging_init import get_logger
logger = get_logger("default")


class Reflector:
    """
    处理决策反思并更新记忆的类。

    该类负责评估交易决策和分析，并根据结果更新相关角色的记忆。
    它使用一个语言模型来生成对各种报告和决策的深入反思。
    """

    def __init__(self, quick_thinking_llm: ChatOpenAI):
        """
        初始化 Reflector。

        Args:
            quick_thinking_llm (ChatOpenAI): 用于生成反思的语言模型。
        """
        self.quick_thinking_llm = quick_thinking_llm
        self.reflection_system_prompt = self._get_reflection_prompt()

    def _get_reflection_prompt(self) -> str:
        """
        获取用于反思的系统提示。

        Returns:
            str: 包含反思指南的系统提示字符串。
        """
        return """
You are an expert financial analyst tasked with reviewing trading decisions/analysis and providing a comprehensive, step-by-step analysis. 
Your goal is to deliver detailed insights into investment decisions and highlight opportunities for improvement, adhering strictly to the following guidelines:

1. Reasoning:
   - For each trading decision, determine whether it was correct or incorrect. A correct decision results in an increase in returns, while an incorrect decision does the opposite.
   - Analyze the contributing factors to each success or mistake. Consider:
     - Market intelligence.
     - Technical indicators.
     - Technical signals.
     - Price movement analysis.
     - Overall market data analysis 
     - News analysis.
     - Social media and sentiment analysis.
     - Fundamental data analysis.
     - Weight the importance of each factor in the decision-making process.

2. Improvement:
   - For any incorrect decisions, propose revisions to maximize returns.
   - Provide a detailed list of corrective actions or improvements, including specific recommendations (e.g., changing a decision from HOLD to BUY on a particular date).

3. Summary:
   - Summarize the lessons learned from the successes and mistakes.
   - Highlight how these lessons can be adapted for future trading scenarios and draw connections between similar situations to apply the knowledge gained.

4. Query:
   - Extract key insights from the summary into a concise sentence of no more than 1000 tokens.
   - Ensure the condensed sentence captures the essence of the lessons and reasoning for easy reference.

Adhere strictly to these instructions, and ensure your output is detailed, accurate, and actionable. You will also be given objective descriptions of the market from a price movements, technical indicator, news, and sentiment perspective to provide more context for your analysis.
"""

    def _extract_current_situation(self, current_state: Dict[str, Any]) -> str:
        """
        从状态中提取当前的市场情况。

        Args:
            current_state (Dict[str, Any]): 包含市场、情绪、新闻和基本面报告的当前状态字典。

        Returns:
            str: 一个包含所有报告的格式化字符串，代表当前的市场情况。
        """
        curr_market_report = current_state["market_report"]
        curr_sentiment_report = current_state["sentiment_report"]
        curr_news_report = current_state["news_report"]
        curr_fundamentals_report = current_state["fundamentals_report"]

        return f"{curr_market_report}\n\n{curr_sentiment_report}\n\n{curr_news_report}\n\n{curr_fundamentals_report}"

    def _reflect_on_component(
        self, component_type: str, report: str, situation: str, returns_losses
    ) -> str:
        """
        为单个组件（角色）生成反思。

        Args:
            component_type (str): 组件的类型（例如，“BULL”，“BEAR”，“TRADER”）。
            report (str): 该组件生成的分析或决策报告。
            situation (str): 当前的市场情况。
            returns_losses: 投资回报或损失的记录。

        Returns:
            str: 由语言模型生成的反思结果。
        """
        messages = [
            ("system", self.reflection_system_prompt),
            (
                "human",
                f"Returns: {returns_losses}\n\nAnalysis/Decision: {report}\n\nObjective Market Reports for Reference: {situation}",
            ),
        ]

        result = self.quick_thinking_llm.invoke(messages).content
        return result

    def reflect_bull_researcher(self, current_state, returns_losses, bull_memory):
        """
        反思“牛市研究员”的分析并更新其记忆。

        Args:
            current_state: 图的当前状态。
            returns_losses: 投资回报或损失。
            bull_memory: 牛市研究员的记忆对象。
        """
        situation = self._extract_current_situation(current_state)
        bull_debate_history = current_state["investment_debate_state"]["bull_history"]

        result = self._reflect_on_component(
            "BULL", bull_debate_history, situation, returns_losses
        )
        bull_memory.add_situations([(situation, result)])

    def reflect_bear_researcher(self, current_state, returns_losses, bear_memory):
        """
        反思“熊市研究员”的分析并更新其记忆。

        Args:
            current_state: 图的当前状态。
            returns_losses: 投资回报或损失。
            bear_memory: 熊市研究员的记忆对象。
        """
        situation = self._extract_current_situation(current_state)
        bear_debate_history = current_state["investment_debate_state"]["bear_history"]

        result = self._reflect_on_component(
            "BEAR", bear_debate_history, situation, returns_losses
        )
        bear_memory.add_situations([(situation, result)])

    def reflect_trader(self, current_state, returns_losses, trader_memory):
        """
        反思“交易员”的决策并更新其记忆。

        Args:
            current_state: 图的当前状态。
            returns_losses: 投资回报或损失。
            trader_memory: 交易员的记忆对象。
        """
        situation = self._extract_current_situation(current_state)
        trader_decision = current_state["trader_investment_plan"]

        result = self._reflect_on_component(
            "TRADER", trader_decision, situation, returns_losses
        )
        trader_memory.add_situations([(situation, result)])

    def reflect_invest_judge(self, current_state, returns_losses, invest_judge_memory):
        """
        反思“投资裁判”的决策并更新其记忆。

        Args:
            current_state: 图的当前状态。
            returns_losses: 投资回报或损失。
            invest_judge_memory: 投资裁判的记忆对象。
        """
        situation = self._extract_current_situation(current_state)
        judge_decision = current_state["investment_debate_state"]["judge_decision"]

        result = self._reflect_on_component(
            "INVEST JUDGE", judge_decision, situation, returns_losses
        )
        invest_judge_memory.add_situations([(situation, result)])

    def reflect_risk_manager(self, current_state, returns_losses, risk_manager_memory):
        """
        反思“风险经理”的决策并更新其记忆。

        Args:
            current_state: 图的当前状态。
            returns_losses: 投资回报或损失。
            risk_manager_memory: 风险经理的记忆对象。
        """
        situation = self._extract_current_situation(current_state)
        judge_decision = current_state["risk_debate_state"]["judge_decision"]

        result = self._reflect_on_component(
            "RISK JUDGE", judge_decision, situation, returns_losses
        )
        risk_manager_memory.add_situations([(situation, result)])
