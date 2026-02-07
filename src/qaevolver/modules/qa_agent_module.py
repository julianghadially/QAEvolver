"""Simple judge module - barebones fact checker without research."""

import dspy
from src.qaevolver.signatures.qa_agent import QAAgent


class QAAgentModule(dspy.Module):
    """QA Agent that answers questions."""

    def __init__(self):
        """Initialize the simple judge module."""
        super().__init__()
        self.agent = dspy.ReAct(QAAgent, tools=[], tool_choice="auto")

    def forward(self, query: str) -> dspy.Prediction:
        """Answer a question."""
        result = self.agent(query=query)

        return dspy.Prediction(response=result.response)
