"""Simple judge signature for direct statement evaluation without research."""

from dspy import Signature, InputField, OutputField
from typing import Literal


class QAAgent(Signature):
    """Evaluate a question and provide an answer."""

    query: str = InputField(desc="The query to respond to")
    response: str = OutputField(desc="Response to the query")