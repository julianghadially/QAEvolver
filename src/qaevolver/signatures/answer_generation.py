"""DSPy signature for answer generation in the multi-hop QA pipeline."""

import dspy


class AnswerGeneration(dspy.Signature):
    """Given a question and gathered evidence, generate a concise answer to the question."""

    question: str = dspy.InputField(desc="The question to answer")
    evidence_summary: str = dspy.InputField(desc="Summary of all gathered evidence relevant to the question")
    answer: str = dspy.OutputField(desc="A concise answer to the question based on the evidence")
