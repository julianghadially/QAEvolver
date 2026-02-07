"""DSPy signatures for query generation in the multi-hop QA pipeline."""

import dspy


class InitialQueryGeneration(dspy.Signature):
    """Given a question, generate a search query to find relevant information."""

    question: str = dspy.InputField(desc="The question to answer")
    query: str = dspy.OutputField(desc="A search query to find relevant information for answering the question")


class FollowUpQueryGeneration(dspy.Signature):
    """Given a question and evidence gathered so far, generate a follow-up search query to find additional information needed to answer the question."""

    question: str = dspy.InputField(desc="The question to answer")
    evidence_summary: str = dspy.InputField(desc="Summary of evidence gathered so far")
    query: str = dspy.OutputField(desc="A follow-up search query to find additional information needed to answer the question")
