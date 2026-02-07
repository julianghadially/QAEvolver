"""DSPy signatures for evidence summarization in the multi-hop QA pipeline."""

import dspy


class EvidenceSummarization(dspy.Signature):
    """Given a question and scraped web content, summarize the key evidence relevant to answering the question."""

    question: str = dspy.InputField(desc="The question to answer")
    scraped_content: str = dspy.InputField(desc="Scraped web page content")
    evidence_summary: str = dspy.OutputField(desc="A summary of the key evidence relevant to answering the question")


class CumulativeEvidenceSummarization(dspy.Signature):
    """Given a question, previously gathered evidence, and new scraped web content, produce a cumulative summary of all evidence relevant to answering the question."""

    question: str = dspy.InputField(desc="The question to answer")
    prior_evidence_summary: str = dspy.InputField(desc="Summary of evidence gathered from previous retrieval steps")
    scraped_content: str = dspy.InputField(desc="Newly scraped web page content")
    evidence_summary: str = dspy.OutputField(desc="A cumulative summary of all evidence relevant to answering the question")
