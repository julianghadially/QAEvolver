"""Multi-hop QA pipeline using DSPy with Serper search and Firecrawl scraping."""

import dspy

from src.qaevolver.signatures.query_generation import (
    InitialQueryGeneration,
    FollowUpQueryGeneration,
)
from src.qaevolver.signatures.evidence_summarization import (
    EvidenceSummarization,
    CumulativeEvidenceSummarization,
)
from src.qaevolver.signatures.answer_generation import AnswerGeneration
from src.qaevolver.modules.retriever import retrieve


class MultiHopQAPipeline(dspy.Module):
    """A 2-hop retrieval QA pipeline.

    Architecture:
        1. Generate initial search query from question
        2. Retrieve (search + scrape) for hop 1
        3. Summarize evidence from hop 1
        4. Generate follow-up query based on question + evidence
        5. Retrieve (search + scrape) for hop 2
        6. Cumulatively summarize evidence from both hops
        7. Generate final answer from question + cumulative evidence
    """

    def __init__(self):
        super().__init__()
        self.generate_initial_query = dspy.Predict(InitialQueryGeneration)
        self.summarize_evidence_1 = dspy.ChainOfThought(EvidenceSummarization)
        self.generate_followup_query = dspy.Predict(FollowUpQueryGeneration)
        self.summarize_evidence_2 = dspy.ChainOfThought(CumulativeEvidenceSummarization)
        self.generate_answer = dspy.ChainOfThought(AnswerGeneration)

    def forward(self, question: str) -> dspy.Prediction:
        """Execute the 2-hop retrieval pipeline.

        Args:
            question: The question to answer.

        Returns:
            dspy.Prediction with answer and intermediate artifacts.
        """
        # Step 1: Generate initial query
        query_1_result = self.generate_initial_query(question=question)
        query_1 = query_1_result.query

        # Step 2: Retrieve for hop 1
        retrieval_1 = retrieve(query_1)
        scraped_content_1 = retrieval_1.scraped_page.markdown if retrieval_1.scraped_page else ""

        if not scraped_content_1:
            scraped_content_1 = "No content retrieved."

        # Step 3: Summarize evidence from hop 1
        evidence_1_result = self.summarize_evidence_1(
            question=question,
            scraped_content=scraped_content_1,
        )
        evidence_summary_1 = evidence_1_result.evidence_summary

        # Step 4: Generate follow-up query
        query_2_result = self.generate_followup_query(
            question=question,
            evidence_summary=evidence_summary_1,
        )
        query_2 = query_2_result.query

        # Step 5: Retrieve for hop 2
        retrieval_2 = retrieve(query_2)
        scraped_content_2 = retrieval_2.scraped_page.markdown if retrieval_2.scraped_page else ""

        if not scraped_content_2:
            scraped_content_2 = "No content retrieved."

        # Step 6: Cumulative evidence summarization
        evidence_2_result = self.summarize_evidence_2(
            question=question,
            prior_evidence_summary=evidence_summary_1,
            scraped_content=scraped_content_2,
        )
        evidence_summary_2 = evidence_2_result.evidence_summary

        # Step 7: Generate final answer
        answer_result = self.generate_answer(
            question=question,
            evidence_summary=evidence_summary_2,
        )

        return dspy.Prediction(
            answer=answer_result.answer,
            query_1=query_1,
            evidence_summary_1=evidence_summary_1,
            query_2=query_2,
            evidence_summary_2=evidence_summary_2,
        )
