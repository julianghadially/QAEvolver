"""Main runner for QAEvolver multi-hop QA evaluation on HotpotQA."""

import json
import os
from pathlib import Path

import dspy

from src.context_.context import openai_key
from src.data_loader import load_hotpotqa_splits
from src.qaevolver.modules.multihop_qa_pipeline import MultiHopQAPipeline
from src.evaluation.hotpotqa_metrics import answer_em, answer_f1, print_hotpotqa_results


def main():
    # Configure DSPy with GPT-4.1
    lm = dspy.LM("openai/gpt-4.1", api_key=openai_key)
    dspy.configure(lm=lm)

    # Load HotpotQA splits
    print("Loading HotpotQA splits...")
    train, val, test = load_hotpotqa_splits()
    print(f"  Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    # Initialize pipeline
    pipeline = MultiHopQAPipeline()

    # Run evaluation on validation set with F1 as primary metric
    print(f"\nEvaluating on {len(val)} validation examples...")
    evaluator = dspy.Evaluate(
        devset=val,
        metric=answer_f1,
        num_threads=4,
        display_progress=True,
        return_all_scores=True,
        return_outputs=True,
    )

    f1_score, outputs = evaluator(pipeline)

    # Compute EM as secondary metric from the same outputs
    em_scores = []
    detailed_results = []

    for example, prediction, score in outputs:
        em = answer_em(example, prediction)
        em_scores.append(em)

        detailed_results.append({
            "id": example.id,
            "question": example.question,
            "gold_answer": example.answer,
            "predicted_answer": prediction.answer,
            "query_1": prediction.query_1,
            "evidence_summary_1": prediction.evidence_summary_1,
            "query_2": prediction.query_2,
            "evidence_summary_2": prediction.evidence_summary_2,
            "f1": score,
            "em": em,
            "type": example.type,
            "level": example.level,
        })

    em_score = sum(em_scores) / len(em_scores) * 100 if em_scores else 0.0

    # Print results
    print_hotpotqa_results(em_score, f1_score, len(val), "Validation")

    # Save detailed results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / "validation_results.json"

    with open(results_path, "w") as f:
        json.dump(
            {
                "metrics": {
                    "em": em_score,
                    "f1": f1_score,
                    "num_examples": len(val),
                },
                "results": detailed_results,
            },
            f,
            indent=2,
        )

    print(f"Detailed results saved to {results_path}")


if __name__ == "__main__":
    main()
