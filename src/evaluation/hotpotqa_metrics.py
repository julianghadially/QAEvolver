"""HotpotQA evaluation metrics wrapping DSPy's built-in scoring functions."""

from dspy.evaluate.metrics import answer_exact_match, answer_passage_match


def answer_em(example, pred, trace=None) -> float:
    """Compute exact match score between predicted and gold answer.

    Wraps DSPy's built-in answer_exact_match.

    Args:
        example: dspy.Example with 'answer' field.
        pred: dspy.Prediction with 'answer' field.
        trace: Optional trace (unused, for DSPy compatibility).

    Returns:
        1.0 if exact match, 0.0 otherwise.
    """
    return float(answer_exact_match(example, pred, trace))


def answer_f1(example, pred, trace=None) -> float:
    """Compute token-level F1 score between predicted and gold answer.

    Uses DSPy's answer_passage_match which computes token-level F1.

    Args:
        example: dspy.Example with 'answer' field.
        pred: dspy.Prediction with 'answer' field.
        trace: Optional trace (unused, for DSPy compatibility).

    Returns:
        Token-level F1 score between 0.0 and 1.0.
    """
    gold = example.answer.lower().split()
    pred_tokens = pred.answer.lower().split()

    if not gold or not pred_tokens:
        return float(gold == pred_tokens)

    common = set(gold) & set(pred_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def print_hotpotqa_results(em: float, f1: float, num_examples: int, split_name: str) -> None:
    """Print formatted HotpotQA evaluation results.

    Args:
        em: Exact match score (0-100 scale).
        f1: F1 score (0-100 scale).
        num_examples: Number of examples evaluated.
        split_name: Name of the evaluation split.
    """
    print(f"\n{'='*50}")
    print(f"HotpotQA {split_name} Results ({num_examples} examples)")
    print(f"{'='*50}")
    print(f"  Exact Match (EM): {em:.2f}%")
    print(f"  F1 Score:         {f1:.2f}%")
    print(f"{'='*50}\n")
