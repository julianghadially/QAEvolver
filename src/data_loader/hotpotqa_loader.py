"""HotpotQA dataset loader using HuggingFace datasets."""

import random

import dspy
from datasets import load_dataset


def load_hotpotqa_splits(
    train_size: int = 800,
    val_size: int = 200,
    test_size: int = 1000,
    seed: int = 42,
) -> tuple[list[dspy.Example], list[dspy.Example], list[dspy.Example]]:
    """Load HotpotQA splits as DSPy Examples.

    Loads the fullwiki config from HuggingFace and creates random samples
    for train, validation, and test splits.

    Args:
        train_size: Number of training examples to sample.
        val_size: Number of validation examples to sample.
        test_size: Number of test examples to sample.
        seed: Random seed for reproducible sampling.

    Returns:
        Tuple of (train, val, test) as lists of dspy.Example.
        Each Example has: id, question (input), answer, type, level (labels).
    """
    dataset = load_dataset("hotpotqa/hotpot_qa", "fullwiki", trust_remote_code=True)

    train_data = list(dataset["train"])
    validation_data = list(dataset["validation"])

    rng = random.Random(seed)

    train_sample = rng.sample(train_data, min(train_size, len(train_data)))
    val_sample = rng.sample(validation_data, min(val_size, len(validation_data)))
    test_sample = rng.sample(train_data, min(test_size, len(train_data)))

    def to_examples(samples: list[dict]) -> list[dspy.Example]:
        examples = []
        for item in samples:
            example = dspy.Example(
                id=item["id"],
                question=item["question"],
                answer=item["answer"],
                type=item["type"],
                level=item["level"],
            ).with_inputs("question")
            examples.append(example)
        return examples

    return to_examples(train_sample), to_examples(val_sample), to_examples(test_sample)
