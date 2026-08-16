import pandas as pd

from .preprocessing import preprocess_text


def build_training_examples(data: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Create transparent training paraphrases from each reviewed knowledge row."""
    texts, labels = [], []
    for _, row in data.iterrows():
        question, keywords = str(row["question"]), str(row["keywords"])
        examples = [f"{question} {keywords}", keywords,
                    f"Please explain {question} {keywords}",
                    f"I need information about {keywords}"]
        texts.extend(preprocess_text(item) for item in examples)
        labels.extend([str(row["intent"])] * len(examples))
    return texts, labels

