from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline

from .preprocessing import preprocess_text
from .safety import detect_emergency, emergency_message


@dataclass
class ChatResponse:
    answer: str
    intent: str
    confidence: float
    source: str | None = None
    emergency: bool = False


class MedicalChatbot:
    def __init__(self, data_path: str | Path):
        self.data_path = Path(data_path)
        self.data = pd.read_csv(self.data_path).fillna("")
        self.data["training_text"] = (
            self.data["question"] + " " + self.data["keywords"]
        ).map(preprocess_text)
        self.model = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
                ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42)),
            ]
        )
        self.model.fit(self.data["training_text"], self.data["intent"])
        self.retriever = TfidfVectorizer(ngram_range=(1, 2))
        self.knowledge_vectors = self.retriever.fit_transform(self.data["training_text"])

    def save_model(self, model_path: str | Path) -> None:
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, model_path)

    def respond(self, question: str) -> ChatResponse:
        question = question.strip()
        if not question:
            return ChatResponse("Please enter a healthcare-related question.", "empty", 0.0)

        reasons = detect_emergency(question)
        if reasons:
            return ChatResponse(emergency_message(reasons), "emergency", 1.0, emergency=True)

        cleaned = preprocess_text(question)
        probabilities = self.model.predict_proba([cleaned])[0]
        predicted_intent = self.model.classes_[probabilities.argmax()]
        intent_confidence = float(probabilities.max())

        query_vector = self.retriever.transform([cleaned])
        similarities = cosine_similarity(query_vector, self.knowledge_vectors)[0]
        best_index = int(similarities.argmax())
        retrieval_score = float(similarities[best_index])

        if retrieval_score < 0.12:
            return ChatResponse(
                "I do not have enough reliable information to answer that question. "
                "Please ask about common symptoms, prevention or general healthcare, or consult a healthcare professional.",
                "unknown",
                retrieval_score,
            )

        row = self.data.iloc[best_index]
        confidence = min(1.0, (intent_confidence + retrieval_score) / 2)
        return ChatResponse(
            answer=str(row["answer"]),
            # Retrieval corrects ambiguous classifier predictions such as
            # symptom vs prevention questions for the same disease.
            intent=str(row["intent"]),
            confidence=confidence,
            source=str(row["source"]) or None,
        )
