from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

from .genetic_selector import GAResult, GeneticFeatureSelector
from .preprocessing import preprocess_text
from .safety import detect_emergency, emergency_message
from .training_data import build_training_examples


@dataclass
class ChatResponse:
    answer: str
    intent: str
    confidence: float
    source: str | None = None
    emergency: bool = False


class MedicalChatbot:
    def __init__(self, data_path: str | Path, use_genetic_algorithm=True,
                 ga_population=8, ga_generations=4):
        self.data_path = Path(data_path)
        self.data = pd.read_csv(self.data_path).fillna("")
        self.data["training_text"] = (
            self.data["question"] + " " + self.data["keywords"]
        ).map(preprocess_text)
        texts, labels = build_training_examples(self.data)
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        vectors = self.vectorizer.fit_transform(texts)
        self.feature_mask = None
        self.ga_result: GAResult | None = None
        if use_genetic_algorithm:
            selector = GeneticFeatureSelector(ga_population, ga_generations, random_state=42)
            feature_names = self.vectorizer.get_feature_names_out()
            # Preserve reviewed knowledge-base vocabulary. The GA searches
            # augmentation-only features without discarding core medical terms.
            reviewed_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
            reviewed_vectorizer.fit(self.data["training_text"])
            reviewed_features = set(reviewed_vectorizer.get_feature_names_out())
            mandatory_mask = [feature in reviewed_features for feature in feature_names]
            self.ga_result = selector.fit(vectors, labels, mandatory_mask)
            self.feature_mask = self.ga_result.mask
            vectors = vectors[:, self.feature_mask]
        self.model = LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42)
        self.model.fit(vectors, labels)
        self.retriever = TfidfVectorizer(ngram_range=(1, 2))
        self.knowledge_vectors = self.retriever.fit_transform(self.data["training_text"])

    def _model_vectors(self, texts):
        vectors = self.vectorizer.transform(texts)
        return vectors[:, self.feature_mask] if self.feature_mask is not None else vectors

    def predict_intent(self, question):
        vectors = self._model_vectors([preprocess_text(question)])
        probabilities = self.model.predict_proba(vectors)[0]
        best = int(probabilities.argmax())
        return str(self.model.classes_[best]), float(probabilities[best])

    def save_model(self, model_path):
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"vectorizer": self.vectorizer, "feature_mask": self.feature_mask,
                     "classifier": self.model, "ga_result": self.ga_result}, model_path)

    def respond(self, question):
        question = question.strip()
        if not question:
            return ChatResponse("Please enter a healthcare-related question.", "empty", 0.0)
        reasons = detect_emergency(question)
        if reasons:
            return ChatResponse(emergency_message(reasons), "emergency", 1.0, emergency=True)
        predicted_intent, intent_confidence = self.predict_intent(question)
        query_vector = self.retriever.transform([preprocess_text(question)])
        similarities = cosine_similarity(query_vector, self.knowledge_vectors)[0]
        best_index = int(similarities.argmax())
        retrieval_score = float(similarities[best_index])
        if retrieval_score < 0.12:
            return ChatResponse(
                "I do not have enough reliable information to answer that question. "
                "Please ask about common symptoms, prevention or general healthcare, or consult a healthcare professional.",
                "unknown", retrieval_score)
        row = self.data.iloc[best_index]
        retrieved_intent = str(row["intent"])
        final_intent = retrieved_intent if retrieval_score >= 0.20 else predicted_intent
        return ChatResponse(str(row["answer"]), final_intent,
                            min(1.0, (intent_confidence + retrieval_score) / 2),
                            str(row["source"]) or None)
