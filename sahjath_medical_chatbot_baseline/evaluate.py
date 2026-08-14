from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from src.chatbot import MedicalChatbot


BASE_DIR = Path(__file__).resolve().parent
bot = MedicalChatbot(BASE_DIR / "data" / "medical_knowledge_base.csv")
test = pd.read_csv(BASE_DIR / "data" / "evaluation_questions.csv")
predictions = [bot.respond(question).intent for question in test["question"]]

print(f"Evaluation questions: {len(test)}")
print(f"Intent accuracy: {accuracy_score(test['expected_intent'], predictions):.3f}")
print(classification_report(test["expected_intent"], predictions, zero_division=0))
