from pathlib import Path

from src.chatbot import MedicalChatbot


BASE_DIR = Path(__file__).resolve().parent
bot = MedicalChatbot(BASE_DIR / "data" / "medical_knowledge_base.csv")
output = BASE_DIR / "models" / "intent_model.joblib"
bot.save_model(output)
print(f"Model trained and saved to: {output}")

