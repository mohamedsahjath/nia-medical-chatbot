from pathlib import Path

from src.chatbot import MedicalChatbot


ROOT = Path(__file__).resolve().parent
bot = MedicalChatbot(ROOT / "data" / "medical_knowledge_base.csv")

diabetes = bot.respond("What symptoms happen with diabetes?")
assert "urination" in diabetes.answer.lower() or "thirst" in diabetes.answer.lower()

emergency = bot.respond("I have severe chest pain and cannot breathe")
assert emergency.emergency is True
assert "1990" in emergency.answer

empty = bot.respond("   ")
assert empty.intent == "empty"

antibiotic = bot.respond("Can antibiotics cure a viral cold?")
assert "do not treat viral" in antibiotic.answer.lower()

print("All chatbot checks passed.")

