from pathlib import Path

from src.chatbot import MedicalChatbot


ROOT = Path(__file__).resolve().parents[1]
BOT = MedicalChatbot(ROOT / "data" / "medical_knowledge_base.csv")


def test_diabetes_question_returns_relevant_answer():
    result = BOT.respond("What symptoms happen with diabetes?")
    assert "urination" in result.answer.lower() or "thirst" in result.answer.lower()


def test_emergency_is_detected():
    result = BOT.respond("I have severe chest pain and cannot breathe")
    assert result.emergency is True
    assert "1990" in result.answer


def test_empty_question_is_handled():
    result = BOT.respond("   ")
    assert result.intent == "empty"

