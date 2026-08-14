import re


EMERGENCY_PATTERNS = {
    "severe breathing difficulty": r"(cannot|can'?t|hard to|difficulty|severe).*breath|choking",
    "chest pain or pressure": r"chest (pain|pressure|tightness)",
    "possible stroke": r"(face droop|one.side weakness|slurred speech|sudden weakness)",
    "loss of consciousness or seizure": r"(unconscious|faint(ed|ing)?|seizure|convulsion)",
    "severe bleeding": r"(heavy|severe|uncontrolled|won't stop|will not stop).*bleed",
    "severe allergic reaction": r"(swollen tongue|throat swelling|anaphylaxis)",
    "self-harm risk": r"(kill myself|suicide|self harm|hurt myself)",
}


def detect_emergency(text: str) -> list[str]:
    normalized = text.lower()
    return [name for name, pattern in EMERGENCY_PATTERNS.items() if re.search(pattern, normalized)]


def emergency_message(reasons: list[str]) -> str:
    reason_text = ", ".join(reasons)
    return (
        f"🚨 **Possible emergency detected: {reason_text}.** "
        "Please seek emergency medical help now or go to the nearest emergency department. "
        "In Sri Lanka, you can call the Suwa Seriya ambulance service on **1990**. "
        "Do not rely on this chatbot for emergency treatment."
    )

