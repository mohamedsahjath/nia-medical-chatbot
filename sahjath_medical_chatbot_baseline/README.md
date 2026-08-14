# AI-Based Medical Chatbot Using NLP — Sahjath Baseline

This branch-ready baseline contains Sahjath's allocated components: dataset preparation, NLP preprocessing, TF-IDF, Logistic Regression, medical retrieval and the emergency safety module. Genetic Algorithm optimization is intentionally left for the next collaboration stage.

This is a complete educational prototype based on the proposed workflow:

`User Input → Text Preprocessing → NLP Processing → Intent Classification → Medical Knowledge Base → Response Generation`

## Main features

- Natural-language healthcare questions
- Lowercasing, tokenization, stop-word removal and lightweight word normalization
- TF-IDF text representation
- Logistic Regression intent classification
- Cosine-similarity knowledge retrieval
- Curated healthcare Q&A dataset with source URLs
- Emergency keyword detection and Sri Lanka 1990 ambulance guidance
- Confidence score and trusted-source link
- Streamlit conversational interface
- Training, evaluation and automated test scripts

## Project structure

```text
ai_medical_chatbot/
├── app.py
├── train_model.py
├── evaluate.py
├── requirements.txt
├── data/medical_knowledge_base.csv
├── data/evaluation_questions.csv
├── src/
│   ├── chatbot.py
│   ├── preprocessing.py
│   └── safety.py
└── tests/test_chatbot.py
```

## Run in Visual Studio Code / Windows

Open the project folder in VS Code, then run:

```powershell
py -m venv .venv
.venv\Scripts\activate
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

The browser normally opens at `http://localhost:8501`.

## Run in Google Colab

Upload the project ZIP and extract it. Then run:

```python
!pip install -r requirements.txt
```

Streamlit is designed to run as a web app; local VS Code execution is the simplest option. Colab can be used for model experiments and evaluation:

```python
!python evaluate.py
```

## Train and evaluate

```powershell
py train_model.py
py evaluate.py
py run_tests.py
```

The included dataset is intentionally small and transparent for an academic prototype. For final research evaluation, collect more independently reviewed examples for every intent, divide them into training and unseen testing sets, and obtain healthcare-professional review.

## Safety and limitations

- The chatbot provides general educational information only.
- It does not diagnose diseases or recommend individual medication doses.
- Keyword emergency detection can miss unusual wording and can produce false alarms.
- Medical information and public-health guidance change; the knowledge base must be reviewed and updated.
- A qualified healthcare professional should validate the dataset before real-world deployment.

## Information sources

Dataset answers are concise paraphrases linked row-by-row to trusted sources, mainly:

- [WHO health fact sheets](https://www.who.int/news-room/fact-sheets)
- [CDC Influenza information](https://www.cdc.gov/flu/)
- [NHS health information](https://www.nhs.uk/conditions/)

## Suggested evaluation criteria

1. Intent classification accuracy on unseen labelled questions.
2. Response relevance rated on a 1–5 scale.
3. Medical correctness reviewed by a qualified professional.
4. Usability and satisfaction collected using an anonymous questionnaire.
5. Emergency-warning recall tested with varied warning-sign phrases.
