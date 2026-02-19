# Quizzinfinity

Adaptive study quiz app with FastAPI + React.

## Features
- Topic or learning-goal input form
- Difficulty slider and configurable question count
- GPT-5-powered generation + verification pass
- Multiple-choice quiz interface with instant right/wrong feedback and explanations
- Per-question "need to study more" flagging (independent of correctness)
- Response analysis by strengths/weaknesses
- Editable study-priority list (mistakes + flags) used as basis for follow-up quiz generation
- SQLite persistence for quizzes, questions, answers, and attempts

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
uvicorn app.main:app --reload --port 8000
```
Then open http://localhost:8000.

Set API key to enable live generation:
```bash
export OPENAI_API_KEY=your_key_here
```

If no API key is set, backend uses deterministic fallback questions.

## Tests
```bash
pytest
```
