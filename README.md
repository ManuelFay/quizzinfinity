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
- Historical performance stats endpoint + modal
- Dataset export/import in JSON format
- SQLite persistence for quizzes, questions, answers, and attempts

## Development setup
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

## Testing
```bash
pytest -q
```

## Maintenance notes
- Frontend API calls should go through `parseApiResponse` + `fetchJsonOrThrow` (`app/static/app.js`) to avoid duplicated fetch/error handling logic.
- Keep shared payload construction centralized (e.g., `buildFormattedAnswers`) to reduce drift between submit/error-analysis flows.
- Avoid broad `try/catch` nesting when one helper can standardize behavior.
