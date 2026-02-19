# AGENTS.md

## Project overview
Quizzinfinity is a local-first adaptive quiz platform:
- Backend: FastAPI + SQLAlchemy + SQLite
- Frontend: React (CDN, static files served by FastAPI)
- LLM: OpenAI Responses API (`gpt-5`) for question generation + verification

Core flow:
1. `POST /api/quizzes/generate` creates a quiz from topic/goal.
2. User answers multiple-choice questions in UI.
3. `POST /api/quizzes/{quiz_id}/submit` stores answers and computes analysis.
4. User gets strengths/weaknesses + a prioritized study-topic list.
5. User can edit study priorities via `POST /api/attempts/{attempt_id}/study-plan`.
6. Follow-up quiz generation can use saved priorities (`followup_from_attempt_id`).

## Repository layout
- `app/main.py` — API routes and app wiring
- `app/models.py` — SQLAlchemy data models
- `app/schemas.py` — Pydantic request/response schemas
- `app/services.py` — OpenAI generation/verification + analytics helpers
- `app/static/` — frontend (`index.html`, `app.js`, `styles.css`)
- `tests/` — pytest suite
- `SPEC.md` — detailed product + system spec
- `TESTS.md` — latest test recap and outcomes

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
uvicorn app.main:app --reload --port 8000
```

## Testing guidance
- Primary test command: `pytest -q`
- Unit/API tests use a mocked service for determinism in `tests/test_app.py`.
- Optional live integration test (`tests/test_openai_live.py`) performs **real** GPT-5 calls and is auto-skipped unless `OPENAI_API_KEY` is set.

## Environment variables
- `OPENAI_API_KEY` — required for live GPT generation/verification.
- `OPENAI_BASE_URL` — optional override for API endpoint.

## Notes for future contributors/agents
- The frontend currently reveals `correct_option_index` and explanation in generate response to support immediate feedback UX.
- Study priorities are persisted and intended to be the basis for follow-up quizzes.
- SQLite DB file path: `data/quizzinfinity.db`.
