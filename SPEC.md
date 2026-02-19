# Quizzinfinity v1 Specification

## 1. Product Goal
Build a local-first study platform that:
1. accepts a user-provided learning target (topic or broad description),
2. generates an initial **diagnostic quiz (~20 hard multiple-choice questions)**,
3. records each response persistently,
4. analyzes strengths/weaknesses by category,
5. provides explanations for right/wrong choices,
6. iteratively generates follow-up quizzes targeting weak areas.

This v1 supports a single local user while preserving data model choices that can later support multiple users.

---

## 2. Functional Requirements

### 2.1 Quiz generation input
- User can enter:
  - `topic` (short text), or
  - `learning_goal` (longer free-text description)
- User can choose:
  - number of questions (default 20; allowed range 5–30)
  - difficulty slider (1–10; default 7)
- User starts generation with one action.

### 2.2 Generated quiz requirements
- Questions must be multiple-choice with exactly 4 options.
- Exactly one option must be correct.
- Questions should be relatively hard by default.
- Questions should span a broad range of subtopics.
- Every question must include:
  - prompt
  - options
  - correct option index
  - category tag
  - short rationale/explanation
  - optional source notes (when available)

### 2.3 Question verification pass
- After generation, run a second "verifier" model call that checks:
  - schema validity,
  - one and only one correct option,
  - option uniqueness/non-trivial distractors,
  - category presence,
  - answerability from general domain knowledge.
- If invalid:
  - regenerate once (v1),
  - if still invalid, return explicit error.

### 2.4 Quiz-taking interface
- Render one question at a time with progress indicator.
- Allow selecting exactly one option.
- Navigation:
  - Next/Previous,
  - submit only when all answered.
- On submit:
  - compute score,
  - persist attempt details,
  - show detailed results.

### 2.5 Results and analysis
- Per-question result includes:
  - selected option,
  - correct option,
  - correctness,
  - explanation (why correct and why distractors are wrong)
- Aggregate analytics include:
  - total score,
  - category-level performance,
  - strongest categories,
  - weakest categories,
  - recommended next focus.

### 2.6 Iterative follow-up
- User can generate next quiz with weak categories emphasized.
- Follow-up generation should:
  - include previous weak categories in prompt,
  - keep same difficulty unless user changes.

### 2.7 Persistence
- Persist all generated quizzes and attempts in SQLite.
- Persist at minimum:
  - quiz metadata,
  - full generated question payload,
  - user answer per question,
  - correctness,
  - timestamps.

---

## 3. Non-Functional Requirements
- Local runnable with a single command.
- API latency target for generation: best effort; UI should show loading state.
- Robust JSON schema validation for LLM output.
- Deterministic backend tests without calling external APIs (mock OpenAI service).
- Clean separation: frontend static assets + Python API backend.

---

## 4. System Design

### 4.1 Stack
- **Backend:** Python 3.11+, FastAPI, SQLAlchemy, Pydantic
- **Database:** SQLite (`data/quizzinfinity.db`)
- **Frontend:** React (CDN, no build step) + plain CSS (served by FastAPI static route)
- **LLM provider:** OpenAI Responses API (`gpt-5` for generation + verification)

### 4.2 High-level flow
1. User submits topic + settings.
2. Backend calls `QuestionGenerator.generate_quiz(...)`.
3. Backend calls `QuestionVerifier.verify(...)`.
4. On success, backend stores quiz + questions.
5. Frontend renders quiz UI.
6. User submits answers.
7. Backend scores + stores attempt.
8. Backend returns analysis and explanations.
9. User may generate targeted follow-up quiz.

---

## 5. API Contract (v1)

### `POST /api/quizzes/generate`
Request:
```json
{
  "topic": "quantum mechanics",
  "learning_goal": "I want to prepare for graduate-level diagnostics",
  "difficulty": 8,
  "question_count": 20,
  "use_web_search": true,
  "followup_from_attempt_id": null
}
```
Response:
```json
{
  "quiz_id": 1,
  "title": "Diagnostic Quiz: quantum mechanics",
  "difficulty": 8,
  "question_count": 20,
  "questions": [ ... ]
}
```

### `POST /api/quizzes/{quiz_id}/submit`
Request:
```json
{
  "answers": [
    {"question_id": 11, "selected_option_index": 2}
  ]
}
```
Response:
```json
{
  "attempt_id": 1,
  "score": 13,
  "total": 20,
  "percentage": 65.0,
  "category_summary": [ ... ],
  "question_results": [ ... ],
  "recommendations": [ ... ]
}
```

### `GET /api/attempts/{attempt_id}`
- Returns prior attempt with full analysis payload.

### `GET /api/health`
- Returns service status and DB connectivity.

---

## 6. Data Model

### Tables
- `quizzes`
  - id, topic, learning_goal, difficulty, question_count, title, created_at
- `questions`
  - id, quiz_id, prompt, options_json, correct_option_index, category, explanation
- `attempts`
  - id, quiz_id, started_at, submitted_at, score, total, percentage
- `attempt_answers`
  - id, attempt_id, question_id, selected_option_index, is_correct

Indexes:
- `questions.quiz_id`
- `attempts.quiz_id`
- `attempt_answers.attempt_id`

---

## 7. Validation Rules
- Input validation:
  - topic OR learning_goal required.
  - difficulty integer in [1,10]
  - question_count integer in [5,30]
- Generated question validation:
  - 4 options exactly
  - unique options
  - valid correct index [0..3]
  - non-empty category and explanation.

---

## 8. Prompting Strategy

### Generation prompt constraints
- Produce hard diagnostic items with realistic distractors.
- Cover broad conceptual and practical subareas.
- Return strict JSON only.

### Verification prompt constraints
- Validate item quality and structural correctness.
- Return pass/fail with reasons and optional repaired payload.

---

## 9. Security / Secrets
- Use `OPENAI_API_KEY` from environment.
- Never expose API key to frontend.
- Server-side LLM calls only.

---

## 10. Test Plan
- Unit tests:
  - schema validation
  - scoring logic
  - category analytics computation
- API tests:
  - generation route (with mocked generator/verifier)
  - submit route scoring and persistence
  - follow-up generation context
- Health test for app startup + DB session.

---

## 11. Future Roadmap
- Multi-user auth (JWT/session)
- Spaced repetition scheduling
- Rich analytics dashboard trends over time
- Source-cited question generation with stricter factuality checks
- Hosted deployment (Docker + Postgres + React build pipeline)
