# Test Recap

## Environment setup
- Command: `python -m pip install -e .[dev]`
- Status: ✅ Pass
- Notes: Installed runtime and test dependencies successfully.

## Automated tests

### 1) Full pytest suite
- Command: `pytest -q`
- Status: ✅ Pass
- Result: `3 passed, 1 skipped`
- Notes:
  - Application unit/API tests passed.
  - Live OpenAI integration test skipped because `OPENAI_API_KEY` was not set in this environment.

### 2) Live GPT-5 integration test only (non-mocked)
- Command: `pytest -q tests/test_openai_live.py -rs`
- Status: ⚠️ Skipped (environment limitation)
- Result: `SKIPPED [1] ... OPENAI_API_KEY not set`
- Notes:
  - This test performs real calls through `LLMQuizService` to GPT-5 when credentials are available.
  - To run for real, set `OPENAI_API_KEY` and re-run the command.

## Coverage of behavior verified by tests
- Quiz generation and submission flow
- Scoring and attempt retrieval
- Study-plan update endpoint
- Health endpoint
- (Optional when key provided) Live generation + verification calls to GPT-5
