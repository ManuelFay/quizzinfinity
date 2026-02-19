import time

from fastapi.testclient import TestClient

from app.database import Base, engine
from app.main import app
from app.schemas import QuestionPayload, VerificationResult
from app.services import QuestionGenerationError


class MockService:
    def generate_questions(self, **kwargs):
        progress_callback = kwargs.get("progress_callback")
        if progress_callback:
            progress_callback(1, kwargs.get("question_count", 2))
            progress_callback(2, kwargs.get("question_count", 2))

        class Plan:
            prompt = "mock prompt"
            difficulty = kwargs.get("difficulty", 9)
            difficulty_rationale = "mock rationale"

        return [
            QuestionPayload(
                prompt="What does Bayes theorem compute?",
                options=[
                    "Derivative of a product",
                    "Posterior probability from prior and likelihood",
                    "Mean absolute error",
                    "Taylor approximation",
                ],
                correct_option_index=1,
                category="probability",
                explanation="Bayes theorem updates belief using observed evidence.",
            ),
            QuestionPayload(
                prompt="Which method is best for high-bias underfit linear model?",
                options=["More features", "More regularization", "Smaller dataset", "Early stopping only"],
                correct_option_index=0,
                category="ml-modeling",
                explanation="Adding relevant features can reduce underfitting bias.",
            ),
        ], Plan()

    def verify_questions(self, questions, progress_callback=None):
        if progress_callback:
            for i in range(len(questions)):
                progress_callback(i + 1, len(questions))
        return VerificationResult(is_valid=True, reasons=[])


def wait_for_job(client: TestClient, job_id: str):
    for _ in range(50):
        status = client.get(f"/api/quizzes/generate/{job_id}")
        assert status.status_code == 200
        payload = status.json()
        if payload["state"] == "completed":
            return payload["result"]
        if payload["state"] == "failed":
            raise AssertionError(payload["error"])
        time.sleep(0.02)
    raise AssertionError("Timed out waiting for generation job")


def setup_module():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


def test_generate_and_submit_quiz(monkeypatch):
    monkeypatch.setattr("app.main.service", MockService())
    client = TestClient(app)

    gen_start = client.post(
        "/api/quizzes/generate",
        json={
            "topic": "machine learning",
            "learning_goal": "",
            "question_count": 5,
            "use_web_search": False,
            "custom_instructions": "make it practical",
        },
    )
    assert gen_start.status_code == 200
    job_id = gen_start.json()["job_id"]
    data = wait_for_job(client, job_id)
    assert data["quiz_id"] > 0
    assert len(data["questions"]) == 2
    assert "correct_option_index" in data["questions"][0]
    assert data["generation_prompt"] == "mock prompt"

    answers = [
        {"question_id": data["questions"][0]["id"], "selected_option_index": 1, "flagged_for_review": False},
        {"question_id": data["questions"][1]["id"], "selected_option_index": 2, "flagged_for_review": True},
    ]
    sub = client.post(f"/api/quizzes/{data['quiz_id']}/submit", json={"answers": answers})
    assert sub.status_code == 200
    payload = sub.json()
    assert payload["score"] == 1
    assert payload["total"] == 2
    assert len(payload["category_summary"]) == 2
    assert len(payload["study_topics"]) == 2
    assert any(x["flagged_for_review"] for x in payload["question_results"])


def test_attempt_fetch_and_study_plan_update(monkeypatch):
    monkeypatch.setattr("app.main.service", MockService())
    client = TestClient(app)

    gen_start = client.post(
        "/api/quizzes/generate",
        json={"topic": "statistics", "learning_goal": "", "difficulty": 7, "question_count": 5},
    )
    gen = wait_for_job(client, gen_start.json()["job_id"])

    answers = [
        {"question_id": gen["questions"][0]["id"], "selected_option_index": 1, "flagged_for_review": False},
        {"question_id": gen["questions"][1]["id"], "selected_option_index": 1, "flagged_for_review": True},
    ]
    sub = client.post(f"/api/quizzes/{gen['quiz_id']}/submit", json={"answers": answers}).json()
    attempt_id = sub["attempt_id"]

    update = client.post(
        f"/api/attempts/{attempt_id}/study-plan",
        json={"topics": [{"topic": "custom weak topic", "priority": 1}, {"topic": "probability", "priority": 2}]},
    )
    assert update.status_code == 200
    updated = update.json()
    assert updated["study_topics"][0]["topic"] == "custom weak topic"
    assert updated["study_topics"][0]["source"] == "user"

    get_attempt = client.get(f"/api/attempts/{attempt_id}")
    assert get_attempt.status_code == 200
    assert get_attempt.json()["attempt_id"] == attempt_id


def test_health_endpoint():
    client = TestClient(app)
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"


class InvalidKeyService:
    def generate_questions(self, **kwargs):
        raise QuestionGenerationError("Invalid OpenAI API key")


def test_invalid_api_key_maps_to_failed_generation_job(monkeypatch):
    monkeypatch.setattr("app.main.service", InvalidKeyService())
    client = TestClient(app)
    response = client.post(
        "/api/quizzes/generate",
        json={"topic": "calculus", "learning_goal": "", "question_count": 5, "use_web_search": False},
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]

    for _ in range(20):
        status = client.get(f"/api/quizzes/generate/{job_id}")
        assert status.status_code == 200
        payload = status.json()
        if payload["state"] == "failed":
            assert "Invalid OpenAI API key" in payload["error"]
            return
        time.sleep(0.02)

    raise AssertionError("Expected failed generation job")
