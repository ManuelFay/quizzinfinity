import os

import pytest

from app.services import LLMQuizService


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_live_generate_and_verify_with_gpt5():
    service = LLMQuizService()
    questions, _plan = service.generate_questions(
        topic="intro linear algebra",
        learning_goal="diagnostic check",
        difficulty=6,
        question_count=5,
        use_web_search=False,
        weak_categories=["eigenvalues"],
    )

    assert len(questions) == 5
    for q in questions:
        assert len(q.options) == 4
        assert 0 <= q.correct_option_index <= 3
        assert q.category.strip()
        assert q.explanation.strip()

    verification = service.verify_questions(questions)
    assert verification.is_valid
