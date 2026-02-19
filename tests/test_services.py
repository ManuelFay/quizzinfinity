import random

from app.schemas import QuestionPayload
from app.services import LLMQuizService


def test_randomize_question_option_order_updates_correct_index():
    question = QuestionPayload(
        prompt="Which value is prime?",
        options=["4", "9", "11", "21"],
        correct_option_index=2,
        category="number theory",
        explanation="11 is the only prime in the list.",
    )

    randomized = LLMQuizService.randomize_question_option_order(question, random.Random(1))

    assert sorted(randomized.options) == sorted(question.options)
    assert randomized.options[randomized.correct_option_index] == "11"


def test_randomize_questions_option_order_keeps_each_correct_option():
    service = LLMQuizService()
    questions = [
        QuestionPayload(
            prompt="Q1",
            options=["A", "B", "C", "D"],
            correct_option_index=0,
            category="c1",
            explanation="A",
        ),
        QuestionPayload(
            prompt="Q2",
            options=["E", "F", "G", "H"],
            correct_option_index=3,
            category="c2",
            explanation="H",
        ),
    ]

    randomized = service.randomize_questions_option_order(questions, random.Random(7))

    assert randomized[0].options[randomized[0].correct_option_index] == "A"
    assert randomized[1].options[randomized[1].correct_option_index] == "H"


def test_generation_prompt_requests_expanded_terms_before_abbreviations():
    service = LLMQuizService()

    plan = service._resolve_generation_plan(
        topic="Networking",
        learning_goal="Understand protocols",
        requested_difficulty=7,
    )

    assert "Avoid unexplained abbreviations" in plan.prompt


def test_generate_error_lesson_prompt_requires_markdown_and_high_level_first():
    service = LLMQuizService()
    service.client = object()
    captured = {}

    def fake_generate_json_with_retry(*, prompt, schema_name, schema, tools=None):
        captured["prompt"] = prompt
        return {"lesson": "## Big Picture\n- Foundations first"}

    service._generate_json_with_retry = fake_generate_json_with_retry

    lesson = service.generate_error_lesson(
        topic="Operating Systems",
        learning_goal="Understand scheduling",
        missed_question_results=[{"prompt": "Q", "options": ["a", "b", "c", "d"]}],
    )

    assert lesson.startswith("## Big Picture")
    assert "Write the lesson in Markdown" in captured["prompt"]
    assert "Structure the teaching from high-level understanding first" in captured["prompt"]
    assert "Do not include a section about why wrong answers were tempting" in captured["prompt"]
