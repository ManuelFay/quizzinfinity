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
