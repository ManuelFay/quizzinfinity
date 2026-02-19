import json
import os
from collections import defaultdict
from typing import List

from openai import OpenAI

from app.schemas import QuestionPayload, VerificationResult


class QuestionGenerationError(RuntimeError):
    pass


class LLMQuizService:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None

    def generate_questions(
        self,
        *,
        topic: str,
        learning_goal: str,
        difficulty: int,
        question_count: int,
        use_web_search: bool,
        weak_categories: List[str] | None = None,
    ) -> List[QuestionPayload]:
        if not self.client:
            return self._fallback_questions(topic or learning_goal, question_count)

        weak_categories_txt = ", ".join(weak_categories or [])
        prompt = (
            "Generate hard diagnostic multiple-choice questions. Return JSON only with key 'questions'. "
            "Each question must include prompt, 4 options, correct_option_index (0-3), category, explanation. "
            f"Topic: {topic}. Learning goal: {learning_goal}. Difficulty (1-10): {difficulty}. "
            f"Question count: {question_count}. "
            f"Prior weak categories to emphasize: {weak_categories_txt or 'none'}"
        )
        tools = [{"type": "web_search_preview"}] if use_web_search else []
        response = self.client.responses.create(model="gpt-5", input=prompt, tools=tools)
        text = response.output_text
        payload = self._extract_json(text)
        questions = [QuestionPayload(**q) for q in payload["questions"]]
        self._validate_questions(questions)
        return questions

    def verify_questions(self, questions: List[QuestionPayload]) -> VerificationResult:
        if not self.client:
            return VerificationResult(is_valid=True, reasons=[])

        verifier_prompt = (
            "Verify this list of quiz questions. Return JSON with {is_valid:boolean,reasons:string[]}. "
            "Check exactly 4 unique options, single valid correct index, non-empty category and explanation. "
            f"Questions: {json.dumps([q.model_dump() for q in questions])}"
        )
        response = self.client.responses.create(model="gpt-5", input=verifier_prompt)
        payload = self._extract_json(response.output_text)
        result = VerificationResult(**payload)
        return result

    @staticmethod
    def _extract_json(text: str):
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise QuestionGenerationError("Model did not return JSON payload")
        return json.loads(text[start : end + 1])

    @staticmethod
    def _validate_questions(questions: List[QuestionPayload]):
        for q in questions:
            if len(set(q.options)) != 4:
                raise QuestionGenerationError("Question has duplicate options")
            if q.correct_option_index < 0 or q.correct_option_index > 3:
                raise QuestionGenerationError("Question has invalid correct option index")
            if not q.category.strip() or not q.explanation.strip():
                raise QuestionGenerationError("Question missing category or explanation")

    @staticmethod
    def _fallback_questions(seed: str, count: int) -> List[QuestionPayload]:
        topic = seed or "general knowledge"
        questions = []
        for idx in range(count):
            questions.append(
                QuestionPayload(
                    prompt=f"[{topic}] Advanced diagnostic item {idx + 1}: Which statement is most accurate?",
                    options=[
                        "A commonly believed but incomplete claim",
                        "A subtly incorrect interpretation",
                        "The most defensible answer based on core principles",
                        "An attractive but invalid shortcut",
                    ],
                    correct_option_index=2,
                    category=f"{topic} fundamentals" if idx < count // 2 else f"{topic} applications",
                    explanation="Option 3 is intentionally the most defensible principle-based choice.",
                )
            )
        return questions


def compute_analysis(question_rows, answer_rows):
    by_category = defaultdict(lambda: {"correct": 0, "total": 0, "flagged": 0})
    question_lookup = {q.id: q for q in question_rows}
    question_results = []

    for answer in answer_rows:
        question = question_lookup[answer.question_id]
        category = question.category
        by_category[category]["total"] += 1
        by_category[category]["correct"] += int(answer.is_correct)
        by_category[category]["flagged"] += int(answer.flagged_for_review)

        question_results.append(
            {
                "question_id": question.id,
                "prompt": question.prompt,
                "category": category,
                "selected_option_index": answer.selected_option_index,
                "correct_option_index": question.correct_option_index,
                "is_correct": answer.is_correct,
                "flagged_for_review": answer.flagged_for_review,
                "explanation": question.explanation,
                "options": json.loads(question.options_json),
            }
        )

    category_summary = []
    for category, stats in by_category.items():
        pct = (stats["correct"] / stats["total"] * 100.0) if stats["total"] else 0
        category_summary.append(
            {
                "category": category,
                "correct": stats["correct"],
                "total": stats["total"],
                "percentage": round(pct, 2),
                "flagged": stats["flagged"],
            }
        )

    category_summary.sort(key=lambda x: x["percentage"], reverse=True)
    strengths = [c["category"] for c in category_summary[:2]]
    weaknesses = [c["category"] for c in category_summary[-2:]] if category_summary else []
    recommendations = [f"Practice deeper exercises in {w}." for w in weaknesses]

    return {
        "category_summary": [{k: v for k, v in c.items() if k != "flagged"} for c in category_summary],
        "strengths": strengths,
        "weaknesses": weaknesses,
        "recommendations": recommendations,
        "question_results": question_results,
        "raw_category_summary": category_summary,
    }


def build_study_topics(raw_category_summary):
    scored = []
    for row in raw_category_summary:
        deficit = 100 - row["percentage"]
        penalty = row.get("flagged", 0) * 15
        priority_score = deficit + penalty
        scored.append((priority_score, row["category"]))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"topic": cat, "priority": idx + 1, "source": "auto"} for idx, (_, cat) in enumerate(scored)]
