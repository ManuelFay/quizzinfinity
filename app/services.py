import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import List

from openai import AuthenticationError, OpenAI

from app.schemas import QuestionPayload, VerificationResult

logger = logging.getLogger(__name__)


class QuestionGenerationError(RuntimeError):
    pass


@dataclass
class GenerationPlan:
    prompt: str
    difficulty: int
    difficulty_rationale: str


class LLMQuizService:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None

    def _resolve_generation_plan(
        self,
        *,
        topic: str,
        learning_goal: str,
        requested_difficulty: int,
        weak_categories: List[str] | None = None,
        flagged_prompts: List[str] | None = None,
        prior_attempt_percentage: float | None = None,
    ) -> GenerationPlan:
        normalized_topic = topic.strip() or "General topic"
        normalized_goal = learning_goal.strip() or "Strengthen deep conceptual mastery"
        weak_categories = weak_categories or []
        flagged_prompts = flagged_prompts or []

        difficulty = requested_difficulty
        rationale = "Using user-selected difficulty."
        if not requested_difficulty:
            difficulty = 9
            rationale = "Defaulting to hard difficulty for diagnostic depth."

        if prior_attempt_percentage is not None:
            if prior_attempt_percentage < 55:
                difficulty = min(difficulty, 6)
                rationale = (
                    "Prior attempt score suggests excessive struggle, so follow-up is eased "
                    "to rebuild fundamentals before returning to hard mode."
                )
            elif prior_attempt_percentage < 70:
                difficulty = min(difficulty, 8)
                rationale = (
                    "Prior attempt indicates partial mastery, so follow-up is slightly eased "
                    "to balance challenge with learning momentum."
                )
            else:
                difficulty = max(difficulty, 9)
                rationale = "Prior performance supports continuing with hard difficulty."

        weak_text = ", ".join(weak_categories) if weak_categories else "none"
        flagged_text = "\n".join(f"- {x}" for x in flagged_prompts) if flagged_prompts else "- none"
        prompt = (
            "Generate diagnostic multiple-choice questions and return JSON only with key 'questions'. "
            "Each question must include prompt, 4 distinct options, correct_option_index (0-3), category, explanation. "
            "Prioritize conceptual rigor, plausible distractors, and explanations that teach. "
            f"Topic: {normalized_topic}. Learning goal: {normalized_goal}. "
            f"Difficulty (1-10): {difficulty}. "
            f"Prior weak categories to emphasize: {weak_text}. "
            "Questions explicitly flagged by learner for extra reinforcement:\n"
            f"{flagged_text}"
        )
        return GenerationPlan(prompt=prompt, difficulty=difficulty, difficulty_rationale=rationale)

    def generate_questions(
        self,
        *,
        topic: str,
        learning_goal: str,
        difficulty: int,
        question_count: int,
        use_web_search: bool,
        weak_categories: List[str] | None = None,
        flagged_prompts: List[str] | None = None,
        prior_attempt_percentage: float | None = None,
    ) -> tuple[List[QuestionPayload], GenerationPlan]:
        if not self.client:
            raise QuestionGenerationError("OPENAI_API_KEY is required to generate quizzes")

        plan = self._resolve_generation_plan(
            topic=topic,
            learning_goal=learning_goal,
            requested_difficulty=difficulty,
            weak_categories=weak_categories,
            flagged_prompts=flagged_prompts,
            prior_attempt_percentage=prior_attempt_percentage,
        )

        prompt = (
            f"{plan.prompt} Question count: {question_count}. "
            "Ensure categories are meaningful and varied where possible."
        )
        tools = [{"type": "web_search_preview"}] if use_web_search else []

        logger.info(
            "Generating quiz via OpenAI (topic=%r, difficulty=%s, question_count=%s, weak_categories=%s, flagged_prompts=%s)",
            topic,
            plan.difficulty,
            question_count,
            len(weak_categories or []),
            len(flagged_prompts or []),
        )

        try:
            response = self.client.responses.create(model="gpt-5", input=prompt, tools=tools)
        except AuthenticationError as exc:
            raise QuestionGenerationError("Invalid OpenAI API key") from exc

        text = response.output_text or ""
        logger.info("OpenAI generation response length=%s", len(text))

        try:
            payload = self._extract_json(text)
        except QuestionGenerationError as first_exc:
            logger.warning("Initial parse failed (%s). Retrying generation once.", first_exc)
            retry_prompt = (
                f"{prompt} IMPORTANT: Return STRICT JSON only. No markdown fences and no prose outside JSON."
            )
            retry_response = self.client.responses.create(model="gpt-5", input=retry_prompt, tools=tools)
            retry_text = retry_response.output_text or ""
            logger.info("OpenAI generation retry response length=%s", len(retry_text))
            payload = self._extract_json(retry_text)

        questions = [QuestionPayload(**q) for q in payload["questions"]]
        self._validate_questions(questions)
        return questions, plan

    def verify_questions(self, questions: List[QuestionPayload]) -> VerificationResult:
        if not self.client:
            raise QuestionGenerationError("OPENAI_API_KEY is required to verify quizzes")

        verifier_prompt = (
            "Verify this list of quiz questions. Return JSON with {is_valid:boolean,reasons:string[]}. "
            "Check exactly 4 unique options, single valid correct index, non-empty category and explanation. "
            f"Questions: {json.dumps([q.model_dump() for q in questions])}"
        )
        logger.info("Verifying %s generated questions", len(questions))
        response = self.client.responses.create(model="gpt-5", input=verifier_prompt)
        payload = self._extract_json(response.output_text or "")
        result = VerificationResult(**payload)
        logger.info("Verification result: is_valid=%s reasons=%s", result.is_valid, len(result.reasons))
        return result

    @staticmethod
    def _extract_json(text: str):
        if not text or not text.strip():
            raise QuestionGenerationError("Model returned empty output while JSON was expected")

        normalized = text.strip()
        if normalized.startswith("```"):
            normalized = normalized.strip("`")
            if normalized.startswith("json"):
                normalized = normalized[4:]

        start = normalized.find("{")
        end = normalized.rfind("}")
        if start == -1 or end == -1 or end < start:
            preview = normalized[:200].replace("\n", " ")
            raise QuestionGenerationError(f"Model did not return a JSON object. Preview: {preview!r}")

        candidate = normalized[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            preview = candidate[:220].replace("\n", " ")
            raise QuestionGenerationError(
                f"Model returned invalid JSON ({exc.msg} at line {exc.lineno}, col {exc.colno}). Preview: {preview!r}"
            ) from exc

    @staticmethod
    def _validate_questions(questions: List[QuestionPayload]):
        for q in questions:
            if len(set(q.options)) != 4:
                raise QuestionGenerationError("Question has duplicate options")
            if q.correct_option_index < 0 or q.correct_option_index > 3:
                raise QuestionGenerationError("Question has invalid correct option index")
            if not q.category.strip() or not q.explanation.strip():
                raise QuestionGenerationError("Question missing category or explanation")


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
