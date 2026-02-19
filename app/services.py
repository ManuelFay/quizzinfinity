import json
import logging
import math
import os
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Callable, List

from openai import AuthenticationError, OpenAI

from app.schemas import QuestionPayload, VerificationResult

logger = logging.getLogger(__name__)


QUESTION_LIST_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "prompt": {"type": "string"},
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "correct_option_index": {"type": "integer", "minimum": 0, "maximum": 3},
                    "category": {"type": "string"},
                    "explanation": {"type": "string"},
                },
                "required": [
                    "prompt",
                    "options",
                    "correct_option_index",
                    "category",
                    "explanation",
                ],
            },
            "minItems": 1,
        }
    },
    "required": ["questions"],
}


CATEGORY_PLAN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "categories": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "focus": {"type": "string"},
                    "question_target": {"type": "integer", "minimum": 1},
                },
                "required": ["name", "focus", "question_target"],
            },
        }
    },
    "required": ["categories"],
}


VERIFICATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "is_valid": {"type": "boolean"},
        "reasons": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["is_valid", "reasons"],
}


ERROR_LESSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "lesson": {"type": "string"},
    },
    "required": ["lesson"],
}




TOPIC_NORMALIZATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "cleaned_topic": {"type": "string"},
    },
    "required": ["cleaned_topic"],
}

QUESTION_REPAIR_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "question": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "prompt": {"type": "string"},
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 4,
                    "maxItems": 4,
                },
                "correct_option_index": {"type": "integer", "minimum": 0, "maximum": 3},
                "category": {"type": "string"},
                "explanation": {"type": "string"},
            },
            "required": ["prompt", "options", "correct_option_index", "category", "explanation"],
        }
    },
    "required": ["question"],
}


class QuestionGenerationError(RuntimeError):
    pass


@dataclass
class GenerationPlan:
    prompt: str
    difficulty: int
    difficulty_rationale: str


@dataclass
class CategoryPlan:
    name: str
    focus: str
    question_target: int


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
        custom_instructions: str = "",
        existing_questions: List[str] | None = None,
    ) -> GenerationPlan:
        normalized_topic = topic.strip() or "General topic"
        normalized_goal = learning_goal.strip() or "Strengthen deep conceptual mastery"
        weak_categories = weak_categories or []
        flagged_prompts = flagged_prompts or []
        existing_questions = existing_questions or []

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
        instruction_text = custom_instructions.strip() or "none"
        existing_questions_text = (
            "\n".join(f"- {question}" for question in existing_questions)
            if existing_questions
            else "- none"
        )
        prompt = (
            "Generate diagnostic multiple-choice questions and return JSON only with key 'questions'. "
            "Each question must include prompt, 4 distinct options, correct_option_index (0-3), category, explanation. "
            "Prioritize conceptual rigor, plausible distractors, and explanations that teach. "
            "Keep option lengths balanced so the correct answer is not noticeably longer than distractors. "
            f"Topic: {normalized_topic}. Learning goal: {normalized_goal}. "
            f"Difficulty (1-10): {difficulty}. "
            f"Prior weak categories to emphasize: {weak_text}. "
            "Questions explicitly flagged by learner for extra reinforcement:\n"
            f"{flagged_text}\n"
            "Questions already asked in prior quizzes. Avoid duplicates or near-duplicates:\n"
            f"{existing_questions_text}\n"
            f"Additional user instructions for follow-up generation: {instruction_text}."
        )
        return GenerationPlan(prompt=prompt, difficulty=difficulty, difficulty_rationale=rationale)


    def normalize_topic(self, topic: str, learning_goal: str = "") -> str:
        raw_topic = topic.strip()
        if not raw_topic:
            return ""
        if not self.client:
            return raw_topic

        prompt = (
            "Clean and standardize the learner's quiz topic label for storage. "
            "Return JSON with {'cleaned_topic': string}. "
            "Keep it concise (2-6 words), title case, and preserve intent. "
            "Do not include extra commentary. "
            f"Raw topic: {raw_topic}. Learning goal context: {learning_goal.strip() or 'none'}."
        )
        payload = self._generate_json_with_retry(
            prompt=prompt,
            schema_name="topic_normalization",
            schema=TOPIC_NORMALIZATION_SCHEMA,
        )
        cleaned_topic = str(payload.get("cleaned_topic", "")).strip()
        return cleaned_topic or raw_topic

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
        custom_instructions: str = "",
        existing_questions: List[str] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
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
            custom_instructions=custom_instructions,
            existing_questions=existing_questions,
        )

        tools = [{"type": "web_search_preview"}] if use_web_search else []
        over_generate_count = question_count + max(2, math.ceil(question_count * 0.25))
        all_questions: List[QuestionPayload] = []
        total = over_generate_count

        logger.info(
            "Generating quiz via OpenAI with planning (topic=%r, difficulty=%s, requested=%s, over_generate=%s)",
            topic,
            plan.difficulty,
            question_count,
            over_generate_count,
        )

        categories = self._plan_categories(
            plan=plan,
            topic=topic,
            learning_goal=learning_goal,
            target_count=over_generate_count,
            weak_categories=weak_categories or [],
            flagged_prompts=flagged_prompts or [],
            custom_instructions=custom_instructions,
            existing_questions=existing_questions or [],
            tools=tools,
        )
        logger.info("Question category plan: %s", [c.__dict__ for c in categories])

        category_requests: List[tuple[int, CategoryPlan, int]] = []
        planned = 0
        for idx, category in enumerate(categories):
            remaining = total - planned
            request_count = min(category.question_target, remaining)
            if request_count <= 0:
                break
            planned += request_count
            category_requests.append((idx, category, request_count))

        def _generate_for_category(category: CategoryPlan, request_count: int) -> List[QuestionPayload]:
            prompt = (
                f"{plan.prompt} "
                f"Category focus: {category.name} ({category.focus}). "
                f"Generate {request_count} questions for this category only. "
                "Ensure questions are conceptually distinct and avoid overlap with each other or other categories."
            )
            payload = self._generate_json_with_retry(
                prompt=prompt,
                schema_name="quiz_questions",
                schema=QUESTION_LIST_SCHEMA,
                tools=tools,
            )
            chunk_questions = [QuestionPayload(**q) for q in payload["questions"]]
            self._validate_questions(chunk_questions)
            return chunk_questions[:request_count]

        completed_chunks: dict[int, List[QuestionPayload]] = {}
        generated_so_far = 0
        with ThreadPoolExecutor(max_workers=min(6, len(category_requests) or 1)) as executor:
            futures = {
                executor.submit(_generate_for_category, category, request_count): idx
                for idx, category, request_count in category_requests
            }
            for future in as_completed(futures):
                idx = futures[future]
                chunk_questions = future.result()
                completed_chunks[idx] = chunk_questions
                generated_so_far += len(chunk_questions)
                if progress_callback:
                    progress_callback(min(generated_so_far, total), total)

        for idx, _, _ in category_requests:
            all_questions.extend(completed_chunks.get(idx, []))

        deduped_questions = self._deduplicate_questions(all_questions)
        randomized_questions = self.randomize_questions_option_order(deduped_questions)
        final_questions = randomized_questions[:question_count]
        logger.info(
            "Generated %s raw questions, %s after dedupe, returning %s",
            len(all_questions),
            len(deduped_questions),
            len(final_questions),
        )

        return final_questions, plan

    def _plan_categories(
        self,
        *,
        plan: GenerationPlan,
        topic: str,
        learning_goal: str,
        target_count: int,
        weak_categories: List[str],
        flagged_prompts: List[str],
        custom_instructions: str,
        existing_questions: List[str],
        tools,
    ) -> List[CategoryPlan]:
        existing_questions_text = (
            "\n".join(f"- {question}" for question in existing_questions)
            if existing_questions
            else "- none"
        )
        planner_prompt = (
            "Plan a diverse quiz blueprint and return JSON only in the form "
            "{'categories':[{'name':str,'focus':str,'question_target':int}]}. "
            "Use 4-6 categories with materially different focus areas. "
            f"Total planned question_target must sum to {target_count}. "
            f"Topic: {topic}. Learning goal: {learning_goal}. Difficulty: {plan.difficulty}. "
            f"Weak categories to cover: {weak_categories}. Flagged prompts: {flagged_prompts}. "
            "Questions already asked in prior quizzes (must avoid duplicates/near-duplicates):\n"
            f"{existing_questions_text}\n"
            f"Additional instructions: {custom_instructions or 'none'}."
        )
        payload = self._generate_json_with_retry(
            prompt=planner_prompt,
            schema_name="quiz_category_plan",
            schema=CATEGORY_PLAN_SCHEMA,
            tools=tools,
        )
        raw_categories = payload.get("categories") or []

        categories = [
            CategoryPlan(
                name=str(item.get("name", "General")).strip() or "General",
                focus=str(item.get("focus", "Core concepts")).strip() or "Core concepts",
                question_target=max(1, int(item.get("question_target", 1))),
            )
            for item in raw_categories
        ]
        if not categories:
            return [CategoryPlan(name="General", focus="Core concepts", question_target=target_count)]

        return self._rebalance_targets(categories, target_count)

    @staticmethod
    def randomize_question_option_order(question: QuestionPayload, rng: random.Random | None = None) -> QuestionPayload:
        rng = rng or random
        option_indices = list(range(len(question.options)))
        rng.shuffle(option_indices)

        randomized_options = [question.options[idx] for idx in option_indices]
        randomized_correct_index = option_indices.index(question.correct_option_index)
        return question.model_copy(
            update={
                "options": randomized_options,
                "correct_option_index": randomized_correct_index,
            }
        )

    def randomize_questions_option_order(
        self,
        questions: List[QuestionPayload],
        rng: random.Random | None = None,
    ) -> List[QuestionPayload]:
        return [self.randomize_question_option_order(question, rng) for question in questions]

    @staticmethod
    def _rebalance_targets(categories: List[CategoryPlan], target_count: int) -> List[CategoryPlan]:
        current = sum(c.question_target for c in categories)
        if current == target_count:
            return categories

        if current < target_count:
            idx = 0
            while current < target_count:
                categories[idx % len(categories)].question_target += 1
                current += 1
                idx += 1
            return categories

        categories = sorted(categories, key=lambda c: c.question_target, reverse=True)
        idx = 0
        while current > target_count and any(c.question_target > 1 for c in categories):
            candidate = categories[idx % len(categories)]
            if candidate.question_target > 1:
                candidate.question_target -= 1
                current -= 1
            idx += 1
        return categories

    def _generate_json_with_retry(self, *, prompt: str, schema_name: str, schema, tools=None):
        logger.info("Prompt sent to gpt-5 (%s): %s", schema_name, prompt)
        try:
            response = self.client.responses.create(
                model="gpt-5",
                input=prompt,
                tools=tools or [],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "strict": True,
                        "schema": schema,
                    }
                },
            )
        except AuthenticationError as exc:
            raise QuestionGenerationError("Invalid OpenAI API key") from exc

        text = response.output_text or ""
        logger.info("OpenAI %s response length=%s", schema_name, len(text))

        try:
            return self._extract_json(text)
        except QuestionGenerationError as first_exc:
            logger.warning("Initial parse failed for %s (%s). Retrying once.", schema_name, first_exc)
            retry_prompt = (
                f"{prompt} IMPORTANT: Return STRICT JSON only. No markdown fences and no prose outside JSON."
            )
            retry_response = self.client.responses.create(
                model="gpt-5",
                input=retry_prompt,
                tools=tools or [],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "strict": True,
                        "schema": schema,
                    }
                },
            )
            retry_text = retry_response.output_text or ""
            logger.info("OpenAI %s retry response length=%s", schema_name, len(retry_text))
            return self._extract_json(retry_text)

    @staticmethod
    def _needs_option_length_rebalance(question: QuestionPayload) -> bool:
        lengths = [len(option.strip()) for option in question.options if option.strip()]
        if len(lengths) < 4:
            return False
        longest = max(lengths)
        shortest = max(1, min(lengths))
        return longest > shortest * 1.6

    def rebalance_question_option_lengths(self, question: QuestionPayload) -> QuestionPayload:
        if not self.client or not self._needs_option_length_rebalance(question):
            return question

        prompt = (
            "Rewrite only the answer choices so all 4 options have comparable length while preserving meaning. "
            "Return JSON with one fixed question object. Keep the same prompt, category, explanation, and correct_option_index. "
            "You may rephrase distractors to be equally plausible and similar in length to the correct option. "
            f"Original question: {json.dumps(question.model_dump())}"
        )
        payload = self._generate_json_with_retry(
            prompt=prompt,
            schema_name="quiz_option_rebalance",
            schema=QUESTION_REPAIR_SCHEMA,
        )
        rebalanced = QuestionPayload(**payload["question"])
        self._validate_questions([rebalanced])
        return rebalanced

    def rebalance_questions_for_option_lengths(
        self,
        questions: List[QuestionPayload],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> List[QuestionPayload]:
        total = len(questions)
        balanced: List[QuestionPayload] = []
        for idx, question in enumerate(questions, start=1):
            maybe_balanced = self.rebalance_question_option_lengths(question)
            balanced.append(maybe_balanced)
            if progress_callback:
                progress_callback(idx, total)
        return balanced

    def verify_questions(
        self,
        questions: List[QuestionPayload],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> VerificationResult:
        if not self.client:
            raise QuestionGenerationError("OPENAI_API_KEY is required to verify quizzes")

        logger.info("Verifying %s generated questions", len(questions))
        total = len(questions)
        for i in range(total):
            if progress_callback:
                progress_callback(i + 1, total)

        verifier_prompt = (
            "Verify this list of quiz questions for correctness, non-ambiguity, and diversity. "
            "Return JSON with {is_valid:boolean,reasons:string[]}. "
            "Check exactly 4 unique options, single valid correct index, non-empty category and explanation, "
            "clear unambiguous wording, avoid near-duplicate questions, and ensure the correct option is not "
            "noticeably longer than distractors. "
            f"Questions: {json.dumps([q.model_dump() for q in questions])}"
        )
        payload = self._generate_json_with_retry(
            prompt=verifier_prompt,
            schema_name="quiz_verification",
            schema=VERIFICATION_SCHEMA,
        )
        result = VerificationResult(**payload)
        logger.info("Verification result: is_valid=%s reasons=%s", result.is_valid, len(result.reasons))
        return result

    def repair_question(self, question: QuestionPayload, reasons: List[str]) -> QuestionPayload | None:
        if not self.client:
            raise QuestionGenerationError("OPENAI_API_KEY is required to repair quizzes")

        repair_prompt = (
            "Repair this multiple-choice quiz question so it is correct, unambiguous, and technically feasible. "
            "Return one fixed question. Keep 4 unique options, a valid correct_option_index, and a clear explanation. "
            "Ensure the correct option is not noticeably longer than the distractors. "
            "Preserve the core learning objective and category when possible. "
            f"Original question: {json.dumps(question.model_dump())}. "
            f"Verification issues to address: {json.dumps(reasons)}"
        )
        payload = self._generate_json_with_retry(
            prompt=repair_prompt,
            schema_name="quiz_question_repair",
            schema=QUESTION_REPAIR_SCHEMA,
        )
        repaired = QuestionPayload(**payload["question"])
        self._validate_questions([repaired])
        return repaired

    def generate_error_lesson(
        self,
        *,
        topic: str,
        learning_goal: str,
        missed_question_results: list[dict],
    ) -> str:
        if not self.client:
            raise QuestionGenerationError("OPENAI_API_KEY is required to generate a lesson")

        prompt = (
            "Create a short remedial lesson tailored to a learner's quiz mistakes. "
            "Return JSON with {'lesson': string}. "
            "The lesson must include: (1) 3-5 concise concept bullets, "
            "(2) why each missed item was tempting/wrong, "
            "(3) a quick checklist for next attempt. "
            f"Topic: {topic or 'General topic'}. Learning goal: {learning_goal or 'Improve mastery'}. "
            f"Missed question data: {json.dumps(missed_question_results)}"
        )
        payload = self._generate_json_with_retry(
            prompt=prompt,
            schema_name="error_lesson",
            schema=ERROR_LESSON_SCHEMA,
        )
        return str(payload["lesson"]).strip()

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

    @staticmethod
    def _deduplicate_questions(questions: List[QuestionPayload]) -> List[QuestionPayload]:
        deduped: List[QuestionPayload] = []
        normalized_prompts: List[str] = []
        for question in questions:
            normalized = " ".join(question.prompt.lower().split())
            is_duplicate = False
            for existing in normalized_prompts:
                if normalized == existing or SequenceMatcher(None, normalized, existing).ratio() >= 0.9:
                    is_duplicate = True
                    break
            if is_duplicate:
                continue
            normalized_prompts.append(normalized)
            deduped.append(question)
        return deduped


def compute_analysis(question_rows, answer_rows):
    by_category = defaultdict(lambda: {"correct": 0, "total": 0, "flagged": 0})
    question_lookup = {q.id: q for q in question_rows}
    question_results = []

    for answer in answer_rows:
        question = question_lookup[answer.question_id]
        category = question.subcategory or question.category
        main_topic = question.main_topic or getattr(question.quiz, "topic", "")
        by_category[category]["total"] += 1
        by_category[category]["correct"] += int(bool(answer.is_correct))
        by_category[category]["flagged"] += int(answer.flagged_for_review)

        question_results.append(
            {
                "question_id": question.id,
                "prompt": question.prompt,
                "main_topic": main_topic,
                "category": category,
                "subcategory": category,
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
