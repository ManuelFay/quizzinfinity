import json
import logging
from datetime import datetime
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.database import Base, engine, get_db
from app.models import Attempt, AttemptAnswer, Question, QuestionHint, Quiz, StudyTopic
from app.schemas import (
    AttemptResponse,
    DatasetExportResponse,
    DatasetImportRequest,
    DatasetImportResponse,
    GenerateQuizRequest,
    GenerateQuizResponse,
    HistoricalStatsResponse,
    QuizGenerationJobResponse,
    ResetStatsResponse,
    QuizGenerationJobStatus,
    QuestionHintRequest,
    StudyPlanUpdateRequest,
    SubmitQuizRequest,
)
from app.services import LLMQuizService, QuestionGenerationError, build_study_topics, compute_analysis


app = FastAPI(title="Quizzinfinity")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)
service = LLMQuizService()

def _ensure_schema_compatibility():
    with engine.begin() as conn:
        column_rows = conn.execute(text("PRAGMA table_info(questions)")).fetchall()
        existing_columns = {row[1] for row in column_rows}
        if "main_topic" not in existing_columns:
            conn.execute(text("ALTER TABLE questions ADD COLUMN main_topic VARCHAR(255) DEFAULT ''"))
        if "subcategory" not in existing_columns:
            conn.execute(text("ALTER TABLE questions ADD COLUMN subcategory VARCHAR(120) DEFAULT ''"))


Base.metadata.create_all(bind=engine)
_ensure_schema_compatibility()

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@dataclass
class GenerationJob:
    payload: GenerateQuizRequest
    state: str = "queued"
    stage: str = "Queued"
    generated_questions: int = 0
    verified_questions: int = 0
    total_questions: int = 0
    stage_detail: str = ""
    category_progress: list[str] = field(default_factory=list)
    error: str = ""
    result: GenerateQuizResponse | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


_generation_jobs: dict[str, GenerationJob] = {}
_generation_jobs_lock = threading.Lock()


def _track_progress(
    job: GenerationJob,
    *,
    stage: str,
    generated: int | None = None,
    verified: int | None = None,
    total: int | None = None,
    stage_detail: str | None = None,
    category_event: str | None = None,
):
    with job.lock:
        job.stage = stage
        if generated is not None:
            job.generated_questions = generated
        if verified is not None:
            job.verified_questions = verified
        if total is not None:
            job.total_questions = total
        if stage_detail is not None:
            job.stage_detail = stage_detail
        if category_event:
            job.category_progress = [category_event, *job.category_progress][:4]




def _prepare_payload_with_cleaned_topic(payload: GenerateQuizRequest) -> GenerateQuizRequest:
    cleaned_topic = service.normalize_topic(payload.topic, payload.learning_goal)
    if cleaned_topic == payload.topic:
        return payload
    return payload.model_copy(update={"topic": cleaned_topic})

def _resolve_followup_context(payload: GenerateQuizRequest, db: Session):
    weak_categories = []
    flagged_prompts = []
    prior_attempt_percentage = None
    existing_questions = []

    if payload.topic.strip() or payload.learning_goal.strip():
        prior_quizzes = db.query(Quiz).all()
        for prior_quiz in prior_quizzes:
            same_topic = payload.topic.strip() and prior_quiz.topic and prior_quiz.topic.strip() == payload.topic.strip()
            same_goal = (
                payload.learning_goal.strip()
                and prior_quiz.learning_goal
                and prior_quiz.learning_goal.strip() == payload.learning_goal.strip()
            )
            if same_topic or same_goal:
                prior_questions = db.query(Question).filter(Question.quiz_id == prior_quiz.id).all()
                existing_questions.extend([q.prompt for q in prior_questions])

    if payload.followup_from_attempt_id:
        prior_topics = (
            db.query(StudyTopic)
            .filter(StudyTopic.attempt_id == payload.followup_from_attempt_id)
            .order_by(StudyTopic.priority.asc())
            .all()
        )
        if prior_topics:
            weak_categories = [t.topic for t in prior_topics]

        prior_attempt = db.query(Attempt).filter(Attempt.id == payload.followup_from_attempt_id).first()
        if prior_attempt:
            prior_attempt_percentage = prior_attempt.percentage
            answers = db.query(AttemptAnswer).filter(AttemptAnswer.attempt_id == prior_attempt.id).all()
            questions = db.query(Question).filter(Question.quiz_id == prior_attempt.quiz_id).all()
            if not weak_categories:
                analysis = compute_analysis(questions, answers)
                weak_categories = analysis["weaknesses"]
            flagged_ids = {a.question_id for a in answers if a.flagged_for_review}
            if flagged_ids:
                flagged_prompts = [q.prompt for q in questions if q.id in flagged_ids]

    return weak_categories, flagged_prompts, prior_attempt_percentage, existing_questions


def _persist_quiz(db: Session, payload: GenerateQuizRequest, questions, plan) -> GenerateQuizResponse:
    quiz = Quiz(
        topic=payload.topic,
        learning_goal=payload.learning_goal,
        difficulty=plan.difficulty,
        question_count=len(questions),
        title=f"Diagnostic Quiz: {payload.topic or payload.learning_goal[:60]}",
    )
    db.add(quiz)
    db.flush()

    for q in questions:
        db.add(
            Question(
                quiz_id=quiz.id,
                prompt=q.prompt,
                options_json=json.dumps(q.options),
                correct_option_index=q.correct_option_index,
                category=q.category,
                main_topic=payload.topic,
                subcategory=q.category,
                explanation=q.explanation,
            )
        )

    db.commit()
    db.refresh(quiz)

    db_questions = db.query(Question).filter(Question.quiz_id == quiz.id).all()
    return GenerateQuizResponse(
        quiz_id=quiz.id,
        title=quiz.title,
        difficulty=quiz.difficulty,
        difficulty_rationale=plan.difficulty_rationale,
        generation_prompt=plan.prompt,
        question_count=quiz.question_count,
        questions=[
            {
                "id": q.id,
                "prompt": q.prompt,
                "options": json.loads(q.options_json),
                "main_topic": q.main_topic or quiz.topic,
                "category": q.subcategory or q.category,
                "subcategory": q.subcategory or q.category,
                "correct_option_index": q.correct_option_index,
                "explanation": q.explanation,
            }
            for q in db_questions
        ],
    )


def _run_generation_job(job_id: str):
    with _generation_jobs_lock:
        job = _generation_jobs.get(job_id)
    if not job:
        return

    with job.lock:
        job.state = "running"
        job.stage = "Resolving follow-up context"
        job.stage_detail = "Collecting prior attempts, weak categories, and flagged prompts"

    db = next(get_db())
    try:
        cleaned_payload = _prepare_payload_with_cleaned_topic(job.payload)
        weak_categories, flagged_prompts, prior_attempt_percentage, existing_questions = _resolve_followup_context(
            cleaned_payload, db
        )
        logger.info(
            "Resolved follow-up context (weak_categories=%s, flagged_prompts=%s, prior_attempt_percentage=%s)",
            len(weak_categories),
            len(flagged_prompts),
            prior_attempt_percentage,
        )

        _track_progress(
            job,
            stage="Generating question chunks",
            generated=0,
            verified=0,
            stage_detail="Planning category mix and starting parallel generation",
        )
        questions, plan = service.generate_questions(
            topic=cleaned_payload.topic,
            learning_goal=cleaned_payload.learning_goal,
            difficulty=cleaned_payload.difficulty,
            question_count=cleaned_payload.question_count,
            use_web_search=cleaned_payload.use_web_search,
            weak_categories=weak_categories,
            flagged_prompts=flagged_prompts,
            prior_attempt_percentage=prior_attempt_percentage,
            custom_instructions=cleaned_payload.custom_instructions,
            existing_questions=existing_questions,
            progress_callback=lambda generated, total, category_event="": _track_progress(
                job,
                stage=f"Generating questions ({generated}/{total})",
                generated=generated,
                total=total,
                stage_detail="Generating category batches in parallel",
                category_event=category_event,
            ),
        )

        _track_progress(
            job,
            stage="Verifying generated questions",
            verified=0,
            stage_detail="Checking answer validity and explanation quality",
        )
        if hasattr(service, "rebalance_questions_for_option_lengths"):
            questions = service.rebalance_questions_for_option_lengths(
                questions,
                progress_callback=lambda fixed, total: _track_progress(
                    job,
                    stage=f"Balancing option lengths ({fixed}/{total})",
                    verified=fixed,
                    total=total,
                    stage_detail="Normalizing option lengths for fairer multiple choice",
                ),
            )
        verification = service.verify_questions(
            questions,
            progress_callback=lambda checked, total: _track_progress(
                job,
                stage=f"Verifying questions ({checked}/{total})",
                verified=checked,
                stage_detail="Running strict schema and quality verification",
            ),
        )
        if verification.is_valid:
            verified_questions = questions
        else:
            logger.warning("Batch verification reported issues; falling back to per-question filtering/repair")
            total_questions = len(questions)
            verified_questions = [None] * total_questions

            def _verify_or_repair(idx: int, question):
                single_result = service.verify_questions([question])
                if single_result.is_valid:
                    return idx, question, "verified"

                repaired = service.repair_question(question, single_result.reasons)
                if repaired:
                    repaired_result = service.verify_questions([repaired])
                    if repaired_result.is_valid:
                        return idx, repaired, "repaired"

                logger.warning(
                    "Dropping question %s after verification failure: %s",
                    idx + 1,
                    "; ".join(single_result.reasons),
                )
                return idx, None, "dropped"

            completed = 0
            with ThreadPoolExecutor(max_workers=min(8, total_questions or 1)) as executor:
                futures = [executor.submit(_verify_or_repair, idx, question) for idx, question in enumerate(questions)]
                for future in as_completed(futures):
                    idx, maybe_question, status = future.result()
                    verified_questions[idx] = maybe_question
                    completed += 1
                    _track_progress(
                        job,
                        stage=f"Filtering questions ({completed}/{total_questions})",
                        verified=completed,
                        total=total_questions,
                        stage_detail="Repairing or dropping questions that failed validation",
                    )
                    if status == "repaired":
                        logger.info("Repaired question %s after verification failure", idx + 1)

            verified_questions = [q for q in verified_questions if q is not None]
            if not verified_questions:
                raise QuestionGenerationError(
                    "Verification failed for all questions: " + "; ".join(verification.reasons)
                )

        _track_progress(job, stage="Persisting quiz", stage_detail="Saving quiz and question records")
        result = _persist_quiz(db, cleaned_payload, verified_questions, plan)

        with job.lock:
            job.result = result
            job.total_questions = len(verified_questions)
            job.generated_questions = len(questions)
            job.verified_questions = len(verified_questions)
            job.stage = "Completed"
            job.stage_detail = "Quiz is ready"
            job.state = "completed"
        logger.info("Quiz generation job %s completed with %s questions", job_id, len(questions))
    except QuestionGenerationError as exc:
        with job.lock:
            job.state = "failed"
            job.error = str(exc)
            job.stage = "Failed"
            job.stage_detail = "Generation job failed"
    except Exception as exc:  # noqa: BLE001
        with job.lock:
            job.state = "failed"
            job.error = f"Quiz generation failed: {exc}"
            job.stage = "Failed"
            job.stage_detail = "Generation job failed"
    finally:
        db.close()


@app.get("/")
def root():
    return FileResponse(static_dir / "index.html")


@app.get("/api/health")
def health(db: Session = Depends(get_db)):
    db.execute(text("SELECT 1"))
    return {"status": "ok"}


@app.post("/api/quizzes/generate", response_model=QuizGenerationJobResponse)
def generate_quiz(payload: GenerateQuizRequest):
    logger.info(
        "Generate quiz request received (topic=%r, followup_from_attempt_id=%s, question_count=%s)",
        payload.topic,
        payload.followup_from_attempt_id,
        payload.question_count,
    )
    job_id = str(uuid.uuid4())
    job = GenerationJob(payload=payload, total_questions=payload.question_count)
    with _generation_jobs_lock:
        _generation_jobs[job_id] = job

    thread = threading.Thread(target=_run_generation_job, args=(job_id,), daemon=True)
    thread.start()
    return {"job_id": job_id}


@app.get("/api/quizzes/generate/{job_id}", response_model=QuizGenerationJobStatus)
def get_generation_job_status(job_id: str):
    with _generation_jobs_lock:
        job = _generation_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Generation job not found")

    with job.lock:
        return {
            "job_id": job_id,
            "state": job.state,
            "stage": job.stage,
            "generated_questions": job.generated_questions,
            "verified_questions": job.verified_questions,
            "total_questions": job.total_questions,
            "stage_detail": job.stage_detail,
            "category_progress": job.category_progress,
            "error": job.error,
            "result": job.result,
        }


@app.post("/api/quizzes/{quiz_id}/submit", response_model=AttemptResponse)
def submit_quiz(quiz_id: int, payload: SubmitQuizRequest, db: Session = Depends(get_db)):
    questions = db.query(Question).filter(Question.quiz_id == quiz_id).all()
    if not questions:
        raise HTTPException(status_code=404, detail="Quiz not found")

    question_ids = {q.id for q in questions}
    answers_by_qid = {a.question_id: a for a in payload.answers}
    if len(answers_by_qid) != len(payload.answers):
        raise HTTPException(status_code=400, detail="Each question can only be submitted once")
    unknown_ids = set(answers_by_qid.keys()) - question_ids
    if unknown_ids:
        raise HTTPException(status_code=400, detail="Submission includes unknown question_id values")

    score = 0
    attempt = Attempt(quiz_id=quiz_id, score=0, total=len(questions), percentage=0)
    db.add(attempt)
    db.flush()

    for q in questions:
        answer = answers_by_qid.get(q.id)
        selected_option_index = answer.selected_option_index if answer else None
        is_correct = selected_option_index == q.correct_option_index if selected_option_index is not None else None
        score += int(bool(is_correct))
        db.add(
            AttemptAnswer(
                attempt_id=attempt.id,
                question_id=q.id,
                selected_option_index=selected_option_index,
                is_correct=is_correct,
                flagged_for_review=answer.flagged_for_review if answer else False,
            )
        )

    percentage = round((score / len(questions)) * 100, 2)
    attempt.score = score
    attempt.percentage = percentage
    db.flush()

    answer_rows = db.query(AttemptAnswer).filter(AttemptAnswer.attempt_id == attempt.id).all()
    analysis = compute_analysis(questions, answer_rows)
    topics = build_study_topics(analysis["raw_category_summary"])
    for t in topics:
        db.add(StudyTopic(attempt_id=attempt.id, topic=t["topic"], priority=t["priority"], source=t["source"]))

    db.commit()

    return {
        "attempt_id": attempt.id,
        "score": score,
        "total": len(questions),
        "percentage": percentage,
        "study_topics": topics,
        **{k: v for k, v in analysis.items() if k != "raw_category_summary"},
    }




@app.post("/api/quizzes/{quiz_id}/questions/{question_id}/hint")
def generate_question_hint(
    quiz_id: int,
    question_id: int,
    payload: QuestionHintRequest,
    db: Session = Depends(get_db),
):
    question = db.query(Question).filter(Question.id == question_id, Question.quiz_id == quiz_id).first()
    quiz = db.query(Quiz).filter(Quiz.id == quiz_id).first()
    if not question or not quiz:
        raise HTTPException(status_code=404, detail="Question or quiz not found")

    selected_option_index = payload.selected_option_index if payload.selected_option_index is not None else -1
    cached_hint = (
        db.query(QuestionHint)
        .filter(
            QuestionHint.question_id == question.id,
            QuestionHint.selected_option_index == selected_option_index,
        )
        .first()
    )
    if cached_hint:
        return {"lesson": cached_hint.lesson, "cached": True}

    hint = service.generate_question_hint(
        topic=quiz.topic,
        learning_goal=quiz.learning_goal,
        question={
            "prompt": question.prompt,
            "category": question.subcategory or question.category,
            "options": json.loads(question.options_json),
            "explanation": question.explanation,
        },
        selected_option_index=payload.selected_option_index,
    )
    db.add(
        QuestionHint(
            question_id=question.id,
            selected_option_index=selected_option_index,
            lesson=hint,
        )
    )
    db.commit()
    return {"lesson": hint, "cached": False}
@app.post("/api/quizzes/{quiz_id}/error-lesson")
def generate_error_lesson(quiz_id: int, payload: SubmitQuizRequest, db: Session = Depends(get_db)):
    questions = db.query(Question).filter(Question.quiz_id == quiz_id).all()
    quiz = db.query(Quiz).filter(Quiz.id == quiz_id).first()
    if not questions or not quiz:
        raise HTTPException(status_code=404, detail="Quiz not found")

    question_ids = {q.id for q in questions}
    answers_by_qid = {a.question_id: a for a in payload.answers}
    if question_ids != set(answers_by_qid.keys()):
        raise HTTPException(status_code=400, detail="All questions must be answered exactly once")

    missed_results = []
    for question in questions:
        answer = answers_by_qid[question.id]
        if answer.selected_option_index != question.correct_option_index:
            missed_results.append(
                {
                    "prompt": question.prompt,
                    "category": question.subcategory or question.category,
                    "selected_option_index": answer.selected_option_index,
                    "correct_option_index": question.correct_option_index,
                    "options": json.loads(question.options_json),
                    "explanation": question.explanation,
                }
            )

    if not missed_results:
        return {"lesson": "Great job—you did not miss any questions in this quiz."}

    lesson = service.generate_error_lesson(
        topic=quiz.topic,
        learning_goal=quiz.learning_goal,
        missed_question_results=missed_results,
    )
    return {"lesson": lesson}


@app.get("/api/stats/history", response_model=HistoricalStatsResponse)
def get_historical_stats(db: Session = Depends(get_db)):
    attempts = db.query(Attempt).all()
    answers = db.query(AttemptAnswer).all()
    questions = db.query(Question).all()
    quizzes = db.query(Quiz).all()

    question_by_id = {q.id: q for q in questions}
    quiz_by_id = {qz.id: qz for qz in quizzes}
    attempt_by_id = {attempt.id: attempt for attempt in attempts}

    total_answers = len(answers)
    correct_answers = sum(1 for answer in answers if answer.is_correct is True)
    global_accuracy = round((correct_answers / total_answers) * 100, 2) if total_answers else 0.0

    category_bucket: dict[str, dict[str, int]] = {}
    missed_questions = []

    for answer in answers:
        question = question_by_id.get(answer.question_id)
        if not question:
            continue

        subcategory = question.subcategory or question.category
        stats = category_bucket.setdefault(subcategory, {"correct": 0, "total": 0})
        stats["total"] += 1
        stats["correct"] += int(answer.is_correct is True)

        if answer.is_correct:
            continue

        options = json.loads(question.options_json)
        selected_text = (
            options[answer.selected_option_index]
            if answer.selected_option_index is not None and 0 <= answer.selected_option_index < len(options)
            else ""
        )
        correct_text = options[question.correct_option_index] if 0 <= question.correct_option_index < len(options) else ""
        attempt = attempt_by_id.get(answer.attempt_id)
        quiz = quiz_by_id.get(attempt.quiz_id) if attempt else None

        missed_questions.append(
            {
                "question_id": question.id,
                "prompt": question.prompt,
                "category": subcategory,
                "quiz_topic": (quiz.topic if quiz else "") or "General",
                "selected_option_index": answer.selected_option_index,
                "selected_option_text": selected_text,
                "correct_option_index": question.correct_option_index,
                "correct_option_text": correct_text,
                "explanation": question.explanation,
            }
        )

    per_category_stats = [
        {
            "category": category,
            "correct": row["correct"],
            "total": row["total"],
            "accuracy_percentage": round((row["correct"] / row["total"]) * 100, 2) if row["total"] else 0.0,
        }
        for category, row in sorted(category_bucket.items(), key=lambda item: item[0].lower())
    ]

    missed_questions.sort(key=lambda row: (row["quiz_topic"].lower(), row["category"].lower(), row["question_id"]))

    return {
        "global_stats": {
            "attempts": len(attempts),
            "questions_answered": total_answers,
            "correct_answers": correct_answers,
            "accuracy_percentage": global_accuracy,
        },
        "per_category_stats": per_category_stats,
        "missed_questions": missed_questions,
    }


def _isoformat(dt: datetime | None) -> str:
    if not dt:
        return ""
    return dt.isoformat()


@app.post("/api/stats/reset", response_model=ResetStatsResponse)
def reset_user_stats(db: Session = Depends(get_db)):
    answer_count = db.query(AttemptAnswer).count()
    study_topic_count = db.query(StudyTopic).count()
    attempt_count = db.query(Attempt).count()

    if answer_count:
        db.query(AttemptAnswer).delete(synchronize_session=False)
    if study_topic_count:
        db.query(StudyTopic).delete(synchronize_session=False)
    if attempt_count:
        db.query(Attempt).delete(synchronize_session=False)
    db.commit()

    return {
        "deleted_attempts": attempt_count,
        "deleted_answers": answer_count,
        "deleted_study_topics": study_topic_count,
    }


@app.get("/api/dataset/export", response_model=DatasetExportResponse)
def export_dataset(db: Session = Depends(get_db)):
    quizzes = db.query(Quiz).order_by(Quiz.id.asc()).all()
    export_rows = []

    for quiz in quizzes:
        questions = db.query(Question).filter(Question.quiz_id == quiz.id).order_by(Question.id.asc()).all()
        question_index_by_id = {question.id: idx + 1 for idx, question in enumerate(questions)}

        attempts = db.query(Attempt).filter(Attempt.quiz_id == quiz.id).order_by(Attempt.id.asc()).all()
        attempt_rows = []
        for attempt in attempts:
            answers = (
                db.query(AttemptAnswer)
                .filter(AttemptAnswer.attempt_id == attempt.id)
                .order_by(AttemptAnswer.id.asc())
                .all()
            )
            attempt_rows.append(
                {
                    "score": attempt.score,
                    "total": attempt.total,
                    "percentage": attempt.percentage,
                    "started_at": _isoformat(attempt.started_at),
                    "submitted_at": _isoformat(attempt.submitted_at),
                    "answers": [
                        {
                            "question_position": question_index_by_id.get(answer.question_id, 0),
                            "selected_option_index": answer.selected_option_index,
                            "is_correct": answer.is_correct,
                            "flagged_for_review": answer.flagged_for_review,
                        }
                        for answer in answers
                        if question_index_by_id.get(answer.question_id)
                    ],
                }
            )

        export_rows.append(
            {
                "topic": quiz.topic,
                "learning_goal": quiz.learning_goal,
                "difficulty": quiz.difficulty,
                "question_count": quiz.question_count,
                "title": quiz.title,
                "created_at": _isoformat(quiz.created_at),
                "questions": [
                    {
                        "prompt": question.prompt,
                        "options": json.loads(question.options_json),
                        "correct_option_index": question.correct_option_index,
                        "main_topic": question.main_topic or quiz.topic,
                        "category": question.subcategory or question.category,
                        "subcategory": question.subcategory or question.category,
                        "explanation": question.explanation,
                    }
                    for question in questions
                ],
                "attempts": attempt_rows,
            }
        )

    return {"format_version": "1.0", "quizzes": export_rows}


@app.post("/api/dataset/import", response_model=DatasetImportResponse)
def import_dataset(payload: DatasetImportRequest, db: Session = Depends(get_db)):
    imported_quizzes = 0
    imported_questions = 0
    imported_attempts = 0
    imported_answers = 0

    for incoming_quiz in payload.quizzes:
        question_count = len(incoming_quiz.questions)
        quiz = Quiz(
            topic=incoming_quiz.topic,
            learning_goal=incoming_quiz.learning_goal,
            difficulty=incoming_quiz.difficulty,
            question_count=question_count,
            title=incoming_quiz.title or f"Imported Quiz: {incoming_quiz.topic or 'General'}",
        )
        db.add(quiz)
        db.flush()

        db_questions = []
        for incoming_question in incoming_quiz.questions:
            question = Question(
                quiz_id=quiz.id,
                prompt=incoming_question.prompt,
                options_json=json.dumps(incoming_question.options),
                correct_option_index=incoming_question.correct_option_index,
                category=incoming_question.category,
                main_topic=incoming_question.main_topic or incoming_quiz.topic,
                subcategory=incoming_question.subcategory or incoming_question.category,
                explanation=incoming_question.explanation,
            )
            db.add(question)
            db_questions.append(question)
            imported_questions += 1

        db.flush()

        for incoming_attempt in incoming_quiz.attempts:
            attempt = Attempt(
                quiz_id=quiz.id,
                score=incoming_attempt.score or 0,
                total=incoming_attempt.total or question_count,
                percentage=incoming_attempt.percentage or 0,
            )
            db.add(attempt)
            db.flush()

            correct_counter = 0
            for incoming_answer in incoming_attempt.answers:
                if incoming_answer.question_position > len(db_questions):
                    continue
                mapped_question = db_questions[incoming_answer.question_position - 1]
                is_correct = (
                    incoming_answer.is_correct
                    if incoming_answer.is_correct is not None
                    else (
                        incoming_answer.selected_option_index == mapped_question.correct_option_index
                        if incoming_answer.selected_option_index is not None
                        else None
                    )
                )
                correct_counter += int(bool(is_correct))
                db.add(
                    AttemptAnswer(
                        attempt_id=attempt.id,
                        question_id=mapped_question.id,
                        selected_option_index=incoming_answer.selected_option_index,
                        is_correct=is_correct,
                        flagged_for_review=incoming_answer.flagged_for_review,
                    )
                )
                imported_answers += 1

            if incoming_attempt.score is None:
                attempt.score = correct_counter
            if incoming_attempt.total is None:
                attempt.total = len(db_questions)
            if incoming_attempt.percentage is None:
                attempt.percentage = round((attempt.score / attempt.total) * 100, 2) if attempt.total else 0.0

            imported_attempts += 1

        imported_quizzes += 1

    db.commit()
    return {
        "imported_quizzes": imported_quizzes,
        "imported_questions": imported_questions,
        "imported_attempts": imported_attempts,
        "imported_answers": imported_answers,
    }


@app.post("/api/attempts/{attempt_id}/study-plan", response_model=AttemptResponse)
def update_study_plan(attempt_id: int, payload: StudyPlanUpdateRequest, db: Session = Depends(get_db)):
    attempt = db.query(Attempt).filter(Attempt.id == attempt_id).first()
    if not attempt:
        raise HTTPException(status_code=404, detail="Attempt not found")

    db.query(StudyTopic).filter(StudyTopic.attempt_id == attempt_id).delete()
    for idx, topic in enumerate(payload.topics, start=1):
        db.add(StudyTopic(attempt_id=attempt_id, topic=topic.topic, priority=idx, source="user"))
    db.commit()
    return _build_attempt_response(attempt, db)


@app.get("/api/attempts/{attempt_id}", response_model=AttemptResponse)
def get_attempt(attempt_id: int, db: Session = Depends(get_db)):
    attempt = db.query(Attempt).filter(Attempt.id == attempt_id).first()
    if not attempt:
        raise HTTPException(status_code=404, detail="Attempt not found")
    return _build_attempt_response(attempt, db)


def _build_attempt_response(attempt: Attempt, db: Session):
    questions = db.query(Question).filter(Question.quiz_id == attempt.quiz_id).all()
    answers = db.query(AttemptAnswer).filter(AttemptAnswer.attempt_id == attempt.id).all()
    topics = (
        db.query(StudyTopic)
        .filter(StudyTopic.attempt_id == attempt.id)
        .order_by(StudyTopic.priority.asc())
        .all()
    )
    analysis = compute_analysis(questions, answers)
    return {
        "attempt_id": attempt.id,
        "score": attempt.score,
        "total": attempt.total,
        "percentage": attempt.percentage,
        "study_topics": [{"topic": t.topic, "priority": t.priority, "source": t.source} for t in topics],
        **{k: v for k, v in analysis.items() if k != "raw_category_summary"},
    }
