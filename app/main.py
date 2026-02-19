import json
import logging
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
from app.models import Attempt, AttemptAnswer, Question, Quiz, StudyTopic
from app.schemas import (
    AttemptResponse,
    GenerateQuizRequest,
    GenerateQuizResponse,
    QuizGenerationJobResponse,
    QuizGenerationJobStatus,
    StudyPlanUpdateRequest,
    SubmitQuizRequest,
)
from app.services import LLMQuizService, QuestionGenerationError, build_study_topics, compute_analysis


app = FastAPI(title="Quizzinfinity")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)
service = LLMQuizService()

Base.metadata.create_all(bind=engine)

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
):
    with job.lock:
        job.stage = stage
        if generated is not None:
            job.generated_questions = generated
        if verified is not None:
            job.verified_questions = verified
        if total is not None:
            job.total_questions = total


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
                "category": q.category,
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

    db = next(get_db())
    try:
        weak_categories, flagged_prompts, prior_attempt_percentage, existing_questions = _resolve_followup_context(
            job.payload, db
        )
        logger.info(
            "Resolved follow-up context (weak_categories=%s, flagged_prompts=%s, prior_attempt_percentage=%s)",
            len(weak_categories),
            len(flagged_prompts),
            prior_attempt_percentage,
        )

        _track_progress(job, stage="Generating question chunks", generated=0, verified=0)
        questions, plan = service.generate_questions(
            topic=job.payload.topic,
            learning_goal=job.payload.learning_goal,
            difficulty=job.payload.difficulty,
            question_count=job.payload.question_count,
            use_web_search=job.payload.use_web_search,
            weak_categories=weak_categories,
            flagged_prompts=flagged_prompts,
            prior_attempt_percentage=prior_attempt_percentage,
            custom_instructions=job.payload.custom_instructions,
            existing_questions=existing_questions,
            progress_callback=lambda generated, total: _track_progress(
                job,
                stage=f"Generating questions ({generated}/{total})",
                generated=generated,
                total=total,
            ),
        )

        _track_progress(job, stage="Verifying generated questions", verified=0)
        verification = service.verify_questions(
            questions,
            progress_callback=lambda checked, total: _track_progress(
                job,
                stage=f"Verifying questions ({checked}/{total})",
                verified=checked,
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
                    )
                    if status == "repaired":
                        logger.info("Repaired question %s after verification failure", idx + 1)

            verified_questions = [q for q in verified_questions if q is not None]
            if not verified_questions:
                raise QuestionGenerationError(
                    "Verification failed for all questions: " + "; ".join(verification.reasons)
                )

        _track_progress(job, stage="Persisting quiz")
        result = _persist_quiz(db, job.payload, verified_questions, plan)

        with job.lock:
            job.result = result
            job.total_questions = len(verified_questions)
            job.generated_questions = len(questions)
            job.verified_questions = len(verified_questions)
            job.stage = "Completed"
            job.state = "completed"
        logger.info("Quiz generation job %s completed with %s questions", job_id, len(questions))
    except QuestionGenerationError as exc:
        with job.lock:
            job.state = "failed"
            job.error = str(exc)
            job.stage = "Failed"
    except Exception as exc:  # noqa: BLE001
        with job.lock:
            job.state = "failed"
            job.error = f"Quiz generation failed: {exc}"
            job.stage = "Failed"
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
    if question_ids != set(answers_by_qid.keys()):
        raise HTTPException(status_code=400, detail="All questions must be answered exactly once")

    score = 0
    attempt = Attempt(quiz_id=quiz_id, score=0, total=len(questions), percentage=0)
    db.add(attempt)
    db.flush()

    for q in questions:
        answer = answers_by_qid[q.id]
        is_correct = answer.selected_option_index == q.correct_option_index
        score += int(is_correct)
        db.add(
            AttemptAnswer(
                attempt_id=attempt.id,
                question_id=q.id,
                selected_option_index=answer.selected_option_index,
                is_correct=is_correct,
                flagged_for_review=answer.flagged_for_review,
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
                    "category": question.category,
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
