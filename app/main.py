import json
import logging
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


@app.get("/")
def root():
    return FileResponse(static_dir / "index.html")


@app.get("/api/health")
def health(db: Session = Depends(get_db)):
    db.execute(text("SELECT 1"))
    return {"status": "ok"}


@app.post("/api/quizzes/generate", response_model=GenerateQuizResponse)
def generate_quiz(payload: GenerateQuizRequest, db: Session = Depends(get_db)):
    logger.info("Generate quiz request received (topic=%r, followup_from_attempt_id=%s, question_count=%s)", payload.topic, payload.followup_from_attempt_id, payload.question_count)
    weak_categories = []
    flagged_prompts = []
    prior_attempt_percentage = None
    effective_difficulty = payload.difficulty

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

    try:
        logger.info("Resolved follow-up context (weak_categories=%s, flagged_prompts=%s, prior_attempt_percentage=%s)", len(weak_categories), len(flagged_prompts), prior_attempt_percentage)
        questions, plan = service.generate_questions(
            topic=payload.topic,
            learning_goal=payload.learning_goal,
            difficulty=payload.difficulty,
            question_count=payload.question_count,
            use_web_search=payload.use_web_search,
            weak_categories=weak_categories,
            flagged_prompts=flagged_prompts,
            prior_attempt_percentage=prior_attempt_percentage,
        )
        effective_difficulty = plan.difficulty
        verification = service.verify_questions(questions)
        logger.info("Quiz generated successfully with %s questions and difficulty=%s", len(questions), effective_difficulty)
        if not verification.is_valid:
            raise QuestionGenerationError("Verification failed: " + "; ".join(verification.reasons))
    except QuestionGenerationError as exc:
        detail = str(exc)
        status = 401 if "api key" in detail.lower() else 500
        raise HTTPException(status_code=status, detail=detail) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Quiz generation failed: {exc}") from exc

    quiz = Quiz(
        topic=payload.topic,
        learning_goal=payload.learning_goal,
        difficulty=effective_difficulty,
        question_count=payload.question_count,
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
    return {
        "quiz_id": quiz.id,
        "title": quiz.title,
        "difficulty": quiz.difficulty,
        "difficulty_rationale": plan.difficulty_rationale,
        "generation_prompt": plan.prompt,
        "question_count": quiz.question_count,
        "questions": [
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
