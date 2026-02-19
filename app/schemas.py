from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class QuestionPayload(BaseModel):
    prompt: str
    options: List[str] = Field(min_length=4, max_length=4)
    correct_option_index: int
    category: str
    explanation: str


class GenerateQuizRequest(BaseModel):
    topic: str = ""
    learning_goal: str = ""
    difficulty: int = Field(default=9, ge=1, le=10)
    question_count: int = Field(default=20, ge=5, le=30)
    use_web_search: bool = True
    followup_from_attempt_id: Optional[int] = None

    @model_validator(mode="after")
    def topic_or_goal_required(self):
        if not self.topic.strip() and not self.learning_goal.strip():
            raise ValueError("Either topic or learning_goal must be provided")
        return self


class QuizQuestionOut(BaseModel):
    id: int
    prompt: str
    options: List[str]
    category: str
    correct_option_index: int
    explanation: str


class GenerateQuizResponse(BaseModel):
    quiz_id: int
    title: str
    difficulty: int
    question_count: int
    generation_prompt: str
    difficulty_rationale: str
    questions: List[QuizQuestionOut]


class AnswerIn(BaseModel):
    question_id: int
    selected_option_index: int = Field(ge=0, le=3)
    flagged_for_review: bool = False


class SubmitQuizRequest(BaseModel):
    answers: List[AnswerIn]


class StudyTopicOut(BaseModel):
    topic: str
    priority: int
    source: str


class StudyTopicUpdateIn(BaseModel):
    topic: str
    priority: int = Field(ge=1)


class StudyPlanUpdateRequest(BaseModel):
    topics: List[StudyTopicUpdateIn]


class QuestionResultOut(BaseModel):
    question_id: int
    prompt: str
    category: str
    selected_option_index: int
    correct_option_index: int
    is_correct: bool
    flagged_for_review: bool
    explanation: str
    options: List[str]


class CategorySummaryOut(BaseModel):
    category: str
    correct: int
    total: int
    percentage: float


class AttemptResponse(BaseModel):
    attempt_id: int
    score: int
    total: int
    percentage: float
    category_summary: List[CategorySummaryOut]
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    study_topics: List[StudyTopicOut]
    question_results: List[QuestionResultOut]


class VerificationResult(BaseModel):
    is_valid: bool
    reasons: List[str] = Field(default_factory=list)
