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
    custom_instructions: str = ""

    @model_validator(mode="after")
    def topic_or_goal_required(self):
        if not self.topic.strip() and not self.learning_goal.strip():
            raise ValueError("Either topic or learning_goal must be provided")
        return self


class QuizQuestionOut(BaseModel):
    id: int
    prompt: str
    options: List[str]
    main_topic: str
    category: str
    subcategory: str
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


class QuizGenerationJobResponse(BaseModel):
    job_id: str


class QuizGenerationJobStatus(BaseModel):
    job_id: str
    state: str
    stage: str
    generated_questions: int = 0
    verified_questions: int = 0
    total_questions: int = 0
    error: str = ""
    result: Optional[GenerateQuizResponse] = None


class AnswerIn(BaseModel):
    question_id: int
    selected_option_index: Optional[int] = Field(default=None, ge=0, le=3)
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
    main_topic: str
    category: str
    subcategory: str
    selected_option_index: Optional[int]
    correct_option_index: int
    is_correct: Optional[bool]
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


class MissedQuestionStatOut(BaseModel):
    question_id: int
    prompt: str
    category: str
    quiz_topic: str
    selected_option_index: Optional[int]
    selected_option_text: str
    correct_option_index: int
    correct_option_text: str
    explanation: str


class HistoricalGlobalStatsOut(BaseModel):
    attempts: int
    questions_answered: int
    correct_answers: int
    accuracy_percentage: float


class HistoricalCategoryStatOut(BaseModel):
    category: str
    correct: int
    total: int
    accuracy_percentage: float


class HistoricalStatsResponse(BaseModel):
    global_stats: HistoricalGlobalStatsOut
    per_category_stats: List[HistoricalCategoryStatOut]
    missed_questions: List[MissedQuestionStatOut]


class ExportAttemptAnswerOut(BaseModel):
    question_position: int
    selected_option_index: Optional[int]
    is_correct: Optional[bool]
    flagged_for_review: bool


class ExportAttemptOut(BaseModel):
    score: int
    total: int
    percentage: float
    started_at: str
    submitted_at: str
    answers: List[ExportAttemptAnswerOut]


class ExportQuestionOut(BaseModel):
    prompt: str
    options: List[str]
    correct_option_index: int
    main_topic: str
    category: str
    subcategory: str
    explanation: str


class ExportQuizOut(BaseModel):
    topic: str
    learning_goal: str
    difficulty: int
    question_count: int
    title: str
    created_at: str
    questions: List[ExportQuestionOut]
    attempts: List[ExportAttemptOut]


class DatasetExportResponse(BaseModel):
    format_version: str
    quizzes: List[ExportQuizOut]


class ImportAttemptAnswerIn(BaseModel):
    question_position: int = Field(ge=1)
    selected_option_index: Optional[int] = Field(default=None, ge=0, le=3)
    is_correct: Optional[bool] = None
    flagged_for_review: bool = False


class ImportAttemptIn(BaseModel):
    score: Optional[int] = None
    total: Optional[int] = None
    percentage: Optional[float] = None
    answers: List[ImportAttemptAnswerIn] = Field(default_factory=list)


class ImportQuestionIn(BaseModel):
    prompt: str
    options: List[str] = Field(min_length=4, max_length=4)
    correct_option_index: int = Field(ge=0, le=3)
    main_topic: str = ""
    category: str
    subcategory: str = ""
    explanation: str


class ImportQuizIn(BaseModel):
    topic: str = ""
    learning_goal: str = ""
    difficulty: int = Field(default=9, ge=1, le=10)
    title: str = "Imported Quiz"
    questions: List[ImportQuestionIn] = Field(min_length=1)
    attempts: List[ImportAttemptIn] = Field(default_factory=list)


class DatasetImportRequest(BaseModel):
    quizzes: List[ImportQuizIn] = Field(default_factory=list)


class DatasetImportResponse(BaseModel):
    imported_quizzes: int
    imported_questions: int
    imported_attempts: int
    imported_answers: int


class VerificationResult(BaseModel):
    is_valid: bool
    reasons: List[str] = Field(default_factory=list)
