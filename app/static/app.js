const { useMemo, useRef, useState } = React;


async function parseApiResponse(res) {
  const raw = await res.text();
  if (!raw) {
    return { ok: res.ok, status: res.status, data: null, detail: 'Empty response from server' };
  }

  try {
    const data = JSON.parse(raw);
    return { ok: res.ok, status: res.status, data, detail: data?.detail };
  } catch (err) {
    return {
      ok: false,
      status: res.status,
      data: null,
      detail: `Server returned non-JSON response (${err.message})`,
    };
  }
}

function App() {
  const [topic, setTopic] = useState('');
  const [learningGoal, setLearningGoal] = useState('');
  const [difficulty, setDifficulty] = useState(9);
  const [questionCount, setQuestionCount] = useState(20);
  const [customInstructions, setCustomInstructions] = useState('');
  const [followupInstructions, setFollowupInstructions] = useState('');
  const [loading, setLoading] = useState(false);
  const [loadingStage, setLoadingStage] = useState('Queued');
  const [progressPct, setProgressPct] = useState(0);
  const [progressMeta, setProgressMeta] = useState('Waiting for worker...');
  const [error, setError] = useState('');
  const [quiz, setQuiz] = useState(null);
  const [answers, setAnswers] = useState({});
  const [flags, setFlags] = useState({});
  const [validated, setValidated] = useState({});
  const [idx, setIdx] = useState(0);
  const [result, setResult] = useState(null);
  const [studyTopics, setStudyTopics] = useState([]);
  const [lessonLoading, setLessonLoading] = useState(false);
  const [errorLesson, setErrorLesson] = useState('');
  const [showLessonModal, setShowLessonModal] = useState(false);
  const [historyStats, setHistoryStats] = useState(null);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [showHistoryModal, setShowHistoryModal] = useState(false);
  const fileInputRef = useRef(null);

  const missedQuestions = useMemo(() => {
    if (!quiz) return [];
    return quiz.questions.filter((q) => validated[q.id] && answers[q.id] !== q.correct_option_index);
  }, [quiz, validated, answers]);

  async function pollGenerationJob(jobId) {
    while (true) {
      const statusRes = await fetch(`/api/quizzes/generate/${jobId}`);
      const statusParsed = await parseApiResponse(statusRes);
      if (!statusParsed.ok || !statusParsed.data) {
        throw new Error(statusParsed.detail || 'Failed to read generation status');
      }

      const status = statusParsed.data;
      setLoadingStage(status.stage || 'Working');

      const total = Math.max(status.total_questions || questionCount, 1);
      const generated = Math.min(status.generated_questions || 0, total);
      const verified = Math.min(status.verified_questions || 0, total);

      let pct = 4;
      if (status.stage?.toLowerCase().includes('resolving')) pct = 8;
      if (status.stage?.toLowerCase().includes('generating')) {
        pct = Math.max(pct, 12 + Math.round((generated / total) * 58));
        setProgressMeta(`Generated ${generated}/${total} questions`);
      }
      if (status.stage?.toLowerCase().includes('verifying')) {
        pct = Math.max(pct, 72 + Math.round((verified / total) * 24));
        setProgressMeta(`Verified ${verified}/${total} questions`);
      }
      if (status.stage?.toLowerCase().includes('persisting')) {
        pct = Math.max(pct, 97);
        setProgressMeta('Saving quiz to local database');
      }
      if (status.state === 'completed') {
        setProgressPct(100);
        setProgressMeta('Done');
        return status.result;
      }
      if (status.state === 'failed') {
        throw new Error(status.error || 'Generation failed');
      }
      if (!status.stage?.toLowerCase().includes('generating') && !status.stage?.toLowerCase().includes('verifying') && !status.stage?.toLowerCase().includes('persisting')) {
        setProgressMeta(status.stage || 'Working');
      }
      setProgressPct(Math.min(pct, 99));
      await new Promise((resolve) => setTimeout(resolve, 700));
    }
  }

  async function generateQuiz(followupFrom = null) {
    setLoading(true);
    setLoadingStage('Creating generation job');
    setProgressPct(2);
    setProgressMeta('Submitting generation request');
    setError('');
    setResult(null);
    setQuiz(null);
    setAnswers({});
    setFlags({});
    setValidated({});
    try {
      const res = await fetch('/api/quizzes/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          topic,
          learning_goal: learningGoal,
          difficulty,
          question_count: questionCount,
          use_web_search: true,
          followup_from_attempt_id: followupFrom,
          custom_instructions: followupFrom ? followupInstructions : customInstructions,
        }),
      });
      const parsed = await parseApiResponse(res);
      if (!parsed.ok || !parsed.data) throw new Error(parsed.detail || 'Generation failed');

      const generatedQuiz = await pollGenerationJob(parsed.data.job_id);
      setQuiz(generatedQuiz);
      setIdx(0);
      setErrorLesson('');
      setShowLessonModal(false);
      setProgressPct(100);
      setProgressMeta('Done');
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function submitQuiz() {
    const formattedAnswers = quiz.questions.map((q) => ({
      question_id: q.id,
      selected_option_index: answers[q.id],
      flagged_for_review: !!flags[q.id],
    }));
    const res = await fetch(`/api/quizzes/${quiz.quiz_id}/submit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ answers: formattedAnswers }),
    });
    const parsed = await parseApiResponse(res);
    if (!parsed.ok || !parsed.data) return setError(parsed.detail || 'Submit failed');
    setResult(parsed.data);
    setStudyTopics(parsed.data.study_topics || []);
  }

  async function requestErrorLesson() {
    if (!quiz) return;
    setLessonLoading(true);
    setError('');
    try {
      const formattedAnswers = quiz.questions.map((q) => ({
        question_id: q.id,
        selected_option_index: answers[q.id],
        flagged_for_review: !!flags[q.id],
      }));
      const res = await fetch(`/api/quizzes/${quiz.quiz_id}/error-lesson`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ answers: formattedAnswers }),
      });
      const parsed = await parseApiResponse(res);
      if (!parsed.ok || !parsed.data) throw new Error(parsed.detail || 'Failed to generate lesson');
      setErrorLesson(parsed.data.lesson || 'No lesson returned.');
      setShowLessonModal(true);
    } catch (e) {
      setError(e.message);
    } finally {
      setLessonLoading(false);
    }
  }

  async function openHistoricalStats() {
    setHistoryLoading(true);
    setError('');
    try {
      const res = await fetch('/api/stats/history');
      const parsed = await parseApiResponse(res);
      if (!parsed.ok || !parsed.data) throw new Error(parsed.detail || 'Failed to load historical stats');
      setHistoryStats(parsed.data);
      setShowHistoryModal(true);
    } catch (e) {
      setError(e.message);
    } finally {
      setHistoryLoading(false);
    }
  }

  async function exportDataset() {
    setError('');
    try {
      const res = await fetch('/api/dataset/export');
      const parsed = await parseApiResponse(res);
      if (!parsed.ok || !parsed.data) throw new Error(parsed.detail || 'Failed to export dataset');

      const blob = new Blob([JSON.stringify(parsed.data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = `quizzinfinity-dataset-${Date.now()}.json`;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
    } catch (e) {
      setError(e.message);
    }
  }

  function triggerImportDataset() {
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
      fileInputRef.current.click();
    }
  }

  async function importDataset(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    setError('');

    try {
      const text = await file.text();
      const payload = JSON.parse(text);
      const res = await fetch('/api/dataset/import', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const parsed = await parseApiResponse(res);
      if (!parsed.ok || !parsed.data) throw new Error(parsed.detail || 'Failed to import dataset');
      const imported = parsed.data;
      alert(`Import complete: ${imported.imported_quizzes} quizzes, ${imported.imported_questions} questions, ${imported.imported_attempts} attempts.`);
    } catch (e) {
      setError(`Import failed: ${e.message}`);
    }
  }

  async function saveStudyPlan() {
    if (!result) return;
    const res = await fetch(`/api/attempts/${result.attempt_id}/study-plan`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        topics: studyTopics.map((t, i) => ({ topic: t.topic, priority: i + 1 })),
      }),
    });
    const parsed = await parseApiResponse(res);
    if (!parsed.ok || !parsed.data) return setError(parsed.detail || 'Failed to save study plan');
    setResult(parsed.data);
    setStudyTopics(parsed.data.study_topics || []);
  }

  function moveTopic(i, dir) {
    const j = i + dir;
    if (j < 0 || j >= studyTopics.length) return;
    const copy = [...studyTopics];
    [copy[i], copy[j]] = [copy[j], copy[i]];
    setStudyTopics(copy.map((t, rank) => ({ ...t, priority: rank + 1 })));
  }

  function validateQuestion() {
    const q = quiz.questions[idx];
    if (answers[q.id] === undefined) return;
    setValidated({ ...validated, [q.id]: true });
  }

  if (!quiz) {
    return (
      <div className="shell">
        <div className="container">
          <h1>Quizzinfinity</h1>
          <p className="small">Beautifully adaptive quizzes that start hard, then adjust based on your first-round analysis.</p>
          {error && <div className="error-banner">{error}</div>}
          <input type="text" placeholder="Topic (e.g. Probability Theory)" value={topic} onChange={(e) => setTopic(e.target.value)} />
          <textarea rows="4" placeholder="What exactly do you want to learn? Add any preferences for easier/harder emphasis." value={learningGoal} onChange={(e) => setLearningGoal(e.target.value)} />
          <textarea rows="3" placeholder="Custom instructions for next quiz generation (optional). These are combined with your analysis + flagged questions." value={customInstructions} onChange={(e) => setCustomInstructions(e.target.value)} />
          <div className="slider-block">
            <label>Difficulty: <strong>{difficulty}</strong>/10</label>
            <input type="range" min="1" max="10" value={difficulty} onChange={(e) => setDifficulty(Number(e.target.value))} />
          </div>
          <div className="row">
            <label>Questions:</label>
            <input type="number" min="5" max="30" value={questionCount} onChange={(e) => setQuestionCount(Number(e.target.value))} style={{ width: '100px' }} />
          </div>
          <button disabled={loading} onClick={() => generateQuiz()}>{loading ? 'Generating...' : 'Generate Diagnostic Quiz'}</button>
          <button onClick={exportDataset}>Export Dataset (JSON)</button>
          <button onClick={triggerImportDataset}>Import Dataset (JSON)</button>
          <input ref={fileInputRef} type="file" accept="application/json" style={{ display: 'none' }} onChange={importDataset} />

          {loading && (
            <div className="progress-card">
              <div className="small">{loadingStage}</div>
              <div className="small">{progressMeta}</div>
              <div className="progress-track">
                <div className="progress-fill" style={{ width: `${progressPct}%` }}>
                  <span>{progressPct}%</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  }

  const q = quiz.questions[idx];
  const selected = answers[q.id];
  const answered = selected !== undefined;
  const isValidated = !!validated[q.id];
  const isCorrect = isValidated && selected === q.correct_option_index;
  const allValidated = quiz.questions.every((x) => validated[x.id]);

  return (
    <div className="shell">
      <div className="container layout">
        <div className="main-panel">
          <h1>{quiz.title}</h1>
          <div className="small">Difficulty: {quiz.difficulty}/10 · {quiz.difficulty_rationale}</div>
          <div className="small">Question {idx + 1} / {quiz.questions.length}</div>
          {error && <div className="error-banner">{error}</div>}
          <div className="question">
            <div className="badge">{q.category}</div>
            <h3>{q.prompt}</h3>
            {q.options.map((opt, i) => (
              <label className={`option ${selected === i ? 'selected' : ''}`} key={i}>
                <input type="radio" checked={selected === i} name={`q-${q.id}`} onChange={() => setAnswers({ ...answers, [q.id]: i })} disabled={isValidated} /> {opt}
              </label>
            ))}

            {!isValidated && <button disabled={!answered} onClick={validateQuestion}>Validate answer</button>}
            {isValidated && (
              <div className={`result-card ${isCorrect ? '' : 'wrong'}`}>
                <div><strong>{isCorrect ? 'Correct ✅' : 'Incorrect ❌'}</strong></div>
                <div className="small">Correct answer: {q.options[q.correct_option_index]}</div>
                <div className="small">Explanation: {q.explanation}</div>
                <label>
                  <input type="checkbox" checked={!!flags[q.id]} onChange={(e) => setFlags({ ...flags, [q.id]: e.target.checked })} /> Need to study more
                </label>
              </div>
            )}
          </div>
          <button onClick={() => setIdx(Math.max(idx - 1, 0))} disabled={idx === 0}>Previous</button>
          <button onClick={() => setIdx(Math.min(idx + 1, quiz.questions.length - 1))} disabled={idx === quiz.questions.length - 1}>Next</button>
          <button onClick={submitQuiz} disabled={!allValidated}>Submit Quiz</button>

          {result && (
            <div>
              <h2>Results: {result.score}/{result.total} ({result.percentage}%)</h2>
              <h3>Category Summary</h3>
              <ul>
                {result.category_summary.map((c) => (
                  <li key={c.category}>{c.category}: {c.correct}/{c.total} ({c.percentage}%)</li>
                ))}
              </ul>
              <p><strong>Strengths:</strong> {result.strengths.join(', ') || '—'}</p>
              <p><strong>Weaknesses:</strong> {result.weaknesses.join(', ') || '—'}</p>
              <button onClick={openHistoricalStats} disabled={historyLoading}>{historyLoading ? 'Loading stats...' : 'Historical Stats'}</button>
              <button onClick={exportDataset}>Export Dataset (JSON)</button>
              <button onClick={triggerImportDataset}>Import Dataset (JSON)</button>

              <h3>Study Priorities (basis for next quiz)</h3>
              <div className="small">Reorder and save. Follow-up prompts include your flagged questions, this analysis, and your custom instructions.</div>
              <ul>
                {studyTopics.map((t, i) => (
                  <li key={`${t.topic}-${i}`}>
                    #{i + 1} {t.topic} ({t.source})
                    <button onClick={() => moveTopic(i, -1)}>↑</button>
                    <button onClick={() => moveTopic(i, 1)}>↓</button>
                  </li>
                ))}
              </ul>
              <button onClick={saveStudyPlan}>Save Study Priority List</button>
              <textarea
                rows="3"
                placeholder="Additional instructions for follow-up quiz generation (optional)"
                value={followupInstructions}
                onChange={(e) => setFollowupInstructions(e.target.value)}
              />
              <button onClick={() => generateQuiz(result.attempt_id)}>Generate Follow-up Quiz from Study Priorities</button>
              <button onClick={requestErrorLesson} disabled={lessonLoading}>{lessonLoading ? 'Generating lesson...' : 'Get LLM Error Analysis + Lesson'}</button>
            </div>
          )}
        </div>

        <aside className="side-panel">
          <h3>Missed questions</h3>
          {missedQuestions.length === 0 ? <div className="small">No missed questions yet.</div> : (
            <ul>
              {missedQuestions.map((mq, i) => <li key={mq.id}>{i + 1}. {mq.prompt}</li>)}
            </ul>
          )}
        </aside>
      </div>


      {showHistoryModal && historyStats && (
        <div className="modal-overlay" onClick={() => setShowHistoryModal(false)}>
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Historical Stats</h3>
              <button className="ghost-btn" onClick={() => setShowHistoryModal(false)}>Close</button>
            </div>
            <div className="small">
              Attempts: {historyStats.global_stats.attempts} · Questions answered: {historyStats.global_stats.questions_answered} · Correct: {historyStats.global_stats.correct_answers} · Accuracy: {historyStats.global_stats.accuracy_percentage}%
            </div>
            <h4>Per category</h4>
            <ul>
              {historyStats.per_category_stats.map((row) => (
                <li key={row.category}>{row.category}: {row.correct}/{row.total} ({row.accuracy_percentage}%)</li>
              ))}
            </ul>
            <h4>Missed questions to review</h4>
            {historyStats.missed_questions.length === 0 ? <div className="small">No missed questions recorded yet.</div> : (
              <ul>
                {historyStats.missed_questions.map((miss) => (
                  <li key={`${miss.question_id}-${miss.selected_option_index}`}>
                    <strong>{miss.quiz_topic}</strong> · {miss.category}<br />
                    {miss.prompt}<br />
                    <span className="small">Your answer: {miss.selected_option_text} | Correct: {miss.correct_option_text}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      )}

      {showLessonModal && (
        <div className="modal-overlay" onClick={() => setShowLessonModal(false)}>
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Error Analysis Lesson</h3>
              <button className="ghost-btn" onClick={() => setShowLessonModal(false)}>Close</button>
            </div>
            <pre className="lesson-content">{errorLesson}</pre>
          </div>
        </div>
      )}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
