const { useEffect, useMemo, useState } = React;

const generationStages = [
  'Analyzing your learning goal',
  'Drafting hard diagnostic questions',
  'Verifying quality and distractors',
  'Finalizing quiz',
];


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
  const [questionCount, setQuestionCount] = useState(20);
  const [loading, setLoading] = useState(false);
  const [loadingStage, setLoadingStage] = useState(0);
  const [error, setError] = useState('');
  const [quiz, setQuiz] = useState(null);
  const [answers, setAnswers] = useState({});
  const [flags, setFlags] = useState({});
  const [idx, setIdx] = useState(0);
  const [result, setResult] = useState(null);
  const [studyTopics, setStudyTopics] = useState([]);

  useEffect(() => {
    if (!loading) return;
    const id = setInterval(() => {
      setLoadingStage((x) => Math.min(x + 1, generationStages.length - 1));
    }, 900);
    return () => clearInterval(id);
  }, [loading]);

  const progressPct = useMemo(() => ((loadingStage + 1) / generationStages.length) * 100, [loadingStage]);

  async function generateQuiz(followupFrom = null) {
    setLoading(true);
    setLoadingStage(0);
    setError('');
    setResult(null);
    setQuiz(null);
    setAnswers({});
    setFlags({});
    try {
      const res = await fetch('/api/quizzes/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          topic,
          learning_goal: learningGoal,
          question_count: questionCount,
          use_web_search: true,
          followup_from_attempt_id: followupFrom,
        }),
      });
      const parsed = await parseApiResponse(res);
      if (!parsed.ok || !parsed.data) throw new Error(parsed.detail || 'Generation failed');
      setQuiz(parsed.data);
      setIdx(0);
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

  if (!quiz) {
    return (
      <div className="shell">
        <div className="container">
          <h1>Quizzinfinity</h1>
          <p className="small">Beautifully adaptive quizzes that start hard, then adjust based on your first-round analysis.</p>
          {error && <div className="error-banner">{error}</div>}
          <input type="text" placeholder="Topic (e.g. Probability Theory)" value={topic} onChange={(e) => setTopic(e.target.value)} />
          <textarea rows="4" placeholder="What exactly do you want to learn? Add any preferences for easier/harder emphasis." value={learningGoal} onChange={(e) => setLearningGoal(e.target.value)} />
          <div className="row">
            <label>Questions:</label>
            <input type="number" min="5" max="30" value={questionCount} onChange={(e) => setQuestionCount(Number(e.target.value))} style={{ width: '100px' }} />
          </div>
          <button disabled={loading} onClick={() => generateQuiz()}>{loading ? 'Generating...' : 'Generate Diagnostic Quiz'}</button>

          {loading && (
            <div className="progress-card">
              <div className="small">{generationStages[loadingStage]}</div>
              <div className="progress-track">
                <div className="progress-fill" style={{ width: `${progressPct}%` }} />
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
  const isCorrect = answered && selected === q.correct_option_index;
  const allAnswered = quiz.questions.every((x) => answers[x.id] !== undefined);

  return (
    <div className="shell">
      <div className="container">
        <h1>{quiz.title}</h1>
        <div className="small">Difficulty: {quiz.difficulty}/10 · {quiz.difficulty_rationale}</div>
        <div className="small">Question {idx + 1} / {quiz.questions.length}</div>
        {error && <div className="error-banner">{error}</div>}
        <div className="question">
          <div className="badge">{q.category}</div>
          <h3>{q.prompt}</h3>
          {q.options.map((opt, i) => (
            <label className={`option ${selected === i ? 'selected' : ''}`} key={i}>
              <input type="radio" checked={selected === i} name={`q-${q.id}`} onChange={() => setAnswers({ ...answers, [q.id]: i })} /> {opt}
            </label>
          ))}

          {answered && (
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
        <button onClick={submitQuiz} disabled={!allAnswered}>Submit Quiz</button>

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

            <h3>Study Priorities (basis for next quiz)</h3>
            <div className="small">Reorder and save. Follow-up prompts include your flagged questions plus this analysis.</div>
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
            <button onClick={() => generateQuiz(result.attempt_id)}>Generate Follow-up Quiz from Study Priorities</button>

            <h3>Prompt Used for Latest Generation</h3>
            <pre className="prompt-box">{quiz.generation_prompt}</pre>
          </div>
        )}
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
