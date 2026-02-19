const { useState } = React;

function App() {
  const [topic, setTopic] = useState("");
  const [learningGoal, setLearningGoal] = useState("");
  const [difficulty, setDifficulty] = useState(7);
  const [questionCount, setQuestionCount] = useState(20);
  const [loading, setLoading] = useState(false);
  const [quiz, setQuiz] = useState(null);
  const [answers, setAnswers] = useState({});
  const [flags, setFlags] = useState({});
  const [idx, setIdx] = useState(0);
  const [result, setResult] = useState(null);
  const [studyTopics, setStudyTopics] = useState([]);

  async function generateQuiz(followupFrom = null) {
    setLoading(true);
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
          difficulty,
          question_count: questionCount,
          use_web_search: true,
          followup_from_attempt_id: followupFrom,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Generation failed');
      setQuiz(data);
      setIdx(0);
    } catch (e) {
      alert(e.message);
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
    const data = await res.json();
    if (!res.ok) return alert(data.detail || 'Submit failed');
    setResult(data);
    setStudyTopics(data.study_topics || []);
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
    const data = await res.json();
    if (!res.ok) return alert(data.detail || 'Failed to save study plan');
    setResult(data);
    setStudyTopics(data.study_topics || []);
    alert('Study priority list saved.');
  }

  function moveTopic(i, dir) {
    const j = i + dir;
    if (j < 0 || j >= studyTopics.length) return;
    const copy = [...studyTopics];
    [copy[i], copy[j]] = [copy[j], copy[i]];
    setStudyTopics(copy.map((t, idx) => ({ ...t, priority: idx + 1 })));
  }

  if (!quiz) {
    return (
      <div className="container">
        <h1>Quizzinfinity</h1>
        <p>Create a hard diagnostic quiz, then review strengths and weaknesses by category.</p>
        <input type="text" placeholder="Topic (e.g. Probability Theory)" value={topic} onChange={(e) => setTopic(e.target.value)} />
        <textarea rows="4" placeholder="Broader learning goal or what you want to study" value={learningGoal} onChange={(e) => setLearningGoal(e.target.value)} />
        <div className="row">
          <label>Difficulty: <strong>{difficulty}</strong></label>
          <input type="range" min="1" max="10" value={difficulty} onChange={(e) => setDifficulty(Number(e.target.value))} />
        </div>
        <div className="row">
          <label>Questions:</label>
          <input type="number" min="5" max="30" value={questionCount} onChange={(e) => setQuestionCount(Number(e.target.value))} style={{width:'100px'}} />
        </div>
        <button disabled={loading} onClick={() => generateQuiz()}>{loading ? 'Generating...' : 'Generate Diagnostic Quiz'}</button>
      </div>
    )
  }

  const q = quiz.questions[idx];
  const selected = answers[q.id];
  const answered = selected !== undefined;
  const isCorrect = answered && selected === q.correct_option_index;
  const allAnswered = quiz.questions.every((x) => answers[x.id] !== undefined);

  return (
    <div className="container">
      <h1>{quiz.title}</h1>
      <div className="small">Question {idx + 1} / {quiz.questions.length}</div>
      <div className="question">
        <div className="badge">{q.category}</div>
        <h3>{q.prompt}</h3>
        {q.options.map((opt, i) => (
          <label className={`option ${selected === i ? 'selected': ''}`} key={i}>
            <input
              type="radio"
              checked={selected === i}
              name={`q-${q.id}`}
              onChange={() => setAnswers({ ...answers, [q.id]: i })}
            /> {opt}
          </label>
        ))}

        {answered && (
          <div className={`result-card ${isCorrect ? '' : 'wrong'}`}>
            <div><strong>{isCorrect ? 'Correct ✅' : 'Incorrect ❌'}</strong></div>
            <div className="small">Correct answer: {q.options[q.correct_option_index]}</div>
            <div className="small">Explanation: {q.explanation}</div>
            <label>
              <input
                type="checkbox"
                checked={!!flags[q.id]}
                onChange={(e) => setFlags({ ...flags, [q.id]: e.target.checked })}
              /> Need to study more
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
          <div className="small">You can reorder this list and save. It includes mistakes + manually flagged items.</div>
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

          <h3>Detailed Review</h3>
          {result.question_results.map((qr) => (
            <div key={qr.question_id} className={`result-card ${qr.is_correct ? '' : 'wrong'}`}>
              <div><strong>{qr.is_correct ? 'Correct' : 'Incorrect'}</strong> — {qr.category}</div>
              <div>{qr.prompt}</div>
              <div className="small">Your answer: {qr.options[qr.selected_option_index]} | Correct answer: {qr.options[qr.correct_option_index]}</div>
              <div className="small">Explanation: {qr.explanation}</div>
              <div className="small">Flagged by you: {qr.flagged_for_review ? 'Yes' : 'No'}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
