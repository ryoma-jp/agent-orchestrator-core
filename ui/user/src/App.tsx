import { useEffect, useState, type FormEvent } from 'react'
import {
  Activity, ArrowUpRight, Ban, Check, ChevronDown, Clock3, Command,
  Copy, KeyRound, LoaderCircle, LogOut, MessageSquareText, Plus, Send,
  Sparkles, Trash2, X,
} from 'lucide-react'

type User = { id: string; email: string; role: string; must_change_password: boolean }
type Task = {
  id: string; instruction: string; status: string; created_at: string; updated_at: string;
  result: string | null; error: { code?: string; message?: string } | null
}
type Message = { id: string; sequence: number; role: 'user' | 'assistant'; content: string; created_at: string }

async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers)
  if (init.body) headers.set('Content-Type', 'application/json')
  if (init.method && init.method !== 'GET') {
    const csrf = document.cookie.match(/(?:^|; )agent_csrf=([^;]+)/)?.[1]
    if (csrf) headers.set('X-CSRF-Token', decodeURIComponent(csrf))
  }
  const response = await fetch(`/api${path}`, { ...init, headers, credentials: 'include' })
  if (!response.ok) {
    const body = await response.json().catch(() => null)
    throw new Error(body?.detail ?? `Request failed (${response.status})`)
  }
  if (response.status === 204) return undefined as T
  return response.json() as Promise<T>
}

const statusLabel: Record<string, string> = {
  queued: '待機中', planning: '計画中', running: '実行中', waiting_approval: '承認待ち',
  needs_attention: '要確認', succeeded: '完了', failed: '失敗', cancelled: '停止',
}

function App() {
  const [user, setUser] = useState<User | null>(null)
  const [apiConnected, setApiConnected] = useState<boolean | null>(null)
  const [tasks, setTasks] = useState<Task[]>([])
  const [active, setActive] = useState<Task | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [showCreate, setShowCreate] = useState(false)
  const [showAccount, setShowAccount] = useState(false)

  const refreshTasks = async () => setTasks(await api<Task[]>('/tasks'))
  const refreshMessages = async (taskId: string) => setMessages(await api<Message[]>(`/tasks/${taskId}/messages`))

  useEffect(() => {
    api<{ status: string }>('/health')
      .then(() => setApiConnected(true))
      .catch(() => setApiConnected(false))
    api<User>('/auth/me').then(async (value) => {
      setUser(value)
      if (!value.must_change_password) await refreshTasks()
    }).catch(() => undefined)
  }, [])

  useEffect(() => {
    if (!active || !user) return
    let alive = true
    const taskId = active.id
    api<Message[]>(`/tasks/${taskId}/messages`)
      .then((value) => { if (alive) setMessages(value) })
      .catch((reason) => { if (alive) setError(reason.message) })
    const stream = new EventSource(`/api/tasks/${taskId}/events`, { withCredentials: true })
    const sync = async () => {
      if (!alive) return
      try {
        const [task, history, updatedTasks] = await Promise.all([
          api<Task>(`/tasks/${taskId}`),
          api<Message[]>(`/tasks/${taskId}/messages`),
          api<Task[]>('/tasks'),
        ])
        if (alive) { setActive(task); setMessages(history); setTasks(updatedTasks) }
      } catch { /* A later event or reconnect will retry the read. */ }
    }
    for (const name of ['task.status_changed', 'conversation.message_added', 'agent.progress', 'tool.completed', 'task.completed', 'task.failed']) {
      stream.addEventListener(name, sync)
    }
    return () => { alive = false; stream.close() }
  }, [active?.id, user])

  const signIn = async (email: string, password: string) => {
    setBusy(true); setError('')
    try {
      const result = await api<{ user: User; csrf_token: string }>('/auth/login', {
        method: 'POST', body: JSON.stringify({ email, password }),
      })
      setUser(result.user)
      if (!result.user.must_change_password) await refreshTasks()
    } catch (reason) { setError((reason as Error).message) }
    finally { setBusy(false) }
  }

  const createTask = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const form = new FormData(event.currentTarget)
    const instruction = String(form.get('instruction') ?? '').trim()
    if (!instruction) return
    setBusy(true); setError('')
    try {
      const task = await api<Task>('/tasks', { method: 'POST', body: JSON.stringify({ instruction }) })
      await refreshTasks(); setActive(task); setShowCreate(false)
    } catch (reason) { setError((reason as Error).message) }
    finally { setBusy(false) }
  }

  const sendMessage = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (!active) return
    const form = event.currentTarget
    const content = String(new FormData(form).get('content') ?? '').trim()
    if (!content) return
    setBusy(true); setError('')
    try {
      await api<Message>(`/tasks/${active.id}/messages`, { method: 'POST', body: JSON.stringify({ content }) })
      form.reset(); await refreshMessages(active.id); await refreshTasks()
    } catch (reason) { setError((reason as Error).message) }
    finally { setBusy(false) }
  }

  const cancelTask = async () => {
    if (!active || !window.confirm('Stop this task? The current run will be cancelled.')) return
    setBusy(true); setError('')
    try {
      const cancelled = await api<Task>(`/tasks/${active.id}/cancel`, { method: 'POST' })
      setActive(cancelled); await refreshTasks()
    } catch (reason) { setError((reason as Error).message) }
    finally { setBusy(false) }
  }

  const deleteTask = async () => {
    if (!active || !window.confirm('Permanently delete this task and its history?')) return
    setBusy(true); setError('')
    try {
      await api(`/tasks/${active.id}`, { method: 'DELETE' })
      setActive(null); setMessages([]); await refreshTasks()
    } catch (reason) { setError((reason as Error).message) }
    finally { setBusy(false) }
  }

  const logout = async () => {
    try { await api('/auth/logout', { method: 'POST' }) } finally {
      setUser(null); setTasks([]); setActive(null); setMessages([])
    }
  }

  if (!user) return <Login onSubmit={signIn} busy={busy} error={error} />
  if (user.must_change_password) return <PasswordGate onDone={async () => {
    const nextUser = await api<User>('/auth/me'); setUser(nextUser); await refreshTasks()
  }} />

  const counts = tasks.reduce<Record<string, number>>((acc, task) => {
    acc[task.status] = (acc[task.status] ?? 0) + 1; return acc
  }, {})

  return <div className="app-shell">
    <aside className="rail">
      <a className="brand" href="/" aria-label="Relay home"><span className="brand-mark"><Command size={17} /></span><span>relay<span className="brand-dot">.</span></span></a>
      <div className="rail-label">WORKSPACE</div>
      <button className="nav-item nav-active"><Activity size={17} /> Your tasks <span className="nav-count">{tasks.length}</span></button>
      <button className="nav-item" onClick={() => setShowCreate(true)}><Plus size={17} /> New task</button>
      <div className="rail-bottom">
        <button className="nav-item" onClick={() => setShowAccount(true)}><KeyRound size={17} /> Account & API key</button>
        <div className="account-row"><div className="avatar">{user.email[0]?.toUpperCase()}</div><div className="account-copy"><strong>{user.email.split('@')[0]}</strong><span>{user.email}</span></div><button className="icon-btn" title="Sign out" onClick={logout}><LogOut size={16} /></button></div>
      </div>
    </aside>

    <main className="main-area">
      <header className="topbar"><div className="crumb">Workspace <span>/</span> Tasks <ChevronDown size={14} /></div><div className={`live-label ${apiConnected === false ? 'is-offline' : ''}`}><i />{apiConnected === null ? 'Connecting' : apiConnected ? 'System ready' : 'API unavailable'}</div></header>
      {error && <div className="toast-error" role="alert"><span>{error}</span><button onClick={() => setError('')}><X size={15} /></button></div>}
      <div className="workspace-grid">
        <section className="task-pane">
          <div className="page-heading"><div><div className="eyebrow">PERSONAL WORKSPACE</div><h1>Your tasks</h1><p>Keep every thread of work moving.</p></div><button className="primary-btn" onClick={() => setShowCreate(true)}><Plus size={17} /> New task</button></div>
          <div className="stats-strip"><div><span className="stat-number">{tasks.length.toString().padStart(2, '0')}</span><span className="stat-caption">TOTAL TASKS</span></div><div><span className="stat-number">{(counts.running ?? 0) + (counts.planning ?? 0)}</span><span className="stat-caption">IN PROGRESS</span></div><div><span className="stat-number">{counts.succeeded ?? 0}</span><span className="stat-caption">COMPLETED</span></div></div>
          <div className="section-heading"><h2>Recent work</h2><span>{tasks.length} items</span></div>
          <div className="task-list">
            {tasks.length === 0 ? <div className="empty-state"><div className="empty-icon"><Sparkles size={22} /></div><h3>A clear desk.</h3><p>Start with a question, a piece of research, or a task you want to move forward.</p><button className="text-btn" onClick={() => setShowCreate(true)}>Create your first task <ArrowUpRight size={15} /></button></div> : tasks.map((task) => <button key={task.id} className={`task-row ${active?.id === task.id ? 'selected' : ''}`} onClick={() => setActive(task)}>
              <span className={`status-dot status-${task.status}`} /><span className="task-row-copy"><strong>{task.instruction}</strong><small><Clock3 size={12} /> {new Date(task.updated_at).toLocaleString('ja-JP', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}</small></span><span className={`status-pill pill-${task.status}`}>{statusLabel[task.status] ?? task.status}</span><ArrowUpRight className="row-arrow" size={16} />
            </button>)}
          </div>
        </section>

        <aside className="detail-pane">
          {active ? <>
            <div className="detail-top"><div><div className="eyebrow">TASK THREAD</div><div className={`status-pill pill-${active.status}`}>{statusLabel[active.status] ?? active.status}</div></div><div className="detail-actions">{['queued', 'planning', 'running', 'waiting_approval'].includes(active.status) ? <button className="icon-btn" title="Cancel task" disabled={busy} onClick={cancelTask}><Ban size={16} /></button> : ['succeeded', 'failed', 'cancelled'].includes(active.status) ? <button className="icon-btn" title="Delete task" disabled={busy} onClick={deleteTask}><Trash2 size={16} /></button> : null}<button className="icon-btn" title="Close task" onClick={() => setActive(null)}><X size={17} /></button></div></div>
            <h2 className="detail-title">{active.instruction}</h2>
            <div className="thread" aria-live="polite">{messages.map((message) => <article className={`message message-${message.role}`} key={message.id}><div className="message-avatar">{message.role === 'user' ? user.email[0].toUpperCase() : <Sparkles size={14} />}</div><div><div className="message-meta">{message.role === 'user' ? 'You' : 'Relay'} <time>{new Date(message.created_at).toLocaleTimeString('ja-JP', { hour: '2-digit', minute: '2-digit' })}</time></div><p>{message.content}</p></div></article>)}
              {active.status === 'queued' && <div className="progress-note"><LoaderCircle size={15} /> Waiting for an available worker</div>}
              {active.result && <article className="result-block"><div className="result-label"><Check size={14} /> RESULT</div><p>{active.result}</p></article>}
              {active.error && <article className="error-block"><strong>{active.error.code ?? 'Task error'}</strong><p>{active.error.message ?? 'The task could not be completed.'}</p></article>}
            </div>
            <form className="reply-box" onSubmit={sendMessage}><textarea name="content" placeholder="Add context or a follow-up..." rows={3} maxLength={20000} required /><div className="reply-actions"><span><MessageSquareText size={14} /> Messages stay in this task</span><button className="send-btn" title="Send message" disabled={busy}><Send size={16} /></button></div></form>
          </> : <div className="detail-empty"><div className="detail-empty-mark"><Activity size={23} /></div><h3>Work, in motion.</h3><p>Select a task to open its conversation, or start a new one.</p><button className="outline-btn" onClick={() => setShowCreate(true)}><Plus size={16} /> Start a task</button></div>}
        </aside>
      </div>
    </main>

    {showCreate && <div className="modal-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) setShowCreate(false) }}><section className="compose-modal" aria-labelledby="compose-title"><div className="modal-head"><div><div className="eyebrow">NEW THREAD</div><h2 id="compose-title">What are we working on?</h2></div><button className="icon-btn" title="Close" onClick={() => setShowCreate(false)}><X size={18} /></button></div><form onSubmit={createTask}><textarea name="instruction" autoFocus rows={5} maxLength={20000} placeholder="Describe the outcome you need..." required /><div className="modal-footer"><span>Each task has its own conversation and history.</span><button className="primary-btn" disabled={busy}><Plus size={16} /> Create task</button></div></form></section></div>}
    {showAccount && <AccountModal user={user} onClose={() => setShowAccount(false)} />}
  </div>
}

function Login({ onSubmit, busy, error }: { onSubmit: (email: string, password: string) => void; busy: boolean; error: string }) {
  return <main className="login-screen"><div className="login-aside"><a className="brand" href="/"><span className="brand-mark"><Command size={17} /></span><span>relay<span className="brand-dot">.</span></span></a><div className="login-manifesto"><div className="eyebrow">A WORKSPACE FOR COMPLEX WORK</div><h1>Make room<br />for the <em>hard</em><br />questions.</h1><div className="manifesto-rule" /><p>One place for research, decisions, and the threads that connect them.</p></div><div className="login-foot">PRIVATE WORKSPACE <span>01 / 01</span></div></div><section className="login-form-wrap"><form className="login-form" onSubmit={(event) => { event.preventDefault(); const data = new FormData(event.currentTarget); onSubmit(String(data.get('email')), String(data.get('password'))) }}><div className="eyebrow">WELCOME BACK</div><h2>Sign in to Relay</h2><p className="login-subtitle">Your work is right where you left it.</p><label>Email address<input name="email" type="email" autoComplete="username" placeholder="you@company.com" required /></label><label>Password<input name="password" type="password" autoComplete="current-password" placeholder="Enter your password" required /></label>{error && <div className="form-error" role="alert">{error}</div>}<button className="primary-btn login-submit" disabled={busy}>{busy ? 'Signing in...' : 'Continue'} <ArrowUpRight size={16} /></button><div className="login-secure"><span /> Encrypted session · Sign-in activity is private</div></form></section></main>
}

function PasswordGate({ onDone }: { onDone: () => Promise<void> }) {
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  return (
    <main className="gate-screen">
      <form
        className="gate-form"
        onSubmit={async (event) => {
          event.preventDefault()
          const data = new FormData(event.currentTarget)
          setBusy(true)
          setError('')
          try {
            await api('/account/password', {
              method: 'POST',
              body: JSON.stringify({
                current_password: data.get('current'),
                new_password: data.get('next'),
              }),
            })
            await onDone()
          } catch (reason) {
            setError((reason as Error).message)
          } finally {
            setBusy(false)
          }
        }}
      >
        <div className="eyebrow">ACCOUNT SECURITY</div>
        <h1>Choose a new password</h1>
        <p>Your administrator set a temporary password. Replace it to continue.</p>
        <label>Temporary password<input name="current" type="password" required /></label>
        <label>New password<input name="next" type="password" minLength={12} required /></label>
        {error && <div className="form-error">{error}</div>}
        <button className="primary-btn" disabled={busy}>Update password <ArrowUpRight size={16} /></button>
      </form>
    </main>
  )
}

function AccountModal({ user, onClose }: { user: User; onClose: () => void }) {
  const [confirmRotation, setConfirmRotation] = useState(false)
  const [issuedKey, setIssuedKey] = useState('')
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [busy, setBusy] = useState(false)

  const rotateKey = async () => {
    setBusy(true); setError(''); setNotice('')
    try {
      const result = await api<{ access_key: string }>('/account/access-key', { method: 'POST' })
      setIssuedKey(result.access_key)
      setConfirmRotation(false)
    } catch (reason) { setError((reason as Error).message) }
    finally { setBusy(false) }
  }

  const changePassword = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const form = event.currentTarget
    const data = new FormData(form)
    setBusy(true); setError(''); setNotice('')
    try {
      await api('/account/password', {
        method: 'POST',
        body: JSON.stringify({ current_password: data.get('current'), new_password: data.get('next') }),
      })
      form.reset(); setNotice('Password updated.')
    } catch (reason) { setError((reason as Error).message) }
    finally { setBusy(false) }
  }

  return <div className="modal-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose() }}>
    <section className="compose-modal account-modal" aria-labelledby="account-title">
      <div className="modal-head"><div><div className="eyebrow">ACCOUNT</div><h2 id="account-title">{user.email}</h2></div><button className="icon-btn" title="Close account settings" onClick={onClose}><X size={18} /></button></div>
      <div className="account-modal-section"><div className="account-modal-title"><KeyRound size={16} /><h3>REST access key</h3></div><p>One active key per account. A replacement immediately invalidates the previous key.</p>
        {issuedKey ? <div className="issued-key"><code>{issuedKey}</code><button className="icon-btn" title="Copy access key" onClick={async () => { try { await navigator.clipboard.writeText(issuedKey); setNotice('Access key copied.') } catch { setError('Clipboard access is unavailable.') } }}><Copy size={16} /></button></div> : <button className="outline-btn" onClick={() => setConfirmRotation(true)}><KeyRound size={15} /> Generate or rotate key</button>}
        {confirmRotation && <div className="key-confirm"><span>This replaces any active key. The new value will only be shown once.</span><div><button className="text-btn" onClick={() => setConfirmRotation(false)}>Cancel</button><button className="primary-btn" disabled={busy} onClick={rotateKey}>Confirm</button></div></div>}
      </div>
      <form className="account-modal-section password-form" onSubmit={changePassword}><div className="account-modal-title"><KeyRound size={16} /><h3>Change password</h3></div><label>Current password<input name="current" type="password" autoComplete="current-password" required /></label><label>New password<input name="next" type="password" minLength={12} autoComplete="new-password" required /></label><button className="primary-btn" disabled={busy}>Update password <ArrowUpRight size={15} /></button></form>
      {error && <div className="form-error">{error}</div>}{notice && <div className="account-notice"><Check size={14} />{notice}</div>}
    </section>
  </div>
}

export default App