import { Fragment, Suspense, lazy, useEffect, useRef, useState, type FormEvent } from 'react'
import { Tree, type NodeRendererProps } from 'react-arborist'
import {
  Activity, ArrowUpRight, Ban, Check, ChevronDown, Command,
  Copy, Download, FileText, Folder, FolderOpen, KeyRound, LoaderCircle,
  LogOut, MessageSquareText, Plus, Save, Send, Sparkles, Trash2, Upload, X,
} from 'lucide-react'

const CodeTextEditor = lazy(() => import('./CodeTextEditor'))

type User = { id: string; email: string; role: string; must_change_password: boolean }
type ToolBudgetApproval = {
  request_id: string; subtask_id: string; subtask_title: string;
  additional_calls: number; reason: string; plan_version: number
}
type Task = {
  id: string; instruction: string; title: string | null; status: string; created_at: string; updated_at: string;
  result: string | null; error: { code?: string; message?: string } | null;
  completion_report: { plan_version: number; unmet_items: { target_type: string; subtask_title: string | null; item_id: string; description: string; evidence: string; deficiency: string }[] } | null;
  pending_tool_approval: ToolBudgetApproval | null
}
type Message = { id: string; sequence: number; role: 'user' | 'assistant'; content: string; created_at: string }
type ProgressItem = { id: string; conversationSequence: number; content: string }
type WorkspaceFile = { kind: 'file'; id: string; path: string; role: string; source_kind: string; size_bytes: number; updated_at: string }
type WorkspaceDirectory = { kind: 'directory'; path: string }
type WorkspaceEntry = WorkspaceFile | WorkspaceDirectory
type ExplorerNodeData = { id: string; name: string; path: string; kind: WorkspaceEntry['kind']; file?: WorkspaceFile; children?: ExplorerNodeData[]; empty?: boolean }

async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers)
  if (init.body && !headers.has('Content-Type')) headers.set('Content-Type', 'application/json')
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
  needs_attention: '要確認', succeeded: '完了', partially_completed: '準完了', failed: '失敗', cancelled: '停止',
}

function App() {
  const [user, setUser] = useState<User | null>(null)
  const [apiConnected, setApiConnected] = useState<boolean | null>(null)
  const [tasks, setTasks] = useState<Task[]>([])
  const [active, setActive] = useState<Task | null>(null)
  const [detailTab, setDetailTab] = useState<'thread' | 'files'>('thread')
  const [messages, setMessages] = useState<Message[]>([])
  const [progress, setProgress] = useState<ProgressItem[]>([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [showCreate, setShowCreate] = useState(false)
  const [showAccount, setShowAccount] = useState(false)

  const refreshTasks = async () => setTasks(await api<Task[]>('/tasks'))
  const refreshMessages = async (taskId: string) => setMessages(await api<Message[]>(`/tasks/${taskId}/messages`))

  useEffect(() => setDetailTab('thread'), [active?.id])

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
    setProgress([])
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
    const recordProgress = (event: Event) => {
      const message = event as MessageEvent<string>
      let payload: Record<string, unknown>
      try { payload = JSON.parse(message.data) as Record<string, unknown> } catch { return }
      const conversationSequence = payload.conversation_sequence
      if (typeof conversationSequence !== 'number' || !Number.isInteger(conversationSequence)) return
      let content = ''
      if (message.type === 'plan.created') {
        const summary = typeof payload.summary === 'string' ? payload.summary : ''
        const count = typeof payload.subtask_count === 'number' ? payload.subtask_count : 0
        content = summary || `${count}件の作業に分けて計画しました。`
      } else if (message.type === 'subtask.status_changed') {
        const title = typeof payload.title === 'string' ? payload.title : '作業項目'
        const status = payload.status
        const action = status === 'running' ? 'に取り組んでいます' : status === 'reviewing' ? 'の成果物を検証しています' : status === 'revising' ? 'の不足部分を修正しています' : status === 'accepted' ? 'の成果物を受け入れました' : status === 'partially_accepted' ? 'は一部未達です' : status === 'succeeded' ? 'が完了しました' : status === 'blocked' ? 'は追加情報待ちです' : status === 'failed' ? 'で問題が発生しました' : ''
        content = action ? `${title}${action}` : ''
      } else if (message.type === 'agent.progress') {
        content = typeof payload.summary === 'string' ? payload.summary : ''
      } else if (message.type === 'quality.checklists_created') {
        const count = typeof payload.subtask_item_count === 'number' ? payload.subtask_item_count : 0
        content = `成果物チェックリストを作成しました（${count}項目）。`
      } else if (message.type === 'quality.review_completed') {
        const verdict = payload.verdict
        content = verdict === 'accepted' ? '成果物の受け入れチェックに合格しました。' : verdict === 'revision_requested' ? '未達項目に絞って修正を依頼しました。' : verdict === 'partially_accepted' ? '修正上限に達した未達項目を記録しました。' : ''
      } else if (message.type === 'task.status_changed') {
        const status = payload.status
        content = status === 'planning' ? '作業内容を整理しています。' : status === 'running' ? 'エージェントが作業を開始しました。' : status === 'needs_attention' ? '追加の情報が必要です。' : ''
      }
      if (!content || !alive) return
      setProgress((current) => current.some((item) => item.id === message.lastEventId)
        ? current
        : [...current, { id: message.lastEventId, conversationSequence, content }])
    }
    for (const name of ['task.status_changed', 'conversation.message_added', 'task.completed', 'task.partially_completed', 'task.failed']) {
      stream.addEventListener(name, sync)
    }
    for (const name of ['plan.created', 'subtask.status_changed', 'agent.progress', 'quality.checklists_created', 'quality.review_completed', 'task.status_changed']) {
      stream.addEventListener(name, recordProgress)
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

  const decideToolBudgetApproval = async (approved: boolean) => {
    if (!active?.pending_tool_approval) return
    setBusy(true); setError('')
    try {
      const updated = await api<Task>(`/tasks/${active.id}/tool-budget-approval`, {
        method: 'POST',
        body: JSON.stringify({
          request_id: active.pending_tool_approval.request_id,
          approved,
        }),
      })
      setActive(updated)
      await refreshTasks()
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

  const latestUserMessageSequence = messages.filter((message) => message.role === 'user').at(-1)?.sequence
  const renderProgress = (conversationSequence: number) => {
    const items = progress.filter((item) => item.conversationSequence === conversationSequence)
    if (items.length === 0) return null
    const isActiveRun = conversationSequence === latestUserMessageSequence
      && active !== null
      && ['queued', 'planning', 'running'].includes(active.status)
    return <details className="progress-section" open={isActiveRun}>
      <summary className="progress-heading"><span><Activity size={14} /> 途中経過 <small>{items.length}件</small></span><ChevronDown className="progress-chevron" size={14} /></summary>
      <ol>{items.map((item) => <li key={item.id}>{item.content}</li>)}</ol>
    </details>
  }

  return <div className="app-shell">
    <aside className="rail">
      <a className="brand" href="/" aria-label="Relay home"><span className="brand-mark"><Command size={17} /></span><span>relay<span className="brand-dot">.</span></span></a>
      <div className="rail-label">WORKSPACE</div>
      <button className="rail-new-task" onClick={() => setShowCreate(true)}><Plus size={17} /> New task</button>
      <nav className="rail-task-list" aria-label="Tasks">
        {tasks.length === 0 ? <p className="rail-empty">No tasks yet</p> : tasks.map((task) => <button key={task.id} className={`rail-task-row ${active?.id === task.id ? 'selected' : ''}`} title={task.title ?? task.instruction} onClick={() => setActive(task)}>
          <span className={`status-dot status-${task.status}`} /><span className="rail-task-copy"><strong>{task.title ?? task.instruction}</strong><span className="rail-task-meta"><small>{new Date(task.updated_at).toLocaleString('ja-JP', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}</small><span className={`status-pill pill-${task.status}`}>{statusLabel[task.status] ?? task.status}</span></span></span>
        </button>)}
      </nav>
      <div className="rail-bottom">
        <button className="nav-item" onClick={() => setShowAccount(true)}><KeyRound size={17} /> Account & API key</button>
        <div className="account-row"><div className="avatar">{user.email[0]?.toUpperCase()}</div><div className="account-copy"><strong>{user.email.split('@')[0]}</strong><span>{user.email}</span></div><button className="icon-btn" title="Sign out" onClick={logout}><LogOut size={16} /></button></div>
      </div>
    </aside>

    <main className="main-area">
      <header className="topbar"><div className="crumb">Workspace <span>/</span> Tasks <ChevronDown size={14} /></div><div className={`live-label ${apiConnected === false ? 'is-offline' : ''}`}><i />{apiConnected === null ? 'Connecting' : apiConnected ? 'System ready' : 'API unavailable'}</div></header>
      {error && <div className="toast-error" role="alert"><span>{error}</span><button onClick={() => setError('')}><X size={15} /></button></div>}
      <div className="workspace-grid">
        <aside className="detail-pane">
          {active ? <>
            <div className="detail-top"><div><div className="eyebrow">TASK THREAD</div><div className={`status-pill pill-${active.status}`}>{statusLabel[active.status] ?? active.status}</div></div><div className="detail-actions">{['queued', 'planning', 'running', 'waiting_approval', 'needs_attention'].includes(active.status) ? <button className="icon-btn" title="Cancel task" disabled={busy} onClick={cancelTask}><Ban size={16} /></button> : ['succeeded', 'partially_completed', 'failed', 'cancelled'].includes(active.status) ? <button className="icon-btn" title="Delete task" disabled={busy} onClick={deleteTask}><Trash2 size={16} /></button> : null}<button className="icon-btn" title="Close task" onClick={() => setActive(null)}><X size={17} /></button></div></div>
            <h2 className="detail-title">{active.title ?? active.instruction}</h2>
            <div className="detail-tabs" role="tablist" aria-label="Task views">
              <button role="tab" aria-selected={detailTab === 'thread'} className={detailTab === 'thread' ? 'selected' : ''} onClick={() => setDetailTab('thread')}><MessageSquareText size={14} /> Conversation</button>
              <button role="tab" aria-selected={detailTab === 'files'} className={detailTab === 'files' ? 'selected' : ''} onClick={() => setDetailTab('files')}><FolderOpen size={14} /> Files</button>
            </div>
            {active.status === 'waiting_approval' && active.pending_tool_approval && <article className="partial-block approval-block" role="alert">
              <strong>追加調査の許可が必要です</strong>
              <p>{active.pending_tool_approval.subtask_title}で追加の読み取り専用調査が必要です（追加{active.pending_tool_approval.additional_calls}回、累積上限100回）。</p>
              <p>{active.pending_tool_approval.reason}</p>
              <div className="reply-actions">
                <button className="primary-btn" type="button" disabled={busy} onClick={() => decideToolBudgetApproval(true)}><Check size={15} /> 許可する</button>
                <button className="outline-btn" type="button" disabled={busy} onClick={() => decideToolBudgetApproval(false)}><X size={15} /> 拒否して部分結果で続行</button>
              </div>
            </article>}
            {detailTab === 'files' ? <WorkspacePanel taskId={active.id} refreshOnCompletion={['succeeded', 'partially_completed', 'failed'].includes(active.status)} /> : <>
            <div className="thread" aria-live="polite">{messages.map((message) => <Fragment key={message.id}><article className={`message message-${message.role}`}><div className="message-avatar">{message.role === 'user' ? user.email[0].toUpperCase() : <Sparkles size={14} />}</div><div><div className="message-meta">{message.role === 'user' ? 'You' : 'Relay'} <time>{new Date(message.created_at).toLocaleTimeString('ja-JP', { hour: '2-digit', minute: '2-digit' })}</time></div><p>{message.content}</p></div></article>{message.role === 'user' && renderProgress(message.sequence)}</Fragment>)}
              {active.status === 'queued' && <div className="progress-note"><LoaderCircle size={15} /> Waiting for an available worker</div>}
              {active.status === 'needs_attention' && <div className="clarification-note" role="status"><MessageSquareText size={15} /> Reply below to continue this task.</div>}
              {active.status === 'partially_completed' && active.completion_report && <article className="partial-block"><strong>未達項目 · 計画 v{active.completion_report.plan_version}</strong><ul>{active.completion_report.unmet_items.map((item) => <li key={`${item.target_type}-${item.subtask_title}-${item.item_id}`}><b>{item.subtask_title ?? '最終回答'}: {item.description}</b>{item.deficiency && <span>{item.deficiency}</span>}{item.evidence && <small>確認内容: {item.evidence}</small>}</li>)}</ul></article>}
              {active.result && !messages.some((message) => message.role === 'assistant' && message.content === active.result) && <article className="result-block"><div className="result-label"><Check size={14} /> RESULT</div><p>{active.result}</p></article>}
              {active.error && <article className="error-block"><strong>{active.error.code ?? 'Task error'}</strong><p>{active.error.message ?? 'The task could not be completed.'}</p></article>}
            </div>
            {active.status !== 'waiting_approval' && <form className="reply-box" onSubmit={sendMessage}><textarea name="content" placeholder={active.status === 'needs_attention' ? 'Answer the questions above to continue...' : 'Add context or a follow-up...'} rows={3} maxLength={20000} required /><div className="reply-actions"><span><MessageSquareText size={14} /> Messages stay in this task</span><button className="send-btn" title="Send message" disabled={busy}><Send size={16} /></button></div></form>}
            </>}
          </> : <div className="detail-empty"><div className="detail-empty-mark"><Activity size={23} /></div><h3>Work, in motion.</h3><p>Select a task to open its conversation, or start a new one.</p><button className="outline-btn" onClick={() => setShowCreate(true)}><Plus size={16} /> Start a task</button></div>}
        </aside>
      </div>
    </main>

    {showCreate && <div className="modal-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) setShowCreate(false) }}><section className="compose-modal" aria-labelledby="compose-title"><div className="modal-head"><div><div className="eyebrow">NEW THREAD</div><h2 id="compose-title">What are we working on?</h2></div><button className="icon-btn" title="Close" onClick={() => setShowCreate(false)}><X size={18} /></button></div><form onSubmit={createTask}><textarea name="instruction" autoFocus rows={5} maxLength={20000} placeholder="Describe the outcome you need..." required /><div className="modal-footer"><span>Each task has its own conversation and history.</span><button className="primary-btn" disabled={busy}><Plus size={16} /> Create task</button></div></form></section></div>}
    {showAccount && <AccountModal user={user} onClose={() => setShowAccount(false)} />}
  </div>
}

function buildExplorerTree(entries: WorkspaceEntry[]): ExplorerNodeData[] {
  const roots: ExplorerNodeData[] = []
  const folders = new Map<string, ExplorerNodeData>()
  for (const entry of entries) {
    const parts = entry.path.split('/')
    let parent = roots
    let currentPath = ''
    for (const [index, name] of parts.entries()) {
      currentPath = currentPath ? `${currentPath}/${name}` : name
      const isLastPart = index === parts.length - 1
      if (isLastPart && entry.kind === 'file') {
        parent.push({ id: `file:${entry.path}`, name, path: entry.path, kind: 'file', file: entry })
        continue
      }
      let folder = folders.get(currentPath)
      if (!folder) {
        folder = { id: `folder:${currentPath}`, name, path: currentPath, kind: 'directory', children: [] }
        folders.set(currentPath, folder)
        parent.push(folder)
      }
      parent = folder.children ?? []
    }
  }
  const markEmptyDirectories = (nodes: ExplorerNodeData[]) => {
    for (const node of nodes) {
      if (node.kind !== 'directory') continue
      node.empty = node.children?.length === 0
      markEmptyDirectories(node.children ?? [])
    }
  }
  markEmptyDirectories(roots)
  return roots
}

function ExplorerTreeRow({ node, style, dragHandle }: NodeRendererProps<ExplorerNodeData>) {
  return <div
    ref={dragHandle}
    style={style}
    className={`explorer-node ${node.isSelected ? 'selected' : ''}`}
    onClick={(event) => node.handleClick(event)}
    onDoubleClick={() => { if (node.isInternal) node.toggle() }}
    title={node.data.file?.path ?? `${node.data.path}${node.data.empty ? ' (empty directory)' : ''}`}
  >
    {node.data.kind === 'file' ? <FileText size={14} /> : node.isOpen ? <FolderOpen size={14} /> : <Folder size={14} />}
    <span>{node.data.name}</span>{node.data.empty && <small className="explorer-empty-label">empty</small>}
  </div>
}

function WorkspacePanel({ taskId, refreshOnCompletion }: { taskId: string; refreshOnCompletion: boolean }) {
  const [files, setFiles] = useState<WorkspaceEntry[]>([])
  const [selected, setSelected] = useState<WorkspaceFile | null>(null)
  const [content, setContent] = useState('')
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const fileTreeElement = useRef<HTMLDivElement>(null)
  const [treeSize, setTreeSize] = useState({ width: 0, height: 0 })

  const refresh = async () => {
    const items = await api<WorkspaceEntry[]>(`/tasks/${taskId}/workspace/files`)
    setFiles(items)
    setSelected((current) => current ? items.find((entry): entry is WorkspaceFile => entry.kind === 'file' && entry.path === current.path) ?? null : null)
  }

  useEffect(() => {
    let alive = true
    setLoading(true)
    api<WorkspaceEntry[]>(`/tasks/${taskId}/workspace/files`)
      .then((items) => { if (alive) setFiles(items) })
      .catch((reason) => { if (alive) setError((reason as Error).message) })
      .finally(() => { if (alive) setLoading(false) })
    setSelected(null)
    setContent('')
    return () => { alive = false }
  }, [taskId])

  useEffect(() => {
    if (!refreshOnCompletion) return
    let alive = true
    api<WorkspaceEntry[]>(`/tasks/${taskId}/workspace/files`)
      .then((items) => {
        if (!alive) return
        setFiles(items)
        setSelected((current) => current ? items.find((entry): entry is WorkspaceFile => entry.kind === 'file' && entry.path === current.path) ?? null : null)
      })
      .catch((reason) => { if (alive) setError((reason as Error).message) })
    return () => { alive = false }
  }, [taskId, refreshOnCompletion])

  useEffect(() => {
    const element = fileTreeElement.current
    if (!element) return
    const bounds = element.getBoundingClientRect()
    setTreeSize({ width: Math.floor(bounds.width), height: Math.floor(bounds.height) })
    const observer = new ResizeObserver(([entry]) => {
      setTreeSize({
        width: Math.floor(entry.contentRect.width),
        height: Math.floor(entry.contentRect.height),
      })
    })
    observer.observe(element)
    return () => observer.disconnect()
  }, [files.length, loading])

  const fileUrl = (path: string) => `/api/tasks/${taskId}/workspace/files/${path.split('/').map(encodeURIComponent).join('/')}`
  const download = async (file: WorkspaceFile) => {
    const response = await fetch(fileUrl(file.path), { credentials: 'include' })
    if (!response.ok) throw new Error(`Download failed (${response.status})`)
    const blob = await response.blob()
    const link = document.createElement('a')
    link.href = URL.createObjectURL(blob)
    link.download = file.path.split('/').at(-1) ?? 'download'
    link.click()
    URL.revokeObjectURL(link.href)
  }

  const selectFile = async (file: WorkspaceFile) => {
    setSelected(file)
    setError('')
    const extension = file.path.split('.').at(-1)?.toLowerCase() ?? ''
    const editable = ['md', 'txt', 'json', 'py', 'ts', 'tsx', 'js', 'jsx', 'css', 'html', 'xml', 'yaml', 'yml', 'toml', 'sh', 'sql', 'csv', 'log'].includes(extension)
    if (!editable || file.size_bytes > 1_000_000) { setContent(''); return }
    try {
      const response = await fetch(fileUrl(file.path), { credentials: 'include' })
      if (!response.ok) throw new Error(`Read failed (${response.status})`)
      setContent(await response.text())
    } catch (reason) { setError((reason as Error).message) }
  }

  const upload = async (event: FormEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(event.currentTarget.files ?? [])
    event.currentTarget.value = ''
    if (selectedFiles.length === 0) return
    setBusy(true); setError('')
    try {
      let lastUploaded: WorkspaceFile | null = null
      for (const file of selectedFiles) {
        lastUploaded = await api<WorkspaceFile>(`/tasks/${taskId}/workspace/files?filename=${encodeURIComponent(file.name)}`, {
          method: 'POST',
          headers: { 'Content-Type': file.type || 'application/octet-stream' },
          body: file,
        })
      }
      await refresh()
      if (lastUploaded) await selectFile(lastUploaded)
    } catch (reason) { setError((reason as Error).message) }
    finally { setBusy(false) }
  }

  const save = async () => {
    if (!selected) return
    setBusy(true); setError('')
    try {
      const updated = await api<WorkspaceFile>(`/tasks/${taskId}/workspace/files/${selected.path.split('/').map(encodeURIComponent).join('/')}?role=${selected.role}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'text/plain; charset=utf-8' },
        body: content,
      })
      await refresh()
      setSelected(updated)
    } catch (reason) { setError((reason as Error).message) }
    finally { setBusy(false) }
  }

  const remove = async (file: WorkspaceFile) => {
    if (!window.confirm(`Delete ${file.path}?`)) return
    setBusy(true); setError('')
    try {
      await api(`/tasks/${taskId}/workspace/files/${file.path.split('/').map(encodeURIComponent).join('/')}`, { method: 'DELETE' })
      if (selected?.path === file.path) { setSelected(null); setContent('') }
      await refresh()
    } catch (reason) { setError((reason as Error).message) }
    finally { setBusy(false) }
  }

  const textFile = selected && ['md', 'txt', 'json', 'py', 'ts', 'tsx', 'js', 'jsx', 'css', 'html', 'xml', 'yaml', 'yml', 'toml', 'sh', 'sql', 'csv', 'log'].includes(selected.path.split('.').at(-1)?.toLowerCase() ?? '') && selected.size_bytes <= 1_000_000
  const fileCount = files.filter((entry) => entry.kind === 'file').length
  const treeData = buildExplorerTree(files)

  return <section className="workspace-panel" aria-label="Task files">
    <div className="workspace-toolbar"><div><div className="eyebrow">TASK WORKSPACE</div><strong>{fileCount} files</strong></div><label className="icon-btn workspace-upload" title="Upload input files"><Upload size={16} /><input type="file" multiple onChange={upload} disabled={busy} /></label></div>
    {error && <div className="workspace-error" role="alert">{error}</div>}
    {loading ? <div className="workspace-empty">Loading files...</div> : files.length === 0 ? <div className="workspace-empty"><FolderOpen size={21} /><strong>No files yet</strong><span>Upload task inputs or save an artifact here.</span></div> : <div className="workspace-browser">
      <div className="file-list" aria-label="Files" ref={fileTreeElement}>
        {treeSize.width > 0 && treeSize.height > 0 && <Tree<ExplorerNodeData>
          data={treeData}
          width={treeSize.width}
          height={treeSize.height}
          rowHeight={27}
          indent={12}
          disableDrag
          disableDrop
          disableEdit
          onSelect={(nodes) => {
            const file = nodes[0]?.data.file
            if (file) void selectFile(file)
          }}
        >{ExplorerTreeRow}</Tree>}
      </div>
      <div className="file-editor">
        {selected ? <>
          <div className="file-editor-head"><strong title={selected.path}>{selected.path}</strong><div><button className="icon-btn" title="Download file" onClick={() => void download(selected).catch((reason) => setError((reason as Error).message))}><Download size={15} /></button><button className="icon-btn file-delete" title="Delete file" disabled={busy} onClick={() => void remove(selected)}><Trash2 size={14} /></button></div></div>
          {textFile ? <><Suspense fallback={<div className="code-editor-host workspace-empty">Loading editor...</div>}><div className="code-editor-host"><CodeTextEditor
            filename={selected.path}
            value={content}
            onChange={(value: string) => setContent(value)}
          /></div></Suspense><button className="primary-btn file-save" disabled={busy} onClick={save}><Save size={15} /> Save</button></> : <div className="workspace-empty">Preview is unavailable. Download to open this file.</div>}
        </> : <div className="workspace-empty"><FolderOpen size={20} /><span>Select a file to inspect it.</span></div>}
      </div>
    </div>}
  </section>
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