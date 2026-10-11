import { useEffect, useState, type FormEvent } from 'react'
import { Activity, ArrowUpRight, Check, CircleAlert, Command, KeyRound, LogOut, Plus, SlidersHorizontal, Users } from 'lucide-react'

type User = { id: string; email: string; role: string; must_change_password: boolean }
type TaskSummary = { id: string; owner_email: string; status: string; created_at: string; updated_at: string; error: { code?: string; message?: string } | null }
type AdminSettings = { manager_model: string; worker_model: string; max_concurrent_agents: number; max_agent_attempts: number; openai_key_configured: boolean; version: number }

async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers)
  if (init.body) headers.set('Content-Type', 'application/json')
  if (init.method && init.method !== 'GET') {
    const token = document.cookie.match(/(?:^|; )agent_csrf=([^;]+)/)?.[1]
    if (token) headers.set('X-CSRF-Token', decodeURIComponent(token))
  }
  const response = await fetch(`/api${path}`, { ...init, headers, credentials: 'include' })
  if (!response.ok) {
    const body = await response.json().catch(() => null)
    throw new Error(body?.detail ?? `Request failed (${response.status})`)
  }
  if (response.status === 204) return undefined as T
  return response.json() as Promise<T>
}

function App() {
  const [user, setUser] = useState<User | null>(null)
  const [apiConnected, setApiConnected] = useState<boolean | null>(null)
  const [tasks, setTasks] = useState<TaskSummary[]>([])
  const [section, setSection] = useState<'operations' | 'accounts' | 'settings'>('operations')
  const [notice, setNotice] = useState('')
  const [error, setError] = useState('')

  const load = async () => setTasks(await api<TaskSummary[]>('/admin/operations/tasks'))
  useEffect(() => {
    api<{ status: string }>('/health')
      .then(() => setApiConnected(true))
      .catch(() => setApiConnected(false))
    api<User>('/auth/me').then(async (current) => {
      if (current.role !== 'admin') throw new Error('Administrator access required')
      setUser(current); await load()
    }).catch(() => undefined)
  }, [])

  const login = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault(); setError('')
    const data = new FormData(event.currentTarget)
    try {
      const result = await api<{ user: User }>('/auth/login', { method: 'POST', body: JSON.stringify({ email: data.get('email'), password: data.get('password') }) })
      if (result.user.role !== 'admin') throw new Error('Administrator access required')
      setUser(result.user); await load()
    } catch (reason) { setError((reason as Error).message) }
  }

  const logout = async () => {
    try { await api('/auth/logout', { method: 'POST' }) } finally { setUser(null); setTasks([]) }
  }

  if (!user) return <main className="admin-login"><div className="admin-login-brand"><span className="admin-mark"><Command size={17} /></span><span>relay<span>.</span><small>CONTROL ROOM</small></span></div><form className="admin-login-form" onSubmit={login}><div className="eyebrow">RESTRICTED ACCESS</div><h1>Administration</h1><p>Sign in with an administrator account.</p><label>Email<input type="email" name="email" autoComplete="username" required /></label><label>Password<input type="password" name="password" autoComplete="current-password" required /></label>{error && <div className="admin-error">{error}</div>}<button className="admin-button">Enter control room <ArrowUpRight size={16} /></button></form><div className="admin-login-stamp">AUTHENTICATED ACCESS ONLY <span>RLY / 01</span></div></main>

  const activeCount = tasks.filter((task) => ['queued', 'planning', 'running', 'waiting_approval'].includes(task.status)).length
  const failedCount = tasks.filter((task) => ['failed', 'needs_attention'].includes(task.status)).length

  return <div className="admin-shell">
    <aside className="admin-rail"><div className="admin-brand"><span className="admin-mark"><Command size={17} /></span><span>relay<span>.</span><small>CONTROL ROOM</small></span></div><div className="admin-caption">OPERATIONS</div><button className={`admin-nav ${section === 'operations' ? 'chosen' : ''}`} onClick={() => setSection('operations')}><Activity size={17} /> System overview</button><button className={`admin-nav ${section === 'accounts' ? 'chosen' : ''}`} onClick={() => setSection('accounts')}><Users size={17} /> User accounts</button><button className={`admin-nav ${section === 'settings' ? 'chosen' : ''}`} onClick={() => setSection('settings')}><SlidersHorizontal size={17} /> Runtime settings</button><div className="admin-rail-bottom"><div className="admin-identity"><div className="admin-avatar">A</div><div><strong>Administrator</strong><span>{user.email}</span></div></div><button className="admin-nav" onClick={logout}><LogOut size={16} /> Sign out</button></div></aside>
    <main className="admin-main"><header className="admin-top"><div><span className="top-kicker">RELAY PLATFORM</span><span className="top-slash">/</span><span>{section === 'operations' ? 'Operations' : section === 'accounts' ? 'Accounts' : 'Settings'}</span></div><div className={`admin-live ${apiConnected === false ? 'is-offline' : ''}`}><i /> {apiConnected === null ? 'Connecting' : apiConnected ? 'API connected' : 'API unavailable'} <button title="Refresh data" onClick={() => load().catch((reason) => setError(reason.message))}><Activity size={15} /></button></div></header>
      {error && <div className="admin-alert"><CircleAlert size={16} />{error}</div>}
      {section === 'operations' ? <section className="admin-content"><div className="admin-heading"><div><div className="eyebrow">LIVE OPERATIONS</div><h1>System overview</h1><p>Operational metadata across user tasks. Conversation content is not shown here.</p></div><div className="admin-date">UPDATED {new Date().toLocaleTimeString('en', { hour: '2-digit', minute: '2-digit' })}</div></div><div className="admin-metrics"><div><span className="metric-label">TOTAL TASKS</span><strong>{String(tasks.length).padStart(2, '0')}</strong><span className="metric-foot">All recorded tasks</span></div><div><span className="metric-label">ACTIVE</span><strong>{String(activeCount).padStart(2, '0')}</strong><span className="metric-foot">Queued or in progress</span></div><div><span className="metric-label">NEEDS ATTENTION</span><strong className={failedCount ? 'metric-warn' : ''}>{String(failedCount).padStart(2, '0')}</strong><span className="metric-foot">Failed or blocked</span></div><button className="metric-accent" onClick={() => setSection('settings')}><KeyRound size={18} /><span>OpenAI API key</span><strong>Manage settings <ArrowUpRight size={14} /></strong></button></div><div className="admin-table-heading"><div><h2>Task activity</h2><span>Recent operational records</span></div><span className="table-total">{tasks.length} records</span></div><div className="admin-table-wrap"><table><thead><tr><th>TASK ID</th><th>OWNER</th><th>STATUS</th><th>LAST UPDATED</th><th>ERROR</th></tr></thead><tbody>{tasks.map((task) => <tr key={task.id}><td className="mono">{task.id.slice(0, 8).toUpperCase()}</td><td>{task.owner_email}</td><td><span className={`admin-status status-${task.status}`}><i />{task.status.replaceAll('_', ' ')}</span></td><td>{new Date(task.updated_at).toLocaleString()}</td><td>{task.error?.code ?? '—'}</td></tr>)}</tbody></table>{tasks.length === 0 && <div className="table-empty">No task activity yet.</div>}</div><div className="privacy-note"><Check size={14} /> User instructions, conversations, and artifacts are excluded from this view.</div></section> : section === 'accounts' ? <Accounts onCreated={async () => { await load(); setNotice('Account created. The user must choose a new password at first sign-in.'); setTimeout(() => setNotice(''), 6000) }} /> : <SettingsPanel />}
      {notice && <div className="admin-toast"><Check size={15} />{notice}</div>}
      <footer className="admin-footer"><span>RELAY CORE <b>0.1.0</b></span><span>SESSION ACTIVE <i /></span></footer>
    </main>
  </div>
}

function Accounts({ onCreated }: { onCreated: () => Promise<void> }) {
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  return <section className="admin-content accounts-content"><div className="admin-heading"><div><div className="eyebrow">ACCESS MANAGEMENT</div><h1>User accounts</h1><p>Create a user account with a temporary password.</p></div></div><div className="account-layout"><div className="account-intro"><div className="account-icon"><Users size={20} /></div><h2>Invite a teammate</h2><p>New users will be required to choose a new password after signing in for the first time.</p><div className="account-policy"><Check size={15} /> Passwords are stored as one-way hashes.</div></div><form className="account-form" onSubmit={async (event) => { event.preventDefault(); const form = event.currentTarget; setBusy(true); setError(''); const data = new FormData(form); try { await api('/admin/users', { method: 'POST', body: JSON.stringify({ email: data.get('email'), initial_password: data.get('password') }) }); form.reset(); await onCreated() } catch (reason) { setError((reason as Error).message) } finally { setBusy(false) } }}><label>Email address<input type="email" name="email" autoComplete="off" placeholder="person@company.com" required /></label><label>Temporary password<input type="password" name="password" minLength={12} autoComplete="new-password" placeholder="At least 12 characters" required /></label>{error && <div className="admin-error">{error}</div>}<button className="admin-button" disabled={busy}><Plus size={16} /> Create account</button></form></div><div className="account-warning"><CircleAlert size={16} /><p>Share temporary passwords through a separate secure channel. This screen does not display them again after submission.</p></div></section>
}

function SettingsPanel() {
  const [settings, setSettings] = useState<AdminSettings | null>(null)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    api<AdminSettings>('/admin/settings')
      .then(setSettings)
      .catch((reason) => setError((reason as Error).message))
  }, [])

  const saveSettings = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const data = new FormData(event.currentTarget)
    setBusy(true); setError(''); setNotice('')
    try {
      const saved = await api<AdminSettings>('/admin/settings', {
        method: 'PUT',
        body: JSON.stringify({
          manager_model: data.get('manager_model'),
          worker_model: data.get('worker_model'),
          max_concurrent_agents: Number(data.get('max_concurrent_agents')),
          max_agent_attempts: Number(data.get('max_agent_attempts')),
        }),
      })
      setSettings(saved); setNotice(`Settings saved · version ${saved.version}`)
    } catch (reason) { setError((reason as Error).message) }
    finally { setBusy(false) }
  }

  const saveApiKey = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const form = event.currentTarget
    const data = new FormData(form)
    setBusy(true); setError(''); setNotice('')
    try {
      await api('/admin/secrets/openai-api-key', {
        method: 'PUT', body: JSON.stringify({ api_key: data.get('api_key') }),
      })
      form.reset()
      setSettings((current) => current ? { ...current, openai_key_configured: true } : current)
      setNotice('OpenAI API key stored in Vault.')
    } catch (reason) { setError((reason as Error).message) }
    finally { setBusy(false) }
  }

  if (!settings) return <section className="admin-content"><div className="admin-heading"><div><div className="eyebrow">CONFIGURATION</div><h1>Runtime settings</h1></div></div>{error || 'Loading settings...'}</section>
  return <section className="admin-content settings-content">
    <div className="admin-heading"><div><div className="eyebrow">CONFIGURATION</div><h1>Runtime settings</h1><p>Updates apply to newly created tasks. Existing tasks keep their saved configuration.</p></div><span className="settings-version">VERSION {settings.version}</span></div>
    {error && <div className="admin-error settings-error">{error}</div>}
    {notice && <div className="settings-notice"><Check size={15} />{notice}</div>}
    <div className="settings-layout">
      <form className="settings-form" onSubmit={saveSettings}>
        <div className="settings-section-title"><SlidersHorizontal size={17} /><div><h2>Agent runtime</h2><span>Model selection and execution limits</span></div></div>
        <label>Manager model ID<input name="manager_model" defaultValue={settings.manager_model} maxLength={200} required /></label>
        <label>Worker model ID<input name="worker_model" defaultValue={settings.worker_model} maxLength={200} required /></label>
        <div className="settings-number-grid"><label>Concurrent agents<input name="max_concurrent_agents" type="number" min={1} max={32} defaultValue={settings.max_concurrent_agents} required /></label><label>Maximum attempts<input name="max_agent_attempts" type="number" min={1} max={3} defaultValue={settings.max_agent_attempts} required /></label></div>
        <button className="admin-button" disabled={busy}><Check size={15} /> Save runtime settings</button>
      </form>
      <section className="secret-panel"><div className="settings-section-title"><KeyRound size={17} /><div><h2>OpenAI API key</h2><span>System-wide secret · Vault KV v2</span></div></div><div className={`secret-state ${settings.openai_key_configured ? 'is-configured' : ''}`}><i />{settings.openai_key_configured ? 'Configured' : 'Not configured'}</div><form onSubmit={saveApiKey}><label>Replace API key<input name="api_key" type="password" autoComplete="new-password" placeholder="Paste a new API key" required /></label><button className="admin-button" disabled={busy}><KeyRound size={15} /> Store key</button></form><div className="secret-foot">The saved value is never returned to this screen.</div></section>
    </div>
  </section>
}

export default App