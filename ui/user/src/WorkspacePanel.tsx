import { Suspense, lazy, useEffect, useRef, useState, type FormEvent } from 'react'
import { Tree, type NodeRendererProps } from 'react-arborist'
import {
  Download,
  FileText,
  Folder,
  FolderOpen,
  Save,
  Trash2,
  Upload,
} from 'lucide-react'

import { api } from './api'

const CodeTextEditor = lazy(() => import('./CodeTextEditor'))

type WorkspaceFile = {
  kind: 'file'
  id: string
  path: string
  role: string
  source_kind: string
  size_bytes: number
  updated_at: string
}
type WorkspaceDirectory = { kind: 'directory'; path: string }
type WorkspaceEntry = WorkspaceFile | WorkspaceDirectory
type ExplorerNodeData = {
  id: string
  name: string
  path: string
  kind: WorkspaceEntry['kind']
  file?: WorkspaceFile
  children?: ExplorerNodeData[]
  empty?: boolean
}

const IMAGE_PREVIEW_MAX_BYTES = 20 * 1024 * 1024
const IMAGE_PREVIEW_EXTENSIONS = new Set(['png', 'jpg', 'jpeg', 'webp', 'gif'])
const IMAGE_PREVIEW_MIME_TYPES = new Set([
  'image/png',
  'image/jpeg',
  'image/webp',
  'image/gif',
])

function isPreviewableImage(file: WorkspaceFile) {
  const extension = file.path.split('.').at(-1)?.toLowerCase() ?? ''
  return IMAGE_PREVIEW_EXTENSIONS.has(extension)
}

function formatModifiedAt(value: string) {
  return new Date(value).toLocaleString('ja-JP', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  })
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
        parent.push({
          id: `file:${entry.path}`,
          name,
          path: entry.path,
          kind: 'file',
          file: entry,
        })
        continue
      }
      let folder = folders.get(currentPath)
      if (!folder) {
        folder = {
          id: `folder:${currentPath}`,
          name,
          path: currentPath,
          kind: 'directory',
          children: [],
        }
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
      if (node.empty) {
        delete node.children
        continue
      }
      markEmptyDirectories(node.children ?? [])
    }
  }
  markEmptyDirectories(roots)
  return roots
}

function expandEmptyDirectoryParents(
  nodes: ExplorerNodeData[],
): Record<string, boolean> {
  const openState: Record<string, boolean> = {}
  const ancestors: string[] = []

  const visit = (entries: ExplorerNodeData[]) => {
    for (const node of entries) {
      if (node.empty) {
        for (const ancestor of ancestors) openState[ancestor] = true
      }
      if (!node.children?.length) continue
      ancestors.push(node.id)
      visit(node.children)
      ancestors.pop()
    }
  }

  visit(nodes)
  return openState
}

function ExplorerTreeRow({ node, style, dragHandle }: NodeRendererProps<ExplorerNodeData>) {
  const file = node.data.file
  const updatedAt = file ? new Date(file.updated_at) : null
  return (
    <div
      ref={dragHandle}
      style={style}
      className={`explorer-node ${node.isSelected ? 'selected' : ''}`}
      onClick={(event) => node.handleClick(event)}
      onDoubleClick={() => {
        if (node.isInternal) node.toggle()
      }}
      title={
        file && updatedAt
          ? `${file.path}\nUpdated ${updatedAt.toLocaleString('ja-JP')}`
          : `${node.data.path}${node.data.empty ? ' (empty directory)' : ''}`
      }
    >
      {node.data.kind === 'file' ? (
        <FileText size={14} />
      ) : node.isOpen ? (
        <FolderOpen size={14} />
      ) : (
        <Folder size={14} />
      )}
      <span>{node.data.name}</span>
      {file && (
        <small className="explorer-file-datetime">
          {formatModifiedAt(file.updated_at)}
        </small>
      )}
      {node.data.empty && <small className="explorer-empty-label">empty</small>}
    </div>
  )
}

export default function WorkspacePanel({
  taskId,
  refreshOnCompletion,
}: {
  taskId: string
  refreshOnCompletion: boolean
}) {
  const [files, setFiles] = useState<WorkspaceEntry[]>([])
  const [selected, setSelected] = useState<WorkspaceFile | null>(null)
  const [content, setContent] = useState('')
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null)
  const [imagePreviewLoading, setImagePreviewLoading] = useState(false)
  const [imagePreviewError, setImagePreviewError] = useState('')
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const fileTreeElement = useRef<HTMLDivElement>(null)
  const [treeSize, setTreeSize] = useState({ width: 0, height: 0 })

  const refresh = async () => {
    const items = await api<WorkspaceEntry[]>(`/tasks/${taskId}/workspace/files`)
    setFiles(items)
    setSelected((current) =>
      current
        ? items.find(
            (entry): entry is WorkspaceFile =>
              entry.kind === 'file' && entry.path === current.path,
          ) ?? null
        : null,
    )
  }

  useEffect(() => {
    let alive = true
    setLoading(true)
    api<WorkspaceEntry[]>(`/tasks/${taskId}/workspace/files`)
      .then((items) => {
        if (alive) setFiles(items)
      })
      .catch((reason) => {
        if (alive) setError((reason as Error).message)
      })
      .finally(() => {
        if (alive) setLoading(false)
      })
    setSelected(null)
    setContent('')
    return () => {
      alive = false
    }
  }, [taskId])

  const previewImage = selected && isPreviewableImage(selected) ? selected : null
  const canPreviewImage =
    previewImage !== null && previewImage.size_bytes <= IMAGE_PREVIEW_MAX_BYTES

  useEffect(() => {
    if (!canPreviewImage || !previewImage) {
      setImagePreviewUrl(null)
      setImagePreviewLoading(false)
      setImagePreviewError('')
      return
    }

    let active = true
    let objectUrl: string | null = null
    setImagePreviewUrl(null)
    setImagePreviewLoading(true)
    setImagePreviewError('')
    const encodedPath = previewImage.path
      .split('/')
      .map(encodeURIComponent)
      .join('/')
    fetch(`/api/tasks/${taskId}/workspace/files/${encodedPath}`, {
      credentials: 'include',
    })
      .then((response) => {
        if (!response.ok) throw new Error(`Image read failed (${response.status})`)
        return response.blob()
      })
      .then((blob) => {
        const mediaType = blob.type.toLowerCase().split(';', 1)[0]
        if (!IMAGE_PREVIEW_MIME_TYPES.has(mediaType)) {
          throw new Error('This file is not a supported raster image.')
        }
        objectUrl = URL.createObjectURL(blob)
        if (!active) {
          URL.revokeObjectURL(objectUrl)
          return
        }
        setImagePreviewUrl(objectUrl)
      })
      .catch((reason: Error) => {
        if (active) setImagePreviewError(reason.message)
      })
      .finally(() => {
        if (active) setImagePreviewLoading(false)
      })

    return () => {
      active = false
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [
    taskId,
    canPreviewImage,
    previewImage?.path,
    previewImage?.size_bytes,
    previewImage?.updated_at,
  ])

  useEffect(() => {
    if (!refreshOnCompletion) return
    let alive = true
    api<WorkspaceEntry[]>(`/tasks/${taskId}/workspace/files`)
      .then((items) => {
        if (!alive) return
        setFiles(items)
        setSelected((current) =>
          current
            ? items.find(
                (entry): entry is WorkspaceFile =>
                  entry.kind === 'file' && entry.path === current.path,
              ) ?? null
            : null,
        )
      })
      .catch((reason) => {
        if (alive) setError((reason as Error).message)
      })
    return () => {
      alive = false
    }
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

  const fileUrl = (path: string) =>
    `/api/tasks/${taskId}/workspace/files/${path
      .split('/')
      .map(encodeURIComponent)
      .join('/')}`
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
    const editable = [
      'md', 'txt', 'json', 'py', 'ts', 'tsx', 'js', 'jsx', 'css', 'html',
      'xml', 'yaml', 'yml', 'toml', 'sh', 'sql', 'csv', 'log',
    ].includes(extension)
    if (!editable || file.size_bytes > 1_000_000) {
      setContent('')
      return
    }
    try {
      const response = await fetch(fileUrl(file.path), { credentials: 'include' })
      if (!response.ok) throw new Error(`Read failed (${response.status})`)
      setContent(await response.text())
    } catch (reason) {
      setError((reason as Error).message)
    }
  }

  const upload = async (event: FormEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(event.currentTarget.files ?? [])
    event.currentTarget.value = ''
    if (selectedFiles.length === 0) return
    setBusy(true)
    setError('')
    try {
      let lastUploaded: WorkspaceFile | null = null
      for (const file of selectedFiles) {
        lastUploaded = await api<WorkspaceFile>(
          `/tasks/${taskId}/workspace/files?filename=${encodeURIComponent(file.name)}`,
          {
            method: 'POST',
            headers: { 'Content-Type': file.type || 'application/octet-stream' },
            body: file,
          },
        )
      }
      await refresh()
      if (lastUploaded) await selectFile(lastUploaded)
    } catch (reason) {
      setError((reason as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const save = async () => {
    if (!selected) return
    setBusy(true)
    setError('')
    try {
      const updated = await api<WorkspaceFile>(
        `/tasks/${taskId}/workspace/files/${selected.path
          .split('/')
          .map(encodeURIComponent)
          .join('/')}?role=${selected.role}`,
        {
          method: 'PUT',
          headers: { 'Content-Type': 'text/plain; charset=utf-8' },
          body: content,
        },
      )
      await refresh()
      setSelected(updated)
    } catch (reason) {
      setError((reason as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const remove = async (file: WorkspaceFile) => {
    if (!window.confirm(`Delete ${file.path}?`)) return
    setBusy(true)
    setError('')
    try {
      await api(
        `/tasks/${taskId}/workspace/files/${file.path
          .split('/')
          .map(encodeURIComponent)
          .join('/')}`,
        { method: 'DELETE' },
      )
      if (selected?.path === file.path) {
        setSelected(null)
        setContent('')
      }
      await refresh()
    } catch (reason) {
      setError((reason as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const textFile =
    selected &&
    [
      'md', 'txt', 'json', 'py', 'ts', 'tsx', 'js', 'jsx', 'css', 'html',
      'xml', 'yaml', 'yml', 'toml', 'sh', 'sql', 'csv', 'log',
    ].includes(selected.path.split('.').at(-1)?.toLowerCase() ?? '') &&
    selected.size_bytes <= 1_000_000
  const fileCount = files.filter((entry) => entry.kind === 'file').length
  const treeData = buildExplorerTree(files)
  const initialOpenState = expandEmptyDirectoryParents(treeData)
  const treeKey = `${taskId}:${Object.keys(initialOpenState).sort().join('|')}`

  return (
    <section className="workspace-panel" aria-label="Task files">
      <div className="workspace-toolbar">
        <div>
          <div className="eyebrow">TASK WORKSPACE</div>
          <strong>{fileCount} files</strong>
        </div>
        <label className="icon-btn workspace-upload" title="Upload input files">
          <Upload size={16} />
          <input type="file" multiple onChange={upload} disabled={busy} />
        </label>
      </div>
      {error && <div className="workspace-error" role="alert">{error}</div>}
      {loading ? (
        <div className="workspace-empty">Loading files...</div>
      ) : files.length === 0 ? (
        <div className="workspace-empty">
          <FolderOpen size={21} />
          <strong>No files yet</strong>
          <span>Upload task inputs or save an artifact here.</span>
        </div>
      ) : (
        <div className="workspace-browser">
          <div className="file-list" aria-label="Files" ref={fileTreeElement}>
            {treeSize.width > 0 && treeSize.height > 0 && (
              <Tree<ExplorerNodeData>
                key={treeKey}
                data={treeData}
                initialOpenState={initialOpenState}
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
              >
                {ExplorerTreeRow}
              </Tree>
            )}
          </div>
          <div className="file-editor">
            {selected ? (
              <>
                <div className="file-editor-head">
                  <strong title={selected.path}>{selected.path}</strong>
                  <div>
                    <button
                      className="icon-btn"
                      title="Download file"
                      onClick={() =>
                        void download(selected).catch((reason) =>
                          setError((reason as Error).message),
                        )
                      }
                    >
                      <Download size={15} />
                    </button>
                    <button
                      className="icon-btn file-delete"
                      title="Delete file"
                      disabled={busy}
                      onClick={() => void remove(selected)}
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                </div>
                {previewImage ? (
                  canPreviewImage ? (
                    <div
                      className={`image-preview ${imagePreviewLoading ? 'is-loading' : ''}`}
                      aria-busy={imagePreviewLoading}
                    >
                      {imagePreviewUrl && (
                        <img
                          src={imagePreviewUrl}
                          alt={selected.path}
                          onLoad={() => setImagePreviewLoading(false)}
                          onError={() => {
                            setImagePreviewLoading(false)
                            setImagePreviewError('The image could not be displayed.')
                            setImagePreviewUrl(null)
                          }}
                        />
                      )}
                      {imagePreviewLoading && <span>Loading image...</span>}
                      {imagePreviewError && (
                        <span role="alert">{imagePreviewError}</span>
                      )}
                    </div>
                  ) : (
                    <div className="workspace-empty">
                      Image preview is limited to 20 MiB. Download to open this file.
                    </div>
                  )
                ) : textFile ? (
                  <>
                    <Suspense
                      fallback={
                        <div className="code-editor-host workspace-empty">
                          Loading editor...
                        </div>
                      }
                    >
                      <div className="code-editor-host">
                        <CodeTextEditor
                          filename={selected.path}
                          value={content}
                          onChange={(value: string) => setContent(value)}
                        />
                      </div>
                    </Suspense>
                    <button
                      className="primary-btn file-save"
                      disabled={busy}
                      onClick={save}
                    >
                      <Save size={15} /> Save
                    </button>
                  </>
                ) : (
                  <div className="workspace-empty">
                    Preview is unavailable. Download to open this file.
                  </div>
                )}
              </>
            ) : (
              <div className="workspace-empty">
                <FolderOpen size={20} />
                <span>Select a file to inspect it.</span>
              </div>
            )}
          </div>
        </div>
      )}
    </section>
  )
}