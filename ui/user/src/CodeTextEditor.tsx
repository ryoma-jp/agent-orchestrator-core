import CodeMirror from '@uiw/react-codemirror'
import type { Extension } from '@codemirror/state'
import { css } from '@codemirror/lang-css'
import { html } from '@codemirror/lang-html'
import { javascript } from '@codemirror/lang-javascript'
import { json } from '@codemirror/lang-json'
import { markdown } from '@codemirror/lang-markdown'
import { python } from '@codemirror/lang-python'
import { sql } from '@codemirror/lang-sql'
import { xml } from '@codemirror/lang-xml'
import { yaml } from '@codemirror/lang-yaml'

type CodeTextEditorProps = {
  filename: string
  value: string
  onChange: (value: string) => void
}

function languageExtensions(filename: string): Extension[] {
  const extension = filename.split('.').at(-1)?.toLowerCase()
  switch (extension) {
    case 'css': return [css().extension]
    case 'html': return [html().extension]
    case 'js': return [javascript().extension]
    case 'jsx': return [javascript({ jsx: true }).extension]
    case 'json': return [json().extension]
    case 'md': return [markdown().extension]
    case 'py': return [python().extension]
    case 'sql': return [sql().extension]
    case 'ts': return [javascript({ typescript: true }).extension]
    case 'tsx': return [javascript({ jsx: true, typescript: true }).extension]
    case 'xml': return [xml().extension]
    case 'yaml':
    case 'yml': return [yaml().extension]
    default: return []
  }
}

export default function CodeTextEditor({
  filename,
  value,
  onChange,
}: CodeTextEditorProps) {
  return <CodeMirror
    value={value}
    width="100%"
    height="100%"
    theme="light"
    extensions={languageExtensions(filename)}
    basicSetup={{
      closeBrackets: true,
      foldGutter: true,
      highlightActiveLine: true,
      lineNumbers: true,
      searchKeymap: true,
    }}
    onChange={onChange}
  />
}