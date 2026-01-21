import React, { useEffect, useState, useCallback, type PointerEvent } from 'react'
import type { NodeProps } from 'reactflow'
import { Handle, Position } from 'reactflow'
import ReactMarkdown from 'react-markdown'
import type { CanvasNodeData } from '../types'
import { BaseCard } from './BaseCard'
import { useCanvasStore } from '../store'
import { useSettingsStore } from '../settingsStore'

// 优化Markdown渲染性能，使用memo缓存
const MarkdownContent = React.memo<{ content: string }>(({ content }) => (
  <ReactMarkdown>{content}</ReactMarkdown>
))

export const MarkdownNode = React.memo<NodeProps<CanvasNodeData>>(function MarkdownNode(props) {
  const { id, data } = props
  const [isEditing, setIsEditing] = useState(false)
  const [draft, setDraft] = useState(data.content ?? '# Markdown')
  const [loading, setLoading] = useState(false)
  const [fileError, setFileError] = useState<string | null>(null)
  const theme = useSettingsStore((s) => s.theme)

  const nodes = useCanvasStore((s) => s.nodes)
  const setNodes = useCanvasStore((s) => s.setNodes)

  const commitTitle = useCallback(
    (nextTitle: string) => {
      const nextNodes = nodes.map((n) => (n.id === id ? { ...n, data: { ...n.data, title: nextTitle } } : n))
      setNodes(nextNodes, true)
    },
    [id, nodes, setNodes],
  )

  const onResizePointerDown = useCallback(
    (e: PointerEvent<HTMLDivElement>) => {
      e.preventDefault()
      e.stopPropagation()
      e.currentTarget.setPointerCapture(e.pointerId)
      const { nodes: currentNodes, setNodes: set } = useCanvasStore.getState()
      const me = currentNodes.find((n) => n.id === id)
      if (!me) return
      const startWidth = (me.style?.width as number) ?? 320
      const startHeight = (me.style?.height as number) ?? 200
      const startX = e.clientX
      const startY = e.clientY

      let lastWidth = startWidth
      let lastHeight = startHeight

      const onMove = (moveEv: globalThis.PointerEvent) => {
        const dx = moveEv.clientX - startX
        const dy = moveEv.clientY - startY
        const w = Math.max(160, startWidth + dx)
        const h = Math.max(120, startHeight + dy)
        lastWidth = w
        lastHeight = h
        const updated = useCanvasStore.getState().nodes.map((n) =>
          n.id === id ? { ...n, style: { ...n.style, width: w, height: h } } : n,
        )
        set(updated, false)
      }

      const onUp = (_upEv: globalThis.PointerEvent) => {
        window.removeEventListener('pointermove', onMove as any)
        window.removeEventListener('pointerup', onUp as any)
        const { setNodes: setFinal } = useCanvasStore.getState()
        const updated = useCanvasStore.getState().nodes.map((n) =>
          n.id === id ? { ...n, style: { ...n.style, width: lastWidth, height: lastHeight } } : n,
        )
        setFinal(updated, true)
      }

      window.addEventListener('pointermove', onMove as any)
      window.addEventListener('pointerup', onUp as any)
    },
    [id],
  )

  const startEdit = useCallback(() => {
    setDraft(data.content ?? '# Markdown')
    setIsEditing(true)
  }, [data.content])

  const commitEdit = useCallback(() => {
    const nextNodes = nodes.map((n) =>
      n.id === id
        ? {
            ...n,
            data: {
              ...n.data,
              content: draft,
            },
          }
        : n,
    )
    setNodes(nextNodes, true)
    setIsEditing(false)
  }, [draft, id, nodes, setNodes])

  const filePath = data.filePath

  useEffect(() => {
    let cancelled = false
    async function run() {
      if (!filePath) return
      if (!window.canvasApi?.readTextFile) return
      setFileError(null)
      setLoading(true)
      try {
        const txt = await window.canvasApi.readTextFile(filePath)
        if (cancelled) return
        // only overwrite editor draft if not currently editing
        if (!isEditing) {
          setDraft(txt)
          const nextNodes = useCanvasStore.getState().nodes.map((n) =>
            n.id === id ? { ...n, data: { ...n.data, content: txt } } : n,
          )
          useCanvasStore.getState().setNodes(nextNodes, true)
        }
      } catch {
        if (!cancelled) setFileError('Failed to read .md file')
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    void run()
    return () => {
      cancelled = true
    }
  }, [filePath, id, isEditing])

  const bindMdFile = useCallback(async () => {
    if (!window.canvasApi) return
    const chosen = await window.canvasApi.openFileDialog({
      title: 'Select Markdown file',
      filters: [{ name: 'Markdown', extensions: ['md', 'markdown'] }],
    })
    if (!chosen) return
    const txt = await window.canvasApi.readTextFile(chosen)
    setDraft(txt)
    const nextNodes = useCanvasStore.getState().nodes.map((n) =>
      n.id === id
        ? {
            ...n,
            data: { ...n.data, filePath: chosen, content: txt, title: n.data?.title ?? 'Markdown' },
          }
        : n,
    )
    useCanvasStore.getState().setNodes(nextNodes, true)
  }, [id])

  const saveMdFile = useCallback(async () => {
    if (!window.canvasApi) return
    const currentNodes = useCanvasStore.getState().nodes
    const me = currentNodes.find((n) => n.id === id)
    const fp = (me?.data as any)?.filePath
    if (!fp) {
      setFileError('No file bound. Use "Bind .md" first.')
      return
    }
    const contentToWrite = isEditing ? draft : ((me?.data as any)?.content ?? draft)
    await window.canvasApi.writeTextFile(fp, contentToWrite)
  }, [draft, id, isEditing])

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <Handle type="target" position={Position.Left} id="left-target" style={{ opacity: 0.02, width: 60, height: 60, top: '50%' }} />
      <Handle type="source" position={Position.Left} id="left-source" style={{ top: '50%', width: 14, height: 14 }} />

      <BaseCard title={data.title ?? 'Markdown'} color={data.color} onTitleChange={commitTitle} selected={props.selected}>
        <div style={{ display: 'flex', gap: 8, marginBottom: 8, alignItems: 'center' }}>
          <button onClick={bindMdFile}>Bind .md</button>
          <button onClick={saveMdFile} disabled={!data.filePath}>
            Save to file
          </button>
          {loading ? <span style={{ fontSize: 12, opacity: 0.7 }}>Loading…</span> : null}
        </div>
        {fileError ? <div style={{ color: '#ef4444', marginBottom: 8 }}>{fileError}</div> : null}
        {isEditing ? (
          <textarea
            value={draft}
            autoFocus
            onChange={(e) => setDraft(e.target.value)}
            onBlur={commitEdit}
            style={{
              width: '100%',
              height: '100%',
              resize: 'none',
              border: 'none',
              outline: 'none',
              background: 'transparent',
              color: 'inherit',
              fontFamily: 'monospace',
              fontSize: 13,
              lineHeight: 1.4,
            }}
          />
        ) : (
          <div
            style={{
              width: '100%',
              height: '100%',
              cursor: 'text',
              fontSize: 13,
              lineHeight: 1.4,
            }}
            onDoubleClick={startEdit}
          >
            <MarkdownContent content={data.content ?? '# 双击编辑 Markdown'} />
          </div>
        )}
      </BaseCard>

      <div
        onPointerDown={onResizePointerDown}
        style={{
          position: 'absolute',
          right: 4,
          bottom: 4,
          width: 14,
          height: 14,
          borderRadius: 3,
          background: theme === 'light' ? 'rgba(15,23,42,0.18)' : 'rgba(255,255,255,0.45)',
          border: theme === 'light' ? '1px solid rgba(15,23,42,0.25)' : '1px solid rgba(0,0,0,0.25)',
          cursor: 'se-resize',
          zIndex: 50,
          pointerEvents: 'all',
        }}
      />

      <Handle type="target" position={Position.Right} id="right-target" style={{ opacity: 0.02, width: 60, height: 60, top: '50%' }} />
      <Handle type="source" position={Position.Right} id="right-source" style={{ top: '50%', width: 14, height: 14 }} />
    </div>
  )
})


