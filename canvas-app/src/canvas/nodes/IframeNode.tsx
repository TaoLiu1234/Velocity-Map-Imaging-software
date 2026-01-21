import React, { useEffect, useMemo, useState, useCallback, type PointerEvent } from 'react'
import type { NodeProps } from 'reactflow'
import { Handle, Position } from 'reactflow'
import type { CanvasNodeData } from '../types'
import { BaseCard } from './BaseCard'
import { useCanvasStore } from '../store'
import { useSettingsStore } from '../settingsStore'

export const IframeNode = React.memo<NodeProps<CanvasNodeData>>(function IframeNode(props) {
  const { id, data } = props
  const theme = useSettingsStore((s) => s.theme)
  const [dataUrl, setDataUrl] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [urlDraft, setUrlDraft] = useState(data.url ?? '')

  const nodes = useCanvasStore((s) => s.nodes)
  const setNodes = useCanvasStore((s) => s.setNodes)

  const commitTitle = useCallback(
    (nextTitle: string) => {
      const nextNodes = nodes.map((n) => (n.id === id ? { ...n, data: { ...n.data, title: nextTitle } } : n))
      setNodes(nextNodes, true)
    },
    [id, nodes, setNodes],
  )

  const filePath = data.filePath
  const url = data.url

  useEffect(() => {
    setUrlDraft(url ?? '')
  }, [url])

  useEffect(() => {
    let cancelled = false
    async function run() {
      setError(null)
      setDataUrl(null)
      if (!filePath) return
      if (!window.canvasApi?.readFileAsDataUrl) {
        setError('canvasApi.readFileAsDataUrl is unavailable')
        return
      }
      try {
        const u = await window.canvasApi.readFileAsDataUrl(filePath)
        if (!cancelled) setDataUrl(u)
      } catch {
        if (!cancelled) setError('Failed to load local html')
      }
    }
    void run()
    return () => {
      cancelled = true
    }
  }, [filePath])

  const pickHtml = useCallback(async () => {
    if (!window.canvasApi) return
    const chosen = await window.canvasApi.openFileDialog({
      title: 'Select local HTML',
      filters: [{ name: 'HTML', extensions: ['html', 'htm'] }],
    })
    if (!chosen) return
    const nextNodes = nodes.map((n) =>
      n.id === id
        ? {
            ...n,
            data: {
              ...n.data,
              filePath: chosen,
              url: undefined,
              title: n.data?.title ?? 'Iframe',
            },
          }
        : n,
    )
    setNodes(nextNodes, true)
  }, [id, nodes, setNodes])

  const commitUrl = useCallback(() => {
    const trimmed = urlDraft.trim()
    if (!trimmed) {
      const nextNodes = nodes.map((n) => (n.id === id ? { ...n, data: { ...n.data, url: undefined } } : n))
      setNodes(nextNodes, true)
      return
    }
    const nextNodes = nodes.map((n) =>
      n.id === id ? { ...n, data: { ...n.data, url: trimmed, filePath: undefined } } : n,
    )
    setNodes(nextNodes, true)
  }, [id, nodes, setNodes, urlDraft])

  const onResizePointerDown = useCallback(
    (e: PointerEvent<HTMLDivElement>) => {
      e.preventDefault()
      e.stopPropagation()
      e.currentTarget.setPointerCapture(e.pointerId)

      const { nodes: currentNodes, setNodes: set } = useCanvasStore.getState()
      const me = currentNodes.find((n) => n.id === id)
      if (!me) return
      const startWidth = (me.style?.width as number) ?? 560
      const startHeight = (me.style?.height as number) ?? 420
      const startX = e.clientX
      const startY = e.clientY

      let lastWidth = startWidth
      let lastHeight = startHeight

      const onMove = (moveEv: globalThis.PointerEvent) => {
        const dx = moveEv.clientX - startX
        const dy = moveEv.clientY - startY
        const w = Math.max(260, startWidth + dx)
        const h = Math.max(220, startHeight + dy)
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

  const title = useMemo(() => data.title ?? 'Iframe', [data.title])

  const iframeSrc = url ? url : dataUrl

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <Handle type="target" position={Position.Left} id="left-target" style={{ opacity: 0.02, width: 60, height: 60, top: '50%' }} />
      <Handle type="source" position={Position.Left} id="left-source" style={{ top: '50%', width: 14, height: 14 }} />

      <BaseCard title={title} color={data.color} onTitleChange={commitTitle} selected={props.selected}>
        <div style={{ display: 'flex', gap: 8, marginBottom: 8, alignItems: 'center' }}>
          <button onClick={pickHtml}>Choose local HTML…</button>
          <input
            value={urlDraft}
            onChange={(e) => setUrlDraft(e.target.value)}
            onBlur={commitUrl}
            placeholder="https://example.com 或 http://localhost:xxxx"
            style={{
              flex: 1,
              minWidth: 220,
              background: 'rgba(255,255,255,0.06)',
              border: '1px solid rgba(255,255,255,0.12)',
              borderRadius: 8,
              color: 'rgba(255,255,255,0.9)',
              padding: '6px 8px',
              fontSize: 12,
              outline: 'none',
            }}
          />
        </div>

        {error ? <div style={{ color: '#ef4444', marginBottom: 8 }}>{error}</div> : null}

        {iframeSrc ? (
          <iframe
            src={iframeSrc}
            style={{ width: '100%', height: '100%', border: 'none', borderRadius: 8, background: 'rgba(255,255,255,0.03)' }}
          />
        ) : (
          <div style={{ opacity: 0.6 }}>Choose a local HTML file or enter a localhost URL</div>
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


