import { useEffect, useMemo, useState, useCallback, type PointerEvent } from 'react'
import type { NodeProps } from 'reactflow'
import { Handle, Position } from 'reactflow'
import type { CanvasNodeData } from '../types'
import { BaseCard } from './BaseCard'
import { useCanvasStore } from '../store'
import { useSettingsStore } from '../settingsStore'

export function ImageNode(props: NodeProps<CanvasNodeData>) {
  const { id, data } = props
  const theme = useSettingsStore((s) => s.theme)
  const [dataUrl, setDataUrl] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

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
        const url = await window.canvasApi.readFileAsDataUrl(filePath)
        if (!cancelled) setDataUrl(url)
      } catch (e) {
        if (!cancelled) setError('Failed to load image')
      }
    }
    void run()
    return () => {
      cancelled = true
    }
  }, [filePath])

  const pickImage = useCallback(async () => {
    if (!window.canvasApi) return
    const chosen = await window.canvasApi.openFileDialog({
      title: 'Select image',
      filters: [{ name: 'Images', extensions: ['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp', 'svg'] }],
    })
    if (!chosen) return
    const nextNodes = nodes.map((n) =>
      n.id === id
        ? {
            ...n,
            data: {
              ...n.data,
              filePath: chosen,
              title: n.data?.title ?? 'Image',
            },
          }
        : n,
    )
    setNodes(nextNodes, true)
  }, [id, nodes, setNodes])

  const onResizePointerDown = useCallback(
    (e: PointerEvent<HTMLDivElement>) => {
      e.preventDefault()
      e.stopPropagation()
      e.currentTarget.setPointerCapture(e.pointerId)

      const { nodes: currentNodes, setNodes: set } = useCanvasStore.getState()
      const me = currentNodes.find((n) => n.id === id)
      if (!me) return
      const startWidth = (me.style?.width as number) ?? 420
      const startHeight = (me.style?.height as number) ?? 280
      const startX = e.clientX
      const startY = e.clientY

      let lastWidth = startWidth
      let lastHeight = startHeight

      const onMove = (moveEv: globalThis.PointerEvent) => {
        const dx = moveEv.clientX - startX
        const dy = moveEv.clientY - startY
        const w = Math.max(220, startWidth + dx)
        const h = Math.max(180, startHeight + dy)
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

  const title = useMemo(() => data.title ?? 'Image', [data.title])

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <Handle type="target" position={Position.Left} id="left-target" style={{ opacity: 0.02, width: 36, height: 36, top: '50%' }} />
      <Handle type="source" position={Position.Left} id="left-source" style={{ top: '50%', width: 14, height: 14 }} />

      <BaseCard title={title} color={data.color} onTitleChange={commitTitle} selected={props.selected}>
        <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
          <button onClick={pickImage}>Choose image…</button>
          {filePath ? (
            <div style={{ fontSize: 12, opacity: 0.7, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {filePath}
            </div>
          ) : null}
        </div>
        <div style={{ width: '100%', height: '100%' }}>
          {error ? <div style={{ color: '#ef4444' }}>{error}</div> : null}
          {dataUrl ? (
            <img
              src={dataUrl}
              alt={title}
              style={{
                width: '100%',
                height: '100%',
                objectFit: 'contain',
                borderRadius: 8,
              }}
              draggable={false}
            />
          ) : (
            <div style={{ opacity: 0.6 }}>No image selected</div>
          )}
        </div>
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

      <Handle type="target" position={Position.Right} id="right-target" style={{ opacity: 0.02, width: 36, height: 36, top: '50%' }} />
      <Handle type="source" position={Position.Right} id="right-source" style={{ top: '50%', width: 14, height: 14 }} />
    </div>
  )
}


