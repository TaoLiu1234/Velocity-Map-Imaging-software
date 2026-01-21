import React, { useState, useCallback, type PointerEvent } from 'react'
import type { NodeProps } from 'reactflow'
import { Handle, Position } from 'reactflow'
import type { CanvasNodeData } from '../types'
import { BaseCard } from './BaseCard'
import { useCanvasStore } from '../store'
import { useSettingsStore } from '../settingsStore'

export const TextNode = React.memo<NodeProps<CanvasNodeData>>(function TextNode(props) {
  const { id, data } = props
  const [isEditing, setIsEditing] = useState(false)
  const [draft, setDraft] = useState(data.content ?? '')
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

  const startEdit = useCallback(() => {
    setDraft(data.content ?? '')
    setIsEditing(true)
  }, [data.content])

  const commitEdit = useCallback(() => {
    const trimmed = draft
    // 更新对应节点内容，并把这次编辑记录进历史
    const nextNodes = nodes.map((n) =>
      n.id === id
        ? {
            ...n,
            data: {
              ...n.data,
              content: trimmed,
            },
          }
        : n,
    )
    setNodes(nextNodes, true)
    setIsEditing(false)
  }, [draft, id, nodes, setNodes])

  // 缓存事件处理函数
  const handleTextareaChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setDraft(e.target.value)
  }, [])

  const handleTextareaBlur = useCallback(() => {
    commitEdit()
  }, [commitEdit])

  const handleTextareaKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && e.shiftKey) {
      // Shift+Enter for new line
      return
    }
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      commitEdit()
    }
    if (e.key === 'Escape') {
      e.preventDefault()
      setIsEditing(false)
    }
  }, [commitEdit])

  const handleDivDoubleClick = useCallback(() => {
    startEdit()
  }, [startEdit])

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

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative', overflow: 'visible' }}>
      {/* Obsidian风格：同一个点既是 source 也是 target（叠两个 handle，视觉上一个点） */}
      {/* 先放一个更大的“隐形 target”，用于更容易接线 */}
      <Handle
        type="target"
        position={Position.Left}
        id="left-target"
        style={{
          top: '50%',
          width: 60,
          height: 60,
          background: 'transparent',
          border: 'none',
          opacity: 0.02,
        }}
      />
      {/* 再放一个可见的 source 点，用于拖出连线 */}
      <Handle
        type="source"
        position={Position.Left}
        id="left-source"
        style={{
          top: '50%',
          width: 14,
          height: 14,
          background: theme === 'light' ? 'rgba(15,23,42,0.45)' : 'rgba(255,255,255,0.75)',
          border: theme === 'light' ? '1px solid rgba(255,255,255,0.9)' : '1px solid rgba(0,0,0,0.35)',
        }}
      />
      <BaseCard title={data.title ?? 'Text'} color={data.color} onTitleChange={commitTitle} selected={props.selected}>
        {isEditing ? (
          <textarea
            value={draft}
            autoFocus
            onChange={handleTextareaChange}
            onBlur={handleTextareaBlur}
            onKeyDown={handleTextareaKeyDown}
            style={{
              width: '100%',
              height: '100%',
              resize: 'none',
              border: 'none',
              outline: 'none',
              background: 'transparent',
              color: 'inherit',
              font: 'inherit',
              lineHeight: 1.35,
            }}
          />
        ) : (
          <div
            style={{ whiteSpace: 'pre-wrap', lineHeight: 1.35, cursor: 'text' }}
            onDoubleClick={handleDivDoubleClick}
          >
            {data.content ?? '双击这里编辑文本'}
          </div>
        )}
      </BaseCard>
      {/* 右下角拖拽改变卡片大小 */}
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
      {/* 右侧同理 */}
      <Handle
        type="target"
        position={Position.Right}
        id="right-target"
        style={{
          top: '50%',
          width: 60,
          height: 60,
          background: 'transparent',
          border: 'none',
          opacity: 0.02,
        }}
      />
      <Handle
        type="source"
        position={Position.Right}
        id="right-source"
        style={{
          top: '50%',
          width: 14,
          height: 14,
          background: theme === 'light' ? 'rgba(15,23,42,0.45)' : 'rgba(255,255,255,0.75)',
          border: theme === 'light' ? '1px solid rgba(255,255,255,0.9)' : '1px solid rgba(0,0,0,0.35)',
        }}
      />
    </div>
  )
})

