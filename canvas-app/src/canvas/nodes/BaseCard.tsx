import React, { useCallback, useEffect, useState, type PropsWithChildren } from 'react'
import { useSettingsStore } from '../settingsStore'

export const BaseCard = React.memo<PropsWithChildren<{
  title?: string
  color?: string
  onTitleChange?: (title: string) => void
  selected?: boolean
}>>(
  function BaseCard({ title, color, onTitleChange, children, selected }) {
    const [editingTitle, setEditingTitle] = useState(false)
    const [titleDraft, setTitleDraft] = useState(title ?? '')

    const { theme } = useSettingsStore()


  useEffect(() => {
    if (!editingTitle) setTitleDraft(title ?? '')
  }, [editingTitle, title])

  const commitTitle = useCallback(() => {
    setEditingTitle(false)
    if (!onTitleChange) return
    onTitleChange(titleDraft)
  }, [onTitleChange, titleDraft])

  // 使用useCallback缓存事件处理函数，避免不必要的重渲染
  const handleTitleDoubleClick = useCallback((e: React.MouseEvent) => {
    if (!onTitleChange) return
    e.stopPropagation()
    setEditingTitle(true)
  }, [onTitleChange])

  const handleTitleBlur = useCallback(() => {
    commitTitle()
  }, [commitTitle])

  const handleTitleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault()
      commitTitle()
    }
    if (e.key === 'Escape') {
      e.preventDefault()
      setEditingTitle(false)
    }
  }, [commitTitle])

  return (
    <div
      style={{
        width: '100%',
        height: '100%',
        background: theme === 'light' ? 'rgba(255, 255, 255, 0.95)' : 'rgba(18, 18, 20, 0.92)',
        border: selected
          ? `2px solid ${color ?? '#38bdf8'}`
          : `1px solid ${theme === 'light' ? 'rgba(15,23,42,0.15)' : 'rgba(255,255,255,0.12)'}`,
        borderRadius: 12,
        overflow: 'hidden',
        boxShadow: selected
          ? theme === 'light'
            ? '0 0 0 1px rgba(56,189,248,0.75), 0 8px 24px rgba(0,0,0,0.15)'
            : '0 0 0 1px rgba(56,189,248,0.75), 0 14px 40px rgba(0,0,0,0.65)'
          : theme === 'light'
          ? '0 4px 16px rgba(0,0,0,0.1)'
          : '0 8px 30px rgba(0,0,0,0.35)',
      }}
    >
      <div
        style={{
          padding: '8px 10px',
          fontSize: 12,
          color: theme === 'light' ? 'rgba(15,23,42,0.8)' : 'rgba(255,255,255,0.72)',
          borderBottom: `1px solid ${theme === 'light' ? 'rgba(15,23,42,0.08)' : 'rgba(255,255,255,0.10)'}`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 8,
          userSelect: 'none',
        }}
      >
        {editingTitle && onTitleChange ? (
          <input
            value={titleDraft}
            autoFocus
            onChange={(e) => setTitleDraft(e.target.value)}
            onBlur={handleTitleBlur}
            onKeyDown={handleTitleKeyDown}
            style={{
              width: '100%',
              background: theme === 'light' ? 'rgba(255,255,255,0.9)' : 'rgba(255,255,255,0.06)',
              border: `1px solid ${theme === 'light' ? 'rgba(15,23,42,0.2)' : 'rgba(255,255,255,0.18)'}`,
              borderRadius: 6,
              color: theme === 'light' ? '#0f172a' : 'rgba(255,255,255,0.92)',
              padding: '2px 6px',
              fontSize: 12,
              outline: 'none',
            }}
          />
        ) : (
          <div
            style={{
              fontWeight: 600,
              color: theme === 'light' ? '#0f172a' : 'rgba(255,255,255,0.85)',
              cursor: onTitleChange ? 'text' : 'default'
            }}
            onDoubleClick={handleTitleDoubleClick}
            title={onTitleChange ? 'Double-click to edit title' : undefined}
          >
            {title ?? 'Untitled'}
          </div>
        )}
      </div>
      <div
        style={{
          // Leave room for the bottom-right resize handle so it doesn't overlap the scrollbar/content.
          padding: 10,
          paddingRight: 26,
          paddingBottom: 52,
          color: theme === 'light' ? '#0f172a' : 'rgba(255,255,255,0.90)',
          fontSize: 13,
          height: 'calc(100% - 56px)',
          boxSizing: 'border-box',
          overflow: 'auto',
          // Keep scrollbar inside the padded area and avoid the resize handle region.
          scrollbarGutter: 'stable both-edges',
        }}
      >
        {children}
      </div>
    </div>
  )
})


