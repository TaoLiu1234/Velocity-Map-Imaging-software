import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react'
import { createPortal } from 'react-dom'
import { useLayerStore, type LayerRule } from './layerStore'
import { useCanvasStore } from './store'
import { useSettingsStore } from './settingsStore'

export const LayerPanel: React.FC = React.memo(() => {
  const {
    layers,
    activeLayerId,
    isPanelVisible,
    createLayer,
    renameLayer,
    deleteLayer,
    addNodesToLayer,
    setActiveLayer,
    setPanelVisible
  } = useLayerStore()

  // 缓存选中的节点ID，避免每次都重新计算
  const selectedNodeIds = useMemo(() => {
    return useCanvasStore.getState().nodes.filter(n => n.selected).map(n => n.id)
  }, [useCanvasStore((s) => s.nodes.filter(n => n.selected).map(n => n.id).join(','))])

  const { theme } = useSettingsStore()

  const [editingLayerId, setEditingLayerId] = useState<string | null>(null)
  const [editingName, setEditingName] = useState('')
  const [isHovered, setIsHovered] = useState(false)
  const hoverRef = useRef(false)
  const [rulesLayerId, setRulesLayerId] = useState<string | null>(null)
  const [rulesDraft, setRulesDraft] = useState<LayerRule[]>([])
  const [newRuleText, setNewRuleText] = useState('')

  const handleMouseEnter = useCallback(() => {
    setIsHovered(true)
    hoverRef.current = true
    setPanelVisible(true)
  }, [setPanelVisible])

  const handleMouseLeave = useCallback(() => {
    setIsHovered(false)
    hoverRef.current = false
    // 减少延迟时间以提高响应性
    setTimeout(() => {
      if (!hoverRef.current) setPanelVisible(false)
    }, 150)
  }, [setPanelVisible])

  const handleCreateLayer = useCallback(() => {
    if (selectedNodeIds.length === 0) {
      alert('请先选择一些卡片')
      return
    }
    createLayer(selectedNodeIds)
  }, [selectedNodeIds, createLayer])

  const handleRenameStart = useCallback((layerId: string, currentName: string) => {
    setEditingLayerId(layerId)
    setEditingName(currentName)
  }, [])

  const handleRenameSave = useCallback(() => {
    if (editingLayerId && editingName.trim()) {
      renameLayer(editingLayerId, editingName.trim())
    }
    setEditingLayerId(null)
    setEditingName('')
  }, [editingLayerId, editingName, renameLayer])

  const handleRenameCancel = useCallback(() => {
    setEditingLayerId(null)
    setEditingName('')
  }, [])

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleRenameSave()
    } else if (e.key === 'Escape') {
      handleRenameCancel()
    }
  }, [handleRenameSave, handleRenameCancel])

  const openRules = useCallback((layerId: string) => {
    const layer = layers.find(l => l.id === layerId)
    if (layer) {
      setRulesLayerId(layerId)
      setRulesDraft(layer.rules ?? [])
      setNewRuleText('')
    }
  }, [layers])

  const closeRules = useCallback(() => {
    setRulesLayerId(null)
    setRulesDraft([])
    setNewRuleText('')
  }, [])

  useEffect(() => {
    if (!rulesLayerId) return
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') closeRules()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [closeRules, rulesLayerId])

  const persistRules = useCallback((nextRules: LayerRule[]) => {
    if (!rulesLayerId) return
    setRulesDraft(nextRules)
    useLayerStore.getState().setLayerRules(rulesLayerId, nextRules)
  }, [rulesLayerId])

  const toggleRule = useCallback((ruleId: string) => {
    const next = rulesDraft.map((r) => r.id === ruleId ? { ...r, done: !r.done } : r)
    persistRules(next)
  }, [rulesDraft, persistRules])

  const addRule = useCallback(() => {
    if (!rulesLayerId) return
    const text = newRuleText.trim()
    if (!text) return
    const next: LayerRule = { id: `rule-${Date.now()}-${Math.random().toString(16).slice(2)}`, text, done: false }
    persistRules([...rulesDraft, next])
    setNewRuleText('')
  }, [newRuleText, rulesDraft, persistRules, rulesLayerId])

  const expanded = isPanelVisible || isHovered || Boolean(rulesLayerId)
  const opacity = expanded ? 1 : 0.25

  const rulesModal = rulesLayerId
    ? createPortal(
        <div
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0,0,0,0.6)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 3000,
          }}
          onMouseDown={(e) => {
            if (e.target === e.currentTarget) closeRules()
          }}
        >
          <div
            style={{
              width: 520,
              height: 640,
              maxWidth: '94vw',
              maxHeight: '94vh',
              overflow: 'auto',
              background: theme === 'light' ? '#ffffff' : '#0f1016',
              color: theme === 'light' ? '#0f172a' : 'rgba(255,255,255,0.95)',
              borderRadius: 16,
              border: `1px solid ${theme === 'light' ? 'rgba(15,23,42,0.15)' : 'rgba(255,255,255,0.2)'}`,
              padding: 20,
              boxShadow: theme === 'light' ? '0 25px 60px rgba(0,0,0,0.15)' : '0 25px 60px rgba(0,0,0,0.8)',
              display: 'flex',
              flexDirection: 'column',
              gap: 16,
            }}
            onMouseDown={(e) => e.stopPropagation()}
          >
            <div style={{ fontWeight: 700, fontSize: 14, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span>层级规则 Checklist</span>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                <div style={{ fontSize: 11, color: theme === 'light' ? 'rgba(15,23,42,0.6)' : 'rgba(255,255,255,0.6)', fontWeight: 500 }}>Esc / 点击空白处关闭</div>
                <button
                  onClick={closeRules}
                  style={{
                    width: 34,
                    height: 34,
                    borderRadius: 8,
                    border: `1px solid ${theme === 'light' ? 'rgba(15,23,42,0.2)' : 'rgba(255,255,255,0.2)'}`,
                    background: theme === 'light' ? 'rgba(15,23,42,0.04)' : 'rgba(255,255,255,0.04)',
                    color: theme === 'light' ? '#0f172a' : 'rgba(255,255,255,0.85)',
                    cursor: 'pointer',
                    fontSize: 18,
                    lineHeight: '32px',
                  }}
                  title="关闭"
                >
                  ×
                </button>
              </div>
            </div>

            <div style={{ display: 'flex', gap: 8, alignItems: 'stretch' }}>
              <input
                value={newRuleText}
                onChange={(e) => setNewRuleText(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') addRule()
                }}
                placeholder="新增一条规则…"
                style={{
                  flex: 1,
                  padding: '10px 12px',
                  borderRadius: 10,
                  border: `1px solid ${theme === 'light' ? 'rgba(15,23,42,0.25)' : 'rgba(255,255,255,0.25)'}`,
                  background: theme === 'light' ? 'rgba(255,255,255,0.9)' : 'rgba(255,255,255,0.06)',
                  color: theme === 'light' ? '#0f172a' : 'rgba(255,255,255,0.92)',
                  outline: 'none',
                  fontSize: 13,
                }}
              />
              <button
                onClick={addRule}
                style={{
                  padding: '10px 14px',
                  borderRadius: 10,
                  border: `1px solid ${theme === 'light' ? 'rgba(79,70,229,0.5)' : 'rgba(255,255,255,0.25)'}`,
                  background: 'rgba(79,70,229,0.9)',
                  color: 'white',
                  cursor: 'pointer',
                  fontSize: 13,
                  whiteSpace: 'nowrap',
                }}
              >
                添加规则
              </button>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {rulesDraft.length === 0 ? (
                <div style={{ padding: '14px 10px', color: theme === 'light' ? 'rgba(15,23,42,0.65)' : 'rgba(255,255,255,0.65)', fontSize: 12, border: `1px dashed ${theme === 'light' ? 'rgba(15,23,42,0.2)' : 'rgba(255,255,255,0.2)'}`, borderRadius: 10 }}>
                  暂无规则，添加一些吧。
                </div>
              ) : (
                rulesDraft.map((rule) => (
                  <label
                    key={rule.id}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 10,
                      padding: '10px 12px',
                      borderRadius: 10,
                      border: `1px solid ${theme === 'light' ? 'rgba(15,23,42,0.12)' : 'rgba(255,255,255,0.12)'}`,
                      background: rule.done ? 'rgba(34,197,94,0.12)' : theme === 'light' ? 'rgba(15,23,42,0.03)' : 'rgba(255,255,255,0.03)',
                      cursor: 'pointer',
                    }}
                  >
                    <input
                      type="checkbox"
                      checked={rule.done}
                      onChange={() => toggleRule(rule.id)}
                      style={{ width: 18, height: 18 }}
                    />
                    <span style={{ flex: 1, color: rule.done ? (theme === 'light' ? 'rgba(15,23,42,0.65)' : 'rgba(255,255,255,0.65)') : (theme === 'light' ? '#0f172a' : 'rgba(255,255,255,0.92)'), textDecoration: rule.done ? 'line-through' : 'none' }}>
                      {rule.text}
                    </span>
                  </label>
                ))
              )}
            </div>
          </div>
        </div>,
        document.body
      )
    : null

  return (
    <>
      <div
        style={{
          position: 'absolute',
          bottom: 20,
          left: '50%',
          transform: 'translateX(-50%)',
          background: theme === 'light' ? 'rgba(255,255,255,0.95)' : 'rgba(15,16,22,0.95)',
          border: `1px solid ${theme === 'light' ? 'rgba(15,23,42,0.15)' : 'rgba(255,255,255,0.18)'}`,
          borderRadius: 8,
          padding: expanded ? '8px 12px' : '6px 10px',
          minWidth: expanded ? 280 : 140,
          maxWidth: expanded ? 400 : 160,
          maxHeight: expanded ? 300 : 32,
          overflow: 'hidden',
          opacity,
          transition: 'opacity 0.15s ease, transform 0.15s ease',
          zIndex: 2000,
          pointerEvents: 'auto',
          boxShadow: expanded
            ? theme === 'light'
              ? '0 8px 24px rgba(0,0,0,0.15)'
              : '0 12px 30px rgba(0,0,0,0.45)'
            : theme === 'light'
            ? '0 4px 12px rgba(0,0,0,0.1)'
            : '0 6px 18px rgba(0,0,0,0.35)',
        }}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
      >
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginBottom: expanded ? 8 : 0,
        paddingBottom: expanded ? 6 : 0,
        borderBottom: expanded ? `1px solid ${theme === 'light' ? 'rgba(15,23,42,0.08)' : 'rgba(255,255,255,0.12)'}` : 'none',
      }}>
        <div style={{
          fontSize: 12,
          fontWeight: 600,
          color: theme === 'light' ? '#0f172a' : 'rgba(255,255,255,0.9)',
          whiteSpace: 'nowrap',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
        }}>
          层级管理 ({layers.length})
        </div>
        {expanded && (
          <button
            onClick={(e) => {
              e.stopPropagation()
              handleCreateLayer()
            }}
            style={{
              padding: '4px 8px',
              borderRadius: 4,
              border: `1px solid ${theme === 'light' ? 'rgba(15,23,42,0.2)' : 'rgba(255,255,255,0.3)'}`,
              background: 'transparent',
              color: theme === 'light' ? '#0f172a' : 'rgba(255,255,255,0.8)',
              cursor: 'pointer',
              fontSize: 11,
            }}
            onMouseEnter={(e) => e.currentTarget.style.background = theme === 'light' ? 'rgba(15,23,42,0.1)' : 'rgba(255,255,255,0.1)'}
            onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
          >
            创建层级
          </button>
        )}
      </div>

      {expanded ? (
        <>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4, maxHeight: 220, overflowY: 'auto', paddingRight: 4 }}>
            {layers.length === 0 ? (
              <div style={{
                padding: '12px',
                textAlign: 'center',
                color: theme === 'light' ? 'rgba(15,23,42,0.6)' : 'rgba(255,255,255,0.5)',
                fontSize: 12,
              }}>
                暂无层级，请选择卡片后创建
              </div>
            ) : (
              layers.map((layer) => (
                <div
                  key={layer.id}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 8,
                    padding: '6px 8px',
                    borderRadius: 4,
                    background: activeLayerId === layer.id
                      ? 'rgba(59,130,246,0.2)'
                      : 'transparent',
                    border: activeLayerId === layer.id
                      ? '1px solid rgba(59,130,246,0.4)'
                      : '1px solid transparent',
                  }}
                >
                  {/* Layer color indicator */}
                  <div
                    style={{
                      width: 12,
                      height: 12,
                      borderRadius: '50%',
                      background: layer.color,
                      flexShrink: 0,
                    }}
                  />

                  {/* Layer name */}
                  <div style={{ flex: 1, minWidth: 0 }}>
                    {editingLayerId === layer.id ? (
                      <input
                        value={editingName}
                        onChange={(e) => setEditingName(e.target.value)}
                        onBlur={handleRenameSave}
                        onKeyDown={handleKeyDown}
                        autoFocus
                        style={{
                          width: '100%',
                          padding: '2px 4px',
                          borderRadius: 3,
                          border: '1px solid rgba(59,130,246,0.5)',
                          background: theme === 'light' ? 'rgba(255,255,255,0.9)' : 'rgba(0,0,0,0.3)',
                          color: theme === 'light' ? '#0f172a' : 'rgba(255,255,255,0.9)',
                          fontSize: 11,
                        }}
                      />
                    ) : (
                      <div
                        style={{
                          fontSize: 11,
                          color: theme === 'light' ? '#0f172a' : 'rgba(255,255,255,0.9)',
                          cursor: 'pointer',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                        }}
                        onDoubleClick={() => handleRenameStart(layer.id, layer.name)}
                      >
                        {layer.name}
                      </div>
                    )}
                    <div style={{
                      fontSize: 9,
                      color: theme === 'light' ? 'rgba(15,23,42,0.6)' : 'rgba(255,255,255,0.5)',
                      marginTop: 1,
                    }}>
                      {layer.nodeIds.length} 个节点
                    </div>
                  </div>

                  {/* Action buttons */}
                  <div style={{ display: 'flex', gap: 2 }}>
                    {/* Rules */}
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        openRules(layer.id)
                      }}
                      style={{
                        padding: '2px 6px',
                        borderRadius: 3,
                        border: `1px solid ${theme === 'light' ? 'rgba(15,23,42,0.2)' : 'rgba(255,255,255,0.2)'}`,
                        background: 'transparent',
                        color: theme === 'light' ? 'rgba(15,23,42,0.7)' : 'rgba(255,255,255,0.75)',
                        cursor: 'pointer',
                        fontSize: 10,
                      }}
                      title="查看/设置规则 checklist"
                    >
                      规则
                    </button>

                    {/* Activate layer */}
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        setActiveLayer(activeLayerId === layer.id ? null : layer.id)
                      }}
                      style={{
                        padding: '2px 6px',
                        borderRadius: 3,
                        border: `1px solid ${theme === 'light' ? 'rgba(15,23,42,0.2)' : 'rgba(255,255,255,0.2)'}`,
                        background: activeLayerId === layer.id
                          ? 'rgba(59,130,246,0.3)'
                          : 'transparent',
                        color: activeLayerId === layer.id
                          ? 'rgba(59,130,246,0.9)'
                          : theme === 'light'
                          ? 'rgba(15,23,42,0.7)'
                          : 'rgba(255,255,255,0.7)',
                        cursor: 'pointer',
                        fontSize: 10,
                      }}
                      title="激活/取消激活层级"
                    >
                      ✓
                    </button>

                    {/* Delete layer */}
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        deleteLayer(layer.id)
                      }}
                      style={{
                        padding: '2px 4px',
                        borderRadius: 3,
                        border: 'none',
                        background: 'transparent',
                        color: 'rgba(239,68,68,0.7)',
                        cursor: 'pointer',
                        fontSize: 10,
                      }}
                      title="删除层级"
                      onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(239,68,68,0.2)'}
                      onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
                    >
                      ×
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>

          {/* Add to existing layer button */}
          {layers.length > 0 && selectedNodeIds.length > 0 && (
            <div style={{
              marginTop: 8,
              paddingTop: 8,
              borderTop: `1px solid ${theme === 'light' ? 'rgba(15,23,42,0.08)' : 'rgba(255,255,255,0.12)'}`,
            }}>
              <div style={{
                fontSize: 11,
                color: theme === 'light' ? 'rgba(15,23,42,0.7)' : 'rgba(255,255,255,0.7)',
                marginBottom: 4,
              }}>
                添加到现有层级:
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                {layers.map((layer) => (
                  <button
                    key={layer.id}
                    onClick={(e) => {
                      e.stopPropagation()
                      if (selectedNodeIds.length > 0) {
                        addNodesToLayer(layer.id, selectedNodeIds)
                      }
                    }}
                    style={{
                      padding: '2px 6px',
                      borderRadius: 3,
                      border: `1px solid ${theme === 'light' ? 'rgba(15,23,42,0.2)' : 'rgba(255,255,255,0.2)'}`,
                      background: 'transparent',
                      color: theme === 'light' ? 'rgba(15,23,42,0.6)' : 'rgba(255,255,255,0.6)',
                      cursor: 'pointer',
                      fontSize: 10,
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.background = theme === 'light' ? 'rgba(15,23,42,0.1)' : 'rgba(255,255,255,0.1)'
                      e.currentTarget.style.color = theme === 'light' ? 'rgba(15,23,42,0.8)' : 'rgba(255,255,255,0.8)'
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.background = 'transparent'
                      e.currentTarget.style.color = theme === 'light' ? 'rgba(15,23,42,0.6)' : 'rgba(255,255,255,0.6)'
                    }}
                  >
                    + {layer.name}
                  </button>
                ))}
              </div>
            </div>
          )}
        </>
      ) : (
        <div style={{ fontSize: 11, color: theme === 'light' ? 'rgba(15,23,42,0.7)' : 'rgba(255,255,255,0.7)' }}>悬停展开层级管理</div>
      )}
      </div>
      {rulesModal}
    </>
  )
})
