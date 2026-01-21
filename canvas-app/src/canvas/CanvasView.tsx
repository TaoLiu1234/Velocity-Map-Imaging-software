import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { performanceMonitor } from './performanceMonitor'

// 创建一个全局的palette缓存，避免每次都重新创建
const paletteCache = new Map<string, any>()

// 定义nodeTypes在组件外部，避免每次渲染都重新创建
const nodeTypes = {
  text: TextNode,
  markdown: MarkdownNode,
  code: CodeNode,
  image: ImageNode,
  pdf: PdfNode,
  iframe: IframeNode,
}

// 链接列表浮动窗口组件
function LinkListPanel() {
  const [isExpanded, setIsExpanded] = useState(false)
  const [hoverTimer, setHoverTimer] = useState<number | null>(null)
  const { theme } = useSettingsStore()
  const links = useLatexLinkStore((s) => s.links)
  const focusNodes = useCallback((nodeIds: string[]) => {
    const nodes = useCanvasStore.getState().nodes
    const next = nodes.map((n) => ({ ...n, selected: nodeIds.includes(n.id) }))
    useCanvasStore.getState().setNodes(next, false)
  }, [])

  const handleMouseEnter = useCallback(() => {
    if (hoverTimer !== null) {
      clearTimeout(hoverTimer)
      setHoverTimer(null)
    }
    setIsExpanded(true)
  }, [hoverTimer])

  const handleMouseLeave = useCallback(() => {
    const timer = window.setTimeout(() => {
      setIsExpanded(false)
      setHoverTimer(null)
    }, 200)
    setHoverTimer(timer)
  }, [])

  // 清理定时器
  useEffect(() => {
    return () => {
      if (hoverTimer !== null) {
        clearTimeout(hoverTimer)
      }
    }
  }, [hoverTimer])

  const palette = useMemo(
    () =>
      theme === 'light'
        ? {
            surface: 'rgba(255,255,255,0.92)',
            border: 'rgba(15,23,42,0.12)',
            textPrimary: '#0f172a',
            textSecondary: 'rgba(15,23,42,0.6)',
          }
        : {
            surface: 'rgba(15, 16, 22, 0.75)',
            border: 'rgba(255,255,255,0.10)',
            textPrimary: 'rgba(255,255,255,0.9)',
            textSecondary: 'rgba(255,255,255,0.65)',
          },
    [theme],
  )

  return (
    <div
      style={{
        position: 'absolute',
        right: 'calc(35% - 259px)', // 贴着canvas区和latex区的边界，往右移动约256px
        top: 360, // 在inspector下方（inspector top: 70 + 估算高度约280-300）
        zIndex: 15,
        display: 'flex',
        flexDirection: 'column', // 垂直排列，展开的内容在下方
        alignItems: 'flex-end',
      }}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      {/* 展开的内容 */}
      {isExpanded && (
        <div
          style={{
            background: palette.surface,
            border: `1px solid ${palette.border}`,
            borderRadius: 12,
            padding: '12px 16px',
            width: 280,
            maxHeight: 300,
            overflowY: 'auto',
            backdropFilter: 'blur(10px)',
            boxShadow: theme === 'light'
              ? '0 8px 24px rgba(0,0,0,0.15)'
              : '0 12px 30px rgba(0,0,0,0.45)',
            marginBottom: 8,
            transition: 'all 0.15s ease',
          }}
        >
          <div style={{
            fontWeight: 600,
            marginBottom: 8,
            color: palette.textPrimary,
            fontSize: 13
          }}>
            LaTeX 链接列表
          </div>
          {links.length === 0 ? (
            <div style={{
              opacity: 0.7,
              color: palette.textSecondary,
              fontSize: 12,
              fontStyle: 'italic'
            }}>
              暂无链接。先选中文本，再选中画布节点，右键创建链接。
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {links.map((link) => (
                <div
                  key={link.id}
                  style={{
                    padding: '8px 10px',
                    borderRadius: 8,
                    background: theme === 'light' ? 'rgba(15,23,42,0.03)' : 'rgba(255,255,255,0.05)',
                    border: `1px solid ${theme === 'light' ? 'rgba(15,23,42,0.08)' : 'rgba(255,255,255,0.1)'}`,
                    cursor: 'pointer',
                  }}
                  onClick={() => focusNodes(link.nodeIds)}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = theme === 'light' ? 'rgba(59,130,246,0.1)' : 'rgba(59,130,246,0.2)'
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = theme === 'light' ? 'rgba(15,23,42,0.03)' : 'rgba(255,255,255,0.05)'
                  }}
                >
                  <div style={{
                    fontSize: 11,
                    color: palette.textPrimary,
                    fontWeight: 500,
                    marginBottom: 4
                  }}>
                    "{link.text.slice(0, 30)}{link.text.length > 30 ? '…' : ''}"
                  </div>
                  <div style={{
                    fontSize: 10,
                    color: palette.textSecondary,
                    opacity: 0.8
                  }}>
                    节点: {link.nodeIds.join(', ')}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* 收缩的竖条 */}
      <div
        style={{
          width: 6,
          height: 120,
          background: palette.surface,
          border: `1px solid ${palette.border}`,
          borderRadius: '3px 0 0 3px',
          cursor: 'pointer',
          backdropFilter: 'blur(10px)',
          boxShadow: theme === 'light'
            ? '0 2px 8px rgba(0,0,0,0.1)'
            : '0 2px 8px rgba(0,0,0,0.3)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          transition: 'all 0.15s ease',
          opacity: isExpanded ? 0.8 : 1,
        }}
        title="LaTeX 链接列表"
      >
        <div
          style={{
            writingMode: 'vertical-rl',
            textOrientation: 'mixed',
            fontSize: 10,
            fontWeight: 600,
            color: palette.textSecondary,
            letterSpacing: 1,
          }}
        >
          链接 ({links.length})
        </div>
      </div>
    </div>
  )
}
import ReactFlow, {
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
  Background,
  BackgroundVariant,
  Controls,
  MarkerType,
  MiniMap,
  ReactFlowProvider,
  useReactFlow,
} from 'reactflow'
import type { Connection, EdgeChange, NodeChange } from 'reactflow'
import 'reactflow/dist/style.css'
import { useCanvasStore, type RFEdge } from './store'
import { useLatexLinkStore } from './latexLinkStore'
import { useLayerStore } from './layerStore'
import { useGlobalUndoStore } from './globalUndoStore'
import { LayerPanel } from './LayerPanel'
import { TextNode } from './nodes/TextNode'
import { MarkdownNode } from './nodes/MarkdownNode'
import { CodeNode } from './nodes/CodeNode'
import { ImageNode } from './nodes/ImageNode'
import { PdfNode } from './nodes/PdfNode'
import { IframeNode } from './nodes/IframeNode'
import type { CanvasNodeData, NodeKind } from './types'
import { exportToDoc, importFromDoc } from './serialization'
import * as htmlToImage from 'html-to-image'
import { useSettingsStore } from './settingsStore'

function newId(prefix: string) {
  return `${prefix}_${Math.random().toString(16).slice(2)}`
}

type EdgeEditState = {
  id: string
  label: string
  weight: string
} | null

type PaneMenuState =
  | {
      x: number
      y: number
    }
  | null

type NodeMenuState =
  | {
      x: number
      y: number
    }
  | null

function strokeWidthForWeight(weight: unknown): number {
  const w = typeof weight === 'number' && Number.isFinite(weight) ? weight : 1
  // Map: 1 -> 2.5px, 5 -> ~6.5px (clamped)
  return Math.max(1.5, Math.min(10, 1.5 + w * 1.0))
}

function CanvasInner() {
  // 启用性能监控（开发模式）
  useEffect(() => {
    if (import.meta.env.DEV) {
      performanceMonitor.enable()
      console.log('[Performance] Monitoring enabled')
    }
  }, [])

  const nodes = useCanvasStore((s) => s.nodes)
  const edges = useCanvasStore((s) => s.edges)
  const setNodes = useCanvasStore((s) => s.setNodes)
  const setEdges = useCanvasStore((s) => s.setEdges)
  const setViewport = useCanvasStore((s) => s.setViewport)
  const globalUndo = useGlobalUndoStore((s) => s.undo)
  const globalRedo = useGlobalUndoStore((s) => s.redo)
  const activeLayerId = useLayerStore((s) => s.activeLayerId)
  const layers = useLayerStore((s) => s.layers)
  const wrapperRef = useRef<HTMLDivElement | null>(null)
  const rf = useReactFlow()
  const [search, setSearch] = useState('')
  const [edgeEdit, setEdgeEdit] = useState<EdgeEditState>(null)
  const [paneMenu, setPaneMenu] = useState<PaneMenuState>(null)
  const [nodeMenu, setNodeMenu] = useState<NodeMenuState>(null)
  const [kindFilter, setKindFilter] = useState<NodeKind | 'all'>('all')
  const [exporting, setExporting] = useState(false)
  const [lastSavePath, setLastSavePath] = useState<string>(() => localStorage.getItem('canvas:lastSavePath') ?? '')
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [toolbarHover, setToolbarHover] = useState(false)

  const { theme, historyLimit, setTheme, setHistoryLimit } = useSettingsStore()

  // 使用全局palette缓存
  const palette = useMemo(() => {
    const cacheKey = `canvas-${theme}`
    if (!paletteCache.has(cacheKey)) {
      paletteCache.set(cacheKey,
        theme === 'light'
          ? {
              canvasBg: '#f8fafc',
              surface: 'rgba(255,255,255,0.92)',
              border: 'rgba(15,23,42,0.12)',
              textPrimary: '#0f172a',
              textSecondary: 'rgba(15,23,42,0.6)',
              accent: '#4f46e5',
            }
          : {
              canvasBg: '#0b0c10',
              surface: 'rgba(15, 16, 22, 0.75)',
              border: 'rgba(255,255,255,0.10)',
              textPrimary: 'rgba(255,255,255,0.9)',
              textSecondary: 'rgba(255,255,255,0.65)',
              accent: '#4f46e5',
            }
      )
    }
    return paletteCache.get(cacheKey)!
  }, [theme])

  const clipboardRef = useRef<{
    nodes: Array<{ id: string; type: any; position: { x: number; y: number }; data: any; style: any }>
    edges: Array<{ id: string; source: string; target: string; sourceHandle?: string | null; targetHandle?: string | null; data?: any; type?: any; markerEnd?: any }>
    anchor: { x: number; y: number }
  } | null>(null)

  const autoSaveTimerRef = useRef<number | null>(null)


  const onNodesChange = useCallback(
    (changes: NodeChange[]) => {
      // 节点位置等变化只更新当前状态，不直接进历史
      setNodes(applyNodeChanges(changes, nodes), false)
    },
    [nodes, setNodes],
  )

  const onNodeDragStop = useCallback(() => {
    // 拖拽结束时，把当前节点状态作为一次完整历史记录
    const current = useCanvasStore.getState().nodes
    useCanvasStore.getState().setNodes(current, true)
  }, [])

  // 优化键盘事件处理，避免在每次渲染时重新创建
  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    const target = e.target as HTMLElement | null
    const tag = target?.tagName
    if (tag === 'TEXTAREA' || tag === 'INPUT' || target?.isContentEditable) {
      // 正在编辑内容时，不拦截 Delete/Backspace，交给输入框自己处理
      return
    }
    if (e.key === 'Delete' || e.key === 'Backspace') {
      // 删除选中的节点和边，作为一次完整操作
      const nextNodes = nodes.filter((n) => !n.selected)
      const nextEdges = edges.filter((e2) => !e2.selected)

      // 从所有层级中移除被删除的节点
      const deletedNodeIds = nodes.filter(n => n.selected).map(n => n.id)
      if (deletedNodeIds.length > 0) {
        const { layers } = useLayerStore.getState()
        layers.forEach(layer => {
          const remainingNodeIds = layer.nodeIds.filter(id => !deletedNodeIds.includes(id))
          if (remainingNodeIds.length !== layer.nodeIds.length) {
            useLayerStore.getState().removeNodesFromLayer(layer.id, deletedNodeIds)
          }
        })
      }

      setNodes(nextNodes, true)
      setEdges(nextEdges, true)
      e.preventDefault()
    }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
      globalUndo()
      e.preventDefault()
    }
    if ((e.ctrlKey || e.metaKey) && (e.key.toLowerCase() === 'y' || (e.shiftKey && e.key.toLowerCase() === 'z'))) {
      globalRedo()
      e.preventDefault()
    }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'c') {
      copySelection()
      e.preventDefault()
    }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'v') {
      pasteSelection()
      e.preventDefault()
    }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'd') {
      duplicateSelection()
      e.preventDefault()
    }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'l') {
      const selectedNodeIds = nodes.filter(n => n.selected).map(n => n.id)
      if (selectedNodeIds.length > 0) {
        // Import dynamically to avoid circular dependency
        import('./layerStore').then(({ useLayerStore }) => {
          useLayerStore.getState().createLayer(selectedNodeIds)
        })
      }
      e.preventDefault()
    }
  }, [nodes, edges, setNodes, setEdges, globalUndo, globalRedo])

  const copySelection = useCallback(() => {
    const curNodes = useCanvasStore.getState().nodes
    const curEdges = useCanvasStore.getState().edges
    const selNodes = curNodes.filter((n) => n.selected)
    if (selNodes.length === 0) return
    const selIds = new Set(selNodes.map((n) => n.id))
    const selEdges = curEdges.filter((e) => selIds.has(e.source) && selIds.has(e.target))
    const minX = Math.min(...selNodes.map((n) => n.position.x))
    const minY = Math.min(...selNodes.map((n) => n.position.y))
    clipboardRef.current = {
      nodes: selNodes.map((n) => ({
        id: n.id,
        type: n.type,
        position: { x: n.position.x, y: n.position.y },
        data: n.data,
        style: n.style,
      })),
      edges: selEdges.map((e) => ({
        id: e.id,
        source: e.source,
        target: e.target,
        sourceHandle: (e as any).sourceHandle ?? null,
        targetHandle: (e as any).targetHandle ?? null,
        data: e.data,
        type: e.type,
        markerEnd: e.markerEnd,
      })),
      anchor: { x: minX, y: minY },
    }
  }, [])

  const pasteSelection = useCallback(() => {
    const clip = clipboardRef.current
    if (!clip) return
    const rect = wrapperRef.current?.getBoundingClientRect()
    const center = rect
      ? rf.screenToFlowPosition({ x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 })
      : { x: 0, y: 0 }

    const offsetX = center.x - clip.anchor.x + 20
    const offsetY = center.y - clip.anchor.y + 20

    const idMap = new Map<string, string>()
    const newNodes = clip.nodes.map((n) => {
      const newId = `${n.id}_${Math.random().toString(16).slice(2)}`
      idMap.set(n.id, newId)
      return {
        id: newId,
        type: n.type,
        position: { x: n.position.x + offsetX, y: n.position.y + offsetY },
        data: n.data,
        style: n.style,
        selected: true,
      }
    })

    const newEdges = clip.edges
      .map((e) => {
        const s = idMap.get(e.source)
        const t = idMap.get(e.target)
        if (!s || !t) return null
        return {
          id: `e_${Math.random().toString(16).slice(2)}`,
          source: s,
          target: t,
          sourceHandle: e.sourceHandle ?? undefined,
          targetHandle: e.targetHandle ?? undefined,
          type: e.type ?? 'smoothstep',
          markerEnd: e.markerEnd,
          data: e.data,
          selected: true,
        } as any
      })
      .filter(Boolean) as any[]

    const curNodes = useCanvasStore.getState().nodes.map((n) => ({ ...n, selected: false }))
    const curEdges = useCanvasStore.getState().edges.map((e) => ({ ...(e as any), selected: false }))
    useCanvasStore.getState().setNodes([...curNodes, ...(newNodes as any)], true)
    useCanvasStore.getState().setEdges([...(curEdges as any), ...(newEdges as any)], true)
  }, [rf])

  const duplicateSelection = useCallback(() => {
    copySelection()
    pasteSelection()
  }, [copySelection, pasteSelection])

  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
      setEdges(applyEdgeChanges(changes, edges), false)
    },
    [edges, setEdges],
  )

  const onEdgeDoubleClick = useCallback(
    (_event: React.MouseEvent, edge: RFEdge) => {
      setEdgeEdit({
        id: edge.id,
        label: edge.data?.label ?? '',
        weight: edge.data?.weight != null ? String(edge.data.weight) : '',
      })
    },
    [],
  )

  const onConnect = useCallback(
    (connection: Connection) => {
      const edge = {
        ...connection,
        id: newId('e'),
        // Use built-in edge types: default/straight/step/smoothstep
        type: 'smoothstep',
        markerEnd: { type: MarkerType.ArrowClosed },
        data: { label: '', weight: 1 },
      }
      setEdges(addEdge(edge, edges))
    },
    [edges, setEdges],
  )

  const addNodeOfKind = useCallback(
    (kind: CanvasNodeData['kind']) => {
      const id = newId('n')
      const base: CanvasNodeData = { kind, title: kind[0].toUpperCase() + kind.slice(1) }
      const colorMap: Record<CanvasNodeData['kind'], string> = {
        text: '#6366f1',
        markdown: '#22c55e',
        code: '#f97316',
        image: '#0ea5e9',
        pdf: '#ef4444',
        iframe: '#eab308',
      }
      let content: string | undefined
      if (kind === 'text') content = '双击编辑文本'
      if (kind === 'markdown') content = '# Markdown 节点\n\n双击编辑内容'
      if (kind === 'code') content = '// Code 节点\n\nfunction demo() {}'
      if (kind === 'image') content = undefined
      if (kind === 'pdf') content = undefined
      if (kind === 'iframe') content = undefined

      // 新建节点放在当前视口中心，避免“创建了但看不到”的错觉
      const rect = wrapperRef.current?.getBoundingClientRect()
      const center = rect
        ? rf.screenToFlowPosition({ x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 })
        : { x: 0, y: 0 }

      const defaultSize =
        kind === 'pdf' || kind === 'iframe'
          ? { width: 560, height: 420 }
          : kind === 'image'
            ? { width: 420, height: 280 }
            : { width: 320, height: 200 }

      const node = {
        id,
        type: kind,
        position: center,
        data: { ...base, content, color: colorMap[kind] } satisfies CanvasNodeData,
        style: defaultSize,
      }
      // 用最新 state 追加，避免闭包导致“点了没加上”
      const current = useCanvasStore.getState().nodes
      useCanvasStore.getState().setNodes([...current, node], true)
      // 强制把视图移动/缩放到新节点，确保你立刻看见
      requestAnimationFrame(() => {
        try {
          rf.fitView({ nodes: [node as any], padding: 0.35, duration: 250 })
        } catch {
          // ignore
        }
      })
    },
    [rf],
  )

  // 优化节点过滤逻辑，使用useMemo缓存结果
  const visibleNodes = useMemo(() => {
    performanceMonitor.startMeasurement('visibleNodes-calculation')
    const searchLower = search.trim().toLowerCase()

    // 基础搜索过滤
    const searchFiltered = searchLower.length === 0
      ? nodes
      : nodes.map((n) => {
          const title = (n.data as any)?.title ?? ''
          const content = (n.data as any)?.content ?? ''
          const matched =
            title.toString().toLowerCase().includes(searchLower) ||
            content.toString().toLowerCase().includes(searchLower)
          return {
            ...n,
            style: {
              ...(n.style ?? {}),
              opacity: matched ? 1 : 0.25,
            },
          }
        })

    // 类型过滤
    const typeFiltered = kindFilter === 'all'
      ? searchFiltered
      : searchFiltered.map((n) => {
          const matched = (n.data as any)?.kind === kindFilter
          return {
            ...n,
            style: {
              ...(n.style ?? {}),
              opacity: matched ? ((n.style as any)?.opacity ?? 1) : 0.15,
            },
          }
        })

    // 层级过滤
    const layerFiltered = activeLayerId
      ? typeFiltered.map((n) => {
          const activeLayer = layers.find(l => l.id === activeLayerId)
          const isInActiveLayer = activeLayer ? activeLayer.nodeIds.includes(n.id) : false
          return {
            ...n,
            style: {
              ...(n.style ?? {}),
              opacity: isInActiveLayer ? ((n.style as any)?.opacity ?? 1) : 0.2,
            },
          }
        })
      : typeFiltered

    performanceMonitor.endMeasurement('visibleNodes-calculation')
    return layerFiltered
  }, [nodes, search, kindFilter, activeLayerId, layers])

  // 优化选中节点和边的查找
  const selectedNode = useMemo(() => nodes.find((n) => n.selected), [nodes])
  const selectedEdge = useMemo(() => edges.find((e) => (e as any).selected), [edges])
  const selectedNodes = useMemo(() => nodes.filter((n) => n.selected), [nodes])

  const updateSelectedNode = useCallback(
    (patch: Partial<CanvasNodeData>) => {
      const current = useCanvasStore.getState().nodes
      const sel = current.find((n) => n.selected)
      if (!sel) return
      const next = current.map((n) => (n.id === sel.id ? { ...n, data: { ...(n.data ?? {}), ...patch } } : n))
      useCanvasStore.getState().setNodes(next, true)
    },
    [],
  )

  const updateSelectedEdge = useCallback(
    (patch: Partial<{ label?: string; weight?: number }>) => {
      const current = useCanvasStore.getState().edges
      const sel = current.find((e) => (e as any).selected)
      if (!sel) return
      const next = current.map((e) => (e.id === sel.id ? { ...e, data: { ...(e.data ?? {}), ...patch } } : e))
      useCanvasStore.getState().setEdges(next, true)
    },
    [],
  )

  const renderEdges = useMemo(() => {
    return edges.map((e) => {
      const label = e.data?.label
      const weight = e.data?.weight
      const strokeWidth = strokeWidthForWeight(weight)
      return {
        ...e,
        type: (e.type as any) ?? 'smoothstep',
        label: label || undefined,
        style: {
          ...(e.style ?? {}),
          strokeWidth,
          stroke: theme === 'light' ? 'rgba(15,23,42,0.6)' : 'rgba(255,255,255,0.75)',
        },
        labelStyle: {
          fontSize: 12,
          fill: theme === 'light' ? '#0f172a' : 'rgba(255,255,255,0.9)',
        },
        labelBgPadding: [6, 3] as any,
        labelBgBorderRadius: 6,
        labelBgStyle: {
          fill: theme === 'light' ? 'rgba(255,255,255,0.95)' : 'rgba(15,16,22,0.85)',
          stroke: theme === 'light' ? 'rgba(15,23,42,0.15)' : 'rgba(255,255,255,0.18)',
          strokeWidth: 1,
        },
      }
    })
  }, [edges, theme])

  const tidyEdges = useCallback(() => {
    const byId = new Map(useCanvasStore.getState().nodes.map((n) => [n.id, n]))
    const next = useCanvasStore.getState().edges.map((e) => {
      const s = byId.get(e.source)
      const t = byId.get(e.target)
      if (!s || !t) return e
      const sx = s.position.x
      const tx = t.position.x
      const preferRight = tx >= sx
      return {
        ...e,
        type: 'smoothstep',
        sourceHandle: preferRight ? 'right-source' : 'left-source',
        targetHandle: preferRight ? 'left-target' : 'right-target',
      }
    })
    useCanvasStore.getState().setEdges(next, true)
  }, [])

  // (reserved) selected edges can be used for future multi-edge operations

  const computeMenuPosition = useCallback(
    (clientX: number, clientY: number) => {
      const rect = wrapperRef.current?.getBoundingClientRect()
      // 预估菜单尺寸与边距，避免超出视口
      const margin = 12
      const maxWidth = 260
      const maxHeight = 380
      let x = clientX
      let y = clientY
      if (rect) {
        const leftLimit = rect.left + margin
        const topLimit = rect.top + margin
        const rightLimit = rect.left + rect.width - margin - maxWidth
        const bottomLimit = rect.top + rect.height - margin - maxHeight
        x = Math.min(Math.max(clientX, leftLimit), Math.max(leftLimit, rightLimit))
        y = Math.min(Math.max(clientY, topLimit), Math.max(topLimit, bottomLimit))
      }
      return { x, y }
    },
    [],
  )

  const alignSelected = useCallback(
    (
      mode:
        | 'left'
        | 'right'
        | 'top'
        | 'bottom'
        | 'h-center'
        | 'v-center'
        | 'grid'
        | 'grid-2'
        | 'grid-3'
        | 'grid-4'
        | 'horizontal-distribute'
        | 'vertical-distribute'
        | 'horizontal-line'
        | 'vertical-line'
        | 'circle'
        | 'arc'
        | 'diagonal',
    ) => {
    const current = useCanvasStore.getState().nodes
    const sel = current.filter((n) => n.selected)
    if (sel.length < 2) return

      if (mode === 'grid' || mode === 'grid-2' || mode === 'grid-3' || mode === 'grid-4') {
        // 矩阵排列：可以指定列数或自动计算
        const count = sel.length
        let cols: number
        if (mode === 'grid-2') cols = 2
        else if (mode === 'grid-3') cols = 3
        else if (mode === 'grid-4') cols = 4
        else cols = Math.ceil(Math.sqrt(count))

        // 计算所有节点的包围盒
    const xs = sel.map((n) => n.position.x)
    const ys = sel.map((n) => n.position.y)
        const widths = sel.map((n) => (n.style?.width as number) ?? 320)
        const heights = sel.map((n) => (n.style?.height as number) ?? 200)
    const minX = Math.min(...xs)
    const minY = Math.min(...ys)
        const maxWidth = Math.max(...widths)
        const maxHeight = Math.max(...heights)

        // 计算间距（节点之间的间距）
        const spacing = 40

        // 起始位置（以第一个节点的左上角为基准）
        const startX = minX
        const startY = minY

        const next = current.map((n) => {
          if (!n.selected) return n
          const nodeIdx = sel.findIndex((s) => s.id === n.id)
          const row = Math.floor(nodeIdx / cols)
          const col = nodeIdx % cols
          const x = startX + col * (maxWidth + spacing)
          const y = startY + row * (maxHeight + spacing)
          return { ...n, position: { x, y } }
        })
        useCanvasStore.getState().setNodes(next, true)
        return
      }

      if (mode === 'horizontal-distribute') {
        // 横向等间距：按 x 坐标排序，然后均匀分布（保持y坐标）
        const sorted = [...sel].sort((a, b) => a.position.x - b.position.x)
        const widths = sorted.map((n) => (n.style?.width as number) ?? 320)
        const leftEdges = sorted.map((n) => n.position.x)
        const rightEdges = sorted.map((n, idx) => n.position.x + widths[idx])
        const minLeft = Math.min(...leftEdges)
        const maxRight = Math.max(...rightEdges)
        const totalWidth = maxRight - minLeft
        const totalNodeWidth = widths.reduce((sum, w) => sum + w, 0)
        const spacing = sorted.length > 1 ? (totalWidth - totalNodeWidth) / (sorted.length - 1) : 0

        let currentX = minLeft
        const next = current.map((n) => {
          if (!n.selected) return n
          const idx = sorted.findIndex((s) => s.id === n.id)
          if (idx === 0) {
            currentX = minLeft
          } else {
            currentX += widths[idx - 1] + spacing
          }
          return { ...n, position: { ...n.position, x: currentX } }
        })
        useCanvasStore.getState().setNodes(next, true)
        return
      }

      if (mode === 'vertical-distribute') {
        // 纵向等间距：按 y 坐标排序，然后均匀分布（保持x坐标）
        const sorted = [...sel].sort((a, b) => a.position.y - b.position.y)
        const heights = sorted.map((n) => (n.style?.height as number) ?? 200)
        const topEdges = sorted.map((n) => n.position.y)
        const bottomEdges = sorted.map((n, i) => n.position.y + heights[i])
        const minTop = Math.min(...topEdges)
        const maxBottom = Math.max(...bottomEdges)
        const totalHeight = maxBottom - minTop
        const totalNodeHeight = heights.reduce((sum, h) => sum + h, 0)
        const spacing = sorted.length > 1 ? (totalHeight - totalNodeHeight) / (sorted.length - 1) : 0

        let currentY = minTop
        const next = current.map((n) => {
          if (!n.selected) return n
          const idx = sorted.findIndex((s) => s.id === n.id)
          if (idx === 0) {
            currentY = minTop
          } else {
            currentY += heights[idx - 1] + spacing
          }
          return { ...n, position: { ...n.position, y: currentY } }
        })
        useCanvasStore.getState().setNodes(next, true)
        return
      }

      if (mode === 'horizontal-line') {
        // 水平排列：所有节点排成一行，y坐标对齐到中心
        const sorted = [...sel].sort((a, b) => a.position.x - b.position.x)
        const widths = sorted.map((n) => (n.style?.width as number) ?? 320)
        const heights = sorted.map((n) => (n.style?.height as number) ?? 200)
        const ys = sorted.map((n) => n.position.y)
        const centerY = (Math.min(...ys) + Math.max(...ys.map((y, i) => y + heights[i]))) / 2
        const spacing = 40

        let currentX = sorted[0].position.x
        const next = current.map((n) => {
          if (!n.selected) return n
          const idx = sorted.findIndex((s) => s.id === n.id)
          if (idx === 0) {
            currentX = sorted[0].position.x
          } else {
            currentX += widths[idx - 1] + spacing
          }
          return { ...n, position: { x: currentX, y: centerY - heights[idx] / 2 } }
        })
        useCanvasStore.getState().setNodes(next, true)
        return
      }

      if (mode === 'vertical-line') {
        // 垂直排列：所有节点排成一列，x坐标对齐到中心
        const sorted = [...sel].sort((a, b) => a.position.y - b.position.y)
        const widths = sorted.map((n) => (n.style?.width as number) ?? 320)
        const heights = sorted.map((n) => (n.style?.height as number) ?? 200)
        const xs = sorted.map((n) => n.position.x)
        const centerX = (Math.min(...xs) + Math.max(...xs.map((x, i) => x + widths[i]))) / 2
        const spacing = 40

        let currentY = sorted[0].position.y
        const next = current.map((n) => {
          if (!n.selected) return n
          const idx = sorted.findIndex((s) => s.id === n.id)
          if (idx === 0) {
            currentY = sorted[0].position.y
          } else {
            currentY += heights[idx - 1] + spacing
          }
          return { ...n, position: { x: centerX - widths[idx] / 2, y: currentY } }
        })
        useCanvasStore.getState().setNodes(next, true)
        return
      }

      if (mode === 'circle') {
        // 圆形排列：节点排列成圆形
        const count = sel.length
        const xs = sel.map((n) => n.position.x)
        const ys = sel.map((n) => n.position.y)
        const widths = sel.map((n) => (n.style?.width as number) ?? 320)
        const heights = sel.map((n) => (n.style?.height as number) ?? 200)
        const centerX = (Math.min(...xs) + Math.max(...xs.map((x, i) => x + widths[i]))) / 2
        const centerY = (Math.min(...ys) + Math.max(...ys.map((y, i) => y + heights[i]))) / 2

        // 计算半径（基于节点包围盒）
        const maxWidth = Math.max(...widths)
        const maxHeight = Math.max(...heights)
        const radius = Math.max(maxWidth, maxHeight) * Math.max(1.2, count * 0.3)

        const next = current.map((n) => {
          if (!n.selected) return n
          const idx = sel.findIndex((s) => s.id === n.id)
          const angle = (idx / count) * Math.PI * 2
          const x = centerX + radius * Math.cos(angle) - widths[idx] / 2
          const y = centerY + radius * Math.sin(angle) - heights[idx] / 2
          return { ...n, position: { x, y } }
        })
        useCanvasStore.getState().setNodes(next, true)
        return
      }

      if (mode === 'arc') {
        // 弧形排列：节点排列成弧形（半圆）
        const count = sel.length
        const sorted = [...sel].sort((a, b) => a.position.x - b.position.x)
        const xs = sorted.map((n) => n.position.x)
        const ys = sorted.map((n) => n.position.y)
        const widths = sorted.map((n) => (n.style?.width as number) ?? 320)
        const heights = sorted.map((n) => (n.style?.height as number) ?? 200)
        const centerX = (Math.min(...xs) + Math.max(...xs.map((x, i) => x + widths[i]))) / 2
        const centerY = Math.max(...ys.map((y, i) => y + heights[i])) + 100

        const maxWidth = Math.max(...widths)
        const radius = maxWidth * Math.max(1.5, count * 0.4)

        const next = current.map((n) => {
          if (!n.selected) return n
          const idx = sorted.findIndex((s) => s.id === n.id)
          const angle = (idx / (count - 1 || 1)) * Math.PI - Math.PI / 2 // 从 -90° 到 90°
          const x = centerX + radius * Math.cos(angle) - widths[idx] / 2
          const y = centerY + radius * Math.sin(angle) - heights[idx] / 2
          return { ...n, position: { x, y } }
        })
        useCanvasStore.getState().setNodes(next, true)
        return
      }

      if (mode === 'diagonal') {
        // 对角线排列：节点沿对角线排列
        const sorted = [...sel].sort((a, b) => a.position.x - b.position.x)
        const xs = sorted.map((n) => n.position.x)
        const ys = sorted.map((n) => n.position.y)
        const widths = sorted.map((n) => (n.style?.width as number) ?? 320)
        const heights = sorted.map((n) => (n.style?.height as number) ?? 200)
        const startX = Math.min(...xs)
        const startY = Math.min(...ys)
        const endX = Math.max(...xs.map((x, i) => x + widths[i]))
        const endY = Math.max(...ys.map((y, i) => y + heights[i]))
        const dx = endX - startX
        const dy = endY - startY

        const next = current.map((n) => {
          if (!n.selected) return n
          const idx = sorted.findIndex((s) => s.id === n.id)
          const t = sorted.length > 1 ? idx / (sorted.length - 1) : 0
          const x = startX + dx * t - widths[idx] / 2
          const y = startY + dy * t - heights[idx] / 2
          return { ...n, position: { x, y } }
        })
        useCanvasStore.getState().setNodes(next, true)
        return
      }

      // 原有的对齐逻辑
      const xs = sel.map((n) => n.position.x)
      const ys = sel.map((n) => n.position.y)
      const widths = sel.map((n) => (n.style?.width as number) ?? 320)
      const heights = sel.map((n) => (n.style?.height as number) ?? 200)
      const minX = Math.min(...xs)
      const maxX = Math.max(...xs.map((x, i) => x + widths[i]))
      const minY = Math.min(...ys)
      const maxY = Math.max(...ys.map((y, i) => y + heights[i]))
    const cx = (minX + maxX) / 2
    const cy = (minY + maxY) / 2

    const next = current.map((n) => {
      if (!n.selected) return n
        const w = (n.style?.width as number) ?? 320
        const h = (n.style?.height as number) ?? 200
      if (mode === 'left') return { ...n, position: { ...n.position, x: minX } }
        if (mode === 'right') return { ...n, position: { ...n.position, x: maxX - w } }
      if (mode === 'top') return { ...n, position: { ...n.position, y: minY } }
        if (mode === 'bottom') return { ...n, position: { ...n.position, y: maxY - h } }
        if (mode === 'h-center') return { ...n, position: { ...n.position, x: cx - w / 2 } }
        return { ...n, position: { ...n.position, y: cy - h / 2 } }
    })
    useCanvasStore.getState().setNodes(next, true)
    },
    [],
  )


  useEffect(() => {
    // Debounced autosave to last saved JSON path, if available.
    if (!lastSavePath || !window.canvasApi?.writeTextFile) return
    if (autoSaveTimerRef.current) window.clearTimeout(autoSaveTimerRef.current)
    autoSaveTimerRef.current = window.setTimeout(async () => {
      try {
        const state = useCanvasStore.getState()
        const doc = exportToDoc(state.nodes, state.edges, lastSavePath, state.viewport)
        await window.canvasApi!.writeTextFile(lastSavePath, JSON.stringify(doc, null, 2))
      } catch {
        // ignore autosave errors
      }
    }, 900)
    return () => {
      if (autoSaveTimerRef.current) window.clearTimeout(autoSaveTimerRef.current)
    }
  }, [nodes, edges, lastSavePath])

  return (
    <div
      ref={wrapperRef}
      style={{ width: '100%', height: '100%', background: palette.canvasBg, position: 'relative' }}
      tabIndex={0}
      onPointerDown={(e) => {
        // 只处理在 canvas 区域内的点击，不拦截其他区域
        if (e.target === wrapperRef.current || wrapperRef.current?.contains(e.target as Node)) {
          wrapperRef.current?.focus()
        }
      }}
      onKeyDown={handleKeyDown}
    >
      {(() => {
        const expanded = toolbarHover || settingsOpen
        return (
          <div
            style={{
              position: 'absolute',
              zIndex: 10,
              top: 10,
              left: 10,
              right: 'calc(35% + 20px)', // 为右侧 LaTeX panel 留出空间
              display: 'flex',
              gap: expanded ? 8 : 6,
              padding: expanded ? 10 : 6,
              borderRadius: 12,
              border: `1px solid ${palette.border}`,
              background: palette.surface,
              backdropFilter: 'blur(10px)',
              flexWrap: expanded ? 'wrap' : 'nowrap',
              color: palette.textPrimary,
              maxHeight: expanded ? 220 : 36,
              overflow: 'hidden',
              transition: 'all 0.25s ease',
              boxShadow: expanded ? '0 12px 30px rgba(0,0,0,0.45)' : '0 8px 20px rgba(0,0,0,0.35)',
            }}
            onMouseEnter={() => setToolbarHover(true)}
            onMouseLeave={() => setToolbarHover(false)}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, minWidth: expanded ? undefined : 120 }}>
              <div style={{ fontWeight: 700, fontSize: 13, whiteSpace: 'nowrap' }}>工具栏</div>
              {!expanded && <div style={{ fontSize: 12, opacity: 0.7 }}>悬停展开</div>}
            </div>

            {expanded && (
              <>
                <button onClick={() => addNodeOfKind('text')}>+ Text</button>
                <button onClick={() => addNodeOfKind('markdown')}>+ Markdown</button>
                <button onClick={() => addNodeOfKind('code')}>+ Code</button>
                <button onClick={() => addNodeOfKind('image')}>+ Image</button>
                <button onClick={() => addNodeOfKind('pdf')}>+ PDF</button>
                <button onClick={() => addNodeOfKind('iframe')}>+ Iframe</button>
                <div style={{ marginLeft: 6, fontSize: 12, opacity: 0.7, userSelect: 'none' }}>Nodes: {nodes.length}</div>
                <input
                  placeholder="搜索标题或内容…"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  style={{
                    marginLeft: 8,
                    minWidth: 220,
                    background: theme === 'light' ? 'rgba(15,23,42,0.04)' : 'rgba(255,255,255,0.06)',
                    border: `1px solid ${palette.border}`,
                    borderRadius: 8,
                    color: palette.textPrimary,
                    padding: '4px 8px',
                    fontSize: 12,
                    outline: 'none',
                  }}
                />
                <select
                  value={kindFilter}
                  onChange={(e) => setKindFilter(e.target.value as any)}
                  style={{
                    background: theme === 'light' ? 'rgba(15,23,42,0.04)' : 'rgba(255,255,255,0.06)',
                    border: `1px solid ${palette.border}`,
                    borderRadius: 8,
                    color: palette.textPrimary,
                    padding: '4px 8px',
                    fontSize: 12,
                    outline: 'none',
                  }}
                >
                  <option value="all" style={{ background: theme === 'light' ? '#ffffff' : '#0f1016', color: palette.textPrimary }}>All</option>
                  <option value="text" style={{ background: theme === 'light' ? '#ffffff' : '#0f1016', color: palette.textPrimary }}>Text</option>
                  <option value="markdown" style={{ background: theme === 'light' ? '#ffffff' : '#0f1016', color: palette.textPrimary }}>Markdown</option>
                  <option value="code" style={{ background: theme === 'light' ? '#ffffff' : '#0f1016', color: palette.textPrimary }}>Code</option>
                  <option value="image" style={{ background: theme === 'light' ? '#ffffff' : '#0f1016', color: palette.textPrimary }}>Image</option>
                  <option value="pdf" style={{ background: theme === 'light' ? '#ffffff' : '#0f1016', color: palette.textPrimary }}>PDF</option>
                  <option value="iframe" style={{ background: theme === 'light' ? '#ffffff' : '#0f1016', color: palette.textPrimary }}>Iframe</option>
                </select>
                <span style={{ width: 1, alignSelf: 'stretch', background: palette.border }} />
                <button onClick={globalUndo}>Undo</button>
                <button onClick={globalRedo}>Redo</button>
                <button
                  onClick={() => setSettingsOpen(true)}
                  style={{
                    marginLeft: 4,
                    padding: '6px 10px',
                    borderRadius: 8,
                    border: `1px solid ${palette.border}`,
                    background: 'transparent',
                    color: palette.textPrimary,
                  }}
                >
                  Settings
                </button>
                <span style={{ width: 1, alignSelf: 'stretch', background: palette.border }} />
              </>
            )}

            {expanded && (
              <>
                <button
                  disabled={exporting}
                  onClick={async () => {
                    if (!window.canvasApi) return
                    const allNodes = useCanvasStore.getState().nodes
                    if (allNodes.length === 0) return

                    const el = wrapperRef.current?.querySelector('.react-flow') as HTMLElement | null
                    if (!el) return

                    const path = await window.canvasApi.saveFileDialog({
                      title: 'Export PNG',
                      filters: [{ name: 'PNG', extensions: ['png'] }],
                    })
                    if (!path) return

                    setExporting(true)
                    try {
                      // 先记录当前视图，再缩放到“包含全部节点的视图”
                      const prevViewport = rf.getViewport()
                      try {
                        rf.fitView({ nodes: allNodes as any, padding: 0.24, minZoom: 0.05, maxZoom: 2 })
                      } catch {
                        // ignore
                      }
                      // 等待一帧让 ReactFlow 完成渲染
                      await new Promise((resolve) => requestAnimationFrame(() => resolve(null)))

                      const dataUrl = await htmlToImage.toPng(el, { pixelRatio: 2, backgroundColor: palette.canvasBg })
                      const b64 = dataUrl.split(',')[1] ?? ''
                      await window.canvasApi.writeBase64File(path, b64)

                      // 导出后恢复原来的视图
                      try {
                        rf.setViewport(prevViewport)
                      } catch {
                        // ignore
                      }
                    } finally {
                      setExporting(false)
                    }
                  }}
                >
                  Export PNG
                </button>
                <button
          disabled={exporting}
          onClick={async () => {
            if (!window.canvasApi) return
            const allNodes = useCanvasStore.getState().nodes
            if (allNodes.length === 0) return

            const el = wrapperRef.current?.querySelector('.react-flow') as HTMLElement | null
            if (!el) return

            const path = await window.canvasApi.saveFileDialog({
              title: 'Export SVG',
              filters: [{ name: 'SVG', extensions: ['svg'] }],
            })
            if (!path) return
            setExporting(true)
            try {
              const prevViewport = rf.getViewport()
              try {
                rf.fitView({ nodes: allNodes as any, padding: 0.24, minZoom: 0.05, maxZoom: 2 })
              } catch {
                // ignore
              }
              await new Promise((resolve) => requestAnimationFrame(() => resolve(null)))

              const dataUrl = await htmlToImage.toSvg(el, { backgroundColor: palette.canvasBg })
              const comma = dataUrl.indexOf(',')
              const svgText = comma >= 0 ? decodeURIComponent(dataUrl.slice(comma + 1)) : dataUrl
              await window.canvasApi.writeTextFile(path, svgText)

              try {
                rf.setViewport(prevViewport)
              } catch {
                // ignore
              }
            } finally {
              setExporting(false)
            }
          }}
                >
                  Export SVG
                </button>
                <button
          onClick={async () => {
            if (!window.canvasApi) return
            const path = await window.canvasApi.saveFileDialog({
              title: 'Save canvas JSON',
              filters: [{ name: 'Canvas JSON', extensions: ['json'] }],
            })
            if (!path) return
            const doc = exportToDoc(nodes, edges, path, rf.getViewport())
            await window.canvasApi.writeTextFile(path, JSON.stringify(doc, null, 2))
            setLastSavePath(path)
            localStorage.setItem('canvas:lastSavePath', path)
          }}
                >
                  Save JSON
                </button>
                <button
          onClick={async () => {
            if (!window.canvasApi) return
            const path = await window.canvasApi.openFileDialog({
              title: 'Open canvas JSON',
              filters: [{ name: 'Canvas JSON', extensions: ['json'] }],
            })
            if (!path) return
            const text = await window.canvasApi.readTextFile(path)
            try {
              const parsed = JSON.parse(text)
              if (parsed.schema !== 'canvas-doc-v1') return
              const { nodes: n, edges: e } = importFromDoc(parsed, path)
              setNodes(n, true)
              setEdges(e, true)
              if (parsed.viewport) {
                const vp = parsed.viewport as { x: number; y: number; zoom: number }
                setViewport(vp, false)
                try {
                  rf.setViewport(vp)
                } catch {
                  // ignore
                }
              }
            } catch {
              // ignore
            }
          }}
                >
                  Load JSON
                </button>
              </>
            )}
          </div>
        )
      })()}

      <ReactFlow
        nodes={visibleNodes}
        edges={renderEdges}
        nodeTypes={nodeTypes}
        onNodesChange={onNodesChange}
        onNodeDragStop={onNodeDragStop}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onEdgeDoubleClick={onEdgeDoubleClick}
        onPaneClick={() => {
          setPaneMenu(null)
          setNodeMenu(null)
        }}
        onPaneContextMenu={(evt) => {
          evt.preventDefault()
          // 如果有选中的节点，显示完整的对齐菜单；否则只显示整理排线
          const current = useCanvasStore.getState().nodes
          const hasSelected = current.some((n) => n.selected)
          const pos = computeMenuPosition(evt.clientX, evt.clientY)
          if (hasSelected) {
            setNodeMenu(pos)
            setPaneMenu(null)
          } else {
            setPaneMenu(pos)
            setNodeMenu(null)
          }
        }}
        onNodeContextMenu={(evt, node) => {
          evt.preventDefault()
          const current = useCanvasStore.getState().nodes
          const clickedNode = current.find((n) => n.id === node.id)
          // 如果点击的节点没有被选中，先选中它
          if (clickedNode && !clickedNode.selected) {
            const next = current.map((n) => ({ ...n, selected: n.id === node.id }))
            useCanvasStore.getState().setNodes(next, false)
          }
          const pos = computeMenuPosition(evt.clientX, evt.clientY)
          setNodeMenu(pos)
          setPaneMenu(null)
        }}
        panOnDrag={[1, 2]} // only right & middle mouse drag to pan
        selectionOnDrag
        elementsSelectable
        nodesDraggable
        // Multi-select: 直接拖拽框选；按住 Ctrl/Shift/Cmd 点击节点可添加/取消选择
        multiSelectionKeyCode={['Control', 'Shift', 'Meta']}
        snapToGrid
        snapGrid={[10, 10]}
        proOptions={{ hideAttribution: true }}
        onMoveEnd={(_event, viewport) => {
          setViewport(viewport, false)
        }}
      >
        <Background
          variant={BackgroundVariant.Dots}
          gap={16}
          size={1}
          color={theme === 'light' ? 'rgba(15,23,42,0.12)' : 'rgba(255,255,255,0.10)'}
        />
        {!exporting && (
          <MiniMap
            pannable
            zoomable
            style={{ background: theme === 'light' ? 'rgba(255,255,255,0.9)' : 'rgba(15,16,22,0.75)' }}
          />
        )}
        {!exporting && <Controls />}
      </ReactFlow>

      {/* Link List Panel */}
      <LinkListPanel />

      {/* Right-side inspector */}
      <div
        style={{
          position: 'absolute',
          right: 10,
          top: 70, // 避开左上角工具栏按钮区域
          width: 280,
          zIndex: 15,
          borderRadius: 12,
          border: `1px solid ${palette.border}`,
          background: palette.surface,
          backdropFilter: 'blur(10px)',
          padding: 10,
          color: palette.textPrimary,
          fontSize: 12,
        }}
      >
        <div style={{ fontWeight: 700, marginBottom: 8 }}>Inspector</div>

        {selectedNode ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <div style={{ opacity: 0.75 }}>Node: {(selectedNode.data as any)?.kind}</div>
            <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              <span>Title</span>
              <input
                value={(selectedNode.data as any)?.title ?? ''}
                onChange={(e) => updateSelectedNode({ title: e.target.value })}
                style={{
                  background: theme === 'light' ? 'rgba(15,23,42,0.04)' : 'rgba(255,255,255,0.06)',
                  border: `1px solid ${palette.border}`,
                  borderRadius: 8,
                  color: palette.textPrimary,
                  padding: '4px 8px',
                  outline: 'none',
                }}
              />
            </label>
            <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              <span>Color</span>
              <input
                type="color"
                value={(selectedNode.data as any)?.color ?? '#ffffff'}
                onChange={(e) => updateSelectedNode({ color: e.target.value })}
                style={{ width: 60, height: 28, padding: 0, border: 'none', background: 'transparent' }}
              />
            </label>
            {(selectedNode.data as any)?.filePath ? (
              <div style={{ opacity: 0.8, wordBreak: 'break-all' }}>
                <div style={{ marginBottom: 4 }}>File</div>
                <div>{(selectedNode.data as any)?.filePath}</div>
              </div>
            ) : null}
            {(selectedNode.data as any)?.url ? (
              <div style={{ opacity: 0.8, wordBreak: 'break-all' }}>
                <div style={{ marginBottom: 4 }}>URL</div>
                <div>{(selectedNode.data as any)?.url}</div>
              </div>
            ) : null}
            <button
              onClick={() => {
                const nextNodes = useCanvasStore.getState().nodes.filter((n) => !n.selected)
                const nextEdges = useCanvasStore
                  .getState()
                  .edges.filter((e) => e.source !== selectedNode.id && e.target !== selectedNode.id)

                // 从所有层级中移除被删除的节点
                const { layers } = useLayerStore.getState()
                const deletedNodeIds = [selectedNode.id]
                layers.forEach(layer => {
                  const remainingNodeIds = layer.nodeIds.filter(id => !deletedNodeIds.includes(id))
                  if (remainingNodeIds.length !== layer.nodeIds.length) {
                    useLayerStore.getState().removeNodesFromLayer(layer.id, deletedNodeIds)
                  }
                })

                useCanvasStore.getState().setNodes(nextNodes, true)
                useCanvasStore.getState().setEdges(nextEdges, true)
              }}
              style={{ background: '#ef4444', border: 'none', color: 'white', borderRadius: 8, padding: '6px 8px' }}
            >
              Delete node
            </button>
          </div>
        ) : selectedEdge ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            <div style={{ opacity: 0.75 }}>Edge</div>
            <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              <span>Label</span>
              <input
                value={(selectedEdge.data as any)?.label ?? ''}
                onChange={(e) => updateSelectedEdge({ label: e.target.value || undefined })}
                style={{
                  background: theme === 'light' ? 'rgba(15,23,42,0.04)' : 'rgba(255,255,255,0.06)',
                  border: `1px solid ${palette.border}`,
                  borderRadius: 8,
                  color: palette.textPrimary,
                  padding: '4px 8px',
                  outline: 'none',
                }}
              />
            </label>
            <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              <span>Weight</span>
              <input
                value={(selectedEdge.data as any)?.weight ?? ''}
                onChange={(e) => updateSelectedEdge({ weight: e.target.value ? Number(e.target.value) : undefined })}
                style={{
                  background: theme === 'light' ? 'rgba(15,23,42,0.04)' : 'rgba(255,255,255,0.06)',
                  border: `1px solid ${palette.border}`,
                  borderRadius: 8,
                  color: palette.textPrimary,
                  padding: '4px 8px',
                  outline: 'none',
                }}
              />
            </label>
            <button
              onClick={() => {
                const nextEdges = useCanvasStore.getState().edges.filter((e) => !(e as any).selected)
                useCanvasStore.getState().setEdges(nextEdges, true)
              }}
              style={{ background: '#ef4444', border: 'none', color: 'white', borderRadius: 8, padding: '6px 8px' }}
            >
              Delete edge
            </button>
          </div>
        ) : (
          <div style={{ opacity: 0.75 }}>Select a node or edge…</div>
        )}
      </div>

      {paneMenu && (
        <div
          style={{
            position: 'absolute',
            left: paneMenu.x,
            top: paneMenu.y,
            zIndex: 60,
            background: palette.surface,
            border: `1px solid ${palette.border}`,
            borderRadius: 10,
            boxShadow: '0 18px 45px rgba(0,0,0,0.6)',
            padding: 6,
            minWidth: 180,
            maxHeight: 260,
            overflowY: 'auto',
            color: palette.textPrimary,
          }}
          onMouseDown={(e) => e.stopPropagation()}
        >
          <button
            onClick={() => {
              tidyEdges()
              setPaneMenu(null)
            }}
            style={{
              width: '100%',
              textAlign: 'left',
              padding: '8px 10px',
              borderRadius: 8,
              border: '1px solid transparent',
              background: 'transparent',
              color: palette.textPrimary,
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = theme === 'light' ? 'rgba(15,23,42,0.05)' : 'rgba(255,255,255,0.08)'
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = 'transparent'
            }}
          >
            整理排线
          </button>
        </div>
      )}

      {nodeMenu && (
        <div
          style={{
            position: 'absolute',
            left: nodeMenu.x,
            top: nodeMenu.y,
            zIndex: 60,
            background: palette.surface,
            border: `1px solid ${palette.border}`,
            borderRadius: 10,
            boxShadow: '0 18px 45px rgba(0,0,0,0.6)',
            padding: 6,
            minWidth: 200,
            maxHeight: 420,
            overflowY: 'auto',
            color: palette.textPrimary,
          }}
          onMouseDown={(e) => e.stopPropagation()}
        >
          <div style={{ padding: '6px 10px', fontSize: 11, color: palette.textSecondary, fontWeight: 600 }}>
            对齐与排列
          </div>
          <div style={{ height: 1, background: palette.border, margin: '4px 0' }} />

          {/* 层级管理选项 */}
          {(() => {
            const hasSelectedNodes = selectedNodes.length > 0
            const activeLayer = layers.find(l => l.id === activeLayerId)
            const selectedNodeIds = selectedNodes.map(n => n.id)

            // 检查选中的节点是否都属于激活层级
            const allInActiveLayer = activeLayer && selectedNodeIds.every(id => activeLayer.nodeIds.includes(id))
            const someInActiveLayer = activeLayer && selectedNodeIds.some(id => activeLayer.nodeIds.includes(id))

            if (!hasSelectedNodes) return null

            return (
              <>
                <div style={{ padding: '4px 10px', fontSize: 10, color: palette.textSecondary, marginTop: 4 }}>
                  层级管理
                </div>

                {/* 创建层级 - 多个选中节点时显示 */}
                {selectedNodes.length >= 2 && (
                  <button
                    onClick={async () => {
                      const { useLayerStore } = await import('./layerStore')
                      useLayerStore.getState().createLayer(selectedNodes.map(n => n.id))
                      setNodeMenu(null)
                    }}
                    style={{
                      width: '100%',
                      textAlign: 'left',
                      padding: '8px 10px',
                      borderRadius: 8,
                      border: '1px solid transparent',
                      background: 'transparent',
                      color: palette.textPrimary,
                      fontSize: 13,
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.background = theme === 'light' ? 'rgba(15,23,42,0.05)' : 'rgba(255,255,255,0.08)'
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.background = 'transparent'
                    }}
                  >
                    创建层级
                  </button>
                )}

                {/* 从激活层级移除 - 当激活层级且选中节点都属于该层级时显示 */}
                {activeLayer && allInActiveLayer && (
                  <button
                    onClick={async () => {
                      const { useLayerStore } = await import('./layerStore')
                      useLayerStore.getState().removeNodesFromLayer(activeLayer.id, selectedNodeIds)
                      setNodeMenu(null)
                    }}
                    style={{
                      width: '100%',
                      textAlign: 'left',
                      padding: '8px 10px',
                      borderRadius: 8,
                      border: '1px solid transparent',
                      background: 'transparent',
                      color: palette.textPrimary,
                      fontSize: 13,
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.background = theme === 'light' ? 'rgba(15,23,42,0.05)' : 'rgba(255,255,255,0.08)'
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.background = 'transparent'
                    }}
                  >
                    从"{activeLayer.name}"移除
                  </button>
                )}

                {/* 添加到激活层级 - 当激活层级且选中节点不属于该层级时显示 */}
                {activeLayer && !someInActiveLayer && (
                  <button
                    onClick={async () => {
                      const { useLayerStore } = await import('./layerStore')
                      useLayerStore.getState().addNodesToLayer(activeLayer.id, selectedNodeIds)
                      setNodeMenu(null)
                    }}
                    style={{
                      width: '100%',
                      textAlign: 'left',
                      padding: '8px 10px',
                      borderRadius: 8,
                      border: '1px solid transparent',
                      background: 'transparent',
                      color: palette.textPrimary,
                      fontSize: 13,
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.background = theme === 'light' ? 'rgba(15,23,42,0.05)' : 'rgba(255,255,255,0.08)'
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.background = 'transparent'
                    }}
                  >
                    添加到"{activeLayer.name}"
                  </button>
                )}

                {/* 添加到现有层级 - 当没有激活层级或有节点不属于激活层级时显示 */}
                {((!activeLayer || !allInActiveLayer) && layers.length > 0) && (
                  <>
                    <div style={{ padding: '4px 10px', fontSize: 10, color: palette.textSecondary }}>
                      添加到层级:
                    </div>
                    {layers
                      .filter((layer) => {
                        const shouldSkipActiveLayerInList = activeLayer && !someInActiveLayer
                        if (shouldSkipActiveLayerInList && layer.id === activeLayer.id) return false
                        return true
                      })
                      .map((layer) => (
                      <button
                        key={layer.id}
                        onClick={async () => {
                          const { useLayerStore } = await import('./layerStore')
                          useLayerStore.getState().addNodesToLayer(layer.id, selectedNodeIds)
                          setNodeMenu(null)
                        }}
                        style={{
                          width: '100%',
                          textAlign: 'left',
                          padding: '6px 10px 6px 20px',
                          borderRadius: 6,
                          border: '1px solid transparent',
                          background: 'transparent',
                          color: palette.textPrimary,
                          fontSize: 12,
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.background = theme === 'light' ? 'rgba(15,23,42,0.05)' : 'rgba(255,255,255,0.06)'
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.background = 'transparent'
                        }}
                      >
                        + {layer.name}
                      </button>
                    ))}
                  </>
                )}

                <div style={{ height: 1, background: palette.border, margin: '4px 0' }} />
              </>
            )
          })()}

          {/* LaTeX 链接选项 */}
          {(() => {
            const latexStore = useLatexLinkStore.getState()
            const selection = latexStore.selection
            const source = latexStore.source
            const selectedNodeIds = selectedNodes.map((n) => n.id)
            const hasTextSelection = selection != null && selection.start !== selection.end
            const hasSelectedNodes = selectedNodeIds.length > 0

            if (!hasTextSelection && !hasSelectedNodes) {
              return null
            }

            // 检查是否存在链接
            let existingLink = null
            if (hasTextSelection && hasSelectedNodes) {
              existingLink = latexStore.getLinksForRange(selection!.start, selection!.end)
              if (existingLink) {
                // 检查是否所有选中的节点都在链接中
                const linkNodeIds = new Set(existingLink.nodeIds)
                const allLinked = selectedNodeIds.every((id) => linkNodeIds.has(id))
                if (!allLinked) {
                  existingLink = null // 部分链接，显示为创建/追加
                }
              }
            } else if (hasSelectedNodes) {
              // 只选中了节点，检查是否有链接
              const links = latexStore.getLinksForNodes(selectedNodeIds)
              if (links.length > 0) {
                existingLink = links[0] // 使用第一个链接
              }
            }

            return (
              <>
                <div style={{ height: 1, background: palette.border, margin: '8px 0' }} />
                <div style={{ padding: '4px 10px', fontSize: 10, color: palette.textSecondary, marginTop: 4 }}>
                  LaTeX 链接
                </div>
                {existingLink ? (
                  <button
                    onClick={() => {
                      if (hasTextSelection && hasSelectedNodes) {
                        latexStore.removeLink({ start: selection!.start, end: selection!.end }, selectedNodeIds)
                      } else if (hasSelectedNodes) {
                        latexStore.removeNodesFromLink(existingLink!.id, selectedNodeIds)
                      }
                      setNodeMenu(null)
                    }}
                    style={{
                      width: '100%',
                      textAlign: 'left',
                      padding: '8px 10px',
                      borderRadius: 8,
                      border: '1px solid transparent',
                      background: 'transparent',
                      color: 'rgba(239,68,68,0.9)',
                      fontSize: 13,
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.background = theme === 'light' ? 'rgba(15,23,42,0.05)' : 'rgba(255,255,255,0.08)'
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.background = 'transparent'
                    }}
                  >
                    {hasTextSelection && hasSelectedNodes ? '取消链接（当前选区 ↔ 选中节点）' : '从链接移除选中节点'}
                  </button>
                ) : hasTextSelection && hasSelectedNodes ? (
                  <button
                    onClick={() => {
                      const text = source.slice(selection!.start, selection!.end)
                      latexStore.createOrAppendLink({ start: selection!.start, end: selection!.end }, text, selectedNodeIds)
                      setNodeMenu(null)
                    }}
                    style={{
                      width: '100%',
                      textAlign: 'left',
                      padding: '8px 10px',
                      borderRadius: 8,
                      border: '1px solid transparent',
                      background: 'transparent',
                      color: 'rgba(96,165,250,0.9)',
                      fontSize: 13,
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.background = theme === 'light' ? 'rgba(15,23,42,0.05)' : 'rgba(255,255,255,0.08)'
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.background = 'transparent'
                    }}
                  >
                    创建/追加链接（当前选区 ↔ 选中节点）
                  </button>
                ) : (
                  <div style={{ padding: '8px 10px', fontSize: 11, color: palette.textSecondary }}>
                    {!hasTextSelection ? '请在 LaTeX 面板中选中文本' : '请先选中节点'}
                  </div>
                )}
              </>
            )
          })()}

          {selectedNodes.length >= 2 ? (
            <>
              <div style={{ padding: '4px 10px', fontSize: 10, color: palette.textSecondary, marginTop: 4 }}>
                对齐
              </div>
              <button
                onClick={() => {
                  alignSelected('left')
                  setNodeMenu(null)
                }}
                style={{
                  width: '100%',
                  textAlign: 'left',
                  padding: '8px 10px',
                  borderRadius: 8,
                  border: '1px solid transparent',
                  background: 'transparent',
                  color: palette.textPrimary,
                  fontSize: 13,
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = theme === 'light' ? 'rgba(15,23,42,0.05)' : 'rgba(255,255,255,0.08)'
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'transparent'
                }}
              >
                左对齐
              </button>
              <button
                onClick={() => {
                  alignSelected('right')
                  setNodeMenu(null)
                }}
                style={{
                  width: '100%',
                  textAlign: 'left',
                  padding: '8px 10px',
                  borderRadius: 8,
                  border: '1px solid transparent',
                  background: 'transparent',
                  color: palette.textPrimary,
                  fontSize: 13,
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = theme === 'light' ? 'rgba(15,23,42,0.05)' : 'rgba(255,255,255,0.08)'
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'transparent'
                }}
              >
                右对齐
              </button>
              <button
                onClick={() => {
                  alignSelected('top')
                  setNodeMenu(null)
                }}
                style={{
                  width: '100%',
                  textAlign: 'left',
                  padding: '8px 10px',
                  borderRadius: 8,
                  border: '1px solid transparent',
                  background: 'transparent',
                  color: palette.textPrimary,
                  fontSize: 13,
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = theme === 'light' ? 'rgba(15,23,42,0.05)' : 'rgba(255,255,255,0.08)'
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'transparent'
                }}
              >
                顶对齐
              </button>
              <button
                onClick={() => {
                  alignSelected('bottom')
                  setNodeMenu(null)
                }}
                style={{
                  width: '100%',
                  textAlign: 'left',
                  padding: '8px 10px',
                  borderRadius: 8,
                  border: '1px solid transparent',
                  background: 'transparent',
                  color: palette.textPrimary,
                  fontSize: 13,
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = theme === 'light' ? 'rgba(15,23,42,0.05)' : 'rgba(255,255,255,0.08)'
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'transparent'
                }}
              >
                底对齐
              </button>
              <button
                onClick={() => {
                  alignSelected('h-center')
                  setNodeMenu(null)
                }}
                style={{
                  width: '100%',
                  textAlign: 'left',
                  padding: '8px 10px',
                  borderRadius: 8,
                  border: '1px solid transparent',
                  background: 'transparent',
                  color: palette.textPrimary,
                  fontSize: 13,
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = theme === 'light' ? 'rgba(15,23,42,0.05)' : 'rgba(255,255,255,0.08)'
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'transparent'
                }}
              >
                水平居中
              </button>
              <button
                onClick={() => {
                  alignSelected('v-center')
                  setNodeMenu(null)
                }}
                style={{
                  width: '100%',
                  textAlign: 'left',
                  padding: '8px 10px',
                  borderRadius: 8,
                  border: '1px solid transparent',
                  background: 'transparent',
                  color: palette.textPrimary,
                  fontSize: 13,
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = theme === 'light' ? 'rgba(15,23,42,0.05)' : 'rgba(255,255,255,0.08)'
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'transparent'
                }}
              >
                垂直居中
              </button>

              <div style={{ height: 1, background: palette.border, margin: '6px 0' }} />
              <div style={{ padding: '4px 10px', fontSize: 10, color: palette.textSecondary }}>分布</div>
              <button
                onClick={() => {
                  alignSelected('horizontal-distribute')
                  setNodeMenu(null)
                }}
                style={{
                  width: '100%',
                  textAlign: 'left',
                  padding: '8px 10px',
                  borderRadius: 8,
                  border: '1px solid transparent',
                  background: 'transparent',
                  color: palette.textPrimary,
                  fontSize: 13,
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = theme === 'light' ? 'rgba(15,23,42,0.05)' : 'rgba(255,255,255,0.08)'
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'transparent'
                }}
              >
                横向等间距
              </button>
              <button
                onClick={() => {
                  alignSelected('vertical-distribute')
                  setNodeMenu(null)
                }}
                style={{
                  width: '100%',
                  textAlign: 'left',
                  padding: '8px 10px',
                  borderRadius: 8,
                  border: '1px solid transparent',
                  background: 'transparent',
                  color: palette.textPrimary,
                  fontSize: 13,
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = theme === 'light' ? 'rgba(15,23,42,0.05)' : 'rgba(255,255,255,0.08)'
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'transparent'
                }}
              >
                纵向等间距
              </button>

              <div style={{ height: 1, background: palette.border, margin: '6px 0' }} />
              <div style={{ padding: '4px 10px', fontSize: 10, color: palette.textSecondary }}>网格排列</div>
              <button
                onClick={() => {
                  alignSelected('grid')
                  setNodeMenu(null)
                }}
                style={{
                  width: '100%',
                  textAlign: 'left',
                  padding: '8px 10px',
                  borderRadius: 8,
                  border: '1px solid transparent',
                  background: 'transparent',
                  color: palette.textPrimary,
                  fontSize: 13,
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = theme === 'light' ? 'rgba(15,23,42,0.05)' : 'rgba(255,255,255,0.08)'
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'transparent'
                }}
              >
                自动网格
              </button>
              <button
                onClick={() => {
                  alignSelected('grid-2')
                  setNodeMenu(null)
                }}
                style={{
                  width: '100%',
                  textAlign: 'left',
                  padding: '8px 10px',
                  borderRadius: 8,
                  border: '1px solid transparent',
                  background: 'transparent',
                  color: palette.textPrimary,
                  fontSize: 13,
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = theme === 'light' ? 'rgba(15,23,42,0.05)' : 'rgba(255,255,255,0.08)'
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'transparent'
                }}
              >
                2 列网格
              </button>
              <button
                onClick={() => {
                  alignSelected('grid-3')
                  setNodeMenu(null)
                }}
                style={{
                  width: '100%',
                  textAlign: 'left',
                  padding: '8px 10px',
                  borderRadius: 8,
                  border: '1px solid transparent',
                  background: 'transparent',
                  color: palette.textPrimary,
                  fontSize: 13,
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = theme === 'light' ? 'rgba(15,23,42,0.05)' : 'rgba(255,255,255,0.08)'
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'transparent'
                }}
              >
                3 列网格
              </button>
              <button
                onClick={() => {
                  alignSelected('grid-4')
                  setNodeMenu(null)
                }}
                style={{
                  width: '100%',
                  textAlign: 'left',
                  padding: '8px 10px',
                  borderRadius: 8,
                  border: '1px solid transparent',
                  background: 'transparent',
                  color: palette.textPrimary,
                  fontSize: 13,
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = theme === 'light' ? 'rgba(15,23,42,0.05)' : 'rgba(255,255,255,0.08)'
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'transparent'
                }}
              >
                4 列网格
              </button>

              <div style={{ height: 1, background: palette.border, margin: '6px 0' }} />
              <div style={{ padding: '4px 10px', fontSize: 10, color: palette.textSecondary }}>线性排列</div>
              <button
                onClick={() => {
                  alignSelected('horizontal-line')
                  setNodeMenu(null)
                }}
                style={{
                  width: '100%',
                  textAlign: 'left',
                  padding: '8px 10px',
                  borderRadius: 8,
                  border: '1px solid transparent',
                  background: 'transparent',
                  color: palette.textPrimary,
                  fontSize: 13,
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = theme === 'light' ? 'rgba(15,23,42,0.05)' : 'rgba(255,255,255,0.08)'
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'transparent'
                }}
              >
                水平一行
              </button>
              <button
                onClick={() => {
                  alignSelected('vertical-line')
                  setNodeMenu(null)
                }}
                style={{
                  width: '100%',
                  textAlign: 'left',
                  padding: '8px 10px',
                  borderRadius: 8,
                  border: '1px solid transparent',
                  background: 'transparent',
                  color: palette.textPrimary,
                  fontSize: 13,
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = theme === 'light' ? 'rgba(15,23,42,0.05)' : 'rgba(255,255,255,0.08)'
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'transparent'
                }}
              >
                垂直一列
              </button>
              <button
                onClick={() => {
                  alignSelected('diagonal')
                  setNodeMenu(null)
                }}
                style={{
                  width: '100%',
                  textAlign: 'left',
                  padding: '8px 10px',
                  borderRadius: 8,
                  border: '1px solid transparent',
                  background: 'transparent',
                  color: palette.textPrimary,
                  fontSize: 13,
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = theme === 'light' ? 'rgba(15,23,42,0.05)' : 'rgba(255,255,255,0.08)'
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'transparent'
                }}
              >
                对角线排列
              </button>

              <div style={{ height: 1, background: palette.border, margin: '6px 0' }} />
              <div style={{ padding: '4px 10px', fontSize: 10, color: palette.textSecondary }}>曲线排列</div>
              <button
                onClick={() => {
                  alignSelected('circle')
                  setNodeMenu(null)
                }}
                style={{
                  width: '100%',
                  textAlign: 'left',
                  padding: '8px 10px',
                  borderRadius: 8,
                  border: '1px solid transparent',
                  background: 'transparent',
                  color: palette.textPrimary,
                  fontSize: 13,
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = theme === 'light' ? 'rgba(15,23,42,0.05)' : 'rgba(255,255,255,0.08)'
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'transparent'
                }}
              >
                圆形排列
              </button>
              <button
                onClick={() => {
                  alignSelected('arc')
                  setNodeMenu(null)
                }}
                style={{
                  width: '100%',
                  textAlign: 'left',
                  padding: '8px 10px',
                  borderRadius: 8,
                  border: '1px solid transparent',
                  background: 'transparent',
                  color: palette.textPrimary,
                  fontSize: 13,
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = theme === 'light' ? 'rgba(15,23,42,0.05)' : 'rgba(255,255,255,0.08)'
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'transparent'
                }}
              >
                弧形排列
              </button>
            </>
          ) : (
            <div style={{ padding: '8px 10px', fontSize: 12, color: palette.textSecondary }}>
              {selectedNodes.length === 0
                ? '请先选择至少 2 个节点'
                : '选择至少 2 个节点以使用对齐功能'}
            </div>
          )}
        </div>
      )}

      {edgeEdit && (
        <div
          style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: 'rgba(0,0,0,0.35)',
            zIndex: 40,
          }}
          onMouseDown={(e) => e.stopPropagation()}
        >
          <div
            style={{
              minWidth: 260,
              padding: 12,
              borderRadius: 10,
              background: palette.surface,
              border: `1px solid ${palette.border}`,
              boxShadow: '0 18px 45px rgba(0,0,0,0.6)',
              display: 'flex',
              flexDirection: 'column',
              gap: 8,
              color: palette.textPrimary,
              fontSize: 13,
            }}
            onMouseDown={(e) => e.stopPropagation()}
          >
            <div style={{ fontWeight: 600, marginBottom: 4 }}>Edit edge</div>
            <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              <span>Label</span>
              <input
                autoFocus
                value={edgeEdit.label}
                onChange={(e) => setEdgeEdit((prev) => (prev ? { ...prev, label: e.target.value } : prev))}
                style={{
                  background: theme === 'light' ? 'rgba(15,23,42,0.04)' : 'rgba(255,255,255,0.05)',
                  border: `1px solid ${palette.border}`,
                  borderRadius: 6,
                  padding: '4px 6px',
                  color: palette.textPrimary,
                  outline: 'none',
                }}
              />
            </label>
            <label style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              <span>Weight (number, optional)</span>
              <input
                value={edgeEdit.weight}
                onChange={(e) => setEdgeEdit((prev) => (prev ? { ...prev, weight: e.target.value } : prev))}
                style={{
                  background: theme === 'light' ? 'rgba(15,23,42,0.04)' : 'rgba(255,255,255,0.05)',
                  border: `1px solid ${palette.border}`,
                  borderRadius: 6,
                  padding: '4px 6px',
                  color: palette.textPrimary,
                  outline: 'none',
                }}
              />
            </label>
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 6, marginTop: 4 }}>
              <button
                onClick={() => setEdgeEdit(null)}
                style={{
                  padding: '4px 8px',
                  borderRadius: 6,
                  border: `1px solid ${palette.border}`,
                  background: 'transparent',
                  color: palette.textSecondary,
                }}
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  if (!edgeEdit) return
                  const weightNum = edgeEdit.weight.trim() ? Number(edgeEdit.weight.trim()) : undefined
                  const next = edges.map((e) =>
                    e.id === edgeEdit.id
                      ? {
                          ...e,
                          data: { ...(e.data ?? {}), label: edgeEdit.label || undefined, weight: weightNum },
                        }
                      : e,
                  )
                  setEdges(next, true)
                  setEdgeEdit(null)
                }}
                style={{
                  padding: '4px 10px',
                  borderRadius: 6,
                  border: 'none',
                  background: palette.accent,
                  color: 'white',
                }}
              >
                Save
              </button>
            </div>
          </div>
        </div>
      )}

      {settingsOpen && (
        <div
          style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: 'rgba(0,0,0,0.35)',
            zIndex: 50,
          }}
          onMouseDown={(e) => {
            // 点击背景层时关闭设置窗口
            if (e.target === e.currentTarget) {
              setSettingsOpen(false)
            }
          }}
        >
          <div
            style={{
              width: 360,
              padding: 14,
              borderRadius: 12,
              background: palette.surface,
              border: `1px solid ${palette.border}`,
              boxShadow: '0 18px 45px rgba(0,0,0,0.6)',
              color: palette.textPrimary,
              display: 'flex',
              flexDirection: 'column',
              gap: 10,
            }}
            onMouseDown={(e) => e.stopPropagation()}
          >
            <div style={{ fontWeight: 700, fontSize: 14 }}>设置</div>
            <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 12 }}>
              <span style={{ color: palette.textSecondary }}>主题</span>
              <select
                value={theme}
                onChange={(e) => setTheme(e.target.value as any)}
                style={{
                  padding: '6px 8px',
                  borderRadius: 8,
                  border: `1px solid ${palette.border}`,
                  background: palette.surface,
                  color: palette.textPrimary,
                  outline: 'none',
                }}
              >
                <option value="light" style={{ background: palette.surface, color: palette.textPrimary }}>
                  浅色
                </option>
                <option value="dark" style={{ background: palette.surface, color: palette.textPrimary }}>
                  深色
                </option>
              </select>
            </label>

            <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 12 }}>
              <span style={{ color: palette.textSecondary }}>撤销记录步数 (10-500)</span>
              <input
                type="number"
                min={10}
                max={500}
                value={historyLimit}
                onChange={(e) => setHistoryLimit(Number(e.target.value))}
                style={{
                  padding: '6px 8px',
                  borderRadius: 8,
                  border: `1px solid ${palette.border}`,
                  background: theme === 'light' ? 'rgba(15,23,42,0.04)' : 'rgba(255,255,255,0.05)',
                  color: palette.textPrimary,
                  outline: 'none',
                }}
              />
            </label>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 4 }}>
              <button
                onClick={() => setSettingsOpen(false)}
                style={{
                  padding: '6px 10px',
                  borderRadius: 8,
                  border: `1px solid ${palette.border}`,
                  background: 'transparent',
                  color: palette.textSecondary,
                }}
              >
                关闭
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Layer Panel */}
      <LayerPanel />
    </div>
  )
}

export function CanvasView() {
  return (
    <ReactFlowProvider>
      <CanvasInner />
    </ReactFlowProvider>
  )
}


