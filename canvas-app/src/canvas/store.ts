import { create } from 'zustand'
import type { Edge, Node, Viewport } from 'reactflow'
import type { CanvasNodeData, EdgeData } from './types'
import { useGlobalUndoStore } from './globalUndoStore'
import { useSettingsStore } from './settingsStore'

export type RFNode = Node<CanvasNodeData>
export type RFEdge = Edge<EdgeData>

type Snapshot = {
  nodes: RFNode[]
  edges: RFEdge[]
  viewport?: Viewport
}

type CanvasState = Snapshot & {
  past: Snapshot[]
  future: Snapshot[]

  setAll: (snap: Snapshot, recordHistory?: boolean) => void
  setNodes: (nodes: RFNode[], recordHistory?: boolean) => void
  setEdges: (edges: RFEdge[], recordHistory?: boolean) => void
  setViewport: (viewport: Viewport, recordHistory?: boolean) => void

  undo: () => void
  redo: () => void
  canUndo: () => boolean
  canRedo: () => boolean
}

function getHistoryLimit() {
  const limit = useSettingsStore.getState().historyLimit
  if (!Number.isFinite(limit) || limit <= 0) return 100
  return Math.min(500, Math.max(10, Math.round(limit)))
}

function pushHistory(state: CanvasState): CanvasState {
  const limit = getHistoryLimit()
  const snapshot: Snapshot = { nodes: state.nodes, edges: state.edges, viewport: state.viewport }
  const past = [...state.past, snapshot]
  const trimmedPast = past.length > limit ? past.slice(past.length - limit) : past
  return { ...state, past: trimmedPast, future: [] }
}

// 优化状态管理，减少不必要的重渲染
export const useCanvasStore = create<CanvasState>((set, get) => ({
  nodes: [],
  edges: [],
  viewport: undefined,
  past: [],
  future: [],

  setAll: (snap, recordHistory = true) =>
    set((state) => {
      const next = recordHistory ? pushHistory(state) : state
      return { ...next, ...snap }
    }),

  setNodes: (nodes, recordHistory = true) => {
    if (recordHistory && !useGlobalUndoStore.getState().isUndoing) {
      // 使用全局撤销系统
      const currentState = get()
      const currentNodes = currentState.nodes
      const currentEdges = currentState.edges
      const currentViewport = currentState.viewport

      useGlobalUndoStore.getState().execute(
        'canvas',
        () => set({ nodes }), // 执行操作
        () => set({ nodes: currentNodes, edges: currentEdges, viewport: currentViewport }) // 撤销操作
      )
    } else {
      set({ nodes })
    }
  },

  setEdges: (edges, recordHistory = true) => {
    if (recordHistory && !useGlobalUndoStore.getState().isUndoing) {
      // 使用全局撤销系统
      const currentState = get()
      const currentNodes = currentState.nodes
      const currentEdges = currentState.edges
      const currentViewport = currentState.viewport

      useGlobalUndoStore.getState().execute(
        'canvas',
        () => set({ edges }), // 执行操作
        () => set({ nodes: currentNodes, edges: currentEdges, viewport: currentViewport }) // 撤销操作
      )
    } else {
      set({ edges })
    }
  },

  setViewport: (viewport, recordHistory = true) => {
    if (recordHistory && !useGlobalUndoStore.getState().isUndoing) {
      // 使用全局撤销系统
      const currentState = get()
      const currentNodes = currentState.nodes
      const currentEdges = currentState.edges
      const currentViewport = currentState.viewport

      useGlobalUndoStore.getState().execute(
        'canvas',
        () => set({ viewport }), // 执行操作
        () => set({ nodes: currentNodes, edges: currentEdges, viewport: currentViewport }) // 撤销操作
      )
    } else {
      set({ viewport })
    }
  },

  undo: () =>
    set((state) => {
      if (state.past.length === 0) return state
      const previous = state.past[state.past.length - 1]
      const current: Snapshot = { nodes: state.nodes, edges: state.edges, viewport: state.viewport }
      const past = state.past.slice(0, -1)
      const limit = getHistoryLimit()
      const future = [current, ...state.future]
      const trimmedFuture = future.length > limit ? future.slice(0, limit) : future
      return { ...state, ...previous, past, future: trimmedFuture }
    }),

  redo: () =>
    set((state) => {
      if (state.future.length === 0) return state
      const nextSnap = state.future[0]
      const current: Snapshot = { nodes: state.nodes, edges: state.edges, viewport: state.viewport }
      const limit = getHistoryLimit()
      const future = state.future.slice(1)
      const past = [...state.past, current]
      const trimmedPast = past.length > limit ? past.slice(past.length - limit) : past
      const trimmedFuture = future.length > limit ? future.slice(0, limit) : future
      return { ...state, ...nextSnap, past: trimmedPast, future: trimmedFuture }
    }),

  canUndo: () => get().past.length > 0,
  canRedo: () => get().future.length > 0,

  // Layer-related helper functions
  getSelectedNodeIds: () => get().nodes.filter(n => n.selected).map(n => n.id),
  getSelectedNodes: () => get().nodes.filter(n => n.selected),
}))


