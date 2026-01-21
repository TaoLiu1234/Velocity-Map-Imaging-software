import { create } from 'zustand'
import { useGlobalUndoStore } from './globalUndoStore'

type LatexLink = {
  id: string
  range: { start: number; end: number }
  text: string
  nodeIds: string[]
}

type LinkSnapshot = {
  links: LatexLink[]
  source: string
}

type LatexLinkStore = {
  links: LatexLink[]
  source: string
  selection: { start: number; end: number } | null
  past: LinkSnapshot[]
  future: LinkSnapshot[]
  isRestoring: boolean // 标记是否正在执行撤销/恢复操作
  setLinks: (links: LatexLink[], recordHistory?: boolean) => void
  setSource: (source: string, recordHistory?: boolean) => void
  setSelection: (selection: { start: number; end: number } | null) => void
  createOrAppendLink: (range: { start: number; end: number }, text: string, nodeIds: string[]) => void
  removeLink: (range: { start: number; end: number }, nodeIds: string[]) => void
  getLinksForRange: (start: number, end: number) => LatexLink | null
  getLinksForNodes: (nodeIds: string[]) => LatexLink[]
  deleteLink: (linkId: string) => void
  removeNodesFromLink: (linkId: string, nodeIds: string[]) => void
  undo: () => void
  redo: () => void
  canUndo: () => boolean
  canRedo: () => boolean
}

const MAX_HISTORY = 100

function newId(prefix: string) {
  return `${prefix}_${Math.random().toString(16).slice(2)}`
}

function pushHistory(state: LatexLinkStore): LatexLinkStore {
  const snapshot: LinkSnapshot = { links: state.links, source: state.source }
  const past = [...state.past, snapshot].slice(-MAX_HISTORY)
  return { ...state, past, future: [] }
}

export const useLatexLinkStore = create<LatexLinkStore>((set, get) => ({
  links: [],
  source: '',
  selection: null,
  past: [],
  future: [],
  isRestoring: false,

  setLinks: (links, recordHistory = true) =>
    set((state) => {
      const next = recordHistory ? pushHistory(state) : state
      return { ...next, links }
    }),
  setSource: (source, recordHistory = true) => {
    if (recordHistory && !useGlobalUndoStore.getState().isUndoing) {
      // 使用全局撤销系统
      const currentState = get()
      const currentSource = currentState.source
      const currentLinks = currentState.links

      useGlobalUndoStore.getState().execute(
        'latex',
        () => set({ source }), // 执行操作
        () => set({ source: currentSource, links: currentLinks }) // 撤销操作
      )
    } else {
      set({ source })
    }
  },
  setSelection: (selection) => set({ selection }),

  createOrAppendLink: (range, text, nodeIds) => {
    if (useGlobalUndoStore.getState().isUndoing) {
      // 如果正在撤销，直接执行操作
      const state = get()
      const idx = state.links.findIndex((l) => l.range.start === range.start && l.range.end === range.end)
      if (idx >= 0) {
        const existing = state.links[idx]
        const merged = Array.from(new Set([...existing.nodeIds, ...nodeIds]))
        const next = [...state.links]
        next[idx] = { ...existing, nodeIds: merged, text }
        set({ links: next })
      } else {
        const id = newId('link')
        set({ links: [...state.links, { id, range, text, nodeIds }] })
      }
      return
    }

    // 使用全局撤销系统
    const currentState = get()
    const currentLinks = currentState.links
    const currentSource = currentState.source

    useGlobalUndoStore.getState().execute(
      'latex',
      () => {
        // 执行操作
        const state = get()
        const idx = state.links.findIndex((l) => l.range.start === range.start && l.range.end === range.end)
        if (idx >= 0) {
          const existing = state.links[idx]
          const merged = Array.from(new Set([...existing.nodeIds, ...nodeIds]))
          const next = [...state.links]
          next[idx] = { ...existing, nodeIds: merged, text }
          set({ links: next })
        } else {
          const id = newId('link')
          set({ links: [...state.links, { id, range, text, nodeIds }] })
        }
      },
      () => {
        // 撤销操作
        set({ links: currentLinks, source: currentSource })
      }
    )
  },

  removeLink: (range, nodeIds) => {
    if (useGlobalUndoStore.getState().isUndoing) {
      // 如果正在撤销，直接执行操作
      const state = get()
      const link = state.links.find((l) => l.range.start === range.start && l.range.end === range.end)
      if (!link) return

      if (nodeIds.length === 0) {
        set({ links: state.links.filter((l) => l.id !== link.id) })
      } else {
        const nodeIdSet = new Set(nodeIds)
        const updated = state.links
          .map((l) => {
            if (l.id !== link.id) return l
            const remaining = l.nodeIds.filter((id) => !nodeIdSet.has(id))
            return remaining.length > 0 ? { ...l, nodeIds: remaining } : null
          })
          .filter((l): l is LatexLink => l !== null)
        set({ links: updated })
      }
      return
    }

    // 使用全局撤销系统
    const currentState = get()
    const currentLinks = currentState.links
    const currentSource = currentState.source

    useGlobalUndoStore.getState().execute(
      'latex',
      () => {
        // 执行操作
        const state = get()
        const link = state.links.find((l) => l.range.start === range.start && l.range.end === range.end)
        if (!link) return

        if (nodeIds.length === 0) {
          set({ links: state.links.filter((l) => l.id !== link.id) })
        } else {
          const nodeIdSet = new Set(nodeIds)
          const updated = state.links
            .map((l) => {
              if (l.id !== link.id) return l
              const remaining = l.nodeIds.filter((id) => !nodeIdSet.has(id))
              return remaining.length > 0 ? { ...l, nodeIds: remaining } : null
            })
            .filter((l): l is LatexLink => l !== null)
          set({ links: updated })
        }
      },
      () => {
        // 撤销操作
        set({ links: currentLinks, source: currentSource })
      }
    )
  },

  getLinksForRange: (start, end) => {
    const { links } = get()
    return links.find((l) => l.range.start === start && l.range.end === end) || null
  },

  getLinksForNodes: (nodeIds) => {
    const { links } = get()
    const nodeIdSet = new Set(nodeIds)
    return links.filter((l) => l.nodeIds.some((id) => nodeIdSet.has(id)))
  },

  deleteLink: (linkId) => {
    if (useGlobalUndoStore.getState().isUndoing) {
      // 如果正在撤销，直接执行操作
      const state = get()
      set({ links: state.links.filter((l) => l.id !== linkId) })
      return
    }

    // 使用全局撤销系统
    const currentState = get()
    const currentLinks = currentState.links
    const currentSource = currentState.source

    useGlobalUndoStore.getState().execute(
      'latex',
      () => {
        // 执行操作
        const state = get()
        set({ links: state.links.filter((l) => l.id !== linkId) })
      },
      () => {
        // 撤销操作
        set({ links: currentLinks, source: currentSource })
      }
    )
  },

  removeNodesFromLink: (linkId, nodeIds) => {
    if (useGlobalUndoStore.getState().isUndoing) {
      // 如果正在撤销，直接执行操作
      const state = get()
      const nodeIdSet = new Set(nodeIds)
      const updated = state.links
        .map((l) => {
          if (l.id !== linkId) return l
          const remaining = l.nodeIds.filter((id) => !nodeIdSet.has(id))
          return remaining.length > 0 ? { ...l, nodeIds: remaining } : null
        })
        .filter((l): l is LatexLink => l !== null)
      set({ links: updated })
      return
    }

    // 使用全局撤销系统
    const currentState = get()
    const currentLinks = currentState.links
    const currentSource = currentState.source

    useGlobalUndoStore.getState().execute(
      'latex',
      () => {
        // 执行操作
        const state = get()
        const nodeIdSet = new Set(nodeIds)
        const updated = state.links
          .map((l) => {
            if (l.id !== linkId) return l
            const remaining = l.nodeIds.filter((id) => !nodeIdSet.has(id))
            return remaining.length > 0 ? { ...l, nodeIds: remaining } : null
          })
          .filter((l): l is LatexLink => l !== null)
        set({ links: updated })
      },
      () => {
        // 撤销操作
        set({ links: currentLinks, source: currentSource })
      }
    )
  },

  undo: () =>
    set((state) => {
      if (state.past.length === 0) return state
      const previous = state.past[state.past.length - 1]
      const current: LinkSnapshot = { links: state.links, source: state.source }
      const past = state.past.slice(0, -1)
      const future = [current, ...state.future].slice(0, MAX_HISTORY)
      return { ...state, ...previous, past, future, isRestoring: true }
    }),

  redo: () =>
    set((state) => {
      if (state.future.length === 0) return state
      const nextSnap = state.future[0]
      const current: LinkSnapshot = { links: state.links, source: state.source }
      const future = state.future.slice(1)
      const past = [...state.past, current].slice(-MAX_HISTORY)
      return { ...state, ...nextSnap, past, future, isRestoring: true }
    }),

  canUndo: () => get().past.length > 0,
  canRedo: () => get().future.length > 0,
}))

export type { LatexLink }

