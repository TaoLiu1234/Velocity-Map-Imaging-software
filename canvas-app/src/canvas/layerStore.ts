import { create } from 'zustand'
import { useGlobalUndoStore } from './globalUndoStore'

export interface Layer {
  id: string
  name: string
  nodeIds: string[]
  color: string
  visible: boolean
  createdAt: number
  rules?: LayerRule[]
}

export type LayerRule = {
  id: string
  text: string
  done: boolean
}

interface LayerState {
  layers: Layer[]
  activeLayerId: string | null
  isPanelVisible: boolean
  nextLayerColorIndex: number

  // Actions
  createLayer: (nodeIds: string[], name?: string) => void
  renameLayer: (layerId: string, name: string) => void
  deleteLayer: (layerId: string) => void
  addNodesToLayer: (layerId: string, nodeIds: string[]) => void
  removeNodesFromLayer: (layerId: string, nodeIds: string[]) => void
  setLayerRules: (layerId: string, rules: LayerRule[]) => void
  setActiveLayer: (layerId: string | null) => void
  toggleLayerVisibility: (layerId: string) => void
  setPanelVisible: (visible: boolean) => void
  getLayerColor: () => string
}

const layerColors = [
  '#3b82f6', // blue
  '#ef4444', // red
  '#10b981', // emerald
  '#f59e0b', // amber
  '#8b5cf6', // violet
  '#06b6d4', // cyan
  '#84cc16', // lime
  '#f97316', // orange
  '#ec4899', // pink
  '#6366f1', // indigo
]

export const useLayerStore = create<LayerState>((set, get) => ({
  layers: [],
  activeLayerId: null,
  isPanelVisible: false,
  nextLayerColorIndex: 0,

  createLayer: (nodeIds, name) => {
    const state = get()
    const layerId = `layer-${Date.now()}-${Math.random().toString(16).slice(2)}`
    const layerName = name || `Layer ${state.layers.length + 1}`
    const color = state.getLayerColor()

    const newLayer: Layer = {
      id: layerId,
      name: layerName,
      nodeIds: [...nodeIds],
      color,
      visible: true,
      createdAt: Date.now(),
      rules: [],
    }

    useGlobalUndoStore.getState().execute(
      'canvas',
      () => set((s) => ({
        layers: [...s.layers, newLayer],
        nextLayerColorIndex: (s.nextLayerColorIndex + 1) % layerColors.length,
      })),
      () => set((s) => ({
        layers: s.layers.filter(l => l.id !== layerId),
        nextLayerColorIndex: state.nextLayerColorIndex,
      }))
    )
  },

  renameLayer: (layerId, name) => {
    const state = get()
    const layer = state.layers.find(l => l.id === layerId)
    if (!layer) return

    const oldName = layer.name

    useGlobalUndoStore.getState().execute(
      'canvas',
      () => set((s) => ({
        layers: s.layers.map(l =>
          l.id === layerId ? { ...l, name } : l
        ),
      })),
      () => set((s) => ({
        layers: s.layers.map(l =>
          l.id === layerId ? { ...l, name: oldName } : l
        ),
      }))
    )
  },

  deleteLayer: (layerId) => {
    const state = get()
    const layerToDelete = state.layers.find(l => l.id === layerId)
    if (!layerToDelete) return

    const currentActiveId = state.activeLayerId

    useGlobalUndoStore.getState().execute(
      'canvas',
      () => set((s) => ({
        layers: s.layers.filter(l => l.id !== layerId),
        activeLayerId: s.activeLayerId === layerId ? null : s.activeLayerId,
      })),
      () => set((s) => ({
        layers: [...s.layers, layerToDelete],
        activeLayerId: currentActiveId,
      }))
    )
  },

  addNodesToLayer: (layerId, nodeIds) => {
    const state = get()
    const layer = state.layers.find(l => l.id === layerId)
    if (!layer) return

    const oldNodeIds = [...layer.nodeIds]
    const newNodeIds = [...new Set([...layer.nodeIds, ...nodeIds])]

    if (newNodeIds.length === oldNodeIds.length) return // No change

    useGlobalUndoStore.getState().execute(
      'canvas',
      () => set((s) => ({
        layers: s.layers.map(l =>
          l.id === layerId ? { ...l, nodeIds: newNodeIds } : l
        ),
      })),
      () => set((s) => ({
        layers: s.layers.map(l =>
          l.id === layerId ? { ...l, nodeIds: oldNodeIds } : l
        ),
      }))
    )
  },

  removeNodesFromLayer: (layerId, nodeIds) => {
    const state = get()
    const layer = state.layers.find(l => l.id === layerId)
    if (!layer) return

    const oldNodeIds = [...layer.nodeIds]
    const newNodeIds = layer.nodeIds.filter(id => !nodeIds.includes(id))

    if (newNodeIds.length === oldNodeIds.length) return // No change

    useGlobalUndoStore.getState().execute(
      'canvas',
      () => set((s) => ({
        layers: s.layers.map(l =>
          l.id === layerId ? { ...l, nodeIds: newNodeIds } : l
        ),
      })),
      () => set((s) => ({
        layers: s.layers.map(l =>
          l.id === layerId ? { ...l, nodeIds: oldNodeIds } : l
        ),
      }))
    )
  },

  setActiveLayer: (layerId) => {
    set({ activeLayerId: layerId })
  },

  setLayerRules: (layerId, rules) => {
    set((s) => ({
      layers: s.layers.map((l) => (l.id === layerId ? { ...l, rules: [...rules] } : l)),
    }))
  },

  toggleLayerVisibility: (layerId) => {
    set((s) => ({
      layers: s.layers.map(l =>
        l.id === layerId ? { ...l, visible: !l.visible } : l
      ),
    }))
  },

  setPanelVisible: (visible) => {
    set({ isPanelVisible: visible })
  },

  getLayerColor: () => {
    const state = get()
    return layerColors[state.nextLayerColorIndex]
  },
}))
