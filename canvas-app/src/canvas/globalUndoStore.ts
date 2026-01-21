import { create } from 'zustand'
import { useSettingsStore } from './settingsStore'

type Operation = {
  id: string
  timestamp: number
  type: 'canvas' | 'latex'
  action: () => void // 执行操作的函数
  undo: () => void   // 撤销操作的函数
}

type GlobalUndoStore = {
  operations: Operation[]
  currentIndex: number // 当前操作索引，-1表示初始状态
  isUndoing: boolean

  // 执行操作（会添加到历史记录）
  execute: (type: 'canvas' | 'latex', action: () => void, undo: () => void) => void

  // 撤销
  undo: () => void

  // 重做
  redo: () => void

  // 是否可以撤销
  canUndo: () => boolean

  // 是否可以重做
  canRedo: () => boolean

  // 重置（用于初始化或清除历史）
  reset: () => void
}

const defaultLimit = 100

function getHistoryLimit() {
  const limit = useSettingsStore.getState().historyLimit
  if (!Number.isFinite(limit) || limit <= 0) return defaultLimit
  return Math.min(500, Math.max(10, Math.round(limit)))
}

export const useGlobalUndoStore = create<GlobalUndoStore>((set, get) => ({
  operations: [],
  currentIndex: -1,
  isUndoing: false,

  execute: (type, action, undo) => {
    const { operations, currentIndex } = get()

    // 如果当前不在最新状态，先截断后续操作
    const newOperations = operations.slice(0, currentIndex + 1)

    // 添加新操作
    const newOperation: Operation = {
      id: `${type}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: Date.now(),
      type,
      action,
      undo
    }

    newOperations.push(newOperation)

    // 限制历史记录数量
    const maxOps = getHistoryLimit()
    if (newOperations.length > maxOps) {
      newOperations.splice(0, newOperations.length - maxOps)
    }

    set({
      operations: newOperations,
      currentIndex: newOperations.length - 1
    })

    // 执行操作
    action()
  },

  undo: () => {
    const { operations, currentIndex } = get()
    if (currentIndex < 0) return

    const operation = operations[currentIndex]
    set({ isUndoing: true, currentIndex: currentIndex - 1 })

    // 执行撤销操作
    operation.undo()

    set({ isUndoing: false })
  },

  redo: () => {
    const { operations, currentIndex } = get()
    if (currentIndex >= operations.length - 1) return

    const nextIndex = currentIndex + 1
    const operation = operations[nextIndex]
    set({ currentIndex: nextIndex })

    // 执行重做操作（重新执行action）
    operation.action()
  },

  canUndo: () => get().currentIndex >= 0,

  canRedo: () => get().currentIndex < get().operations.length - 1,

  reset: () => set({ operations: [], currentIndex: -1, isUndoing: false })
}))
