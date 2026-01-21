import { create } from 'zustand'

export type ThemeMode = 'dark' | 'light'

type SettingsState = {
  theme: ThemeMode
  historyLimit: number
  latexPackages: string[] // Custom LaTeX packages, e.g. ["geometry", "fancyhdr", "listings"]
  setTheme: (theme: ThemeMode) => void
  setHistoryLimit: (limit: number) => void
  setLatexPackages: (packages: string[]) => void
}

const clampHistoryLimit = (limit: number) => {
  if (!Number.isFinite(limit)) return 50
  return Math.min(500, Math.max(10, Math.round(limit)))
}

// Load latexPackages from localStorage
const loadLatexPackages = (): string[] => {
  try {
    const stored = localStorage.getItem('latex:packages')
    if (stored) {
      const parsed = JSON.parse(stored)
      return Array.isArray(parsed) ? parsed : []
    }
  } catch {
    // Ignore parse errors
  }
  return []
}

export const useSettingsStore = create<SettingsState>((set) => ({
  theme: 'dark',
  historyLimit: 100,
  latexPackages: loadLatexPackages(),
  setTheme: (theme) => set({ theme }),
  setHistoryLimit: (limit) => set({ historyLimit: clampHistoryLimit(limit) }),
  setLatexPackages: (packages) => {
    localStorage.setItem('latex:packages', JSON.stringify(packages))
    set({ latexPackages: packages })
  },
}))

