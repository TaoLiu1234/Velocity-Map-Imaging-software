import { CanvasView } from './canvas/CanvasView'
import { LatexPanel } from './canvas/LatexPanel'
import { useSettingsStore } from './canvas/settingsStore'
import { useEffect } from 'react'

export function App() {
  const { theme } = useSettingsStore()
  const palette =
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

  // Ensure any uncovered pixels (e.g. around the app container) match the current theme.
  useEffect(() => {
    document.documentElement.style.background = palette.canvasBg
    document.body.style.background = palette.canvasBg
  }, [palette.canvasBg])

  return (
    <div
      style={{
        display: 'flex',
        width: '100vw',
        height: '100vh',
        background: palette.canvasBg,
        color: palette.textPrimary,
      }}
    >
      <div style={{ flex: 2, minWidth: 0, height: '100%', position: 'relative', overflow: 'hidden' }}>
        <CanvasView />
      </div>
      <div
        style={{
          width: 1,
          background: palette.border,
          boxShadow: theme === 'light' ? 'none' : '0 0 0 1px rgba(15,23,42,1)',
          flexShrink: 0,
        }}
      />
      <div style={{ flex: 1.35, minWidth: 320, maxWidth: 640, height: '100%', position: 'relative', overflow: 'hidden' }}>
        <LatexPanel />
      </div>
    </div>
  )
}


