import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useCanvasStore } from './store'
import { useLatexLinkStore, type LatexLink } from './latexLinkStore'
import { useLayerStore } from './layerStore'
import { useSettingsStore } from './settingsStore'
import katex from 'katex'
import 'katex/dist/katex.min.css'
// LaTeX 渲染函数（使用 KaTeX）
// 注意：KaTeX 只能用于"快速预览"，与 TeX 引擎（xelatex/pdflatex）排版可能存在差异。
function renderLatex(text: string, theme: 'light' | 'dark'): string {
  const escapeHtml = (str: string) => {
    const div = document.createElement('div')
    div.textContent = str
    return div.innerHTML
  }

  // 分割文本，保留公式标记
  const parts: Array<{ type: 'text' | 'inline' | 'block'; content: string }> = []
  let lastIndex = 0

  // 先找块级公式 $$...$$
  const blockRegex = /\$\$([^$]+)\$\$/g
  let blockMatch
  while ((blockMatch = blockRegex.exec(text)) !== null) {
    if (blockMatch.index > lastIndex) {
      parts.push({ type: 'text', content: text.slice(lastIndex, blockMatch.index) })
    }
    parts.push({ type: 'block', content: blockMatch[1] })
    lastIndex = blockMatch.index + blockMatch[0].length
  }
  if (lastIndex < text.length) {
    parts.push({ type: 'text', content: text.slice(lastIndex) })
  }

  // 如果没找到块级公式，处理整个文本
  if (parts.length === 0) {
    parts.push({ type: 'text', content: text })
  }

  // 渲染每个部分
  return parts
    .map((part) => {
      if (part.type === 'block') {
        try {
          const html = katex.renderToString(part.content.trim(), { displayMode: true, throwOnError: false })
          const bgColor = theme === 'light' ? 'rgba(79,70,229,0.1)' : 'rgba(59,130,246,0.15)'
          // 用 <span>（display:block）避免在 <span> 容器中插入 <div> 造成非法 DOM 结构与排版漂移
          return `<span class="latex-block" style="display:block; background: ${bgColor}; padding: 8px; border-radius: 4px; margin: 6px 0; text-align: center; overflow-x: auto;">${html}</span>`
        } catch {
          return escapeHtml(`$$${part.content}$$`)
        }
      }
      // 处理文本中的行内公式
      const inlineParts: Array<{ type: 'text' | 'inline'; content: string }> = []
      let lastIdx = 0
      const inlineRegex = /\$([^$]+)\$/g
      let inlineMatch
      while ((inlineMatch = inlineRegex.exec(part.content)) !== null) {
        if (inlineMatch.index > lastIdx) {
          inlineParts.push({ type: 'text', content: part.content.slice(lastIdx, inlineMatch.index) })
        }
        inlineParts.push({ type: 'inline', content: inlineMatch[1] })
        lastIdx = inlineMatch.index + inlineMatch[0].length
      }
      if (lastIdx < part.content.length) {
        inlineParts.push({ type: 'text', content: part.content.slice(lastIdx) })
      }

      if (inlineParts.length === 0) {
        inlineParts.push({ type: 'text', content: part.content })
      }

      return inlineParts
        .map((p) => {
          if (p.type === 'inline') {
            try {
              const html = katex.renderToString(p.content.trim(), { displayMode: false, throwOnError: false })
              const bgColor = theme === 'light' ? 'rgba(79,70,229,0.1)' : 'rgba(59,130,246,0.15)'
              return `<span class="latex-inline" style="background: ${bgColor}; padding: 2px 4px; border-radius: 3px;">${html}</span>`
            } catch {
              return escapeHtml(`$${p.content}$`)
            }
          }
          return escapeHtml(p.content).replace(/\n/g, '<br>')
        })
        .join('')
    })
    .join('')
}

export function LatexPanel() {
  const [localSource, setLocalSource] = useState<string>('%% LaTeX 文本示例\n\n行内公式: $E = mc^2$\n\n块级公式:\n\n$$\\int_{-\\infty}^{\\infty} e^{-x^2} dx = \\sqrt{\\pi}$$')
  const [selectionStart, setSelectionStart] = useState<number | null>(null)
  const [selectionEnd, setSelectionEnd] = useState<number | null>(null)

  const { theme, latexPackages, setLatexPackages } = useSettingsStore()
  const palette = useMemo(
    () =>
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
          },
    [theme],
  )

  const storeSource = useLatexLinkStore((s) => s.source)
  const setSource = useLatexLinkStore((s) => s.setSource)
  const setSelection = useLatexLinkStore((s) => s.setSelection)
  const source = storeSource || localSource

  // 初始化 store source
  useEffect(() => {
    if (!storeSource && localSource) {
      setSource(localSource, false) // 初始化时不记录历史
    }
  }, []) // 只在初始化时执行

  // 同步 localSource 变化到 store - 使用防抖避免频繁记录
  const [pendingSourceUpdate, setPendingSourceUpdate] = useState<string | null>(null)

  useEffect(() => {
    if (localSource !== storeSource) {
      setPendingSourceUpdate(localSource)
    }
  }, [localSource, storeSource])

  // 延迟记录历史，等待自动调整完成
  useEffect(() => {
    if (pendingSourceUpdate !== null) {
      const timer = setTimeout(() => {
        setSource(pendingSourceUpdate, true) // 用户编辑时记录历史
        setPendingSourceUpdate(null)
      }, 100) // 100ms 延迟，等待自动调整完成

      return () => clearTimeout(timer)
    }
  }, [pendingSourceUpdate, setSource])

  // 同步 selection 到 store
  useEffect(() => {
    if (selectionStart != null && selectionEnd != null) {
      setSelection({ start: selectionStart, end: selectionEnd })
    } else {
      setSelection(null)
    }
  }, [selectionStart, selectionEnd, setSelection])
  const [contextMenu, setContextMenu] = useState<{
    x: number
    y: number
    linkId: string | null
    hasSelection?: boolean
    hasSelectedNodes?: boolean
    nestedLinks?: Array<{ id: string; text: string; range: { start: number; end: number } }>
  } | null>(null)

  // 编辑区和预览区的大小比例（0-1之间）
  const [panelRatio, setPanelRatio] = useState(0.5)
  const [isDragging, setIsDragging] = useState(false)

  // 预览区的缩放和拖拽状态
  const [previewZoom, setPreviewZoom] = useState(1)
  const [previewPan, setPreviewPan] = useState({ x: 0, y: 0 })
  const [isPreviewDragging, setIsPreviewDragging] = useState(false)
  const [previewDragStart, setPreviewDragStart] = useState({ x: 0, y: 0 })
  const [exporting, setExporting] = useState(false)
  const [showPackageSettings, setShowPackageSettings] = useState(false)
  const [previewMode, setPreviewMode] = useState<'fast' | 'tex'>('fast')
  const [texPreviewDataUrl, setTexPreviewDataUrl] = useState<string | null>(null)
  const texPreviewTimerRef = useRef<number | null>(null)

  const links = useLatexLinkStore((s) => s.links)
  const setLinks = useLatexLinkStore((s) => s.setLinks)
  const createOrAppendLinkStore = useLatexLinkStore((s) => s.createOrAppendLink)
  const isRestoring = useLatexLinkStore((s) => s.isRestoring)
  const getLinksForRange = useLatexLinkStore((s) => s.getLinksForRange)

  const activeLayerId = useLayerStore((s) => s.activeLayerId)
  const layers = useLayerStore((s) => s.layers)

  const textareaRef = useRef<HTMLTextAreaElement | null>(null)
  const prevSourceRef = useRef<string>('') // set in effect below
  const contextMenuRef = useRef<HTMLDivElement | null>(null)
  const previewRef = useRef<HTMLDivElement | null>(null)
  const texPreviewIframeRef = useRef<HTMLIFrameElement | null>(null)

  const nodes = useCanvasStore((s) => s.nodes)
  const selectedNodes = useMemo(() => nodes.filter((n) => n.selected), [nodes])
  const activeNodeIds = useMemo(() => new Set(selectedNodes.map((n) => n.id)), [selectedNodes])
  const allNodeIds = useMemo(() => new Set(nodes.map((n) => n.id)), [nodes])

  // 节点删除：自动断链（删掉不存在的 nodeId；如果 link 空了就删）
  useEffect(() => {
    const currentLinks = useLatexLinkStore.getState().links
    const updated = currentLinks
      .map((l) => ({ ...l, nodeIds: l.nodeIds.filter((id) => allNodeIds.has(id)) }))
      .filter((l) => l.nodeIds.length > 0)
    if (updated.length !== currentLinks.length || currentLinks.some((l, i) => l.nodeIds.length !== updated[i]?.nodeIds.length)) {
      setLinks(updated)
    }
  }, [allNodeIds, setLinks])

  // 文本增删：用公共前缀/后缀的轻量 diff，让链接 range 自动漂移
  // 当链接的文本完全被删除或不匹配时取消链接，否则调整范围
  useEffect(() => {
    // 如果正在执行撤销/恢复操作，跳过自动调整
    if (isRestoring) return

    const prev = prevSourceRef.current
    const next = source
    if (prev === '') {
      prevSourceRef.current = next
      return
    }
    if (prev === next) return

    let prefix = 0
    while (prefix < prev.length && prefix < next.length && prev[prefix] === next[prefix]) prefix++

    let suffix = 0
    while (
      suffix < prev.length - prefix &&
      suffix < next.length - prefix &&
      prev[prev.length - 1 - suffix] === next[next.length - 1 - suffix]
    ) {
      suffix++
    }

    const prevMidLen = prev.length - prefix - suffix
    const nextMidLen = next.length - prefix - suffix
    const delta = nextMidLen - prevMidLen
    const changeStart = prefix
    const changeEndPrev = prefix + prevMidLen

    const currentLinks = useLatexLinkStore.getState().links
    const out: LatexLink[] = []
    for (const l of currentLinks) {
      let s = l.range.start
      let e = l.range.end

      // 计算调整后的范围
      let adjustedStart = s
      let adjustedEnd = e

      // 在变更区之后：整体平移
      if (s >= changeEndPrev) {
        adjustedStart = s + delta
        adjustedEnd = e + delta
      } else if (e <= changeStart) {
        // 在变更区之前：不变
        adjustedStart = s
        adjustedEnd = e
      } else {
        // 变更发生在 range 内或跨越 range
        if (s < changeStart && e > changeEndPrev) {
          // 变更在 range 中间：只调整 end
          adjustedEnd = e + delta
        } else if (s >= changeStart && e <= changeEndPrev) {
          // 变更完全覆盖 range：调整范围
          if (delta < 0) {
            // 删除：收缩范围
            const shrinkAmount = Math.min(e - s, -delta)
            adjustedEnd = Math.max(s, e - shrinkAmount)
          } else {
            // 插入：扩展范围
            adjustedEnd = e + delta
          }
        } else if (s < changeStart && e > changeStart && e <= changeEndPrev) {
          // 变更从 range 中间开始：调整 end
          adjustedEnd = Math.max(s, e + delta)
        } else if (s >= changeStart && s < changeEndPrev && e > changeEndPrev) {
          // 变更在 range 中间结束：调整 start 和 end
          adjustedStart = s + delta
          adjustedEnd = e + delta
        }
      }

      // clamp
      adjustedStart = Math.max(0, Math.min(adjustedStart, next.length))
      adjustedEnd = Math.max(0, Math.min(adjustedEnd, next.length))

      // 如果调整后的范围无效或为空，删除链接
      if (adjustedEnd <= adjustedStart) {
        continue
      }

      const adjustedText = next.slice(adjustedStart, adjustedEnd)

      // 如果调整后的文本为空或只包含空白字符，删除链接
      if (adjustedText.length === 0 || adjustedText.trim().length === 0) {
        continue
      }

      // 检查是否与原始链接文本完全不匹配（说明链接的文本被完全替换了）
      const originalText = l.text
      const currentText = next.slice(adjustedStart, adjustedEnd)

      // 如果调整后的文本与原始链接文本完全不同，且长度差异很大，可能被替换了
      if (originalText.length > 0 && currentText.length > 0 &&
          originalText !== currentText &&
          Math.abs(originalText.length - currentText.length) > Math.max(originalText.length, currentText.length) * 0.5) {
        // 文本被大幅修改，删除链接
        continue
      }

      out.push({ ...l, range: { start: adjustedStart, end: adjustedEnd }, text: adjustedText })
    }

    // 只在有变化时更新，不记录历史（历史记录由延迟的文本更新处理）
    if (out.length !== currentLinks.length || currentLinks.some((l, i) => l.range.start !== out[i]?.range.start || l.range.end !== out[i]?.range.end)) {
      setLinks(out, false) // false = 不记录历史，等待统一记录
    }

    prevSourceRef.current = next
  }, [source, setLinks, isRestoring])

  // 恢复完成后重置 isRestoring 标志
  useEffect(() => {
    if (isRestoring) {
      // 使用 setTimeout 确保状态更新完成后再重置标志
      setTimeout(() => {
        useLatexLinkStore.setState({ isRestoring: false })
      }, 0)
    }
  }, [isRestoring])

  const onTextSelect = useCallback((e: React.SyntheticEvent<HTMLTextAreaElement> | React.MouseEvent<HTMLTextAreaElement> | React.KeyboardEvent<HTMLTextAreaElement>) => {
    const el = e.currentTarget as HTMLTextAreaElement
    const start = el.selectionStart
    const end = el.selectionEnd
    setSelectionStart(start)
    setSelectionEnd(end)
  }, [])

  // 查找当前选区对应的链接
  const findLinkForCurrentSelection = useCallback(() => {
    if (selectionStart == null || selectionEnd == null) return null
    const start = Math.min(selectionStart, selectionEnd)
    const end = Math.max(selectionStart, selectionEnd)
    return getLinksForRange(start, end)
  }, [selectionStart, selectionEnd, getLinksForRange])

  const createOrAppendLink = useCallback(() => {
    if (selectionStart == null || selectionEnd == null) return
    if (selectionStart === selectionEnd) return
    if (selectedNodes.length === 0) return

    const start = Math.min(selectionStart, selectionEnd)
    const end = Math.max(selectionStart, selectionEnd)
    const text = source.slice(start, end)
    const nodeIds = selectedNodes.map((n) => n.id)

    createOrAppendLinkStore({ start, end }, text, nodeIds)
  }, [selectionStart, selectionEnd, selectedNodes, source, createOrAppendLinkStore])



  const deleteLink = useCallback(
    (linkId: string) => {
      useLatexLinkStore.getState().deleteLink(linkId)
      setContextMenu(null)
    },
    [],
  )


  const focusNodes = useCallback((ids: string[]) => {
    const current = useCanvasStore.getState().nodes
    const next = current.map((n) => ({ ...n, selected: ids.includes(n.id) }))
    useCanvasStore.getState().setNodes(next, true)
  }, [])

  // 拖拽分隔符的处理逻辑
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    setIsDragging(true)
    e.preventDefault()
  }, [])

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!isDragging) return

    // 获取 LaTeX panel 的容器
    const container = document.querySelector('[data-latex-panel]') as HTMLElement
    if (!container) return

    const rect = container.getBoundingClientRect()
    const newRatio = Math.max(0.2, Math.min(0.8, (e.clientY - rect.top) / rect.height))
    setPanelRatio(newRatio)
  }, [isDragging])

  const handleMouseUp = useCallback(() => {
    setIsDragging(false)
  }, [])

  // 预览区的鼠标事件处理
  const handlePreviewMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button === 0 || e.button === 1) { // 左键或中键
      setIsPreviewDragging(true)
      setPreviewDragStart({ x: e.clientX - previewPan.x, y: e.clientY - previewPan.y })
      e.preventDefault()
    }
  }, [previewPan])

  const handlePreviewWheel = useCallback((e: WheelEvent) => {
    e.preventDefault()
    const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1
    setPreviewZoom((prevZoom) => Math.max(0.1, Math.min(3, prevZoom * zoomFactor)))
  }, [])

  const handlePreviewMouseMove = useCallback((e: MouseEvent) => {
    if (isPreviewDragging) {
      setPreviewPan({
        x: e.clientX - previewDragStart.x,
        y: e.clientY - previewDragStart.y
      })
    }
  }, [isPreviewDragging, previewDragStart])

  const handlePreviewMouseUp = useCallback(() => {
    setIsPreviewDragging(false)
  }, [])

  // 添加全局鼠标事件监听
  useEffect(() => {
    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove)
      document.addEventListener('mouseup', handleMouseUp)
      return () => {
        document.removeEventListener('mousemove', handleMouseMove)
        document.removeEventListener('mouseup', handleMouseUp)
      }
    }
  }, [isDragging, handleMouseMove, handleMouseUp])

  useEffect(() => {
    if (isPreviewDragging) {
      document.addEventListener('mousemove', handlePreviewMouseMove)
      document.addEventListener('mouseup', handlePreviewMouseUp)
      return () => {
        document.removeEventListener('mousemove', handlePreviewMouseMove)
        document.removeEventListener('mouseup', handlePreviewMouseUp)
      }
    }
  }, [isPreviewDragging, handlePreviewMouseMove, handlePreviewMouseUp])

  // 添加预览区的滚轮事件监听（使用原生事件以支持 preventDefault）
  useEffect(() => {
    if (!previewRef.current) return
    
    const element = previewRef.current
    element.addEventListener('wheel', handlePreviewWheel, { passive: false })
    
    return () => {
      element.removeEventListener('wheel', handlePreviewWheel)
    }
  }, [handlePreviewWheel])

  // “准确(TeX)”预览：用与导出相同的 TeX 引擎渲染成 PDF，再以内嵌 PDF 显示
  useEffect(() => {
    if (previewMode !== 'tex') return
    if (!window.canvasApi?.renderLatexToPdfDataUrl) return

    if (texPreviewTimerRef.current) window.clearTimeout(texPreviewTimerRef.current)

    texPreviewTimerRef.current = window.setTimeout(async () => {
      try {
        // 复用导出时的模板与包
        const basePackages = ['amsmath, amssymb, amsfonts', 'graphicx', 'hyperref']
        const customPackages = latexPackages.filter((p) => p.trim()).map((p) => p.trim())
        const allPackages = [...basePackages, ...customPackages]
        const packagesSection = allPackages.map((pkg) => `\\usepackage{${pkg}}`).join('\n')

        const latexDocument = `\\documentclass[UTF8]{ctexart}
${packagesSection}
\\pagestyle{empty}
\\begin{document}
${source}
\\end{document}`

        const res = await window.canvasApi!.renderLatexToPdfDataUrl(latexDocument)
        if (res.success && res.dataUrl) {
          setTexPreviewDataUrl(res.dataUrl)
        } else {
          setTexPreviewDataUrl(null)
        }
      } catch {
        setTexPreviewDataUrl(null)
      }
    }, 450)

    return () => {
      if (texPreviewTimerRef.current) window.clearTimeout(texPreviewTimerRef.current)
    }
  }, [previewMode, source, latexPackages])



  // 为不同嵌套层级定义颜色
  const getLinkColors = (level: number, active: boolean) => {
    const colors = [
      { bg: 'rgba(96,165,250,0.18)', border: 'rgba(96,165,250,0.8)', activeBg: 'rgba(56,189,248,0.25)', activeBorder: 'rgba(56,189,248,0.8)' }, // 蓝色 - 最内层
      { bg: 'rgba(168,85,247,0.18)', border: 'rgba(168,85,247,0.8)', activeBg: 'rgba(139,69,193,0.25)', activeBorder: 'rgba(139,69,193,0.8)' }, // 紫色 - 第二层
      { bg: 'rgba(236,72,153,0.18)', border: 'rgba(236,72,153,0.8)', activeBg: 'rgba(219,39,119,0.25)', activeBorder: 'rgba(219,39,119,0.8)' }, // 粉色 - 第三层
      { bg: 'rgba(34,197,94,0.18)', border: 'rgba(34,197,94,0.8)', activeBg: 'rgba(22,163,74,0.25)', activeBorder: 'rgba(22,163,74,0.8)' }, // 绿色 - 第四层
    ]
    const colorSet = colors[Math.min(level - 1, colors.length - 1)]
    return active ? { bg: colorSet.activeBg, border: colorSet.activeBorder } : { bg: colorSet.bg, border: colorSet.border }
  }


  // 点击外部关闭菜单
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (contextMenu && contextMenuRef.current && !contextMenuRef.current.contains(e.target as Node)) {
        setContextMenu(null)
      }
    }
    if (contextMenu) {
      document.addEventListener('mousedown', handleClickOutside)
      return () => {
        document.removeEventListener('mousedown', handleClickOutside)
      }
    }
  }, [contextMenu])


  return (
    <div
      data-latex-panel
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        width: '100%',
        background: theme === 'light' ? '#f8fafc' : '#020617',
        color: theme === 'light' ? '#0f172a' : 'rgba(248,250,252,0.9)',
        fontSize: 13,
        borderLeft: `1px solid ${theme === 'light' ? 'rgba(15,23,42,0.15)' : 'rgba(148,163,184,0.35)'}`,
        position: 'relative',
        zIndex: 1,
        pointerEvents: 'auto',
        overflow: 'hidden',
      }}
      onClick={(e) => {
        // 点击外部关闭菜单，但点击菜单本身不关闭
        if (e.target === e.currentTarget) {
          setContextMenu(null)
        }
      }}
    >
      <div
        style={{
          padding: '8px 10px',
          borderBottom: `1px solid ${theme === 'light' ? 'rgba(15,23,42,0.08)' : 'rgba(148,163,184,0.35)'}`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 8,
          flexWrap: 'wrap',
        }}
      >
        <div style={{ fontWeight: 600 }}>LaTeX Panel</div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <select
            value={previewMode}
            onChange={(e) => setPreviewMode(e.target.value as any)}
            style={{
              padding: '4px 8px',
              borderRadius: 6,
              border: `1px solid ${theme === 'light' ? 'rgba(15,23,42,0.15)' : 'rgba(148,163,184,0.35)'}`,
              background: 'transparent',
              color: palette.textPrimary,
              fontSize: 11,
              cursor: 'pointer',
              outline: 'none',
            }}
            title={previewMode === 'tex' ? '使用 TeX 引擎渲染，和导出一致，但会更慢' : '使用 KaTeX 快速预览，可能与导出略有差异'}
          >
            <option value="fast">预览：快速(KaTeX)</option>
            <option value="tex">预览：准确(TeX)</option>
          </select>
          <button
            onClick={() => setShowPackageSettings(!showPackageSettings)}
            style={{
              padding: '4px 8px',
              borderRadius: 6,
              border: `1px solid ${theme === 'light' ? 'rgba(15,23,42,0.15)' : 'rgba(148,163,184,0.35)'}`,
              background: showPackageSettings ? palette.accent : 'transparent',
              color: showPackageSettings ? 'white' : palette.textPrimary,
              fontSize: 11,
              cursor: 'pointer',
            }}
          >
            {showPackageSettings ? '隐藏包设置' : '包设置'}
          </button>
          <button
            disabled={exporting}
            onClick={async () => {
              if (!window.canvasApi) {
                alert('canvasApi 不可用')
                return
              }

              const path = await window.canvasApi.saveFileDialog({
                title: 'Export LaTeX to PDF',
                filters: [{ name: 'PDF', extensions: ['pdf'] }],
              })
              if (!path) return

              setExporting(true)
              try {
                // 将 LaTeX 内容转换为完整的 LaTeX 文档
                // 使用 ctexart + xelatex，原生支持 UTF-8 中文
                // 基础包
                const basePackages = [
                  'amsmath, amssymb, amsfonts',
                  'graphicx',
                  'hyperref',
                ]
                // 用户自定义包
                const customPackages = latexPackages.filter(p => p.trim()).map(p => p.trim())
                const allPackages = [...basePackages, ...customPackages]
                
                const packagesSection = allPackages.map(pkg => `\\usepackage{${pkg}}`).join('\n')
                
                const latexDocument = `\\documentclass[UTF8]{ctexart}
${packagesSection}
\\pagestyle{empty}
\\begin{document}
${source}
\\end{document}`

                // 调用 Python 渲染
                const result = await window.canvasApi.renderLatexToPdf(latexDocument, path)
                
                if (result.success) {
                  alert('PDF 导出成功！')
                } else {
                  alert(`PDF 导出失败：\n${result.error || '未知错误'}`)
                }
              } catch (error) {
                console.error('LaTeX PDF export failed:', error)
                alert(`PDF导出失败: ${error}`)
              } finally {
                setExporting(false)
              }
            }}
            style={{
              padding: '4px 8px',
              borderRadius: 6,
              border: `1px solid ${theme === 'light' ? 'rgba(15,23,42,0.15)' : 'rgba(148,163,184,0.35)'}`,
              background: 'transparent',
              color: palette.textPrimary,
              fontSize: 11,
              cursor: 'pointer',
              opacity: exporting ? 0.6 : 1,
            }}
          >
            {exporting ? '渲染中...' : '导出 PDF (LaTeX)'}
          </button>
          <div style={{ fontSize: 11, opacity: 0.6 }}>
            操作：选中文本 + 选中节点 → 按住 <b>Ctrl</b> 右键（编辑区/预览区高亮）创建/追加；已有链接可右键删除或移除节点
          </div>
        </div>
      </div>

      {showPackageSettings && (
        <div
          style={{
            padding: '10px',
            borderBottom: `1px solid ${theme === 'light' ? 'rgba(15,23,42,0.08)' : 'rgba(148,163,184,0.35)'}`,
            background: theme === 'light' ? 'rgba(15,23,42,0.02)' : 'rgba(255,255,255,0.02)',
          }}
        >
          <div style={{ fontSize: 11, fontWeight: 600, marginBottom: 8 }}>自定义 LaTeX 包</div>
          <div style={{ fontSize: 10, opacity: 0.7, marginBottom: 8 }}>
            添加包名（例如：geometry, fancyhdr, listings, tikz）。每行一个包，支持选项（例如：geometry[a4paper, margin=1in]）
          </div>
          <details style={{ marginBottom: 8, fontSize: 10 }}>
            <summary style={{ cursor: 'pointer', opacity: 0.8, marginBottom: 4 }}>如何查找需要的包？</summary>
            <div style={{ padding: '8px', background: theme === 'light' ? 'rgba(15,23,42,0.03)' : 'rgba(255,255,255,0.03)', borderRadius: 4, marginTop: 4 }}>
              <div style={{ marginBottom: 4 }}><strong>常用资源：</strong></div>
              <ul style={{ margin: 0, paddingLeft: 20, lineHeight: 1.6 }}>
                <li><strong>CTAN</strong> (Comprehensive TeX Archive Network): <a href="https://ctan.org" target="_blank" rel="noopener noreferrer" style={{ color: palette.accent }}>ctan.org</a> - 搜索包名和文档</li>
                <li><strong>Overleaf</strong>: <a href="https://www.overleaf.com/learn" target="_blank" rel="noopener noreferrer" style={{ color: palette.accent }}>overleaf.com/learn</a> - LaTeX 教程和包使用示例</li>
                <li><strong>LaTeX Stack Exchange</strong>: <a href="https://tex.stackexchange.com" target="_blank" rel="noopener noreferrer" style={{ color: palette.accent }}>tex.stackexchange.com</a> - 问答社区</li>
              </ul>
              <div style={{ marginTop: 8, marginBottom: 4 }}><strong>常用包分类：</strong></div>
              <ul style={{ margin: 0, paddingLeft: 20, lineHeight: 1.6 }}>
                <li><strong>页面布局：</strong> geometry, fancyhdr, titlesec</li>
                <li><strong>数学公式：</strong> amsmath, amssymb, amsfonts（已包含）, mathtools</li>
                <li><strong>代码高亮：</strong> listings, minted</li>
                <li><strong>图表：</strong> tikz, pgfplots, graphicx（已包含）</li>
                <li><strong>表格：</strong> booktabs, longtable, tabularx</li>
                <li><strong>参考文献：</strong> biblatex, natbib</li>
                <li><strong>其他：</strong> hyperref（已包含）, siunitx, algorithm2e</li>
              </ul>
            </div>
          </details>
          <textarea
            value={latexPackages.join('\n')}
            onChange={(e) => {
              const lines = e.target.value.split('\n').filter(line => line.trim())
              setLatexPackages(lines)
            }}
            placeholder="geometry&#10;fancyhdr&#10;listings&#10;tikz"
            style={{
              width: '100%',
              minHeight: 80,
              padding: 6,
              fontSize: 11,
              fontFamily: 'monospace',
              background: theme === 'light' ? 'rgba(255,255,255,0.9)' : 'rgba(15,23,42,0.95)',
              border: `1px solid ${palette.border}`,
              borderRadius: 6,
              color: palette.textPrimary,
              resize: 'vertical',
              outline: 'none',
            }}
          />
          <div style={{ marginTop: 8, fontSize: 10, opacity: 0.6 }}>
            <div style={{ marginBottom: 4 }}>常用包示例：</div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
              {['geometry', 'fancyhdr', 'listings', 'tikz', 'algorithm2e', 'booktabs', 'siunitx', 'biblatex'].map((pkg) => (
                <button
                  key={pkg}
                  onClick={() => {
                    if (!latexPackages.includes(pkg)) {
                      setLatexPackages([...latexPackages, pkg])
                    }
                  }}
                  style={{
                    padding: '2px 6px',
                    fontSize: 10,
                    borderRadius: 4,
                    border: `1px solid ${palette.border}`,
                    background: 'transparent',
                    color: palette.textPrimary,
                    cursor: 'pointer',
                  }}
                >
                  + {pkg}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      <div
        style={{
          display: 'flex',
          flex: 1,
          minHeight: 0,
          flexDirection: 'column',
          position: 'relative'
        }}
      >
        <div
          style={{
            flex: panelRatio,
            minHeight: 0,
            borderBottom: `1px solid ${theme === 'light' ? 'rgba(15,23,42,0.08)' : 'rgba(148,163,184,0.25)'}`,
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          <div style={{ padding: '6px 8px', fontSize: 11, opacity: 0.75 }}>编辑</div>
          <textarea
            ref={textareaRef}
            value={localSource}
            onChange={(e) => {
              setLocalSource(e.target.value)
              const el = e.currentTarget
              setSelectionStart(el.selectionStart)
              setSelectionEnd(el.selectionEnd)
            }}
            onSelect={onTextSelect}
            onMouseUp={onTextSelect}
            onKeyUp={onTextSelect}
            onClick={(e) => {
              e.stopPropagation()
              onTextSelect(e)
            }}
            onMouseDown={(e) => {
              e.stopPropagation()
            }}
            style={{
              flex: 1,
              resize: 'none',
              border: 'none',
              outline: 'none',
              padding: 8,
              background: theme === 'light' ? 'rgba(255,255,255,0.9)' : 'rgba(15,23,42,0.95)',
              color: 'inherit',
              fontFamily: 'monospace',
              fontSize: 13,
              lineHeight: 1.45,
              width: '100%',
              boxSizing: 'border-box',
              pointerEvents: 'auto',
              cursor: 'text',
              position: 'relative',
              zIndex: 10,
            }}
            placeholder="在这里输入 LaTeX 文本..."
            onContextMenu={(e) => {
              e.preventDefault()
              e.stopPropagation()
              const link = findLinkForCurrentSelection()
              setContextMenu({
                x: e.clientX,
                y: e.clientY,
                linkId: link ? link.id : null,
                hasSelection: selectionStart != null && selectionEnd != null && selectionStart !== selectionEnd,
                hasSelectedNodes: selectedNodes.length > 0,
              })
            }}
          />
        </div>

        {/* 可拖拽的分隔符 */}
        <div
          style={{
            height: 4,
            background: theme === 'light' ? 'rgba(15,23,42,0.08)' : 'rgba(148,163,184,0.25)',
            cursor: 'row-resize',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            position: 'relative',
            flexShrink: 0,
          }}
          onMouseDown={(e) => {
            handleMouseDown(e)
            e.preventDefault()
            e.stopPropagation()
          }}
        >
          <div
            style={{
              width: 20,
              height: 2,
              background: theme === 'light' ? 'rgba(15,23,42,0.3)' : 'rgba(148,163,184,0.5)',
              borderRadius: 1,
            }}
          />
        </div>

        <div
          style={{
            flex: 1 - panelRatio,
            minHeight: 0,
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          <div style={{ padding: '6px 8px', fontSize: 11, opacity: 0.75 }}>预览 / 链接高亮</div>
          <div
            ref={previewRef}
            style={{
              flex: 1,
              position: 'relative',
              overflow: 'hidden',
              cursor: isPreviewDragging ? 'grabbing' : 'grab',
            }}
            onMouseDown={handlePreviewMouseDown}
          >
            {previewMode === 'tex' ? (
              <div style={{ position: 'absolute', inset: 0, background: 'transparent' }}>
                {texPreviewDataUrl ? (
                  <iframe
                    ref={texPreviewIframeRef}
                    src={texPreviewDataUrl}
                    style={{ width: '100%', height: '100%', border: 'none', background: 'transparent' }}
                    title="TeX Preview"
                  />
                ) : (
                  <div style={{ padding: 10, fontSize: 12, opacity: 0.7 }}>
                    正在用 TeX 引擎生成预览…（首次可能较慢）
                  </div>
                )}
              </div>
            ) : null}
            <div
              style={{
                padding: 8,
                fontFamily: 'monospace',
                whiteSpace: 'pre-wrap',
                lineHeight: 1.45,
                transform: `translate(${previewPan.x}px, ${previewPan.y}px) scale(${previewZoom})`,
                transformOrigin: 'top left',
                width: '100%',
                minHeight: '100%',
                userSelect: 'none',
                display: previewMode === 'tex' ? 'none' : undefined,
              }}
              onContextMenu={(e) => {
                e.preventDefault()
                // 找到点击位置对应的所有链接（嵌套链接）
                const target = e.target as HTMLElement
                const allLinkSpans: NodeListOf<Element> = target.querySelectorAll('[data-link-id]')
                const clickedLinkSpan = target.closest('[data-link-id]')

                // 收集所有链接 span，包括嵌套的
                const allSpans = Array.from(allLinkSpans)
                if (clickedLinkSpan && !allSpans.includes(clickedLinkSpan)) {
                  allSpans.push(clickedLinkSpan)
                }

                if (allSpans.length > 0) {
                  // 有链接时，收集所有相关的链接
                  const allLinks: Array<{ id: string; text: string; range: { start: number; end: number } }> = []

                  allSpans.forEach(span => {
                    const linkId = span.getAttribute('data-link-id')
                    const link = links.find(l => l.id === linkId)
                    if (link && !allLinks.some(l => l.id === link.id)) {
                      allLinks.push({
                        id: link.id,
                        text: link.text,
                        range: link.range
                      })
                    }
                  })

                  // 按范围大小排序（小的在前）
                  allLinks.sort((a, b) => (a.range.end - a.range.start) - (b.range.end - b.range.start))

                  setContextMenu({
                    x: e.clientX,
                    y: e.clientY,
                    linkId: allLinks.length === 1 ? allLinks[0].id : null,
                    nestedLinks: allLinks.length > 1 ? allLinks : undefined
                  })
                }
              }}
            >
            {/* 智能文本渲染：按链接边界分割，保持文档流 */}
            {(() => {
              // 按所有链接的边界分割文本
              const splitPoints = new Set<number>()
              splitPoints.add(0)
              splitPoints.add(source.length)
              links.forEach(link => {
                splitPoints.add(link.range.start)
                splitPoints.add(link.range.end)
              })

              const sortedPoints = Array.from(splitPoints).sort((a, b) => a - b)
              const segments: Array<{
                text: string
                start: number
                end: number
                link?: LatexLink
                level: number
              }> = []

              for (let i = 0; i < sortedPoints.length - 1; i++) {
                const start = sortedPoints[i]
                const end = sortedPoints[i + 1]

                if (start >= end) continue

                const segmentText = source.slice(start, end)

                // 找到包含这个范围的所有链接
                let containingLinks = links.filter(link =>
                  link.range.start <= start && link.range.end >= end
                )

                // 如果有激活的层级，只显示该层级的链接
                if (activeLayerId) {
                  const activeLayer = layers.find(l => l.id === activeLayerId)
                  if (activeLayer) {
                    containingLinks = containingLinks.filter(link =>
                      link.nodeIds.some(nodeId => activeLayer.nodeIds.includes(nodeId))
                    )
                  }
                }

                if (containingLinks.length > 0) {
                  // 选择最内层的链接（范围最小的）
                  const innermostLink = containingLinks.sort((a, b) =>
                    (a.range.end - a.range.start) - (b.range.end - b.range.start)
                  )[0]

                  segments.push({
                    text: segmentText,
                    start,
                    end,
                    link: innermostLink,
                    level: containingLinks.length
                  })
                } else {
                  segments.push({
                    text: segmentText,
                    start,
                    end,
                    level: 0
                  })
                }
              }

              return (
                <div
                  style={{ position: 'relative' }}
                  onContextMenu={(e) => {
                    e.preventDefault()
                    const target = e.target as HTMLElement
                    const clickedSpan = target.closest('[data-link-id]')

                    if (clickedSpan) {
                      const linkId = clickedSpan.getAttribute('data-link-id')
                      const clickedLink = links.find(l => l.id === linkId)

                      if (clickedLink) {
                        // 对于嵌套链接，显示所有相关的层级
                        // 由于segment.level表示有多少层嵌套，我们需要找到所有包含这个链接的更大链接
                        const relevantLinks: Array<{ id: string; text: string; range: { start: number; end: number } }> = []

                        // 添加当前点击的链接
                        relevantLinks.push({
                          id: clickedLink.id,
                          text: clickedLink.text,
                          range: clickedLink.range
                        })

                        // 找到所有包含这个链接的更大链接（外层）
                        const containingLinks = links.filter(otherLink =>
                          otherLink.id !== clickedLink.id &&
                          otherLink.range.start <= clickedLink.range.start &&
                          otherLink.range.end >= clickedLink.range.end &&
                          (otherLink.range.end - otherLink.range.start) > (clickedLink.range.end - clickedLink.range.start)
                        )

                        // 添加所有外层链接
                        containingLinks.forEach(link => {
                          relevantLinks.push({
                            id: link.id,
                            text: link.text,
                            range: link.range
                          })
                        })

                        // 按范围大小排序（小的在前）
                        relevantLinks.sort((a, b) => (a.range.end - a.range.start) - (b.range.end - b.range.start))

                        setContextMenu({
                          x: e.clientX,
                          y: e.clientY,
                          linkId: relevantLinks.length === 1 ? relevantLinks[0].id : null,
                          nestedLinks: relevantLinks.length > 1 ? relevantLinks : undefined
                        })
                      }
                    }
                  }}
                >
                  {segments.map((segment, idx) => {
                    const html = renderLatex(segment.text, theme)

                    if (segment.link) {
                      const colors = getLinkColors(segment.level, segment.link.nodeIds.some((id) => activeNodeIds.has(id)))
                      return (
                        <span
                          key={`segment-${idx}`}
                          data-link-id={segment.link.id}
                          onClick={(e) => {
                            e.stopPropagation()
                            focusNodes(segment.link!.nodeIds)
                          }}
                          style={{
                            background: colors.bg,
                            cursor: 'pointer',
                            position: 'relative',
                            display: 'inline',
                            // boxShadow 不参与布局计算，避免引入断行/偏移
                            boxShadow: `0 0 0 1px ${colors.border}`,
                            borderRadius: '2px',
                          }}
                          title={`层级${segment.level}: ${segment.link.text}`}
                          dangerouslySetInnerHTML={{ __html: html }}
                        />
                      )
                    }

                    return (
                      <span
                        key={`segment-${idx}`}
                        style={{ position: 'relative', display: 'inline' }}
                        dangerouslySetInnerHTML={{ __html: html }}
                      />
                    )
                  })}
                </div>
              )
            })()}
            </div>
          </div>
        </div>
      </div>

      <div
        style={{
          borderTop: `1px solid ${theme === 'light' ? 'rgba(15,23,42,0.08)' : 'rgba(148,163,184,0.35)'}`,
          padding: 8,
          maxHeight: 120,
          overflowY: 'auto',
          fontSize: 11,
        }}
      >
        <div style={{ fontWeight: 600, marginBottom: 4 }}>链接列表</div>
        {links.length === 0 ? (
          <div style={{ opacity: 0.7, color: theme === 'light' ? 'rgba(15,23,42,0.6)' : 'inherit' }}>暂无链接。先选中文本，再选中画布节点，点击上方按钮创建链接。</div>
        ) : (
          <ul style={{ margin: 0, paddingLeft: 16 }}>
            {links.map((l) => (
              <li
                key={l.id}
                style={{ marginBottom: 2 }}
                onContextMenu={(e) => {
                  e.preventDefault()
                  setContextMenu({ x: e.clientX, y: e.clientY, linkId: l.id })
                }}
              >
                <span
                  style={{ cursor: 'pointer', color: '#38bdf8' }}
                  onClick={() => focusNodes(l.nodeIds)}
                >
                  "{l.text.slice(0, 24)}{l.text.length > 24 ? '…' : ''}"
                </span>{' '}
                → {l.nodeIds.join(', ')}
              </li>
            ))}
          </ul>
        )}
      </div>

      {contextMenu && (
        <div
          ref={contextMenuRef}
          style={{
            position: 'fixed',
            left: contextMenu.x,
            top: contextMenu.y,
            zIndex: 1000,
            background: theme === 'light' ? 'rgba(255,255,255,0.98)' : 'rgba(15,16,22,0.97)',
            border: `1px solid ${theme === 'light' ? 'rgba(15,23,42,0.15)' : 'rgba(255,255,255,0.18)'}`,
            borderRadius: 8,
            padding: 4,
            minWidth: 180,
            boxShadow: theme === 'light' ? '0 8px 24px rgba(0,0,0,0.1)' : '0 8px 24px rgba(0,0,0,0.4)',
          }}
          onMouseDown={(e) => e.stopPropagation()}
        >
          {contextMenu.nestedLinks && contextMenu.nestedLinks.length > 1 ? (
            // 嵌套链接菜单
            <>
              <div style={{ padding: '6px 10px', fontSize: 11, color: theme === 'light' ? 'rgba(15,23,42,0.7)' : 'rgba(255,255,255,0.7)', fontWeight: 600 }}>
                选择要操作的链接：
              </div>
              <div style={{ height: 1, background: theme === 'light' ? 'rgba(15,23,42,0.08)' : 'rgba(255,255,255,0.12)', margin: '4px 0' }} />
              {contextMenu.nestedLinks.map((link) => (
                <button
                  key={link.id}
                  onClick={() => {
                    // 直接删除选中的链接，不显示二级菜单
                    deleteLink(link.id)
                    setContextMenu(null)
                  }}
                  style={{
                    width: '100%',
                    textAlign: 'left',
                    padding: '8px 10px',
                    borderRadius: 4,
                    border: 'none',
                    background: 'transparent',
                    color: theme === 'light' ? '#0f172a' : 'rgba(255,255,255,0.9)',
                    cursor: 'pointer',
                    fontSize: 11,
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = theme === 'light' ? 'rgba(15,23,42,0.05)' : 'rgba(255,255,255,0.1)'
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = 'transparent'
                  }}
                >
                  <div style={{ fontWeight: 600, marginBottom: 2 }}>
                    "{link.text.slice(0, 20)}{link.text.length > 20 ? '…' : ''}"
                  </div>
                  <div style={{ fontSize: 10, opacity: 0.7 }}>
                    位置: {link.range.start}-{link.range.end}
                  </div>
                </button>
              ))}
            </>
          ) : contextMenu.linkId ? (
            <>
              <button
                onClick={() => {
                  // 直接删除指定的链接
                  deleteLink(contextMenu.linkId!)
                  setContextMenu(null)
                }}
                style={{
                  width: '100%',
                  textAlign: 'left',
                  padding: '6px 10px',
                  borderRadius: 4,
                  border: 'none',
                  background: 'transparent',
                  color: 'rgba(239,68,68,0.9)',
                  cursor: 'pointer',
                  fontSize: 12,
                }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.background = theme === 'light' ? 'rgba(15,23,42,0.05)' : 'rgba(255,255,255,0.1)'
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.background = 'transparent'
                    }}
              >
                删除此链接
              </button>
            </>
          ) : contextMenu.hasSelection && contextMenu.hasSelectedNodes ? (
            <button
              onClick={() => {
                createOrAppendLink()
                setContextMenu(null)
              }}
              style={{
                width: '100%',
                textAlign: 'left',
                padding: '6px 10px',
                borderRadius: 4,
                border: 'none',
                background: 'transparent',
                color: theme === 'light' ? '#0f172a' : 'rgba(255,255,255,0.9)',
                cursor: 'pointer',
                fontSize: 12,
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = theme === 'light' ? 'rgba(15,23,42,0.05)' : 'rgba(255,255,255,0.1)'
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = 'transparent'
              }}
            >
              创建/追加链接（当前选区 ↔ 选中节点）
            </button>
          ) : (
            <div style={{ padding: '6px 10px', fontSize: 11, color: theme === 'light' ? 'rgba(15,23,42,0.6)' : 'rgba(255,255,255,0.6)' }}>
              {!contextMenu.hasSelection && !contextMenu.hasSelectedNodes
                ? '请先选中文本和节点'
                : !contextMenu.hasSelection
                  ? '请先选中文本'
                  : '请先选中节点'}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
