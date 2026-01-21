import type { CanvasDocV1 } from './types'
import type { RFEdge, RFNode } from './store'
import type { Viewport } from 'reactflow'
import { MarkerType } from 'reactflow'

function dirname(p: string): string {
  const s = p.replace(/\\/g, '/')
  const idx = s.lastIndexOf('/')
  return idx >= 0 ? s.slice(0, idx) : ''
}

function join(base: string, rel: string): string {
  const b = base.replace(/\\/g, '/').replace(/\/+$/, '')
  const r = rel.replace(/\\/g, '/').replace(/^\/+/, '')
  return b ? `${b}/${r}` : r
}

function isAbs(p: string): boolean {
  return /^[a-zA-Z]:[\\/]/.test(p) || p.startsWith('\\\\') || p.startsWith('/')
}

function normalize(p: string): string {
  const parts = p.replace(/\\/g, '/').split('/')
  const out: string[] = []
  for (const part of parts) {
    if (!part || part === '.') continue
    if (part === '..') out.pop()
    else out.push(part)
  }
  const prefix = /^[a-zA-Z]:/.test(parts[0] ?? '') ? `${parts[0]}/` : p.startsWith('/') ? '/' : ''
  return prefix + out.join('/')
}

function relative(fromDir: string, toPath: string): string {
  const from = normalize(fromDir.replace(/\\/g, '/'))
  const to = normalize(toPath.replace(/\\/g, '/'))
  // windows drive handling: only relativize if same drive
  const fromDrive = (from.match(/^([a-zA-Z]:)\//)?.[1] ?? '').toLowerCase()
  const toDrive = (to.match(/^([a-zA-Z]:)\//)?.[1] ?? '').toLowerCase()
  if (fromDrive && toDrive && fromDrive !== toDrive) return toPath
  const a = from.replace(/^[a-zA-Z]:\//, '').split('/').filter(Boolean)
  const b = to.replace(/^[a-zA-Z]:\//, '').split('/').filter(Boolean)
  let i = 0
  while (i < a.length && i < b.length && a[i] === b[i]) i++
  const ups = a.length - i
  const downs = b.slice(i)
  const rel = [...Array(ups).fill('..'), ...downs].join('/')
  return rel || '.'
}

export function exportToDoc(
  nodes: RFNode[],
  edges: RFEdge[],
  canvasFilePath?: string,
  viewport?: Viewport,
): CanvasDocV1 {
  const now = new Date().toISOString()
  const baseDir = canvasFilePath ? dirname(canvasFilePath) : ''
  return {
    schema: 'canvas-doc-v1',
    createdAt: now,
    updatedAt: now,
    viewport: viewport
      ? {
          x: viewport.x,
          y: viewport.y,
          zoom: viewport.zoom,
        }
      : undefined,
    nodes: nodes.map((n) => ({
      id: n.id,
      kind: n.data?.kind ?? 'text',
      title: n.data?.title,
      position: n.position,
      size: { width: (n.style as any)?.width, height: (n.style as any)?.height },
      data:
        n.data && (n.data.kind === 'image' || n.data.kind === 'pdf' || n.data.kind === 'iframe' || n.data.kind === 'markdown')
          ? {
              // keep JSON small for file-backed / embedded nodes
              title: n.data.title,
              filePath:
                n.data.filePath && canvasFilePath && isAbs(n.data.filePath)
                  ? relative(baseDir, n.data.filePath)
                  : n.data.filePath,
              url: n.data.url,
              color: n.data.color,
            }
          : n.data
            ? { ...n.data, kind: undefined }
            : undefined,
    })),
    edges: edges.map((e) => ({
      id: e.id,
      source: e.source,
      target: e.target,
      type: e.type as any,
      markerEnd: e.markerEnd ? 'arrow' : undefined,
      data: e.data,
    })),
  }
}

export function importFromDoc(doc: CanvasDocV1, canvasFilePath?: string): { nodes: RFNode[]; edges: RFEdge[] } {
  const baseDir = canvasFilePath ? dirname(canvasFilePath) : ''
  const nodes: RFNode[] = doc.nodes.map((n) => ({
    id: n.id,
    type: n.kind,
    position: n.position,
    data: {
      kind: n.kind,
      title: n.title,
      ...(n.data as any),
      filePath:
        (n.data as any)?.filePath && canvasFilePath && !isAbs((n.data as any).filePath)
          ? normalize(join(baseDir, (n.data as any).filePath))
          : (n.data as any)?.filePath,
    },
    style: { width: n.size?.width ?? 320, height: n.size?.height ?? 200 },
  }))
  const edges: RFEdge[] = doc.edges.map((e) => ({
    id: e.id,
    source: e.source,
    target: e.target,
    type: e.type ?? 'default',
    markerEnd: e.markerEnd === 'arrow' ? { type: MarkerType.ArrowClosed } : undefined,
    data: e.data,
  }))
  return { nodes, edges }
}


