export type NodeKind = 'text' | 'markdown' | 'code' | 'image' | 'pdf' | 'iframe'

export type CanvasNodeData = {
  kind: NodeKind
  title?: string
  // content is used for text/markdown/code; filePath used for local files (markdown/image/pdf)
  content?: string
  filePath?: string
  language?: string // code
  url?: string // iframe
  color?: string
}

export type EdgeData = {
  label?: string
  weight?: number
}

export type CanvasDocV1 = {
  schema: 'canvas-doc-v1'
  createdAt: string
  updatedAt: string
  viewport?: { x: number; y: number; zoom: number }
  nodes: Array<{
    id: string
    kind: NodeKind
    title?: string
    position: { x: number; y: number }
    size?: { width?: number; height?: number }
    data?: Omit<CanvasNodeData, 'kind'>
  }>
  edges: Array<{
    id: string
    source: string
    target: string
    type?: 'straight' | 'default'
    markerEnd?: 'arrow'
    data?: EdgeData
  }>
}


