import { contextBridge, ipcRenderer } from 'electron'

export type OpenFileOptions = {
  title?: string
  filters?: Array<{ name: string; extensions: string[] }>
}

export type SaveFileOptions = {
  title?: string
  defaultPath?: string
  filters?: Array<{ name: string; extensions: string[] }>
}

contextBridge.exposeInMainWorld('canvasApi', {
  readTextFile: (filePath: string) => ipcRenderer.invoke('fs:readText', filePath) as Promise<string>,
  readFileAsDataUrl: (filePath: string) => ipcRenderer.invoke('fs:readAsDataUrl', filePath) as Promise<string>,
  writeTextFile: (filePath: string, content: string) =>
    ipcRenderer.invoke('fs:writeText', filePath, content) as Promise<boolean>,
  writeBase64File: (filePath: string, base64: string) =>
    ipcRenderer.invoke('fs:writeBase64', filePath, base64) as Promise<boolean>,
  openFileDialog: (opts: OpenFileOptions) => ipcRenderer.invoke('dialog:openFile', opts) as Promise<string | null>,
  saveFileDialog: (opts: SaveFileOptions) => ipcRenderer.invoke('dialog:saveFile', opts) as Promise<string | null>,
  renderLatexToPdf: (latexContent: string, outputPath: string) =>
    ipcRenderer.invoke('latex:renderToPdf', latexContent, outputPath) as Promise<{ success: boolean; error?: string }>,
  renderLatexToPdfDataUrl: (latexContent: string) =>
    ipcRenderer.invoke('latex:renderToPdfDataUrl', latexContent) as Promise<{ success: boolean; dataUrl?: string; error?: string }>,
})


