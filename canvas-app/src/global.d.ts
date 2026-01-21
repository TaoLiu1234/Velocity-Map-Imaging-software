export {}

declare global {
  interface Window {
    canvasApi?: {
      readTextFile: (filePath: string) => Promise<string>
      readFileAsDataUrl: (filePath: string) => Promise<string>
      writeTextFile: (filePath: string, content: string) => Promise<boolean>
      writeBase64File: (filePath: string, base64: string) => Promise<boolean>
      openFileDialog: (opts: { title?: string; filters?: Array<{ name: string; extensions: string[] }> }) => Promise<string | null>
      saveFileDialog: (opts: {
        title?: string
        defaultPath?: string
        filters?: Array<{ name: string; extensions: string[] }>
      }) => Promise<string | null>
      renderLatexToPdf: (latexContent: string, outputPath: string) => Promise<{ success: boolean; error?: string }>
      renderLatexToPdfDataUrl: (latexContent: string) => Promise<{ success: boolean; dataUrl?: string; error?: string }>
    }
  }
}


