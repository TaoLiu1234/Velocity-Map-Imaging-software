import { app, BrowserWindow, ipcMain, dialog } from 'electron'
import path from 'node:path'
import fs from 'node:fs/promises'
import { spawn } from 'node:child_process'
import os from 'node:os'

const DEV_URL = 'http://localhost:5173'

function createWindow() {
  const win = new BrowserWindow({
    width: 1400,
    height: 900,
    backgroundColor: '#0b0c10',
    autoHideMenuBar: true,
    frame: true,
    webPreferences: {
      // In dev build, main/preload are compiled into `dist-electron/` and run as CommonJS.
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  })

  if (!app.isPackaged) {
    void win.loadURL(DEV_URL)
    win.webContents.openDevTools({ mode: 'detach' })
  } else {
    // TODO: packaged load from file
    void win.loadFile(path.join(app.getAppPath(), 'dist', 'index.html'))
  }

  win.webContents.on('did-fail-load', (_event, errorCode, errorDescription, validatedURL) => {
    // Helps debug "black window" cases where the renderer never loads.
    // eslint-disable-next-line no-console
    console.error('[did-fail-load]', { errorCode, errorDescription, validatedURL })
  })

  win.webContents.on('render-process-gone', (_event, details) => {
    // eslint-disable-next-line no-console
    console.error('[render-process-gone]', details)
  })

  return win
}

app.whenReady().then(() => {
  createWindow()

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})

ipcMain.handle('fs:readText', async (_evt, filePath: string) => {
  return await fs.readFile(filePath, 'utf-8')
})

ipcMain.handle('fs:writeText', async (_evt, filePath: string, content: string) => {
  await fs.writeFile(filePath, content, 'utf-8')
  return true
})

ipcMain.handle('fs:writeBase64', async (_evt, filePath: string, base64: string) => {
  // base64 without dataURL prefix
  const buf = Buffer.from(base64, 'base64')
  await fs.writeFile(filePath, buf)
  return true
})

function guessMime(filePath: string): string {
  const ext = path.extname(filePath).toLowerCase()
  if (ext === '.png') return 'image/png'
  if (ext === '.jpg' || ext === '.jpeg') return 'image/jpeg'
  if (ext === '.gif') return 'image/gif'
  if (ext === '.webp') return 'image/webp'
  if (ext === '.bmp') return 'image/bmp'
  if (ext === '.svg') return 'image/svg+xml'
  if (ext === '.pdf') return 'application/pdf'
  if (ext === '.html' || ext === '.htm') return 'text/html'
  return 'application/octet-stream'
}

ipcMain.handle('fs:readAsDataUrl', async (_evt, filePath: string) => {
  const buf = await fs.readFile(filePath)
  const mime = guessMime(filePath)
  const b64 = buf.toString('base64')
  return `data:${mime};base64,${b64}`
})

ipcMain.handle('dialog:openFile', async (_evt, opts: { title?: string; filters?: Electron.FileFilter[] }) => {
  const res = await dialog.showOpenDialog({
    title: opts.title ?? 'Open',
    properties: ['openFile'],
    filters: opts.filters,
  })
  if (res.canceled) return null
  return res.filePaths[0] ?? null
})

ipcMain.handle('dialog:saveFile', async (_evt, opts: { title?: string; defaultPath?: string; filters?: Electron.FileFilter[] }) => {
  const res = await dialog.showSaveDialog({
    title: opts.title ?? 'Save',
    defaultPath: opts.defaultPath,
    filters: opts.filters,
  })
  if (res.canceled) return null
  return res.filePath ?? null
})

ipcMain.handle('latex:renderToPdf', async (_evt, latexContent: string, outputPath: string) => {
  // Get the path to the Python script
  // In dev: __dirname is dist-electron/, so scripts/ is at ../scripts/
  // In production: __dirname is the app root, so scripts/ is at ./scripts/
  const scriptPath = app.isPackaged
    ? path.join(process.resourcesPath, 'scripts', 'render_latex.py')
    : path.join(__dirname, '..', 'scripts', 'render_latex.py')
  
  // Create a temporary file for LaTeX content
  const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'latex-'))
  const tmpTexFile = path.join(tmpDir, 'input.tex')
  
  try {
    // Write LaTeX content to temporary file
    await fs.writeFile(tmpTexFile, latexContent, 'utf-8')
    
    // Determine Python command
    const pythonCmd = process.platform === 'win32' ? 'python' : 'python3'
    
    // Run Python script
    return await new Promise<{ success: boolean; error?: string }>((resolve) => {
      const pythonProcess = spawn(pythonCmd, [scriptPath, tmpTexFile, outputPath], {
        cwd: path.dirname(scriptPath),
        stdio: ['ignore', 'pipe', 'pipe'],
      })
      
      let stdout = ''
      let stderr = ''
      
      pythonProcess.stdout.on('data', (data) => {
        stdout += data.toString()
      })
      
      pythonProcess.stderr.on('data', (data) => {
        stderr += data.toString()
      })
      
      pythonProcess.on('close', async (code) => {
        try {
          // Clean up temporary directory
          await fs.rm(tmpDir, { recursive: true, force: true })
        } catch {
          // Ignore cleanup errors
        }
        
        if (code !== 0) {
          resolve({
            success: false,
            error: stderr || stdout || `Python script exited with code ${code}`,
          })
          return
        }
        
        // Parse JSON output from Python script
        try {
          const result = JSON.parse(stdout.trim())
          resolve(result)
        } catch {
          resolve({
            success: false,
            error: `Failed to parse Python output: ${stdout}`,
          })
        }
      })
      
      pythonProcess.on('error', async (err) => {
        try {
          await fs.rm(tmpDir, { recursive: true, force: true })
        } catch {
          // Ignore cleanup errors
        }
        resolve({
          success: false,
          error: `Failed to spawn Python process: ${err.message}. Make sure Python is installed and accessible.`,
        })
      })
    })
  } catch (error) {
    try {
      await fs.rm(tmpDir, { recursive: true, force: true })
    } catch {
      // Ignore cleanup errors
    }
    return {
      success: false,
      error: `Failed to create temporary file: ${error instanceof Error ? error.message : String(error)}`,
    }
  }
})

// Render LaTeX to a temporary PDF and return a data URL for preview
ipcMain.handle('latex:renderToPdfDataUrl', async (_evt, latexContent: string) => {
  const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'latex-prev-'))
  const outPdf = path.join(tmpDir, 'preview.pdf')

  // Reuse existing handler logic by spawning python directly here (keeps it self-contained)
  const scriptPath = app.isPackaged
    ? path.join(process.resourcesPath, 'scripts', 'render_latex.py')
    : path.join(__dirname, '..', 'scripts', 'render_latex.py')

  const tmpTexFile = path.join(tmpDir, 'input.tex')
  await fs.writeFile(tmpTexFile, latexContent, 'utf-8')
  const pythonCmd = process.platform === 'win32' ? 'python' : 'python3'

  const result = await new Promise<{ success: boolean; error?: string }>((resolve) => {
    const pythonProcess = spawn(pythonCmd, [scriptPath, tmpTexFile, outPdf], {
      cwd: path.dirname(scriptPath),
      stdio: ['ignore', 'pipe', 'pipe'],
    })

    let stdout = ''
    let stderr = ''
    pythonProcess.stdout.on('data', (data) => (stdout += data.toString()))
    pythonProcess.stderr.on('data', (data) => (stderr += data.toString()))

    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        resolve({ success: false, error: stderr || stdout || `Python exited with ${code}` })
        return
      }
      try {
        const parsed = JSON.parse(stdout.trim())
        resolve(parsed)
      } catch {
        resolve({ success: false, error: `Failed to parse Python output: ${stdout}` })
      }
    })

    pythonProcess.on('error', (err) => {
      resolve({ success: false, error: `Failed to spawn Python: ${err.message}` })
    })
  })

  if (!result.success) {
    await fs.rm(tmpDir, { recursive: true, force: true }).catch(() => {})
    return { success: false, error: result.error }
  }

  try {
    const buf = await fs.readFile(outPdf)
    const b64 = buf.toString('base64')
    await fs.rm(tmpDir, { recursive: true, force: true }).catch(() => {})
    return { success: true, dataUrl: `data:application/pdf;base64,${b64}` }
  } catch (e) {
    await fs.rm(tmpDir, { recursive: true, force: true }).catch(() => {})
    return { success: false, error: e instanceof Error ? e.message : String(e) }
  }
})


