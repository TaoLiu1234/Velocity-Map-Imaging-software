"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const electron_1 = require("electron");
const node_path_1 = __importDefault(require("node:path"));
const promises_1 = __importDefault(require("node:fs/promises"));
const node_child_process_1 = require("node:child_process");
const node_os_1 = __importDefault(require("node:os"));
const DEV_URL = 'http://localhost:5173';
function createWindow() {
    const win = new electron_1.BrowserWindow({
        width: 1400,
        height: 900,
        backgroundColor: '#0b0c10',
        autoHideMenuBar: true,
        frame: true,
        webPreferences: {
            // In dev build, main/preload are compiled into `dist-electron/` and run as CommonJS.
            preload: node_path_1.default.join(__dirname, 'preload.js'),
            contextIsolation: true,
            nodeIntegration: false,
        },
    });
    if (!electron_1.app.isPackaged) {
        void win.loadURL(DEV_URL);
        win.webContents.openDevTools({ mode: 'detach' });
    }
    else {
        // TODO: packaged load from file
        void win.loadFile(node_path_1.default.join(electron_1.app.getAppPath(), 'dist', 'index.html'));
    }
    win.webContents.on('did-fail-load', (_event, errorCode, errorDescription, validatedURL) => {
        // Helps debug "black window" cases where the renderer never loads.
        // eslint-disable-next-line no-console
        console.error('[did-fail-load]', { errorCode, errorDescription, validatedURL });
    });
    win.webContents.on('render-process-gone', (_event, details) => {
        // eslint-disable-next-line no-console
        console.error('[render-process-gone]', details);
    });
    return win;
}
electron_1.app.whenReady().then(() => {
    createWindow();
    electron_1.app.on('activate', () => {
        if (electron_1.BrowserWindow.getAllWindows().length === 0)
            createWindow();
    });
});
electron_1.app.on('window-all-closed', () => {
    if (process.platform !== 'darwin')
        electron_1.app.quit();
});
electron_1.ipcMain.handle('fs:readText', async (_evt, filePath) => {
    return await promises_1.default.readFile(filePath, 'utf-8');
});
electron_1.ipcMain.handle('fs:writeText', async (_evt, filePath, content) => {
    await promises_1.default.writeFile(filePath, content, 'utf-8');
    return true;
});
electron_1.ipcMain.handle('fs:writeBase64', async (_evt, filePath, base64) => {
    // base64 without dataURL prefix
    const buf = Buffer.from(base64, 'base64');
    await promises_1.default.writeFile(filePath, buf);
    return true;
});
function guessMime(filePath) {
    const ext = node_path_1.default.extname(filePath).toLowerCase();
    if (ext === '.png')
        return 'image/png';
    if (ext === '.jpg' || ext === '.jpeg')
        return 'image/jpeg';
    if (ext === '.gif')
        return 'image/gif';
    if (ext === '.webp')
        return 'image/webp';
    if (ext === '.bmp')
        return 'image/bmp';
    if (ext === '.svg')
        return 'image/svg+xml';
    if (ext === '.pdf')
        return 'application/pdf';
    if (ext === '.html' || ext === '.htm')
        return 'text/html';
    return 'application/octet-stream';
}
electron_1.ipcMain.handle('fs:readAsDataUrl', async (_evt, filePath) => {
    const buf = await promises_1.default.readFile(filePath);
    const mime = guessMime(filePath);
    const b64 = buf.toString('base64');
    return `data:${mime};base64,${b64}`;
});
electron_1.ipcMain.handle('dialog:openFile', async (_evt, opts) => {
    const res = await electron_1.dialog.showOpenDialog({
        title: opts.title ?? 'Open',
        properties: ['openFile'],
        filters: opts.filters,
    });
    if (res.canceled)
        return null;
    return res.filePaths[0] ?? null;
});
electron_1.ipcMain.handle('dialog:saveFile', async (_evt, opts) => {
    const res = await electron_1.dialog.showSaveDialog({
        title: opts.title ?? 'Save',
        defaultPath: opts.defaultPath,
        filters: opts.filters,
    });
    if (res.canceled)
        return null;
    return res.filePath ?? null;
});
electron_1.ipcMain.handle('latex:renderToPdf', async (_evt, latexContent, outputPath) => {
    // Get the path to the Python script
    // In dev: __dirname is dist-electron/, so scripts/ is at ../scripts/
    // In production: __dirname is the app root, so scripts/ is at ./scripts/
    const scriptPath = electron_1.app.isPackaged
        ? node_path_1.default.join(process.resourcesPath, 'scripts', 'render_latex.py')
        : node_path_1.default.join(__dirname, '..', 'scripts', 'render_latex.py');
    // Create a temporary file for LaTeX content
    const tmpDir = await promises_1.default.mkdtemp(node_path_1.default.join(node_os_1.default.tmpdir(), 'latex-'));
    const tmpTexFile = node_path_1.default.join(tmpDir, 'input.tex');
    try {
        // Write LaTeX content to temporary file
        await promises_1.default.writeFile(tmpTexFile, latexContent, 'utf-8');
        // Determine Python command
        const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';
        // Run Python script
        return await new Promise((resolve) => {
            const pythonProcess = (0, node_child_process_1.spawn)(pythonCmd, [scriptPath, tmpTexFile, outputPath], {
                cwd: node_path_1.default.dirname(scriptPath),
                stdio: ['ignore', 'pipe', 'pipe'],
            });
            let stdout = '';
            let stderr = '';
            pythonProcess.stdout.on('data', (data) => {
                stdout += data.toString();
            });
            pythonProcess.stderr.on('data', (data) => {
                stderr += data.toString();
            });
            pythonProcess.on('close', async (code) => {
                try {
                    // Clean up temporary directory
                    await promises_1.default.rm(tmpDir, { recursive: true, force: true });
                }
                catch {
                    // Ignore cleanup errors
                }
                if (code !== 0) {
                    resolve({
                        success: false,
                        error: stderr || stdout || `Python script exited with code ${code}`,
                    });
                    return;
                }
                // Parse JSON output from Python script
                try {
                    const result = JSON.parse(stdout.trim());
                    resolve(result);
                }
                catch {
                    resolve({
                        success: false,
                        error: `Failed to parse Python output: ${stdout}`,
                    });
                }
            });
            pythonProcess.on('error', async (err) => {
                try {
                    await promises_1.default.rm(tmpDir, { recursive: true, force: true });
                }
                catch {
                    // Ignore cleanup errors
                }
                resolve({
                    success: false,
                    error: `Failed to spawn Python process: ${err.message}. Make sure Python is installed and accessible.`,
                });
            });
        });
    }
    catch (error) {
        try {
            await promises_1.default.rm(tmpDir, { recursive: true, force: true });
        }
        catch {
            // Ignore cleanup errors
        }
        return {
            success: false,
            error: `Failed to create temporary file: ${error instanceof Error ? error.message : String(error)}`,
        };
    }
});
// Render LaTeX to a temporary PDF and return a data URL for preview
electron_1.ipcMain.handle('latex:renderToPdfDataUrl', async (_evt, latexContent) => {
    const tmpDir = await promises_1.default.mkdtemp(node_path_1.default.join(node_os_1.default.tmpdir(), 'latex-prev-'));
    const outPdf = node_path_1.default.join(tmpDir, 'preview.pdf');
    // Reuse existing handler logic by spawning python directly here (keeps it self-contained)
    const scriptPath = electron_1.app.isPackaged
        ? node_path_1.default.join(process.resourcesPath, 'scripts', 'render_latex.py')
        : node_path_1.default.join(__dirname, '..', 'scripts', 'render_latex.py');
    const tmpTexFile = node_path_1.default.join(tmpDir, 'input.tex');
    await promises_1.default.writeFile(tmpTexFile, latexContent, 'utf-8');
    const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';
    const result = await new Promise((resolve) => {
        const pythonProcess = (0, node_child_process_1.spawn)(pythonCmd, [scriptPath, tmpTexFile, outPdf], {
            cwd: node_path_1.default.dirname(scriptPath),
            stdio: ['ignore', 'pipe', 'pipe'],
        });
        let stdout = '';
        let stderr = '';
        pythonProcess.stdout.on('data', (data) => (stdout += data.toString()));
        pythonProcess.stderr.on('data', (data) => (stderr += data.toString()));
        pythonProcess.on('close', (code) => {
            if (code !== 0) {
                resolve({ success: false, error: stderr || stdout || `Python exited with ${code}` });
                return;
            }
            try {
                const parsed = JSON.parse(stdout.trim());
                resolve(parsed);
            }
            catch {
                resolve({ success: false, error: `Failed to parse Python output: ${stdout}` });
            }
        });
        pythonProcess.on('error', (err) => {
            resolve({ success: false, error: `Failed to spawn Python: ${err.message}` });
        });
    });
    if (!result.success) {
        await promises_1.default.rm(tmpDir, { recursive: true, force: true }).catch(() => { });
        return { success: false, error: result.error };
    }
    try {
        const buf = await promises_1.default.readFile(outPdf);
        const b64 = buf.toString('base64');
        await promises_1.default.rm(tmpDir, { recursive: true, force: true }).catch(() => { });
        return { success: true, dataUrl: `data:application/pdf;base64,${b64}` };
    }
    catch (e) {
        await promises_1.default.rm(tmpDir, { recursive: true, force: true }).catch(() => { });
        return { success: false, error: e instanceof Error ? e.message : String(e) };
    }
});
