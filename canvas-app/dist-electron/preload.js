"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const electron_1 = require("electron");
electron_1.contextBridge.exposeInMainWorld('canvasApi', {
    readTextFile: (filePath) => electron_1.ipcRenderer.invoke('fs:readText', filePath),
    readFileAsDataUrl: (filePath) => electron_1.ipcRenderer.invoke('fs:readAsDataUrl', filePath),
    writeTextFile: (filePath, content) => electron_1.ipcRenderer.invoke('fs:writeText', filePath, content),
    writeBase64File: (filePath, base64) => electron_1.ipcRenderer.invoke('fs:writeBase64', filePath, base64),
    openFileDialog: (opts) => electron_1.ipcRenderer.invoke('dialog:openFile', opts),
    saveFileDialog: (opts) => electron_1.ipcRenderer.invoke('dialog:saveFile', opts),
    renderLatexToPdf: (latexContent, outputPath) => electron_1.ipcRenderer.invoke('latex:renderToPdf', latexContent, outputPath),
    renderLatexToPdfDataUrl: (latexContent) => electron_1.ipcRenderer.invoke('latex:renderToPdfDataUrl', latexContent),
});
