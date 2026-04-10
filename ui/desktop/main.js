const { app, BrowserWindow, ipcMain, Menu, shell } = require('electron');
const http = require('http');
const path = require('path');
const fs = require('fs');
const { randomUUID } = require('crypto');
const { spawn } = require('child_process');
const readline = require('readline');

let pyProcess = null;
let bridgeServer = null;
let bridgeBaseUrl = null;
let stdoutReader = null;
let stderrReader = null;
let mainWindow = null;
const pendingRequests = new Map();
let runtimePaths = null;
let backendReady = false;
let backendReadinessPromise = null;
let firstSuccessfulBackendResponseLogged = false;

const STARTUP_BACKEND_READY_TIMEOUT_MS = 25000;
const NORMAL_BACKEND_COMMAND_TIMEOUT_MS = 6000;

function ensureDir(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

function resolveRuntimePaths() {
  if (runtimePaths) {
    return runtimePaths;
  }

  const runtimeRoot = path.join(app.getPath('appData'), 'EarLoop', 'engine-runtime');
  const dataDir = path.join(runtimeRoot, 'data');
  const logsDir = path.join(runtimeRoot, 'logs');
  const paths = {
    runtimeRoot,
    dataDir,
    logsDir,
    domainStatePath: path.join(dataDir, 'domain-state.json'),
    userMetaPath: path.join(dataDir, 'user-meta.json'),
    eventLogPath: path.join(logsDir, 'session-events.jsonl'),
    audioDiagnosticLogPath: path.join(logsDir, 'audio-diagnostics.jsonl'),
    runtimeLogPath: path.join(logsDir, 'runtime.log'),
    desktopBridgeLogPath: path.join(logsDir, 'desktop-bridge.log'),
  };

  ensureDir(runtimeRoot);
  ensureDir(dataDir);
  ensureDir(logsDir);
  runtimePaths = paths;
  return paths;
}

function writeDesktopLog(event, payload = {}) {
  try {
    const paths = resolveRuntimePaths();
    const row = {
      timestampUtc: new Date().toISOString(),
      event,
      ...payload,
    };
    fs.appendFileSync(paths.desktopBridgeLogPath, `${JSON.stringify(row)}\n`, 'utf-8');
  } catch (error) {
    console.error('Failed to write desktop bridge log:', error);
  }
}

function buildBackendEnv() {
  const paths = resolveRuntimePaths();
  const diagnosticPreviewMode = process.argv.includes('--audio-diagnostic-mode') ? '1' : (process.env.EARLOOP_AUDIO_DIAGNOSTIC_MODE || '0');
  return {
    ...process.env,
    EARLOOP_RUNTIME_ROOT: paths.runtimeRoot,
    EARLOOP_ENGINE_STATE_PATH: paths.domainStatePath,
    EARLOOP_ENGINE_USER_META_PATH: paths.userMetaPath,
    EARLOOP_ENGINE_EVENT_LOG_PATH: paths.eventLogPath,
    EARLOOP_AUDIO_DIAGNOSTIC_LOG_PATH: paths.audioDiagnosticLogPath,
    EARLOOP_RUNTIME_LOG_PATH: paths.runtimeLogPath,
    EARLOOP_DESKTOP_BRIDGE_LOG_PATH: paths.desktopBridgeLogPath,
    EARLOOP_AUDIO_DIAGNOSTIC_MODE: diagnosticPreviewMode,
    EARLOOP_APP_IS_PACKAGED: app.isPackaged ? '1' : '0',
    EARLOOP_STRICT_DESKTOP_MODE: app.isPackaged ? '1' : '0',
  };
}

function cleanupReaders() {
  if (stdoutReader) {
    stdoutReader.removeAllListeners();
    stdoutReader.close();
    stdoutReader = null;
  }
  if (stderrReader) {
    stderrReader.removeAllListeners();
    stderrReader.close();
    stderrReader = null;
  }
}

function rejectPendingRequests(error) {
  for (const [requestId, pending] of pendingRequests.entries()) {
    clearTimeout(pending.timeout);
    pending.reject(error);
    pendingRequests.delete(requestId);
  }
}

function getBackendPath() {
  if (app.isPackaged) {
    return path.join(process.resourcesPath, 'backend', 'run_server.exe');
  }
  return path.join(__dirname, 'backend', 'run_server.exe');
}

function getFrontendPath() {
  if (app.isPackaged) {
    return path.join(process.resourcesPath, 'ui-dist', 'index.html');
  }
  return path.join(__dirname, '..', 'dist', 'index.html');
}

function getSplashPath() {
  return path.join(__dirname, 'splash.html');
}

function getDriversDir() {
  if (app.isPackaged) {
    return path.join(process.resourcesPath, 'drivers');
  }
  return path.join(__dirname, '..', '..', 'dist', 'driver');
}

function getVirtualCableInstallerPath() {
  return path.join(getDriversDir(), 'VBCABLE_Setup_x64.exe');
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function updateSplashStatus(win, title, subtitle, appearance = 'loading') {
  if (!win || win.isDestroyed()) {
    return;
  }
  const safeTitle = JSON.stringify(escapeHtml(title));
  const safeSubtitle = JSON.stringify(escapeHtml(subtitle));
  const safeAppearance = JSON.stringify(appearance);
  win.webContents.executeJavaScript(`
    (() => {
      const titleEl = document.getElementById('splash-title');
      const subtitleEl = document.getElementById('splash-subtitle');
      if (titleEl) titleEl.textContent = ${safeTitle};
      if (subtitleEl) subtitleEl.textContent = ${safeSubtitle};
      document.body.dataset.state = ${safeAppearance};
    })();
  `).catch(() => {});
}

function broadcastWindowState(win) {
  win.webContents.send('window-state-changed', {
    isMaximized: win.isMaximized(),
  });
}

function startPython() {
  if (pyProcess && !pyProcess.killed && pyProcess.exitCode === null) {
    return;
  }
  const backendPath = getBackendPath();
  const paths = resolveRuntimePaths();
  console.log('Starting backend:', backendPath);
  writeDesktopLog('backend.spawn.start', {
    backendPath,
    runtimePaths: paths,
  });

  pyProcess = spawn(backendPath, [], {
    stdio: ['pipe', 'pipe', 'pipe'],
    cwd: paths.runtimeRoot,
    env: buildBackendEnv(),
  });

  stdoutReader = readline.createInterface({ input: pyProcess.stdout });
  stderrReader = readline.createInterface({ input: pyProcess.stderr });
  backendReady = false;
  backendReadinessPromise = null;
  firstSuccessfulBackendResponseLogged = false;

  stdoutReader.on('line', (line) => {
    if (!line.trim()) return;
    try {
      const response = JSON.parse(line);
      const requestId = response.request_id;
      if (!requestId || !pendingRequests.has(requestId)) {
        console.log(`PY STDOUT: ${line}`);
        writeDesktopLog('backend.stdout.unmatched', { line });
        return;
      }
      const pending = pendingRequests.get(requestId);
      clearTimeout(pending.timeout);
      pendingRequests.delete(requestId);
      if (!firstSuccessfulBackendResponseLogged) {
        firstSuccessfulBackendResponseLogged = true;
        writeDesktopLog('backend.first_response', {
          requestId,
          ok: Boolean(response.ok),
          command: pending.command,
        });
      }
      writeDesktopLog('backend.response', {
        requestId,
        command: pending.command,
        ok: Boolean(response.ok),
        error: response.error?.message ?? null,
      });
      pending.resolve(response);
    } catch (error) {
      console.log(`PY STDOUT: ${line}`);
      console.error('Backend response parse error:', error);
      writeDesktopLog('backend.stdout.parse_error', {
        line,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  });

  stderrReader.on('line', (line) => {
    if (!line.trim()) return;
    console.log(`PY STDERR: ${line}`);
    writeDesktopLog('backend.stderr', { line });
  });

  pyProcess.on('error', (err) => {
    console.error('Python process error:', err);
    writeDesktopLog('backend.process_error', {
      error: err instanceof Error ? err.message : String(err),
    });
    backendReady = false;
    backendReadinessPromise = null;
    rejectPendingRequests(err);
  });

  pyProcess.on('exit', (code, signal) => {
    console.error(`Python process exited: code=${code} signal=${signal}`);
    writeDesktopLog('backend.exit', { code, signal });
    backendReady = false;
    backendReadinessPromise = null;
    rejectPendingRequests(new Error(`Python process exited: code=${code} signal=${signal}`));
    cleanupReaders();
    pyProcess = null;
  });
}

function sendEngineCommandRaw(command, payload = {}, options = {}) {
  startPython();
  if (!pyProcess || pyProcess.exitCode !== null || pyProcess.killed) {
    writeDesktopLog('bridge.command.rejected', {
      command,
      reason: 'backend_not_running',
    });
    return Promise.reject(new Error('Python backend is not running'));
  }

  const requestId = randomUUID();
  const timeoutMs = options.timeoutMs ?? NORMAL_BACKEND_COMMAND_TIMEOUT_MS;
  const stage = options.stage ?? 'runtime';
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      pendingRequests.delete(requestId);
      writeDesktopLog('bridge.command.timeout', { command, requestId, stage, timeoutMs });
      reject(new Error(`Engine request timed out: ${command}`));
    }, timeoutMs);

    pendingRequests.set(requestId, { resolve, reject, timeout, command, stage });
    writeDesktopLog('bridge.command.send', {
      command,
      requestId,
      payload,
      stage,
      timeoutMs,
    });

    pyProcess.stdin.write(`${JSON.stringify({
      request_id: requestId,
      command,
      payload,
    })}\n`, 'utf-8', (error) => {
      if (!error) return;
      clearTimeout(timeout);
      pendingRequests.delete(requestId);
      writeDesktopLog('bridge.command.write_error', {
        command,
        requestId,
        error: error.message,
      });
      reject(error);
    });
  });
}

async function ensureBackendReady() {
  if (backendReady) {
    return;
  }
  if (backendReadinessPromise) {
    writeDesktopLog('backend.ready.wait_reuse', {});
    return backendReadinessPromise;
  }

  startPython();
  writeDesktopLog('backend.ready.wait_start', {
    timeoutMs: STARTUP_BACKEND_READY_TIMEOUT_MS,
  });

  backendReadinessPromise = (async () => {
    let attempt = 0;
    const startedAt = Date.now();
    while (Date.now() - startedAt < STARTUP_BACKEND_READY_TIMEOUT_MS) {
      attempt += 1;
      try {
        writeDesktopLog('backend.ready.attempt', {
          attempt,
          elapsedMs: Date.now() - startedAt,
        });
        const response = await sendEngineCommandRaw('get_engine_status', {}, {
          timeoutMs: Math.max(1500, STARTUP_BACKEND_READY_TIMEOUT_MS - (Date.now() - startedAt)),
          stage: 'startup_readiness',
        });
        if (!response.ok) {
          throw new Error(response.error?.message || 'Backend readiness command failed');
        }
        backendReady = true;
        writeDesktopLog('backend.ready.success', {
          attempt,
          elapsedMs: Date.now() - startedAt,
        });
        return;
      } catch (error) {
        const elapsedMs = Date.now() - startedAt;
        writeDesktopLog('backend.ready.retry', {
          attempt,
          elapsedMs,
          error: error instanceof Error ? error.message : String(error),
        });
        if (elapsedMs >= STARTUP_BACKEND_READY_TIMEOUT_MS) {
          break;
        }
        await new Promise((resolve) => setTimeout(resolve, 350));
      }
    }

    const error = new Error(`Backend startup timed out after ${STARTUP_BACKEND_READY_TIMEOUT_MS}ms`);
    writeDesktopLog('backend.ready.timeout', {
      timeoutMs: STARTUP_BACKEND_READY_TIMEOUT_MS,
      error: error.message,
    });
    throw error;
  })();

  try {
    await backendReadinessPromise;
  } finally {
    if (!backendReady) {
      backendReadinessPromise = null;
    }
  }
}

async function sendEngineCommand(command, payload = {}) {
  await ensureBackendReady();
  return sendEngineCommandRaw(command, payload, {
    timeoutMs: NORMAL_BACKEND_COMMAND_TIMEOUT_MS,
    stage: 'runtime',
  });
}

function readJsonBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on('data', (chunk) => chunks.push(Buffer.from(chunk)));
    req.on('end', () => {
      try {
        const raw = Buffer.concat(chunks).toString('utf-8').trim();
        resolve(raw ? JSON.parse(raw) : {});
      } catch (error) {
        reject(error);
      }
    });
    req.on('error', reject);
  });
}

function writeJson(res, statusCode, payload) {
  res.statusCode = statusCode;
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  res.end(JSON.stringify(payload));
}

async function startBridgeServer() {
  if (bridgeServer && bridgeBaseUrl) {
    return bridgeBaseUrl;
  }

  bridgeServer = http.createServer(async (req, res) => {
    if (!req.url) {
      writeJson(res, 400, { ok: false, error: 'Missing request URL' });
      return;
    }

    if (req.url === '/__engine/health') {
      try {
        writeDesktopLog('bridge.http.health', {});
        await ensureBackendReady();
        writeJson(res, 200, { ok: true, pid: pyProcess?.pid ?? null });
      } catch (error) {
        writeJson(res, 503, { ok: false, error: error instanceof Error ? error.message : 'Engine health check failed' });
      }
      return;
    }

    if (req.url === '/__engine/command' && req.method === 'POST') {
      try {
        const body = await readJsonBody(req);
        const command = typeof body?.command === 'string' ? body.command : '';
        const payload = body?.payload && typeof body.payload === 'object' ? body.payload : {};
        writeDesktopLog('bridge.http.command', {
          command,
          payload,
        });

        if (!command) {
          writeJson(res, 400, { ok: false, error: 'command is required' });
          return;
        }

        const response = await sendEngineCommand(command, payload);
        writeJson(res, 200, {
          ok: Boolean(response.ok),
          data: response.result,
          error: response.ok ? undefined : (response.error?.message || 'Engine command failed'),
        });
      } catch (error) {
        writeDesktopLog('bridge.http.command_error', {
          error: error instanceof Error ? error.message : String(error),
        });
        writeJson(res, 500, { ok: false, error: error instanceof Error ? error.message : 'Engine bridge request failed' });
      }
      return;
    }

    writeJson(res, 404, { ok: false, error: 'Not found' });
  });

  await new Promise((resolve, reject) => {
    bridgeServer.once('error', reject);
    bridgeServer.listen(0, '127.0.0.1', () => resolve());
  });

  const address = bridgeServer.address();
  if (!address || typeof address === 'string') {
    throw new Error('Failed to resolve desktop engine bridge address');
  }

  bridgeBaseUrl = `http://127.0.0.1:${address.port}`;
  process.env.EARLOOP_ENGINE_BRIDGE_URL = bridgeBaseUrl;
  console.log('Desktop engine bridge:', bridgeBaseUrl);
  return bridgeBaseUrl;
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 1100,
    minHeight: 720,
    frame: false,
    autoHideMenuBar: true,
    backgroundColor: '#05060a',
    show: true,
    webPreferences: {
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
    }
  });

  win.on('maximize', () => broadcastWindowState(win));
  win.on('unmaximize', () => broadcastWindowState(win));
  win.on('enter-full-screen', () => broadcastWindowState(win));
  win.on('leave-full-screen', () => broadcastWindowState(win));

  console.log('Loading splash:', getSplashPath());
  win.loadFile(getSplashPath());
  return win;
}

async function loadMainUi(win) {
  const frontendPath = getFrontendPath();
  console.log('Loading UI:', frontendPath);
  await win.loadFile(frontendPath);
  broadcastWindowState(win);
}

app.whenReady().then(async () => {
  writeDesktopLog('desktop.app.ready', {
    runtimePaths: resolveRuntimePaths(),
    isPackaged: app.isPackaged,
  });
  Menu.setApplicationMenu(null);
  mainWindow = createWindow();
  updateSplashStatus(mainWindow, 'EarLoop', 'Starting backend runtime...');
  try {
    startPython();
    await startBridgeServer();
    updateSplashStatus(mainWindow, 'EarLoop', 'Waiting for backend readiness...');
    await ensureBackendReady();
    updateSplashStatus(mainWindow, 'EarLoop', 'Loading interface...');
    if (mainWindow && !mainWindow.isDestroyed()) {
      writeDesktopLog('desktop.ui.load_allowed', {
        backendReady,
      });
      await loadMainUi(mainWindow);
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    writeDesktopLog('desktop.startup.failed', { error: message });
    updateSplashStatus(mainWindow, 'Backend startup failed', message, 'error');
  }
});

ipcMain.handle('window:minimize', () => {
  const win = BrowserWindow.getFocusedWindow() || mainWindow;
  win?.minimize();
});

ipcMain.handle('window:maximize-toggle', () => {
  const win = BrowserWindow.getFocusedWindow() || mainWindow;
  if (!win) return { isMaximized: false };
  if (win.isMaximized()) {
    win.unmaximize();
  } else {
    win.maximize();
  }
  return { isMaximized: win.isMaximized() };
});

ipcMain.handle('window:close', () => {
  const win = BrowserWindow.getFocusedWindow() || mainWindow;
  win?.close();
});

ipcMain.handle('window:get-state', () => {
  const win = BrowserWindow.getFocusedWindow() || mainWindow;
  return { isMaximized: win?.isMaximized() ?? false };
});

ipcMain.handle('desktop:get-flags', () => ({
  forceAudioSetupScreen: process.env.EARLOOP_FORCE_AUDIO_SETUP_SCREEN === '1',
}));

ipcMain.handle('desktop:install-virtual-cable', async () => {
  if (process.platform !== 'win32') {
    writeDesktopLog('desktop.install_virtual_cable.unsupported', {
      platform: process.platform,
    });
    return {
      ok: false,
      error: 'Virtual cable installer is only supported on Windows',
    };
  }

  const installerPath = getVirtualCableInstallerPath();
  if (!fs.existsSync(installerPath)) {
    writeDesktopLog('desktop.install_virtual_cable.missing', {
      installerPath,
    });
    return {
      ok: false,
      error: `VB-Cable installer not found: ${installerPath}`,
    };
  }

  try {
    const child = spawn(installerPath, [], {
      detached: true,
      stdio: 'ignore',
    });
    child.unref();
    writeDesktopLog('desktop.install_virtual_cable.started', {
      installerPath,
      pid: child.pid ?? null,
    });
    return {
      ok: true,
      installerPath,
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    writeDesktopLog('desktop.install_virtual_cable.failed', {
      installerPath,
      error: message,
    });
    return {
      ok: false,
      error: message,
    };
  }
});

ipcMain.handle('desktop:open-sound-settings', async () => {
  if (process.platform !== 'win32') {
    writeDesktopLog('desktop.open_sound_settings.unsupported', {
      platform: process.platform,
    });
    return {
      ok: false,
      error: 'Sound settings shortcut is only supported on Windows',
    };
  }

  try {
    await shell.openExternal('ms-settings:sound');
    writeDesktopLog('desktop.open_sound_settings.started', {});
    return { ok: true };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    writeDesktopLog('desktop.open_sound_settings.failed', {
      error: message,
    });
    return {
      ok: false,
      error: message,
    };
  }
});

app.on('window-all-closed', () => {
  if (bridgeServer) {
    bridgeServer.close();
    bridgeServer = null;
  }
  bridgeBaseUrl = null;
  cleanupReaders();
  if (pyProcess) {
    pyProcess.kill();
  }
  mainWindow = null;
  app.quit();
});
