const { app, BrowserWindow, ipcMain, Menu, shell } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

const { DesktopController } = require('./desktop-controller');

let mainWindow = null;
const desktopController = new DesktopController({ app });

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
  desktopController.writeDesktopLog('desktop.app.ready', {
    runtimePaths: desktopController.resolveRuntimePaths(),
    isPackaged: app.isPackaged,
  });
  Menu.setApplicationMenu(null);
  mainWindow = createWindow();
  const unsubscribeControllerStatus = desktopController.subscribeStatus((status) => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('desktop-controller:status-changed', status);
    }
  });
  updateSplashStatus(mainWindow, 'EarLoop', 'Starting backend runtime...');
  try {
    await desktopController.initialize();
    updateSplashStatus(mainWindow, 'EarLoop', 'Waiting for backend readiness...');
    await desktopController.ensureAudioBridgeReady();
    updateSplashStatus(mainWindow, 'EarLoop', 'Loading interface...');
    if (mainWindow && !mainWindow.isDestroyed()) {
      desktopController.writeDesktopLog('desktop.ui.load_allowed', {
        backendReady: desktopController.getStatus().audioBridge.ready,
      });
      await loadMainUi(mainWindow);
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    desktopController.writeDesktopLog('desktop.startup.failed', { error: message });
    updateSplashStatus(mainWindow, 'Backend startup failed', message, 'error');
  }
  mainWindow?.on('closed', () => {
    unsubscribeControllerStatus();
  });
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

ipcMain.on('desktop-controller:get-client-config-sync', (event) => {
  event.returnValue = desktopController.getClientConfig();
});

ipcMain.handle('desktop-controller:get-status', () => desktopController.getStatus());
ipcMain.handle('desktop-controller:probe', () => desktopController.probe());
ipcMain.handle('desktop-controller:start-audio-bridge', async () => {
  desktopController.startAudioBridge();
  await desktopController.ensureAudioBridgeReady();
  return desktopController.getStatus();
});
ipcMain.handle('desktop-controller:stop-audio-bridge', async () => {
  await desktopController.stopAudioBridge();
  return desktopController.getStatus();
});
ipcMain.handle('desktop-controller:restart-audio-bridge', () => desktopController.restartAudioBridge());
ipcMain.handle('desktop-controller:get-ml-worker-status', () => desktopController.getMlWorkerStatus());
ipcMain.handle('desktop-controller:start-ml-worker', () => desktopController.startMlWorker());
ipcMain.handle('desktop-controller:stop-ml-worker', () => desktopController.stopMlWorker());
ipcMain.handle('desktop-controller:send-ml-command', (_event, payload) => {
  const command = typeof payload?.command === 'string' ? payload.command : '';
  const data = payload?.payload && typeof payload.payload === 'object' ? payload.payload : {};
  return desktopController.sendMlCommand(command, data);
});

ipcMain.handle('desktop:install-virtual-cable', async () => {
  if (process.platform !== 'win32') {
    desktopController.writeDesktopLog('desktop.install_virtual_cable.unsupported', {
      platform: process.platform,
    });
    return {
      ok: false,
      error: 'Virtual cable installer is only supported on Windows',
    };
  }

  const installerPath = getVirtualCableInstallerPath();
  if (!fs.existsSync(installerPath)) {
    desktopController.writeDesktopLog('desktop.install_virtual_cable.missing', {
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
    desktopController.writeDesktopLog('desktop.install_virtual_cable.started', {
      installerPath,
      pid: child.pid ?? null,
    });
    return {
      ok: true,
      installerPath,
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    desktopController.writeDesktopLog('desktop.install_virtual_cable.failed', {
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
    desktopController.writeDesktopLog('desktop.open_sound_settings.unsupported', {
      platform: process.platform,
    });
    return {
      ok: false,
      error: 'Sound settings shortcut is only supported on Windows',
    };
  }

  try {
    await shell.openExternal('ms-settings:sound');
    desktopController.writeDesktopLog('desktop.open_sound_settings.started', {});
    return { ok: true };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    desktopController.writeDesktopLog('desktop.open_sound_settings.failed', {
      error: message,
    });
    return {
      ok: false,
      error: message,
    };
  }
});

app.on('window-all-closed', () => {
  void desktopController.shutdown();
  mainWindow = null;
  app.quit();
});
