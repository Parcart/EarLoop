const { contextBridge, ipcRenderer } = require('electron');

function getControllerClientConfigSync() {
  return ipcRenderer.sendSync('desktop-controller:get-client-config-sync');
}

function resolveInternalBridgeBaseUrl() {
  return getControllerClientConfigSync()?.internalBridgeBaseUrl ?? null;
}

function parseControllerResponse(raw) {
  const payload = JSON.parse(raw);
  if (typeof payload !== 'object' || payload === null || typeof payload.ok !== 'boolean') {
    throw new Error('Desktop controller returned invalid response');
  }
  return payload;
}

async function sendControllerCommand(command, payload) {
  const baseUrl = resolveInternalBridgeBaseUrl();
  if (!baseUrl) {
    return {
      ok: false,
      error: 'Desktop controller bridge is unavailable',
    };
  }
  const response = await fetch(`${baseUrl}/__engine/command`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      command,
      payload,
    }),
  });
  if (!response.ok) {
    return {
      ok: false,
      error: `Desktop controller HTTP error: ${response.status}`,
    };
  }
  return parseControllerResponse(await response.text());
}

function sendControllerCommandSync(command, payload) {
  const baseUrl = resolveInternalBridgeBaseUrl();
  if (!baseUrl) {
    return {
      ok: false,
      error: 'Desktop controller bridge is unavailable',
    };
  }
  const xhr = new XMLHttpRequest();
  xhr.open('POST', `${baseUrl}/__engine/command`, false);
  xhr.setRequestHeader('Content-Type', 'application/json');
  xhr.send(JSON.stringify({
    command,
    payload,
  }));
  if (xhr.status < 200 || xhr.status >= 300) {
    return {
      ok: false,
      error: `Desktop controller HTTP error: ${xhr.status}`,
    };
  }
  return parseControllerResponse(xhr.responseText);
}

contextBridge.exposeInMainWorld('earloopDesktop', {
  isDesktop: true,
  controller: {
    isAvailable: true,
    sendCommand: (command, payload) => sendControllerCommand(command, payload),
    sendCommandSync: (command, payload) => sendControllerCommandSync(command, payload),
    probe: () => ipcRenderer.invoke('desktop-controller:probe'),
    getStatus: () => ipcRenderer.invoke('desktop-controller:get-status'),
    startAudioBridge: () => ipcRenderer.invoke('desktop-controller:start-audio-bridge'),
    stopAudioBridge: () => ipcRenderer.invoke('desktop-controller:stop-audio-bridge'),
    restartAudioBridge: () => ipcRenderer.invoke('desktop-controller:restart-audio-bridge'),
    getMlWorkerStatus: () => ipcRenderer.invoke('desktop-controller:get-ml-worker-status'),
    startMlWorker: () => ipcRenderer.invoke('desktop-controller:start-ml-worker'),
    stopMlWorker: () => ipcRenderer.invoke('desktop-controller:stop-ml-worker'),
    sendMlCommand: (command, payload) => ipcRenderer.invoke('desktop-controller:send-ml-command', { command, payload }),
    onStatusChange: (listener) => {
      const wrappedListener = (_event, payload) => listener(payload);
      ipcRenderer.on('desktop-controller:status-changed', wrappedListener);
      return () => ipcRenderer.removeListener('desktop-controller:status-changed', wrappedListener);
    },
  },
  flags: {
    forceAudioSetupScreen: process.env.EARLOOP_FORCE_AUDIO_SETUP_SCREEN === '1',
  },
  platformTools: {
    getFlags: () => ipcRenderer.invoke('desktop:get-flags'),
    installVirtualCable: () => ipcRenderer.invoke('desktop:install-virtual-cable'),
    openSoundSettings: () => ipcRenderer.invoke('desktop:open-sound-settings'),
  },
  windowControls: {
    minimize: () => ipcRenderer.invoke('window:minimize'),
    toggleMaximize: () => ipcRenderer.invoke('window:maximize-toggle'),
    close: () => ipcRenderer.invoke('window:close'),
    getState: () => ipcRenderer.invoke('window:get-state'),
    onStateChange: (listener) => {
      const wrappedListener = (_event, payload) => listener(payload);
      ipcRenderer.on('window-state-changed', wrappedListener);
      return () => ipcRenderer.removeListener('window-state-changed', wrappedListener);
    },
  },
});
