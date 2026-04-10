const { contextBridge, ipcRenderer } = require('electron');

const engineBridgeBaseUrl = process.env.EARLOOP_ENGINE_BRIDGE_URL || null;

contextBridge.exposeInMainWorld('earloopDesktop', {
  isDesktop: true,
  engineBridgeBaseUrl,
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
