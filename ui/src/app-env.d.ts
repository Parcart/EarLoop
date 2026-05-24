declare global {
  const __APP_VERSION__: string;

  interface Window {
    earloopDesktop?: {
      isDesktop: boolean;
      controller?: {
        isAvailable: boolean;
        sendCommand: (command: string, payload?: unknown) => Promise<{ ok: boolean; data?: unknown; error?: string }>;
        sendCommandSync: (command: string, payload?: unknown) => { ok: boolean; data?: unknown; error?: string };
        probe: () => Promise<boolean>;
        getStatus: () => Promise<{
          audioBridge: {
            state: string;
            transport: string;
            ready: boolean;
            pid: number | null;
            commandInFlightCount: number;
            lastError: string | null;
          };
          domainRuntime: {
            availability: string;
            implementation: string;
            lastError: string | null;
          };
          mlWorker: {
            state: string;
            availability: string;
            lastError: string | null;
          };
          runtime: {
            isPackaged: boolean;
            backendPath: string | null;
            runtimePaths: Record<string, string> | null;
          };
        }>;
        startAudioBridge: () => Promise<unknown>;
        stopAudioBridge: () => Promise<unknown>;
        restartAudioBridge: () => Promise<unknown>;
        getMlWorkerStatus: () => Promise<{ state: string; availability: string; lastError: string | null }>;
        startMlWorker: () => Promise<{ ok: boolean; error?: string; status?: unknown }>;
        stopMlWorker: () => Promise<{ ok: boolean; error?: string; status?: unknown }>;
        sendMlCommand: (command: string, payload?: unknown) => Promise<{ ok: boolean; error?: string; status?: unknown }>;
        onStatusChange: (listener: (payload: {
          audioBridge: {
            state: string;
            transport: string;
            ready: boolean;
            pid: number | null;
            commandInFlightCount: number;
            lastError: string | null;
          };
          domainRuntime: {
            availability: string;
            implementation: string;
            lastError: string | null;
          };
          mlWorker: {
            state: string;
            availability: string;
            lastError: string | null;
          };
          runtime: {
            isPackaged: boolean;
            backendPath: string | null;
            runtimePaths: Record<string, string> | null;
          };
        }) => void) => () => void;
      };
      flags?: {
        forceAudioSetupScreen: boolean;
      };
      platformTools?: {
        getFlags: () => Promise<{ forceAudioSetupScreen: boolean }>;
        installVirtualCable: () => Promise<{ ok: boolean; error?: string; installerPath?: string }>;
        openSoundSettings: () => Promise<{ ok: boolean; error?: string }>;
      };
      windowControls?: {
        minimize: () => Promise<void>;
        toggleMaximize: () => Promise<{ isMaximized: boolean }>;
        close: () => Promise<void>;
        getState: () => Promise<{ isMaximized: boolean }>;
        onStateChange: (listener: (payload: { isMaximized: boolean }) => void) => () => void;
      };
    };
  }
}

export {};
