declare global {
  const __APP_VERSION__: string;

  interface Window {
    earloopDesktop?: {
      isDesktop: boolean;
      engineBridgeBaseUrl: string | null;
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
