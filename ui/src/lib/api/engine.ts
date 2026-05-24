import { adapterEngineApi } from "@/lib/api/engine.adapter";
import { mockEngineApi } from "@/lib/api/engine.mock";

export type {
  AudioDevicesCatalog,
  CreateSessionInput,
  DeleteProfileInput,
  EngineApi,
  EngineBridgeTransport,
  EngineConfig,
  EngineDomainState,
  EngineProviderName,
  EngineStatus,
  EngineTransportCommand,
  EngineTransportPayloadMap,
  EngineTransportRequest,
  EngineTransportResponse,
  EngineTransportResponseMap,
  GenerateNextPairInput,
  MainRuntimeState,
  PreviewSessionTargetInput,
  SaveProfileInput,
  SaveProfileFromSessionInput,
  SaveProfileResult,
  SetActiveProfileInput,
  SetProcessingEnabledInput,
  SessionPreviewStatus,
  SessionSnapshot,
  UpdateEngineConfigInput,
  UpdateProfileInput,
} from "@/lib/api/engine.types";

import type { EngineApi, EngineProviderName } from "@/lib/api/engine.types";

type EngineProviderResolution = {
  provider: EngineProviderName;
  reason: string;
};

function resolveEngineProvider(): EngineProviderResolution {
  const viteEnv = (import.meta as ImportMeta & { env?: Record<string, string | undefined> }).env;
  const hasDesktopController = Boolean(window.earloopDesktop?.isDesktop && window.earloopDesktop.controller?.isAvailable);
  if (viteEnv?.VITE_ENGINE_PROVIDER === "adapter") {
    if (hasDesktopController) {
      return { provider: "adapter", reason: "VITE_ENGINE_PROVIDER=adapter and desktop controller is available" };
    }
    return { provider: "adapter", reason: "VITE_ENGINE_PROVIDER=adapter and HTTP/Vite bridge is expected" };
  }
  if (viteEnv?.VITE_ENGINE_PROVIDER === "mock") {
    return { provider: "mock", reason: "VITE_ENGINE_PROVIDER=mock" };
  }
  if (hasDesktopController) {
    return { provider: "adapter", reason: "desktop controller is available" };
  }
  if (window.earloopDesktop?.isDesktop) {
    return { provider: "mock", reason: "window.earloopDesktop exists but controller is unavailable" };
  }
  return { provider: "mock", reason: "desktop shell is unavailable" };
}

const providers: Record<EngineProviderName, EngineApi> = {
  mock: mockEngineApi,
  adapter: adapterEngineApi,
};

const engineProviderResolution = resolveEngineProvider();
export const engineProviderName = engineProviderResolution.provider;
export const engineApi = providers[engineProviderName];

if (typeof window !== "undefined") {
  const payload = {
    provider: engineProviderName,
    reason: engineProviderResolution.reason,
    isDesktop: Boolean(window.earloopDesktop?.isDesktop),
    hasController: Boolean(window.earloopDesktop?.controller),
    controllerAvailable: Boolean(window.earloopDesktop?.controller?.isAvailable),
  };
  if (engineProviderName === "mock") {
    console.warn("[engine] mock runtime active", payload);
  } else {
    console.info("[engine] adapter runtime active", payload);
  }
}

export const {
  createSession,
  deleteProfile,
  generateNextPair,
  getDomainState,
  getEngineConfig,
  getMainState,
  getEngineStatus,
  listAudioDevices,
  listProfiles,
  previewSessionTarget,
  saveProfile,
  saveProfileFromSession,
  setActiveProfile,
  setProcessingEnabled,
  startSession,
  updateEngineConfig,
  updateProfile,
} = engineApi;
