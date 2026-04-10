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

function resolveEngineProviderName(): EngineProviderName {
  const viteEnv = (import.meta as ImportMeta & { env?: Record<string, string | undefined> }).env;
  if (viteEnv?.VITE_ENGINE_PROVIDER === "adapter") return "adapter";
  if (viteEnv?.VITE_ENGINE_PROVIDER === "mock") return "mock";
  if (window.earloopDesktop?.isDesktop && window.earloopDesktop.engineBridgeBaseUrl) return "adapter";
  return "mock";
}

const providers: Record<EngineProviderName, EngineApi> = {
  mock: mockEngineApi,
  adapter: adapterEngineApi,
};

export const engineProviderName = resolveEngineProviderName();
export const engineApi = providers[engineProviderName];

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
