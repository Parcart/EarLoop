import { mockEngineApi } from "@/lib/api/engine.mock";

import type {
  AudioDevicesCatalog,
  CreateSessionInput,
  DeleteProfileInput,
  EngineApi,
  EngineBridgeTransport,
  EngineConfig,
  EngineDomainState,
  EngineStatus,
  EngineTransportCommand,
  EngineTransportRequest,
  EngineTransportResponse,
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
import type { SavedProfile } from "@/lib/types/ui";

function resolveDesktopController() {
  return window.earloopDesktop?.controller ?? null;
}

type SyncBridgeTransport = EngineBridgeTransport & {
  requestSync: <TCommand extends EngineTransportCommand>(
    message: EngineTransportRequest<TCommand>,
  ) => EngineTransportResponse<TCommand>;
};

function isDesktopBridgeMode(): boolean {
  return Boolean(window.earloopDesktop?.isDesktop && resolveDesktopController()?.isAvailable);
}

function resolveHttpBridgeBaseUrl(): string | null {
  if (typeof window === "undefined" || !window.location) {
    return null;
  }
  return window.location.origin;
}

function reportTransportIssue(command: EngineTransportCommand, error: string, fallbackUsed: boolean) {
  const detail = {
    command,
    error,
    fallbackUsed,
    isDesktop: isDesktopBridgeMode(),
    timestampUtc: new Date().toISOString(),
  };
  console.error(`[engine-adapter] ${command}: ${error}`);
  window.dispatchEvent(new CustomEvent("earloop:transport-error", { detail }));
}

function createDesktopControllerTransport(): SyncBridgeTransport {
  return {
    async request<TCommand extends EngineTransportCommand>(message: EngineTransportRequest<TCommand>) {
      const controller = resolveDesktopController();
      if (!controller) {
        return {
          ok: false,
          error: "Desktop controller is not connected",
        };
      }
      return controller.sendCommand(message.command, message.payload) as Promise<EngineTransportResponse<TCommand>>;
    },
    requestSync<TCommand extends EngineTransportCommand>(message: EngineTransportRequest<TCommand>) {
      const controller = resolveDesktopController();
      if (!controller) {
        return {
          ok: false,
          error: "Desktop controller is not connected",
        };
      }
      return controller.sendCommandSync(message.command, message.payload) as EngineTransportResponse<TCommand>;
    },
  };
}

function createHttpBridgeTransport(): SyncBridgeTransport {
  return {
    async request<TCommand extends EngineTransportCommand>(message: EngineTransportRequest<TCommand>) {
      const baseUrl = resolveHttpBridgeBaseUrl();
      if (!baseUrl) {
        return {
          ok: false,
          error: "HTTP engine bridge is unavailable",
        };
      }
      const response = await fetch(`${baseUrl}/__engine/command`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          command: message.command,
          payload: message.payload,
        }),
      });
      if (!response.ok) {
        return {
          ok: false,
          error: `HTTP engine bridge error: ${response.status}`,
        };
      }
      return await response.json() as EngineTransportResponse<TCommand>;
    },
    requestSync<TCommand extends EngineTransportCommand>(message: EngineTransportRequest<TCommand>) {
      const baseUrl = resolveHttpBridgeBaseUrl();
      if (!baseUrl) {
        return {
          ok: false,
          error: "HTTP engine bridge is unavailable",
        };
      }
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${baseUrl}/__engine/command`, false);
      xhr.setRequestHeader("Content-Type", "application/json");
      xhr.send(JSON.stringify({
        command: message.command,
        payload: message.payload,
      }));
      if (xhr.status < 200 || xhr.status >= 300) {
        return {
          ok: false,
          error: `HTTP engine bridge error: ${xhr.status}`,
        };
      }
      return JSON.parse(xhr.responseText) as EngineTransportResponse<TCommand>;
    },
  };
}

function createDefaultTransport(): SyncBridgeTransport {
  if (isDesktopBridgeMode()) {
    return createDesktopControllerTransport();
  }
  return createHttpBridgeTransport();
}

function createStubEngineTransport(): SyncBridgeTransport {
  return {
    requestSync() {
      return {
        ok: false,
        error: "Local engine bridge is not connected",
      };
    },
    async request() {
      return {
        ok: false,
        error: "Local engine bridge is not connected",
      };
    },
  };
}

async function requestWithFallback<TResponse>(
  runTransport: () => Promise<TResponse>,
  runFallback: () => TResponse,
): Promise<TResponse> {
  try {
    return await runTransport();
  } catch {
    return runFallback();
  }
}

function callTransportSync<TResponse>(
  transport: SyncBridgeTransport,
  message: EngineTransportRequest,
  runFallback: () => TResponse,
): TResponse {
  const response = transport.requestSync(message as never) as EngineTransportResponse;
  if (!response.ok || response.data === undefined) {
    reportTransportIssue(message.command, response.error ?? "Unknown bridge error", true);
    return runFallback();
  }
  return response.data as TResponse;
}

export function createEngineAdapter(
  transport: SyncBridgeTransport = createDefaultTransport(),
  fallback: EngineApi = mockEngineApi,
): EngineApi {
  return {
    getMainState(): MainRuntimeState {
      return callTransportSync(transport, { command: "getMainState", payload: undefined }, () => fallback.getMainState());
    },
    getEngineStatus(): EngineStatus {
      return callTransportSync(transport, { command: "getEngineStatus", payload: undefined }, () => fallback.getEngineStatus());
    },
    listAudioDevices(): AudioDevicesCatalog {
      return callTransportSync(transport, { command: "listAudioDevices", payload: undefined }, () => fallback.listAudioDevices());
    },
    getDomainState(): EngineDomainState {
      return callTransportSync(transport, { command: "getDomainState", payload: undefined }, () => fallback.getDomainState());
    },
    getEngineConfig(): EngineConfig {
      return callTransportSync(transport, { command: "getEngineConfig", payload: undefined }, () => fallback.getEngineConfig());
    },
    updateEngineConfig(input: UpdateEngineConfigInput): EngineConfig {
      return callTransportSync(transport, { command: "updateEngineConfig", payload: input }, () => fallback.updateEngineConfig(input));
    },
    listProfiles(): SavedProfile[] {
      return callTransportSync(transport, { command: "listProfiles", payload: undefined }, () => fallback.listProfiles());
    },
    previewSessionTarget(input: PreviewSessionTargetInput): SessionPreviewStatus {
      const response = transport.requestSync({ command: "previewSessionTarget", payload: input });
      if (!response.ok || response.data === undefined) {
        const error = response.error ?? "Unknown bridge error";
        reportTransportIssue("previewSessionTarget", error, !isDesktopBridgeMode());
        if (isDesktopBridgeMode()) {
          return {
            sessionId: input.sessionId,
            target: input.target,
            label: null,
            processorMode: "passthrough",
            eqCurveReady: false,
            eqBandCount: null,
            preampDb: null,
            appliedToAudio: false,
            applyStatus: "failed",
            lastError: `Desktop bridge error: ${error}`,
            diagnostics: {
              transportError: error,
              fallbackUsed: false,
            },
          };
        }
        return fallback.previewSessionTarget(input);
      }
      return response.data;
    },
    setActiveProfile(input: SetActiveProfileInput): MainRuntimeState {
      return callTransportSync(transport, { command: "setActiveProfile", payload: input }, () => fallback.setActiveProfile(input));
    },
    setProcessingEnabled(input: SetProcessingEnabledInput): MainRuntimeState {
      return callTransportSync(transport, { command: "setProcessingEnabled", payload: input }, () => fallback.setProcessingEnabled(input));
    },
    updateProfile(input: UpdateProfileInput): SavedProfile[] {
      return callTransportSync(transport, { command: "updateProfile", payload: input }, () => fallback.updateProfile(input));
    },
    deleteProfile(input: DeleteProfileInput): SavedProfile[] {
      return callTransportSync(transport, { command: "deleteProfile", payload: input }, () => fallback.deleteProfile(input));
    },
    createSession(input: CreateSessionInput): SessionSnapshot {
      return callTransportSync(transport, { command: "createSession", payload: input }, () => fallback.createSession(input));
    },
    startSession(input: CreateSessionInput): SessionSnapshot {
      return callTransportSync(transport, { command: "startSession", payload: input }, () => fallback.startSession(input));
    },
    generateNextPair(input: GenerateNextPairInput): SessionSnapshot {
      return callTransportSync(transport, { command: "generateNextPair", payload: input }, () => fallback.generateNextPair(input));
    },
    saveProfile(input: SaveProfileInput): SaveProfileResult {
      return callTransportSync(transport, { command: "saveProfile", payload: input }, () => fallback.saveProfile(input));
    },
    saveProfileFromSession(input: SaveProfileFromSessionInput): SaveProfileResult {
      return callTransportSync(transport, { command: "saveProfileFromSession", payload: input }, () => fallback.saveProfileFromSession(input));
    },
  };
}

export const engineAdapterTransport = createDefaultTransport();
export const adapterEngineApi = createEngineAdapter(engineAdapterTransport);

export async function probeEngineAdapter(): Promise<boolean> {
  const controller = resolveDesktopController();
  if (!controller) {
    return false;
  }
  const result = await requestWithFallback(
    () => controller.probe(),
    () => false,
  );

  return result;
}

export { createStubEngineTransport };
