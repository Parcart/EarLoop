import { mockEngineApi } from "@/lib/api/engine.mock";

import type {
  AudioDevicesCatalog,
  CreateSessionInput,
  DeleteProfileInput,
  EngineApi,
  EngineConfig,
  EngineDomainState,
  EngineBridgeTransport,
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

const ENGINE_BRIDGE_COMMAND_ENDPOINT = "/__engine/command";
const ENGINE_BRIDGE_HEALTH_ENDPOINT = "/__engine/health";

type SyncBridgeTransport = EngineBridgeTransport & {
  requestSync: <TCommand extends EngineTransportCommand>(
    message: EngineTransportRequest<TCommand>,
  ) => EngineTransportResponse<TCommand>;
};

function parseBridgeResponse<TCommand extends EngineTransportCommand>(raw: string): EngineTransportResponse<TCommand> {
  const payload = JSON.parse(raw) as EngineTransportResponse<TCommand>;
  if (typeof payload !== "object" || payload === null || typeof payload.ok !== "boolean") {
    throw new Error("Engine bridge returned invalid response");
  }
  return payload;
}

function createHttpEngineTransport(): SyncBridgeTransport {
  return {
    async request<TCommand extends EngineTransportCommand>(message: EngineTransportRequest<TCommand>) {
      const response = await fetch(ENGINE_BRIDGE_COMMAND_ENDPOINT, {
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
          error: `Engine bridge HTTP error: ${response.status}`,
        };
      }

      return parseBridgeResponse<TCommand>(await response.text());
    },
    requestSync<TCommand extends EngineTransportCommand>(message: EngineTransportRequest<TCommand>) {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", ENGINE_BRIDGE_COMMAND_ENDPOINT, false);
      xhr.setRequestHeader("Content-Type", "application/json");
      xhr.send(JSON.stringify({
        command: message.command,
        payload: message.payload,
      }));

      if (xhr.status < 200 || xhr.status >= 300) {
        return {
          ok: false,
          error: `Engine bridge HTTP error: ${xhr.status}`,
        };
      }

      return parseBridgeResponse<TCommand>(xhr.responseText);
    },
  };
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
    console.error(`[engine-adapter] ${response.error ?? "Unknown bridge error"}`);
    return runFallback();
  }
  return response.data as TResponse;
}

export function createEngineAdapter(
  transport: SyncBridgeTransport = createHttpEngineTransport(),
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
      return callTransportSync(transport, { command: "previewSessionTarget", payload: input }, () => fallback.previewSessionTarget(input));
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

export const engineAdapterTransport = createHttpEngineTransport();
export const adapterEngineApi = createEngineAdapter(engineAdapterTransport);

export async function probeEngineAdapter(): Promise<boolean> {
  const result = await requestWithFallback(
    () => fetch(ENGINE_BRIDGE_HEALTH_ENDPOINT).then(async (response) => {
      if (!response.ok) {
        throw new Error(`Engine bridge health HTTP error: ${response.status}`);
      }
      const payload = await response.json() as { ok?: boolean };
      if (!payload.ok) {
        throw new Error("Engine bridge health check failed");
      }
      return true;
    }),
    () => false,
  );

  return result;
}
