import type { ListeningTarget, PairData, PerceptualParams, PipelineCatalog, PipelineConfig, SavedProfile } from "@/lib/types/ui";
import type { EngineConfig, EngineDomainState } from "@/lib/storage/engine-storage";

export type EngineStatus = {
  defaultProfileName: string;
  defaultProfileParams: PerceptualParams;
  defaultPipelineConfig: PipelineConfig;
  initialIteration: number;
  initialProgress: number;
  pipelineCatalog: PipelineCatalog;
  sessionDefaultProfileId: string;
  startupSteps: readonly string[];
  audioStatus?: {
    status: "idle" | "applied" | "failed" | "pending_restart" | "device_unavailable" | string;
    activeConfig: EngineConfig["audio"] | null;
    desiredConfig: EngineConfig["audio"] | null;
    lastError: string | null;
    lastAppliedAt: string | null;
  } | null;
  runtimeProfileStatus?: {
    activeProfileId: string | null;
    activeProfileName: string | null;
    processorMode: string;
    eqCurveReady: boolean;
    eqBandCount: number | null;
    preampDb: number | null;
    appliedToAudio: boolean;
    applyStatus: string;
    lastError: string | null;
  } | null;
};

export type { EngineConfig, EngineDomainState } from "@/lib/storage/engine-storage";

export type MainRuntimeState = {
  processingEnabled: boolean;
  activeProfileId: string | null;
  activeProfileName: string | null;
  audioStatus?: EngineStatus["audioStatus"];
  runtimeProfileStatus?: EngineStatus["runtimeProfileStatus"];
};

export type SessionPreviewStatus = {
  sessionId: string | null;
  target: ListeningTarget | null;
  label: string | null;
  processorMode: string;
  eqCurveReady: boolean;
  eqBandCount: number | null;
  preampDb: number | null;
  appliedToAudio: boolean;
  applyStatus: string;
  lastError: string | null;
};

export type AudioDeviceOption = {
  deviceId: string;
  label: string;
  kind: "input" | "output";
  isDefault: boolean;
  hostapi: string | null;
  maxInputChannels: number | null;
  maxOutputChannels: number | null;
  defaultSampleRate: string | null;
  compatibleDeviceIds?: string[];
  compatibleSampleRates?: string[];
};

export type AudioDevicesCatalog = {
  inputs: AudioDeviceOption[];
  outputs: AudioDeviceOption[];
  status: string;
  error: string | null;
};

export type CreateSessionInput = {
  profiles?: SavedProfile[];
  pipelineConfig?: PipelineConfig;
  sessionBaseProfileId: string;
  sessionBaseParams?: PerceptualParams;
  sessionBaseLabel?: string;
  iteration?: number;
  progress?: number;
  pairVersion?: number;
};

export type SessionSnapshot = {
  sessionId: string;
  status: string;
  sessionBaseProfile: SavedProfile | null;
  sessionBaseParams: PerceptualParams;
  sessionBaseLabel: string;
  sessionPipelineConfig: PipelineConfig;
  iteration: number;
  progress: number;
  pairVersion: number;
  lastFeedback: string;
  lastSelectedTarget: ListeningTarget;
  pairA: PairData;
  pairB: PairData;
};

export type UpdateProfileInput = {
  profileId: string;
  name: string;
  params: PerceptualParams;
};

export type DeleteProfileInput = {
  profileId: string;
};

export type GenerateNextPairInput = {
  profiles?: SavedProfile[];
  pipelineConfig: PipelineConfig;
  sessionBaseProfileId: string;
  sessionBaseParams?: PerceptualParams;
  sessionBaseLabel?: string;
  iteration: number;
  progress: number;
  pairVersion: number;
  selectedTarget: ListeningTarget;
  feedback: string;
};

export type SaveProfileInput = {
  name: string;
  finalChoice: ListeningTarget;
  pairA: PairData;
  pairB: PairData;
  pipelineConfig: PipelineConfig;
  sessionBaseParams: PerceptualParams;
};

export type SaveProfileFromSessionInput = {
  name: string;
  finalChoice: ListeningTarget;
};

export type SaveProfileResult = {
  profileId: string;
  profiles: SavedProfile[];
  savedProfile: SavedProfile;
};

export type UpdateEngineConfigInput = {
  config: Partial<EngineConfig>;
};

export type SetActiveProfileInput = {
  profileId: string;
};

export type SetProcessingEnabledInput = {
  enabled: boolean;
};

export type PreviewSessionTargetInput = {
  sessionId: string;
  target: ListeningTarget;
};

export type EngineApi = {
  getMainState: () => MainRuntimeState;
  getEngineStatus: () => EngineStatus;
  listAudioDevices: () => AudioDevicesCatalog;
  getDomainState: () => EngineDomainState;
  getEngineConfig: () => EngineConfig;
  updateEngineConfig: (input: UpdateEngineConfigInput) => EngineConfig;
  listProfiles: () => SavedProfile[];
  previewSessionTarget: (input: PreviewSessionTargetInput) => SessionPreviewStatus;
  setActiveProfile: (input: SetActiveProfileInput) => MainRuntimeState;
  setProcessingEnabled: (input: SetProcessingEnabledInput) => MainRuntimeState;
  updateProfile: (input: UpdateProfileInput) => SavedProfile[];
  deleteProfile: (input: DeleteProfileInput) => SavedProfile[];
  createSession: (input: CreateSessionInput) => SessionSnapshot;
  startSession: (input: CreateSessionInput) => SessionSnapshot;
  generateNextPair: (input: GenerateNextPairInput) => SessionSnapshot;
  saveProfile: (input: SaveProfileInput) => SaveProfileResult;
  saveProfileFromSession: (input: SaveProfileFromSessionInput) => SaveProfileResult;
};

export type EngineProviderName = "mock" | "adapter";

export type EngineTransportCommand =
  | "getMainState"
  | "getEngineStatus"
  | "listAudioDevices"
  | "getDomainState"
  | "getEngineConfig"
  | "updateEngineConfig"
  | "listProfiles"
  | "previewSessionTarget"
  | "setActiveProfile"
  | "setProcessingEnabled"
  | "updateProfile"
  | "deleteProfile"
  | "createSession"
  | "startSession"
  | "generateNextPair"
  | "saveProfile"
  | "saveProfileFromSession";

export type EngineTransportPayloadMap = {
  getMainState: undefined;
  getEngineStatus: undefined;
  listAudioDevices: undefined;
  getDomainState: undefined;
  getEngineConfig: undefined;
  updateEngineConfig: UpdateEngineConfigInput;
  listProfiles: undefined;
  previewSessionTarget: PreviewSessionTargetInput;
  setActiveProfile: SetActiveProfileInput;
  setProcessingEnabled: SetProcessingEnabledInput;
  updateProfile: UpdateProfileInput;
  deleteProfile: DeleteProfileInput;
  createSession: CreateSessionInput;
  startSession: CreateSessionInput;
  generateNextPair: GenerateNextPairInput;
  saveProfile: SaveProfileInput;
  saveProfileFromSession: SaveProfileFromSessionInput;
};

export type EngineTransportResponseMap = {
  getMainState: MainRuntimeState;
  getEngineStatus: EngineStatus;
  listAudioDevices: AudioDevicesCatalog;
  getDomainState: EngineDomainState;
  getEngineConfig: EngineConfig;
  updateEngineConfig: EngineConfig;
  listProfiles: SavedProfile[];
  previewSessionTarget: SessionPreviewStatus;
  setActiveProfile: MainRuntimeState;
  setProcessingEnabled: MainRuntimeState;
  updateProfile: SavedProfile[];
  deleteProfile: SavedProfile[];
  createSession: SessionSnapshot;
  startSession: SessionSnapshot;
  generateNextPair: SessionSnapshot;
  saveProfile: SaveProfileResult;
  saveProfileFromSession: SaveProfileResult;
};

export type EngineTransportRequest<TCommand extends EngineTransportCommand = EngineTransportCommand> = {
  command: TCommand;
  payload: EngineTransportPayloadMap[TCommand];
};

export type EngineTransportResponse<TCommand extends EngineTransportCommand = EngineTransportCommand> = {
  ok: boolean;
  data?: EngineTransportResponseMap[TCommand];
  error?: string;
};

export type EngineBridgeTransport = {
  request: <TCommand extends EngineTransportCommand>(
    message: EngineTransportRequest<TCommand>,
  ) => Promise<EngineTransportResponse<TCommand>>;
};
