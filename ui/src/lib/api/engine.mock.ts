import {
  DEFAULT_PROFILE_NAME,
  INITIAL_ITERATION,
  INITIAL_PROGRESS,
  SESSION_DEFAULT_PROFILE_ID,
  defaultPipelineConfig,
  defaultProfileParams,
  pipelineCatalog,
} from "@/lib/mock/profiles";
import { loadEngineDomainState, saveEngineDomainState } from "@/lib/storage/engine-storage";
import { buildPair } from "@/lib/session/helpers";

import type {
  AudioDevicesCatalog,
  CreateSessionInput,
  DeleteProfileInput,
  EngineApi,
  EngineConfig,
  EngineDomainState,
  EngineStatus,
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
import type { PerceptualParams, SavedProfile } from "@/lib/types/ui";

const startupSteps = ["Инициализация аудиомоста", "Подключение модели персонализации", "Генерация первых вариантов"] as const;
let engineDomainState = loadEngineDomainState();
let currentSessionSnapshot: SessionSnapshot | null = null;
const mockAudioDevices: AudioDevicesCatalog = {
  status: "mock",
  error: null,
  inputs: [
    {
      deviceId: "mock-input-cable",
      label: "CABLE Output (VB-Audio Virtual Cable)",
      kind: "input",
      isDefault: true,
      hostapi: "Windows WASAPI",
      maxInputChannels: 2,
      maxOutputChannels: 0,
      defaultSampleRate: "48000",
      compatibleDeviceIds: ["mock-output-razer", "mock-output-realtek"],
      compatibleSampleRates: ["48000", "44100"],
    },
    {
      deviceId: "mock-input-blackhole",
      label: "Microphone (Razer Barracuda X 2.4)",
      kind: "input",
      isDefault: false,
      hostapi: "Windows WASAPI",
      maxInputChannels: 2,
      maxOutputChannels: 0,
      defaultSampleRate: "48000",
      compatibleDeviceIds: ["mock-output-razer", "mock-output-realtek"],
      compatibleSampleRates: ["48000", "44100"],
    },
  ],
  outputs: [
    {
      deviceId: "mock-output-razer",
      label: "Динамики (Razer Barracuda X 2.4)",
      kind: "output",
      isDefault: true,
      hostapi: "Windows WASAPI",
      maxInputChannels: 0,
      maxOutputChannels: 2,
      defaultSampleRate: "48000",
      compatibleDeviceIds: ["mock-input-cable", "mock-input-blackhole"],
      compatibleSampleRates: ["48000", "44100"],
    },
    {
      deviceId: "mock-output-realtek",
      label: "Наушники (Realtek)",
      kind: "output",
      isDefault: false,
      hostapi: "Windows WASAPI",
      maxInputChannels: 0,
      maxOutputChannels: 2,
      defaultSampleRate: "48000",
      compatibleDeviceIds: ["mock-input-cable", "mock-input-blackhole"],
      compatibleSampleRates: ["48000", "44100"],
    },
  ],
};

function cloneProfile(profile: SavedProfile): SavedProfile {
  return {
    ...profile,
    params: { ...profile.params },
    pipelineConfig: { ...profile.pipelineConfig },
  };
}

function cloneProfiles(profiles: SavedProfile[]): SavedProfile[] {
  return profiles.map(cloneProfile);
}

function cloneEngineConfig(config: EngineConfig): EngineConfig {
  return {
    audio: { ...config.audio },
    defaults: { ...config.defaults },
    runtime: {
      processingEnabled: config.runtime?.processingEnabled ?? true,
    },
  };
}

function cloneDomainState(): EngineDomainState {
  return {
    profiles: cloneProfiles(engineDomainState.profiles),
    config: cloneEngineConfig(engineDomainState.config),
  };
}

function persistDomainState(nextState: EngineDomainState): EngineDomainState {
  engineDomainState = saveEngineDomainState(nextState);
  return cloneDomainState();
}

function resolveProfiles(profiles?: SavedProfile[]): SavedProfile[] {
  return profiles ? cloneProfiles(profiles) : cloneProfiles(engineDomainState.profiles);
}

function resolveSessionBaseProfile(profiles: SavedProfile[], sessionBaseProfileId: string) {
  return profiles.find((profile) => profile.id === sessionBaseProfileId) ?? null;
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function blendParams(base: PerceptualParams, target: PerceptualParams, ratio: number): PerceptualParams {
  return {
    bass: Number(clamp(base.bass + (target.bass - base.bass) * ratio, -1, 1).toFixed(2)),
    tilt: Number(clamp(base.tilt + (target.tilt - base.tilt) * ratio, -1, 1).toFixed(2)),
    presence: Number(clamp(base.presence + (target.presence - base.presence) * ratio, -1, 1).toFixed(2)),
    air: Number(clamp(base.air + (target.air - base.air) * ratio, -1, 1).toFixed(2)),
    lowmid: Number(clamp(base.lowmid + (target.lowmid - base.lowmid) * ratio, -1, 1).toFixed(2)),
    sparkle: Number(clamp(base.sparkle + (target.sparkle - base.sparkle) * ratio, -1, 1).toFixed(2)),
  };
}

function applyFeedback(params: PerceptualParams, feedback: string): PerceptualParams {
  switch (feedback) {
    case "more_bass":
      return {
        ...params,
        bass: Number(clamp(params.bass + 0.14, -1, 1).toFixed(2)),
        tilt: Number(clamp(params.tilt - 0.03, -1, 1).toFixed(2)),
        lowmid: Number(clamp(params.lowmid + 0.04, -1, 1).toFixed(2)),
      };
    case "less_harsh":
      return {
        ...params,
        tilt: Number(clamp(params.tilt - 0.08, -1, 1).toFixed(2)),
        presence: Number(clamp(params.presence - 0.12, -1, 1).toFixed(2)),
        air: Number(clamp(params.air - 0.04, -1, 1).toFixed(2)),
        lowmid: Number(clamp(params.lowmid + 0.06, -1, 1).toFixed(2)),
        sparkle: Number(clamp(params.sparkle - 0.08, -1, 1).toFixed(2)),
      };
    case "more_air":
      return {
        ...params,
        tilt: Number(clamp(params.tilt + 0.03, -1, 1).toFixed(2)),
        presence: Number(clamp(params.presence + 0.05, -1, 1).toFixed(2)),
        air: Number(clamp(params.air + 0.14, -1, 1).toFixed(2)),
        sparkle: Number(clamp(params.sparkle + 0.07, -1, 1).toFixed(2)),
      };
    case "warmer":
      return {
        ...params,
        bass: Number(clamp(params.bass + 0.06, -1, 1).toFixed(2)),
        tilt: Number(clamp(params.tilt - 0.06, -1, 1).toFixed(2)),
        presence: Number(clamp(params.presence - 0.07, -1, 1).toFixed(2)),
        air: Number(clamp(params.air - 0.05, -1, 1).toFixed(2)),
        lowmid: Number(clamp(params.lowmid + 0.1, -1, 1).toFixed(2)),
        sparkle: Number(clamp(params.sparkle - 0.04, -1, 1).toFixed(2)),
      };
    default:
      return { ...params };
  }
}

function buildSessionSnapshot({
  profiles,
  pipelineConfig,
  sessionBaseProfileId,
  sessionBaseParams,
  sessionBaseLabel,
  iteration = INITIAL_ITERATION,
  progress = INITIAL_PROGRESS,
  pairVersion = 0,
  selectedTarget = "base",
  feedback = "none",
}: CreateSessionInput & { selectedTarget?: "base" | "A" | "B"; feedback?: string }): SessionSnapshot {
  const resolvedProfiles = resolveProfiles(profiles);
  const sessionBaseProfile = resolveSessionBaseProfile(resolvedProfiles, sessionBaseProfileId);
  const resolvedSessionBaseParams = sessionBaseParams ?? sessionBaseProfile?.params ?? defaultProfileParams;
  const resolvedSessionBaseLabel = sessionBaseLabel ?? sessionBaseProfile?.name ?? "С нуля";
  const sessionPipelineConfig = pipelineConfig ?? sessionBaseProfile?.pipelineConfig ?? defaultPipelineConfig;
  const seed = pairVersion + 1;

  return {
    sessionId: "mock-session",
    status: "started",
    sessionBaseProfile,
    sessionBaseParams: { ...resolvedSessionBaseParams },
    sessionBaseLabel: resolvedSessionBaseLabel,
    sessionPipelineConfig: { ...sessionPipelineConfig },
    iteration,
    progress,
    pairVersion,
    lastFeedback: feedback,
    lastSelectedTarget: selectedTarget,
    pairA: buildPair(seed, "A", resolvedSessionBaseParams),
    pairB: buildPair(seed, "B", resolvedSessionBaseParams),
  };
}

function getEngineStatus(): EngineStatus {
  return {
    defaultProfileName: DEFAULT_PROFILE_NAME,
    defaultPipelineConfig: { ...defaultPipelineConfig },
    defaultProfileParams: { ...defaultProfileParams },
    initialIteration: INITIAL_ITERATION,
    initialProgress: INITIAL_PROGRESS,
    pipelineCatalog: {
      models: pipelineCatalog.models.map((option) => ({ ...option })),
      eqs: pipelineCatalog.eqs.map((option) => ({ ...option })),
      mappers: pipelineCatalog.mappers.map((option) => ({ ...option })),
      generators: pipelineCatalog.generators.map((option) => ({ ...option })),
      strategies: pipelineCatalog.strategies.map((option) => ({ ...option })),
    },
    sessionDefaultProfileId: SESSION_DEFAULT_PROFILE_ID,
    startupSteps,
    audioStatus: {
      status: "idle",
      activeConfig: null,
      desiredConfig: cloneEngineConfig(engineDomainState.config).audio,
      lastError: null,
      lastAppliedAt: null,
    },
    runtimeProfileStatus: {
      activeProfileId: engineDomainState.config.defaults.activeProfileId,
      activeProfileName: engineDomainState.profiles.find((profile) => profile.id === engineDomainState.config.defaults.activeProfileId)?.name ?? null,
      processorMode: engineDomainState.config.runtime?.processingEnabled === false ? "passthrough" : "eq",
      eqCurveReady: true,
      eqBandCount: 12,
      preampDb: -2.5,
      appliedToAudio: false,
      applyStatus: engineDomainState.config.runtime?.processingEnabled === false ? "bypass" : "ready",
      lastError: null,
    },
  };
}

function getMainState(): MainRuntimeState {
  const activeProfile = engineDomainState.profiles.find((profile) => profile.id === engineDomainState.config.defaults.activeProfileId) ?? null;
  const engineStatus = getEngineStatus();
  return {
    processingEnabled: engineDomainState.config.runtime?.processingEnabled ?? true,
    activeProfileId: activeProfile?.id ?? null,
    activeProfileName: activeProfile?.name ?? null,
    audioStatus: engineStatus.audioStatus,
    runtimeProfileStatus: engineStatus.runtimeProfileStatus,
  };
}

function previewSessionTarget({ sessionId, target }: PreviewSessionTargetInput): SessionPreviewStatus {
  const snapshot = currentSessionSnapshot;
  if (!snapshot || snapshot.sessionId !== sessionId) {
    return {
      sessionId,
      target,
      label: null,
      processorMode: "passthrough",
      eqCurveReady: false,
      eqBandCount: null,
      preampDb: null,
      appliedToAudio: false,
      applyStatus: "failed",
      lastError: "No active session preview is available",
    };
  }

  return {
    sessionId,
    target,
    label: target === "A" ? "Вариант A" : target === "B" ? "Вариант B" : snapshot.sessionBaseLabel,
    processorMode: "eq",
    eqCurveReady: true,
    eqBandCount: 12,
    preampDb: -2.1,
    appliedToAudio: true,
    applyStatus: "applied",
    lastError: null,
  };
}

function listAudioDevices(): AudioDevicesCatalog {
  return {
    status: mockAudioDevices.status,
    error: mockAudioDevices.error,
    inputs: mockAudioDevices.inputs.map((device) => ({ ...device })),
    outputs: mockAudioDevices.outputs.map((device) => ({ ...device })),
  };
}

function getDomainState(): EngineDomainState {
  return cloneDomainState();
}

function getEngineConfig(): EngineConfig {
  return cloneEngineConfig(engineDomainState.config);
}

function updateEngineConfig({ config }: UpdateEngineConfigInput): EngineConfig {
  const nextConfig: EngineConfig = {
    audio: {
      ...engineDomainState.config.audio,
      ...config.audio,
    },
    defaults: {
      ...engineDomainState.config.defaults,
      ...config.defaults,
    },
    runtime: {
      processingEnabled: config.runtime?.processingEnabled ?? engineDomainState.config.runtime?.processingEnabled ?? true,
    },
  };

  if (!engineDomainState.profiles.some((profile) => profile.id === nextConfig.defaults.activeProfileId)) {
    nextConfig.defaults.activeProfileId = engineDomainState.profiles[0]?.id ?? SESSION_DEFAULT_PROFILE_ID;
  }

  persistDomainState({
    ...engineDomainState,
    config: nextConfig,
  });

  return cloneEngineConfig(nextConfig);
}

function setActiveProfile({ profileId }: SetActiveProfileInput): MainRuntimeState {
  const nextActiveProfileId = engineDomainState.profiles.some((profile) => profile.id === profileId)
    ? profileId
    : engineDomainState.profiles[0]?.id ?? SESSION_DEFAULT_PROFILE_ID;
  persistDomainState({
    ...engineDomainState,
    config: {
      ...engineDomainState.config,
      defaults: {
        ...engineDomainState.config.defaults,
        activeProfileId: nextActiveProfileId,
      },
    },
  });
  return getMainState();
}

function setProcessingEnabled({ enabled }: SetProcessingEnabledInput): MainRuntimeState {
  persistDomainState({
    ...engineDomainState,
    config: {
      ...engineDomainState.config,
      runtime: {
        processingEnabled: enabled,
      },
    },
  });
  return getMainState();
}

function listProfiles(): SavedProfile[] {
  return cloneProfiles(engineDomainState.profiles);
}

function updateProfile({ profileId, name, params }: UpdateProfileInput): SavedProfile[] {
  const nextProfiles = engineDomainState.profiles.map((profile) => (
    profile.id === profileId
      ? { ...profile, name, params: { ...params } }
      : cloneProfile(profile)
  ));

  persistDomainState({
    ...engineDomainState,
    profiles: nextProfiles,
  });

  return cloneProfiles(nextProfiles);
}

function deleteProfile({ profileId }: DeleteProfileInput): SavedProfile[] {
  const nextProfiles = engineDomainState.profiles.filter((profile) => profile.id !== profileId).map(cloneProfile);
  const nextActiveProfileId = engineDomainState.config.defaults.activeProfileId === profileId
    ? (nextProfiles[0]?.id ?? SESSION_DEFAULT_PROFILE_ID)
    : engineDomainState.config.defaults.activeProfileId;

  persistDomainState({
    profiles: nextProfiles,
    config: {
      ...engineDomainState.config,
      defaults: {
        ...engineDomainState.config.defaults,
        activeProfileId: nextActiveProfileId,
      },
    },
  });

  return cloneProfiles(nextProfiles);
}

function createSession(input: CreateSessionInput): SessionSnapshot {
  currentSessionSnapshot = buildSessionSnapshot(input);
  return currentSessionSnapshot;
}

function startSession(input: CreateSessionInput): SessionSnapshot {
  currentSessionSnapshot = buildSessionSnapshot(input);
  return currentSessionSnapshot;
}

function generateNextPair({
  profiles,
  pipelineConfig,
  sessionBaseProfileId,
  sessionBaseParams,
  sessionBaseLabel,
  iteration,
  progress,
  pairVersion,
  selectedTarget,
  feedback,
}: GenerateNextPairInput): SessionSnapshot {
  const currentSnapshot = buildSessionSnapshot({
    profiles,
    pipelineConfig,
    sessionBaseProfileId,
    sessionBaseParams,
    sessionBaseLabel,
    iteration,
    progress,
    pairVersion,
    selectedTarget: "base",
    feedback: "none",
  });
  const selectedParams = selectedTarget === "A"
    ? currentSnapshot.pairA.params
    : selectedTarget === "B"
      ? currentSnapshot.pairB.params
      : currentSnapshot.sessionBaseParams;
  const selectedLabel = selectedTarget === "A"
    ? "вариант A"
    : selectedTarget === "B"
      ? "вариант B"
      : currentSnapshot.sessionBaseLabel;
  const nextBaseParams = applyFeedback(
    selectedTarget === "base"
      ? { ...selectedParams }
      : blendParams(currentSnapshot.sessionBaseParams, selectedParams, 0.72),
    feedback,
  );
  currentSessionSnapshot = buildSessionSnapshot({
    profiles,
    pipelineConfig,
    sessionBaseProfileId,
    sessionBaseParams: nextBaseParams,
    sessionBaseLabel: selectedLabel,
    iteration: iteration + 1,
    progress: Math.min(100, progress + 5),
    pairVersion: pairVersion + 1,
    selectedTarget,
    feedback,
  });
  return currentSessionSnapshot;
}

function saveProfile({
  name,
  finalChoice,
  pairA,
  pairB,
  pipelineConfig,
  sessionBaseParams,
}: SaveProfileInput): SaveProfileResult {
  const profileId = `${name.toLowerCase().split(" ").filter(Boolean).join("-")}-${Date.now()}`;
  const params = finalChoice === "A" ? pairA.params : finalChoice === "B" ? pairB.params : sessionBaseParams;
  const savedProfile = {
    id: profileId,
    name,
    params: { ...params },
    pipelineConfig: { ...pipelineConfig },
  };

  const nextProfiles = [...cloneProfiles(engineDomainState.profiles), savedProfile];
  persistDomainState({
    profiles: nextProfiles,
    config: {
      ...engineDomainState.config,
      defaults: {
        ...engineDomainState.config.defaults,
        activeProfileId: profileId,
      },
    },
  });

  return {
    profileId,
    savedProfile,
    profiles: cloneProfiles(nextProfiles),
  };
}

function saveProfileFromSession({
  name,
  finalChoice,
}: SaveProfileFromSessionInput): SaveProfileResult {
  const snapshot = currentSessionSnapshot;
  if (!snapshot) {
    throw new Error("No active session to save from");
  }
  const params = finalChoice === "A"
    ? snapshot.pairA.params
    : finalChoice === "B"
      ? snapshot.pairB.params
      : snapshot.sessionBaseParams;
  const profileId = `${name.toLowerCase().split(" ").filter(Boolean).join("-")}-${Date.now()}`;
  const savedProfile = {
    id: profileId,
    name,
    params: { ...params },
    pipelineConfig: { ...snapshot.sessionPipelineConfig },
  };
  const nextProfiles = [...cloneProfiles(engineDomainState.profiles), savedProfile];
  persistDomainState({
    profiles: nextProfiles,
    config: {
      ...engineDomainState.config,
      defaults: {
        ...engineDomainState.config.defaults,
        activeProfileId: profileId,
      },
    },
  });
  return {
    profileId,
    savedProfile,
    profiles: cloneProfiles(nextProfiles),
  };
}

export const mockEngineApi: EngineApi = {
  getMainState,
  getEngineStatus,
  listAudioDevices,
  getDomainState,
  getEngineConfig,
  updateEngineConfig,
  listProfiles,
  previewSessionTarget,
  setActiveProfile,
  setProcessingEnabled,
  updateProfile,
  deleteProfile,
  createSession,
  startSession,
  generateNextPair,
  saveProfile,
  saveProfileFromSession,
};

export {
  createSession,
  deleteProfile,
  generateNextPair,
  getDomainState,
  getEngineConfig,
  getEngineStatus,
  listAudioDevices,
  listProfiles,
  saveProfile,
  saveProfileFromSession,
  startSession,
  updateEngineConfig,
  updateProfile,
};
