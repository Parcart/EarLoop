import {
  SESSION_DEFAULT_PROFILE_ID,
  defaultPipelineConfig,
  initialSavedProfiles,
} from "@/lib/mock/profiles";

import type { SavedProfile } from "@/lib/types/ui";

export type EngineAudioConfig = {
  inputDeviceId: string;
  outputDeviceId: string;
  sampleRate: string;
  channels: string;
};

export type EngineConfig = {
  audio: EngineAudioConfig;
  defaults: {
    activeProfileId: string;
  };
  runtime?: {
    processingEnabled: boolean;
  };
};

export type EngineDomainState = {
  profiles: SavedProfile[];
  config: EngineConfig;
  session?: {
    sessionId: string;
    status: string;
    sessionBaseProfileId: string;
    sessionBaseParams: SavedProfile["params"];
    sessionBaseLabel: string;
    currentBaseParams: SavedProfile["params"];
    currentBaseLabel: string;
    pipelineConfig: SavedProfile["pipelineConfig"];
    iteration: number;
    progress: number;
    pairVersion: number;
    lastFeedback: string;
    lastSelectedTarget: "base" | "A" | "B";
    history?: Array<{
      iteration: number;
      selectedTarget: "base" | "A" | "B";
      feedback: string;
      baseLabel: string;
    }>;
  } | null;
};

const ENGINE_STORAGE_KEY = "earloop.engine.domain";

const defaultEngineDomainState: EngineDomainState = {
  profiles: initialSavedProfiles.map((profile) => ({
    ...profile,
    params: { ...profile.params },
    pipelineConfig: { ...profile.pipelineConfig },
  })),
  config: {
    audio: {
      inputDeviceId: "CABLE Output (VB-Audio Virtual Cable)",
      outputDeviceId: "Динамики (Razer Barracuda X 2.4)",
      sampleRate: "48000",
      channels: "2",
    },
    defaults: {
      activeProfileId: initialSavedProfiles[0]?.id ?? SESSION_DEFAULT_PROFILE_ID,
    },
    runtime: {
      processingEnabled: true,
    },
  },
};

function cloneProfiles(profiles: SavedProfile[]): SavedProfile[] {
  return profiles.map((profile) => ({
    ...profile,
    params: { ...profile.params },
    pipelineConfig: { ...(profile.pipelineConfig ?? defaultPipelineConfig) },
  }));
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

function cloneSessionState(session: NonNullable<EngineDomainState["session"]>): NonNullable<EngineDomainState["session"]> {
  return {
    sessionId: session.sessionId,
    status: session.status,
    sessionBaseProfileId: session.sessionBaseProfileId,
    sessionBaseParams: { ...session.sessionBaseParams },
    sessionBaseLabel: session.sessionBaseLabel,
    currentBaseParams: { ...(session.currentBaseParams ?? session.sessionBaseParams) },
    currentBaseLabel: session.currentBaseLabel ?? session.sessionBaseLabel,
    pipelineConfig: { ...session.pipelineConfig },
    iteration: session.iteration,
    progress: session.progress,
    pairVersion: session.pairVersion,
    lastFeedback: session.lastFeedback ?? "none",
    lastSelectedTarget: session.lastSelectedTarget ?? "base",
    history: session.history?.map((entry) => ({ ...entry })),
  };
}

function cloneEngineDomainState(state: EngineDomainState): EngineDomainState {
  return {
    profiles: cloneProfiles(state.profiles),
    config: cloneEngineConfig(state.config),
    session: state.session ? cloneSessionState(state.session) : null,
  };
}

function sanitizeEngineDomainState(value: unknown): EngineDomainState {
  if (!value || typeof value !== "object") {
    return cloneEngineDomainState(defaultEngineDomainState);
  }

  const candidate = value as Partial<EngineDomainState>;
  const profiles = Array.isArray(candidate.profiles) && candidate.profiles.length > 0
    ? cloneProfiles(candidate.profiles as SavedProfile[])
    : cloneProfiles(defaultEngineDomainState.profiles);

  const fallbackActiveProfileId = profiles[0]?.id ?? SESSION_DEFAULT_PROFILE_ID;
  const configCandidate = candidate.config as Partial<EngineConfig> | undefined;

  return {
    profiles,
    config: {
      audio: {
        inputDeviceId: configCandidate?.audio?.inputDeviceId ?? defaultEngineDomainState.config.audio.inputDeviceId,
        outputDeviceId: configCandidate?.audio?.outputDeviceId ?? defaultEngineDomainState.config.audio.outputDeviceId,
        sampleRate: configCandidate?.audio?.sampleRate ?? defaultEngineDomainState.config.audio.sampleRate,
        channels: configCandidate?.audio?.channels ?? defaultEngineDomainState.config.audio.channels,
      },
      defaults: {
        activeProfileId: profiles.some((profile) => profile.id === configCandidate?.defaults?.activeProfileId)
          ? (configCandidate?.defaults?.activeProfileId ?? fallbackActiveProfileId)
          : fallbackActiveProfileId,
      },
      runtime: {
        processingEnabled: configCandidate?.runtime?.processingEnabled ?? true,
      },
    },
    session: candidate.session && typeof candidate.session === "object"
      ? cloneSessionState(candidate.session as NonNullable<EngineDomainState["session"]>)
      : null,
  };
}

export function loadEngineDomainState(): EngineDomainState {
  if (typeof window === "undefined") {
    return cloneEngineDomainState(defaultEngineDomainState);
  }

  try {
    const raw = window.localStorage.getItem(ENGINE_STORAGE_KEY);
    if (!raw) return cloneEngineDomainState(defaultEngineDomainState);
    return sanitizeEngineDomainState(JSON.parse(raw));
  } catch {
    return cloneEngineDomainState(defaultEngineDomainState);
  }
}

export function saveEngineDomainState(state: EngineDomainState): EngineDomainState {
  const nextState = sanitizeEngineDomainState(state);
  if (typeof window !== "undefined") {
    window.localStorage.setItem(ENGINE_STORAGE_KEY, JSON.stringify(nextState));
  }
  return cloneEngineDomainState(nextState);
}

export function getDefaultEngineDomainState(): EngineDomainState {
  return cloneEngineDomainState(defaultEngineDomainState);
}
