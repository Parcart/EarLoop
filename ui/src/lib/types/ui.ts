export type Screen = "home" | "settings" | "session";
export type ThemeMode = "dark" | "light" | "system";
export type ListeningTarget = "base" | "A" | "B";
export type SessionPhase = "idle" | "starting" | "ready";
export type PairKey = "A" | "B";
export type WavePreset = "compact" | "default" | "expanded";

export type PerceptualParams = {
  bass: number;
  tilt: number;
  presence: number;
  air: number;
  lowmid: number;
  sparkle: number;
};

export type PairData = {
  id: PairKey;
  params: PerceptualParams;
  score: number;
};

export type PipelineConfig = {
  modelId: string;
  eqId: string;
  mapperId: string;
  generatorId?: string;
  strategyId?: string;
};

export type PipelineOption = {
  id: string;
  label: string;
};

export type PipelineCatalog = {
  models: PipelineOption[];
  eqs: PipelineOption[];
  mappers: PipelineOption[];
  generators: PipelineOption[];
  strategies: PipelineOption[];
};

export type SavedProfile = {
  id: string;
  name: string;
  params: PerceptualParams;
  pipelineConfig: PipelineConfig;
};

export type AudioDevices = {
  input: string;
  output: string;
  sampleRate: string;
  channels: string;
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
  supportsLoopback?: boolean | null;
  loopbackInputDeviceId?: string | null;
  loopbackEndpointId?: string | null;
};
