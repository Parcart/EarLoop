import type { PerceptualParams, PipelineCatalog, PipelineConfig, SavedProfile } from "@/lib/types/ui";

export const INITIAL_ITERATION = 12;
export const INITIAL_PROGRESS = 62;
export const DEFAULT_PROFILE_NAME = "Новый профиль";
export const SESSION_DEFAULT_PROFILE_ID = "__default__";

export const eqBands = [
  25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630,
  800, 1000, 1600, 2500, 4000, 6300, 10000, 16000,
];

export const defaultProfileParams: PerceptualParams = {
  bass: 0.18,
  tilt: -0.08,
  presence: 0.24,
  air: 0.16,
  lowmid: -0.12,
  sparkle: 0.2,
};

export const defaultPipelineConfig: PipelineConfig = {
  modelId: "preference-linear-v1",
  eqId: "parametric-eq-v1",
  mapperId: "perceptual-mapper-v1",
  generatorId: "candidate-pool-v1",
  strategyId: "adaptive-updater-v1",
};

export const pipelineCatalog: PipelineCatalog = {
  models: [
    { id: "preference-linear-v1", label: "Preference Linear v1" },
    { id: "preference-hybrid-v2", label: "Preference Hybrid v2" },
    { id: "listener-embedding-v1", label: "Listener Embedding v1" },
  ],
  eqs: [
    { id: "parametric-eq-v1", label: "Parametric EQ v1" },
    { id: "minimum-phase-eq-v2", label: "Minimum Phase EQ v2" },
    { id: "wideband-eq-v1", label: "Wideband EQ v1" },
  ],
  mappers: [
    { id: "perceptual-mapper-v1", label: "Perceptual Mapper v1" },
    { id: "smooth-curve-mapper-v2", label: "Smooth Curve Mapper v2" },
    { id: "target-blend-mapper-v1", label: "Target Blend Mapper v1" },
  ],
  generators: [
    { id: "candidate-pool-v1", label: "Candidate Pool v1" },
    { id: "local-search-v2", label: "Local Search v2" },
    { id: "guided-random-v1", label: "Guided Random v1" },
  ],
  strategies: [
    { id: "adaptive-updater-v1", label: "Adaptive Updater v1" },
    { id: "confidence-weighted-v2", label: "Confidence Weighted v2" },
    { id: "fast-explore-v1", label: "Fast Explore v1" },
  ],
};

export const profileParamMeta: Array<{ key: keyof PerceptualParams; label: string }> = [
  { key: "bass", label: "Bass" },
  { key: "tilt", label: "Tilt" },
  { key: "presence", label: "Presence" },
  { key: "air", label: "Air" },
  { key: "lowmid", label: "Low Mid" },
  { key: "sparkle", label: "Sparkle" },
];

export const sliderClassName =
  "[&>span:first-child]:h-2.5 [&>span:first-child]:bg-white/10 [&>span:first-child>span]:bg-violet-300 [&_[role=slider]]:h-4 [&_[role=slider]]:w-4 [&_[role=slider]]:border-violet-200 [&_[role=slider]]:bg-[#0b0c11] [&_[role=slider]]:shadow-[0_0_0_4px_rgba(196,181,253,0.18)]";

export const initialSavedProfiles: SavedProfile[] = [
  {
    id: "wave",
    name: "Моя волна",
    params: { bass: 0.34, tilt: -0.06, presence: 0.41, air: 0.22, lowmid: -0.18, sparkle: 0.3 },
    pipelineConfig: {
      modelId: "preference-linear-v1",
      eqId: "parametric-eq-v1",
      mapperId: "perceptual-mapper-v1",
      generatorId: "candidate-pool-v1",
      strategyId: "adaptive-updater-v1",
    },
  },
  {
    id: "night",
    name: "Night Drive",
    params: { bass: 0.52, tilt: -0.12, presence: 0.18, air: 0.09, lowmid: -0.22, sparkle: 0.12 },
    pipelineConfig: {
      modelId: "preference-hybrid-v2",
      eqId: "minimum-phase-eq-v2",
      mapperId: "smooth-curve-mapper-v2",
      generatorId: "local-search-v2",
      strategyId: "confidence-weighted-v2",
    },
  },
  {
    id: "soft",
    name: "Soft Focus",
    params: { bass: 0.12, tilt: 0.04, presence: -0.08, air: 0.28, lowmid: 0.12, sparkle: 0.34 },
    pipelineConfig: {
      modelId: "listener-embedding-v1",
      eqId: "wideband-eq-v1",
      mapperId: "target-blend-mapper-v1",
      generatorId: "guided-random-v1",
      strategyId: "fast-explore-v1",
    },
  },
];
