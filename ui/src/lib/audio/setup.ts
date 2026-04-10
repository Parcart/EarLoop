import type { AudioDevicesCatalog, EngineStatus } from "@/lib/api/engine.types";
import type { AudioDeviceOption, AudioDevices } from "@/lib/types/ui";

export type BackendMode = "real" | "mock" | "unknown";
export type CaptureSourceType = "input" | "render_loopback";
export type CaptureSourceSupportState = "supported" | "unsupported";
const WASAPI_FALLBACK_HOSTAPI = "Windows WASAPI";
const LOOPBACK_SUPPORT_REASON = "WASAPI loopback пока не поддерживается текущим runtime EarLoop";

export type RouteValidation = {
  valid: boolean;
  reason: string;
};

export type CaptureSourceOption = {
  deviceId: string;
  label: string;
  displayLabel: string;
  sourceType: CaptureSourceType;
  supportState: CaptureSourceSupportState;
  isAvailable: boolean;
  isSupported: boolean;
  supportReason: string | null;
  hostapi: string | null;
  device: AudioDeviceOption;
};

export type AudioSetupEvaluation = {
  captureCandidates: AudioDeviceOption[];
  captureSourceOptions: CaptureSourceOption[];
  availableInputSources: CaptureSourceOption[];
  availableLoopbackSources: CaptureSourceOption[];
  wasapiInputs: AudioDeviceOption[];
  wasapiOutputs: AudioDeviceOption[];
  selectedCaptureCandidate: AudioDeviceOption | null;
  selectedCaptureSource: CaptureSourceOption | null;
  selectedCaptureSourceType: CaptureSourceType | null;
  activeCaptureSourceType: CaptureSourceType | null;
  selectedCaptureSourceTypeLabel: string | null;
  activeCaptureSourceTypeLabel: string | null;
  selectedCaptureStrategyLabel: string | null;
  activeCaptureStrategyLabel: string | null;
  selectedInput: AudioDeviceOption | null;
  selectedOutput: AudioDeviceOption | null;
  selectedCaptureLabel: string | null;
  activeInputLabel: string | null;
  activeOutputLabel: string | null;
  selectedLoopbackEndpointId: string | null;
  activeLoopbackEndpointId: string | null;
  isActiveRouteDifferentFromAttempted: boolean;
  shouldExplainFallbackActiveRoute: boolean;
  resolvedCaptureOptions: CaptureSourceOption[];
  resolvedOutputOptions: AudioDeviceOption[];
  hasVirtualCable: boolean;
  hasCompatibleInputSource: boolean;
  hasSupportedSystemAudioSource: boolean;
  shouldRecommendVBCable: boolean;
  shouldOfferVBCableInstall: boolean;
  sourceAssistTitle: string;
  sourceAssistDescription: string;
  sourceAssistStatusText: string;
  sourceAssistTone: "success" | "warning" | "error";
  routeValidation: RouteValidation;
  isRouteValid: boolean;
  isEvaluationReady: boolean;
  routeError: string | null;
  statusLabel: string;
  statusToneClassName: string;
  runtimeState: "ok" | "error";
  backendMode: BackendMode;
  logsPath: string;
  routeModeHint: string | null;
};

export type AudioDiagnosticsPayload = {
  catalog: AudioDevicesCatalog;
  devices: AudioDevices;
  engineStatus?: EngineStatus;
  forceAudioSetupScreen?: boolean;
};

function resolveConfigCaptureDeviceId(
  config: { captureDeviceId?: string | null; inputDeviceId?: string | null } | null | undefined,
) {
  return config?.captureDeviceId ?? config?.inputDeviceId ?? null;
}

function resolveConfigCaptureSourceType(
  catalog: AudioDevicesCatalog,
  config: { captureSourceType?: CaptureSourceType | null; captureDeviceId?: string | null; inputDeviceId?: string | null } | null | undefined,
): CaptureSourceType | null {
  if (config?.captureSourceType === "input" || config?.captureSourceType === "render_loopback") {
    return config.captureSourceType;
  }
  return resolveCaptureSourceType(catalog, resolveConfigCaptureDeviceId(config));
}

export function isWasapiDevice(device: AudioDeviceOption) {
  return device.hostapi?.includes("WASAPI") ?? false;
}

export function isVirtualCableInputDevice(device: AudioDeviceOption) {
  return device.kind === "input" && device.label.toLowerCase().includes("cable output");
}

export function isVirtualCableDevice(device: AudioDeviceOption) {
  const label = device.label.toLowerCase();
  return label.includes("vb-audio") || label.includes("cable ");
}

export function isPhysicalOutputDevice(device: AudioDeviceOption) {
  return device.kind === "output" && !isVirtualCableDevice(device);
}

export function isMockDeviceId(deviceId: string | null | undefined) {
  return String(deviceId ?? "").trim().toLowerCase().startsWith("mock-");
}

export function resolveWasapiInputs(catalog: AudioDevicesCatalog) {
  return catalog.inputs.filter(isWasapiDevice);
}

export function resolveWasapiOutputs(catalog: AudioDevicesCatalog) {
  return catalog.outputs.filter(isWasapiDevice);
}

function formatCaptureSourceDisplayLabel(device: AudioDeviceOption, sourceType: CaptureSourceType) {
  if (sourceType === "render_loopback") {
    return `${device.label} · Loopback`;
  }
  if (isVirtualCableDevice(device)) {
    return `${device.label} · Виртуальный вход`;
  }
  return device.label;
}

function buildCaptureSourceOption(device: AudioDeviceOption, sourceType: CaptureSourceType): CaptureSourceOption {
  const isSupported = sourceType === "input" || Boolean(device.supportsLoopback);
  return {
    deviceId: device.deviceId,
    label: device.label,
    displayLabel: formatCaptureSourceDisplayLabel(device, sourceType),
    sourceType,
    supportState: isSupported ? "supported" : "unsupported",
    isAvailable: true,
    isSupported,
    supportReason: isSupported ? null : LOOPBACK_SUPPORT_REASON,
    hostapi: device.hostapi,
    device,
  };
}

function createMissingDeviceOption(
  deviceId: string,
  kind: "input" | "output",
): AudioDeviceOption {
  return {
    deviceId,
    label: `Недоступное устройство (${deviceId})`,
    kind,
    isDefault: false,
    hostapi: WASAPI_FALLBACK_HOSTAPI,
    maxInputChannels: kind === "input" ? 0 : null,
    maxOutputChannels: kind === "output" ? 0 : null,
    defaultSampleRate: null,
    compatibleDeviceIds: [],
    compatibleSampleRates: [],
  };
}

function ensureDeviceOption(
  options: AudioDeviceOption[],
  deviceId: string | null | undefined,
  kind: "input" | "output",
) {
  if (!deviceId) return options;
  if (options.some((option) => option.deviceId === deviceId)) return options;
  return [...options, createMissingDeviceOption(deviceId, kind)];
}

function ensureCaptureSourceOption(
  options: CaptureSourceOption[],
  catalog: AudioDevicesCatalog,
  deviceId: string | null | undefined,
) {
  if (!deviceId) return options;
  if (options.some((option) => option.deviceId === deviceId)) return options;
  const inputDevice = resolveWasapiInputs(catalog).find((option) => option.deviceId === deviceId);
  if (inputDevice) return [...options, buildCaptureSourceOption(inputDevice, "input")];
  const outputDevice = resolveWasapiOutputs(catalog).find((option) => option.deviceId === deviceId);
  if (outputDevice) return [...options, buildCaptureSourceOption(outputDevice, "render_loopback")];
  const missingOption: CaptureSourceOption = {
    deviceId,
    label: `Недоступное устройство (${deviceId})`,
    displayLabel: `Недоступное устройство (${deviceId})`,
    sourceType: "input",
    supportState: "supported",
    isAvailable: false,
    isSupported: true,
    supportReason: null,
    hostapi: WASAPI_FALLBACK_HOSTAPI,
    device: createMissingDeviceOption(deviceId, "input"),
  };
  return [...options, missingOption];
}

export function resolveWasapiCaptureCandidates(catalog: AudioDevicesCatalog) {
  const byId = new Map<string, AudioDeviceOption>();
  for (const device of resolveWasapiInputs(catalog)) {
    byId.set(device.deviceId, device);
  }
  for (const device of resolveWasapiOutputs(catalog)) {
    if (!byId.has(device.deviceId)) {
      byId.set(device.deviceId, device);
    }
  }
  return [...byId.values()];
}

export function resolveCaptureSourceOptions(catalog: AudioDevicesCatalog) {
  return [
    ...resolveWasapiInputs(catalog).map((device) => buildCaptureSourceOption(device, "input")),
    ...resolveWasapiOutputs(catalog)
      .filter((device) => !resolveWasapiInputs(catalog).some((input) => input.deviceId === device.deviceId))
      .map((device) => buildCaptureSourceOption(device, "render_loopback")),
  ];
}

export function resolvePreferredInputId(
  preferredId: string | null | undefined,
  fallbackId: string | null | undefined,
  options: AudioDeviceOption[],
  { preferVirtualCable = false }: { preferVirtualCable?: boolean } = {},
) {
  if (preferredId && options.some((option) => option.deviceId === preferredId)) return preferredId;
  if (preferredId) {
    const byLabel = options.find((option) => option.label === preferredId);
    if (byLabel) return byLabel.deviceId;
  }
  if (fallbackId && options.some((option) => option.deviceId === fallbackId)) return fallbackId;
  if (fallbackId) {
    const byLabel = options.find((option) => option.label === fallbackId);
    if (byLabel) return byLabel.deviceId;
  }
  if (preferVirtualCable) {
    const virtualCable = options.find(isVirtualCableInputDevice);
    if (virtualCable) return virtualCable.deviceId;
  }
  return options.find((option) => option.isDefault)?.deviceId ?? options[0]?.deviceId ?? "";
}

export function resolvePreferredOutputId(
  preferredId: string | null | undefined,
  fallbackId: string | null | undefined,
  options: AudioDeviceOption[],
) {
  if (preferredId && options.some((option) => option.deviceId === preferredId)) return preferredId;
  if (preferredId) {
    const byLabel = options.find((option) => option.label === preferredId);
    if (byLabel) return byLabel.deviceId;
  }
  if (fallbackId && options.some((option) => option.deviceId === fallbackId)) return fallbackId;
  if (fallbackId) {
    const byLabel = options.find((option) => option.label === fallbackId);
    if (byLabel) return byLabel.deviceId;
  }
  return options.find((option) => option.isDefault)?.deviceId ?? options[0]?.deviceId ?? "";
}

function resolveRequestedCaptureCandidate(
  captureDeviceId: string,
  candidates: AudioDeviceOption[],
) {
  return candidates.find((option) => option.deviceId === captureDeviceId) ?? candidates[0] ?? null;
}

function resolveRequestedCaptureSource(
  captureDeviceId: string,
  options: CaptureSourceOption[],
) {
  return options.find((option) => option.deviceId === captureDeviceId) ?? options[0] ?? null;
}

function resolveRequestedInput(
  captureSource: CaptureSourceOption | null,
  wasapiInputs: AudioDeviceOption[],
) {
  if (!captureSource || captureSource.sourceType !== "input") return null;
  return wasapiInputs.find((option) => option.deviceId === captureSource.deviceId) ?? null;
}

function resolveCaptureSourceType(catalog: AudioDevicesCatalog, deviceId: string | null | undefined): CaptureSourceType | null {
  if (!deviceId) return null;
  if (resolveWasapiInputs(catalog).some((device) => device.deviceId === deviceId)) return "input";
  if (resolveWasapiOutputs(catalog).some((device) => device.deviceId === deviceId)) return "render_loopback";
  return null;
}

function resolveCompatibleOutputsForInput(
  input: AudioDeviceOption | null,
  wasapiOutputs: AudioDeviceOption[],
) {
  if (!input?.compatibleDeviceIds?.length) {
    return wasapiOutputs;
  }
  const compatibleOutputs = wasapiOutputs.filter((option) => input.compatibleDeviceIds?.includes(option.deviceId));
  return compatibleOutputs.length > 0 ? compatibleOutputs : wasapiOutputs;
}

function resolveSourceAssistMeta({
  hasSupportedSystemAudioSource,
  hasVirtualCable,
  hasLoopbackSources,
}: {
  hasSupportedSystemAudioSource: boolean;
  hasVirtualCable: boolean;
  hasLoopbackSources: boolean;
}) {
  if (hasSupportedSystemAudioSource) {
    return {
      title: "Источники системного звука",
      description: hasVirtualCable
        ? "Для захвата системного звука можно использовать VB-Cable, другие виртуальные входы или совместимые устройства вывода через WASAPI loopback. Если подходящий источник уже есть, установка VB-Cable не обязательна."
        : "Для захвата системного звука можно использовать совместимые виртуальные входы или устройства вывода через WASAPI loopback. Если подходящий источник уже есть, установка VB-Cable не обязательна.",
      statusText: "Доступные источники обнаружены",
      tone: "success" as const,
      shouldOfferVBCableInstall: false,
      shouldRecommendVBCable: false,
    };
  }

  if (hasLoopbackSources) {
    return {
      title: "Источники системного звука",
      description: "В системе видны устройства вывода для WASAPI loopback, но не все из них доступны для текущего режима захвата. Если нужный маршрут не проходит проверку, используйте VB-Cable или другой виртуальный вход.",
      statusText: "Нужен совместимый источник захвата",
      tone: "warning" as const,
      shouldOfferVBCableInstall: true,
      shouldRecommendVBCable: true,
    };
  }

  return {
    title: "Источники системного звука",
    description: "Если подходящих виртуальных входов и loopback-источников нет, установите VB-Cable. После установки он появится в списке Источник захвата и даст стабильный сценарий для системного звука.",
    statusText: "Совместимые источники не найдены",
    tone: "error" as const,
    shouldOfferVBCableInstall: true,
    shouldRecommendVBCable: true,
  };
}

export function pickAutoOutputIdForCapture({
  captureDeviceId,
  currentOutputId,
  preferredOutputId,
  fallbackOutputId,
  catalog,
}: {
  captureDeviceId: string;
  currentOutputId?: string | null;
  preferredOutputId?: string | null;
  fallbackOutputId?: string | null;
  catalog: AudioDevicesCatalog;
}) {
  const wasapiInputs = resolveWasapiInputs(catalog);
  const wasapiOutputs = resolveWasapiOutputs(catalog);
  const captureSources = resolveCaptureSourceOptions(catalog);
  const requestedCaptureSource = resolveRequestedCaptureSource(captureDeviceId, captureSources);
  const requestedInput = resolveRequestedInput(requestedCaptureSource, wasapiInputs);
  const compatibleOutputs = resolveCompatibleOutputsForInput(requestedInput, wasapiOutputs);

  const findOutput = (deviceId: string | null | undefined) => {
    if (!deviceId) return null;
    return compatibleOutputs.find((option) => option.deviceId === deviceId)
      ?? compatibleOutputs.find((option) => option.label === deviceId)
      ?? null;
  };

  const requestedCurrentOutput = findOutput(currentOutputId);
  if (requestedCurrentOutput) {
    return requestedCurrentOutput.deviceId;
  }

  const requestedPreferredOutput = findOutput(preferredOutputId);
  if (requestedPreferredOutput && isPhysicalOutputDevice(requestedPreferredOutput)) {
    return requestedPreferredOutput.deviceId;
  }

  const requestedFallbackOutput = findOutput(fallbackOutputId);
  if (requestedFallbackOutput && isPhysicalOutputDevice(requestedFallbackOutput)) {
    return requestedFallbackOutput.deviceId;
  }

  return compatibleOutputs.find(isPhysicalOutputDevice)?.deviceId
    ?? compatibleOutputs.find((option) => option.isDefault)?.deviceId
    ?? compatibleOutputs[0]?.deviceId
    ?? "";
}

export function resolveAudioPair(
  inputDeviceId: string,
  outputDeviceId: string,
  catalog: AudioDevicesCatalog,
) {
  return resolveAudioPairWithAutoOutput({
    captureDeviceId: inputDeviceId,
    outputDeviceId,
    preferredOutputId: outputDeviceId,
    catalog,
  });
}

export function resolveAudioPairWithAutoOutput({
  captureDeviceId,
  outputDeviceId,
  preferredOutputId,
  fallbackOutputId,
  catalog,
}: {
  captureDeviceId: string;
  outputDeviceId: string;
  preferredOutputId?: string | null;
  fallbackOutputId?: string | null;
  catalog: AudioDevicesCatalog;
}) {
  const captureCandidates = resolveWasapiCaptureCandidates(catalog);
  const captureSourceOptions = resolveCaptureSourceOptions(catalog);
  const wasapiInputs = resolveWasapiInputs(catalog);
  const wasapiOutputs = resolveWasapiOutputs(catalog);
  const selectedCaptureCandidate = resolveRequestedCaptureCandidate(captureDeviceId, captureCandidates);
  const selectedCaptureSource = resolveRequestedCaptureSource(captureDeviceId, captureSourceOptions);
  const selectedInput = resolveRequestedInput(selectedCaptureSource, wasapiInputs);
  const filteredOutputs = resolveCompatibleOutputsForInput(selectedInput, wasapiOutputs);
  const resolvedOutputId = pickAutoOutputIdForCapture({
    captureDeviceId,
    currentOutputId: outputDeviceId,
    preferredOutputId,
    fallbackOutputId,
    catalog,
  });
  const resolvedOutput = filteredOutputs.find((option) => option.deviceId === resolvedOutputId)
    ?? filteredOutputs.find((option) => option.deviceId === outputDeviceId)
    ?? wasapiOutputs.find((option) => option.deviceId === resolvedOutputId)
    ?? wasapiOutputs.find((option) => option.deviceId === outputDeviceId)
    ?? filteredOutputs[0]
    ?? null;
  const filteredInputs = resolvedOutput?.compatibleDeviceIds?.length
    ? wasapiInputs.filter((option) => resolvedOutput.compatibleDeviceIds?.includes(option.deviceId))
    : wasapiInputs;
  const resolvedInput = filteredInputs.find((option) => option.deviceId === captureDeviceId)
    ?? selectedInput
    ?? filteredInputs[0]
    ?? null;
  const sharedSampleRate = resolvedInput?.compatibleSampleRates?.find((value) => resolvedOutput?.compatibleSampleRates?.includes(value))
    ?? resolvedOutput?.defaultSampleRate
    ?? resolvedInput?.defaultSampleRate
    ?? "48000";

  return {
    input: selectedCaptureCandidate?.deviceId ?? captureDeviceId,
    output: resolvedOutput?.deviceId ?? outputDeviceId,
    sampleRate: sharedSampleRate,
  };
}

export function formatAudioDeviceLabel(device: AudioDeviceOption) {
  return device.kind === "input" && isVirtualCableInputDevice(device)
    ? "CABLE Output (VB-Audio Virtual Cable)"
    : device.label;
}

export function resolveAudioDeviceLabel(
  catalog: AudioDevicesCatalog,
  deviceId: string | null | undefined,
) {
  const resolvedId = String(deviceId ?? "");
  if (!resolvedId) return "Not selected";
  const device = [...catalog.inputs, ...catalog.outputs].find((option) => option.deviceId === resolvedId);
  return device ? formatAudioDeviceLabel(device) : `Недоступное устройство (${resolvedId})`;
}

function resolveConfigRouteLabels(
  catalog: AudioDevicesCatalog,
  config: { captureDeviceId?: string; inputDeviceId?: string; outputDeviceId: string } | null,
) {
  if (!config) {
    return {
      inputLabel: null,
      outputLabel: null,
    };
  }
  return {
    inputLabel: resolveAudioDeviceLabel(catalog, resolveConfigCaptureDeviceId(config)),
    outputLabel: resolveAudioDeviceLabel(catalog, config.outputDeviceId),
  };
}

export function resolveBackendMode(engineStatus?: EngineStatus): BackendMode {
  const mode = engineStatus?.backendMode;
  if (mode === "real" || mode === "mock") return mode;
  return "unknown";
}

export function resolveLogsPath(engineStatus?: EngineStatus) {
  return engineStatus?.logPaths?.logsDir ?? engineStatus?.logPaths?.runtimeRoot ?? "Unavailable";
}

export function isSampleRateMismatchReason(reason: string | null | undefined) {
  const value = String(reason ?? "").toLowerCase();
  return (
    value.includes("sample rate")
    || value.includes("частот")
    || value.includes("битрейт")
    || value.includes("no compatible sample rate")
  );
}

function formatCaptureSourceTypeLabel(sourceType: CaptureSourceType | null | undefined) {
  if (sourceType === "render_loopback") return "Render loopback";
  if (sourceType === "input") return "Вход / виртуальный вход";
  return null;
}

function formatCaptureStrategyLabel(sourceType: CaptureSourceType | null | undefined) {
  if (sourceType === "render_loopback") return "WASAPI loopback";
  if (sourceType === "input") return "WASAPI input";
  return null;
}

function resolveLoopbackEndpointId(
  catalog: AudioDevicesCatalog,
  sourceType: CaptureSourceType | null | undefined,
  deviceId: string | null | undefined,
) {
  if (sourceType !== "render_loopback" || !deviceId) return null;
  return resolveWasapiOutputs(catalog).find((device) => device.deviceId === deviceId)?.loopbackEndpointId ?? null;
}

export function validateAudioRoute({
  selectedCaptureSource,
  selectedInput,
  selectedOutput,
  backendMode,
  runtimeStatus,
}: {
  selectedCaptureSource: CaptureSourceOption | null;
  selectedInput: AudioDeviceOption | null;
  selectedOutput: AudioDeviceOption | null;
  backendMode: BackendMode;
  runtimeStatus: EngineStatus["audioStatus"] | null | undefined;
}): RouteValidation {
  if (!selectedCaptureSource) {
    return { valid: false, reason: "no capture device selected" };
  }
  if (!selectedOutput) {
    return { valid: false, reason: "no output device selected" };
  }
  if (selectedCaptureSource.sourceType === "render_loopback" && selectedCaptureSource.deviceId === selectedOutput.deviceId) {
    return {
      valid: false,
      reason: "Нельзя использовать одно и то же устройство для loopback-захвата и вывода",
    };
  }
  if (backendMode === "mock") {
    return { valid: false, reason: "mock runtime active" };
  }
  if (runtimeStatus?.status === "failed" || runtimeStatus?.status === "device_unavailable") {
    return {
      valid: false,
      reason: runtimeStatus.lastError ?? "backend audio error",
    };
  }
  if (!selectedCaptureSource.isSupported) {
    return {
      valid: false,
      reason: selectedCaptureSource.supportReason ?? LOOPBACK_SUPPORT_REASON,
    };
  }
  if (selectedCaptureSource.sourceType === "input" && !selectedInput) {
    return { valid: false, reason: "selected capture device has no input channels" };
  }
  return { valid: true, reason: "-" };
}

export function evaluateAudioSetupEnvironment(
  catalog: AudioDevicesCatalog,
  devices: AudioDevices,
  engineStatus?: EngineStatus,
): AudioSetupEvaluation {
  const runtimeStatus = engineStatus?.audioStatus;
  const attemptedInputId = resolveConfigCaptureDeviceId(runtimeStatus?.desiredConfig) ?? devices.input;
  const attemptedOutputId = runtimeStatus?.desiredConfig?.outputDeviceId ?? devices.output;
  const captureCandidates = ensureDeviceOption(resolveWasapiCaptureCandidates(catalog), attemptedInputId, "input");
  const captureSourceOptions = ensureCaptureSourceOption(resolveCaptureSourceOptions(catalog), catalog, attemptedInputId);
  const wasapiInputs = resolveWasapiInputs(catalog);
  const wasapiOutputs = ensureDeviceOption(resolveWasapiOutputs(catalog), attemptedOutputId, "output");
  const selectedCaptureCandidate = captureCandidates.find((option) => option.deviceId === devices.input) ?? null;
  const selectedCaptureSource = captureSourceOptions.find((option) => option.deviceId === devices.input) ?? null;
  const selectedInput = resolveRequestedInput(selectedCaptureSource, wasapiInputs);
  const selectedOutput = wasapiOutputs.find((option) => option.deviceId === devices.output) ?? null;
  const resolvedCaptureOptions = captureSourceOptions;
  const resolvedOutputOptions = selectedInput?.compatibleDeviceIds?.length
    ? wasapiOutputs.filter((option) => selectedInput.compatibleDeviceIds?.includes(option.deviceId))
    : wasapiOutputs;
  const hasVirtualCable = wasapiInputs.some(isVirtualCableInputDevice);
  const availableInputSources = captureSourceOptions.filter((option) => option.sourceType === "input");
  const availableLoopbackSources = captureSourceOptions.filter((option) => option.sourceType === "render_loopback");
  const hasCompatibleInputSource = availableInputSources.length > 0;
  const hasSupportedSystemAudioSource = captureSourceOptions.some((option) => option.isSupported);
  const backendMode = resolveBackendMode(engineStatus);
  const activeConfig = runtimeStatus?.activeConfig ?? null;
  const activeCaptureSourceType = resolveConfigCaptureSourceType(catalog, activeConfig);
  const activeRouteLabels = resolveConfigRouteLabels(catalog, activeConfig);
  const sourceAssistMeta = resolveSourceAssistMeta({
    hasSupportedSystemAudioSource,
    hasVirtualCable,
    hasLoopbackSources: availableLoopbackSources.length > 0,
  });
  const routeValidation = validateAudioRoute({
    selectedCaptureSource,
    selectedInput,
    selectedOutput,
    backendMode,
    runtimeStatus,
  });
  const isRouteValid = routeValidation.valid;
  const routeError = routeValidation.valid ? null : routeValidation.reason;
  const runtimeState = runtimeStatus?.status === "failed" || runtimeStatus?.status === "device_unavailable" ? "error" : "ok";
  const isEvaluationReady = backendMode !== "unknown" || (runtimeStatus?.status ?? "idle") !== "idle" || captureSourceOptions.length === 0 || wasapiOutputs.length === 0;
  const statusLabel = isRouteValid ? "Готово" : "Нужна настройка";
  const isActiveRouteDifferentFromAttempted = Boolean(
    activeConfig
    && (
    resolveConfigCaptureDeviceId(activeConfig) !== devices.input
      || activeConfig.outputDeviceId !== devices.output
    )
  );
  const shouldExplainFallbackActiveRoute = !isRouteValid && isActiveRouteDifferentFromAttempted;
  const statusToneClassName = isRouteValid
    ? "border-emerald-400/20 bg-emerald-400/10 text-emerald-100"
    : "border-amber-400/20 bg-amber-400/10 text-amber-100";
  const selectedCaptureLabel = selectedCaptureSource ? resolveAudioDeviceLabel(catalog, selectedCaptureSource.deviceId) : null;
  const selectedCaptureSourceTypeLabel = formatCaptureSourceTypeLabel(selectedCaptureSource?.sourceType ?? null);
  const activeCaptureSourceTypeLabel = formatCaptureSourceTypeLabel(activeCaptureSourceType);
  const selectedCaptureStrategyLabel = formatCaptureStrategyLabel(selectedCaptureSource?.sourceType ?? null);
  const activeCaptureStrategyLabel = formatCaptureStrategyLabel(activeCaptureSourceType);
  const selectedLoopbackEndpointId = resolveLoopbackEndpointId(catalog, selectedCaptureSource?.sourceType ?? null, selectedCaptureSource?.deviceId ?? null);
  const activeLoopbackEndpointId = resolveLoopbackEndpointId(catalog, activeCaptureSourceType, resolveConfigCaptureDeviceId(activeConfig));
  const routeModeHint = selectedCaptureSource?.sourceType === "render_loopback" ? "Способ захвата: WASAPI loopback" : null;

  return {
    captureCandidates,
    captureSourceOptions,
    availableInputSources,
    availableLoopbackSources,
    wasapiInputs,
    wasapiOutputs,
    selectedCaptureCandidate,
    selectedCaptureSource,
    selectedCaptureSourceType: selectedCaptureSource?.sourceType ?? null,
    activeCaptureSourceType,
    selectedCaptureSourceTypeLabel,
    activeCaptureSourceTypeLabel,
    selectedCaptureStrategyLabel,
    activeCaptureStrategyLabel,
    selectedInput,
    selectedOutput,
    selectedCaptureLabel,
    activeInputLabel: activeRouteLabels.inputLabel,
    activeOutputLabel: activeRouteLabels.outputLabel,
    selectedLoopbackEndpointId,
    activeLoopbackEndpointId,
    isActiveRouteDifferentFromAttempted,
    shouldExplainFallbackActiveRoute,
    resolvedCaptureOptions,
    resolvedOutputOptions: resolvedOutputOptions.length > 0 ? resolvedOutputOptions : wasapiOutputs,
    hasVirtualCable,
    hasCompatibleInputSource,
    hasSupportedSystemAudioSource,
    shouldRecommendVBCable: sourceAssistMeta.shouldRecommendVBCable,
    shouldOfferVBCableInstall: sourceAssistMeta.shouldOfferVBCableInstall,
    sourceAssistTitle: sourceAssistMeta.title,
    sourceAssistDescription: sourceAssistMeta.description,
    sourceAssistStatusText: sourceAssistMeta.statusText,
    sourceAssistTone: sourceAssistMeta.tone,
    routeValidation,
    isRouteValid,
    isEvaluationReady,
    routeError,
    statusLabel,
    statusToneClassName,
    runtimeState,
    backendMode,
    logsPath: resolveLogsPath(engineStatus),
    routeModeHint,
  };
}

export function buildAudioDiagnosticsText({
  catalog,
  devices,
  engineStatus,
  forceAudioSetupScreen = false,
}: AudioDiagnosticsPayload) {
  const evaluation = evaluateAudioSetupEnvironment(catalog, devices, engineStatus);
  return [
    "EarLoop Audio Diagnostics",
    `Attempted capture type: ${evaluation.selectedCaptureSourceType ?? "-"}`,
    `Attempted capture type label: ${evaluation.selectedCaptureSourceTypeLabel ?? "-"}`,
    `Attempted capture strategy: ${evaluation.selectedCaptureStrategyLabel ?? "-"}`,
    `Attempted input: ${resolveAudioDeviceLabel(catalog, devices.input)}`,
    `Attempted output: ${resolveAudioDeviceLabel(catalog, devices.output)}`,
    `Active capture type: ${evaluation.activeCaptureSourceType ?? "-"}`,
    `Active capture type label: ${evaluation.activeCaptureSourceTypeLabel ?? "-"}`,
    `Active capture strategy: ${evaluation.activeCaptureStrategyLabel ?? "-"}`,
    `Active input: ${evaluation.activeInputLabel ?? "-"}`,
    `Active output: ${evaluation.activeOutputLabel ?? "-"}`,
    `Selected loopback endpoint: ${evaluation.selectedLoopbackEndpointId ?? "-"}`,
    `Active loopback endpoint: ${evaluation.activeLoopbackEndpointId ?? "-"}`,
    `Virtual cable: ${evaluation.hasVirtualCable ? "detected" : "not detected"}`,
    `Input sources: ${evaluation.availableInputSources.length}`,
    `Loopback sources: ${evaluation.availableLoopbackSources.length}`,
    `Runtime: ${evaluation.runtimeState}`,
    `Backend mode: ${evaluation.backendMode}`,
    `Route: ${evaluation.isRouteValid ? "valid" : "invalid"}`,
    `Reason: ${evaluation.routeValidation.reason}`,
    `Fallback active route: ${evaluation.shouldExplainFallbackActiveRoute ? "yes" : "no"}`,
    `Logs: ${evaluation.logsPath}`,
    `App version: ${typeof __APP_VERSION__ === "string" ? __APP_VERSION__ : "unknown"}`,
    `Force audio setup screen: ${forceAudioSetupScreen ? "true" : "false"}`,
  ].join("\n");
}
