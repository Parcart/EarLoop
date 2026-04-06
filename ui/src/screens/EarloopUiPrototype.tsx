import { useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties } from "react";
import { ChevronDown, Copy, Minus, Square, X } from "lucide-react";

import { Sidebar } from "@/components/layout/Sidebar";
import { APP_ONBOARDING_VERSION } from "@/features/onboarding/app-onboarding";
import { OnboardingOverlay } from "@/features/onboarding/OnboardingOverlay";
import { AppOnboardingProvider, useAppOnboarding } from "@/features/onboarding/useAppOnboarding";
import { HomeScreen } from "@/features/home/HomeScreen";
import { SettingsScreen } from "@/features/settings/SettingsScreen";
import { SessionCompareScreen } from "@/features/session/SessionCompareScreen";
import { SessionIntroDialog } from "@/features/session/SessionIntroDialog";
import { SessionResultScreen } from "@/features/session/SessionResultScreen";
import { SessionStartScreen } from "@/features/session/SessionStartScreen";
import {
  createSession,
  deleteProfile,
  generateNextPair,
  getEngineConfig,
  getMainState,
  getEngineStatus,
  listAudioDevices,
  listProfiles,
  previewSessionTarget,
  saveProfileFromSession,
  setActiveProfile,
  setProcessingEnabled,
  startSession,
  updateEngineConfig,
  updateProfile,
} from "@/lib/api/engine";
import type { AudioDevicesCatalog, EngineStatus, MainRuntimeState, SessionPreviewStatus } from "@/lib/api/engine.types";
import { loadUiSettings, saveUiSettings } from "@/lib/storage/ui-settings";

import type {
  AudioDevices,
  ListeningTarget,
  PairKey,
  PerceptualParams,
  PipelineConfig,
  Screen,
  SessionPhase,
  ThemeMode,
  WavePreset,
} from "@/lib/types/ui";

function resolveInitialDeviceId(
  preferredId: string | null | undefined,
  fallbackId: string,
  options: AudioDevicesCatalog["inputs"] | AudioDevicesCatalog["outputs"],
) {
  if (preferredId && options.some((option) => option.deviceId === preferredId)) return preferredId;
  if (preferredId) {
    const byLabel = options.find((option) => option.label === preferredId);
    if (byLabel) return byLabel.deviceId;
  }
  if (options.some((option) => option.deviceId === fallbackId)) return fallbackId;
  const byFallbackLabel = options.find((option) => option.label === fallbackId);
  if (byFallbackLabel) return byFallbackLabel.deviceId;
  return options.find((option) => option.isDefault)?.deviceId ?? options[0]?.deviceId ?? fallbackId;
}

function isWasapiDevice(option: AudioDevicesCatalog["inputs"][number] | AudioDevicesCatalog["outputs"][number]) {
  return option.hostapi?.includes("WASAPI") ?? false;
}

function isVirtualCableInputDevice(option: AudioDevicesCatalog["inputs"][number]) {
  return option.label.toLowerCase().includes("cable output");
}

function hasCompatiblePartners(option: AudioDevicesCatalog["inputs"][number] | AudioDevicesCatalog["outputs"][number]) {
  return (option.compatibleDeviceIds?.length ?? 0) > 0;
}

function resolveUiInputOptions(options: AudioDevicesCatalog["inputs"]) {
  const wasapiCableOptions = options.filter((option) => isWasapiDevice(option) && isVirtualCableInputDevice(option));
  if (wasapiCableOptions.length > 0) return wasapiCableOptions;
  const wasapiOptions = options.filter(isWasapiDevice);
  return wasapiOptions.length > 0 ? wasapiOptions : options;
}

function resolveUiOutputOptions(options: AudioDevicesCatalog["outputs"]) {
  const wasapiOptions = options.filter(isWasapiDevice);
  const compatibleWasapiOptions = wasapiOptions.filter(hasCompatiblePartners);
  if (compatibleWasapiOptions.length > 0) return compatibleWasapiOptions;
  return wasapiOptions.length > 0 ? wasapiOptions : options;
}

function resolveCompatiblePair(
  inputDeviceId: string,
  outputDeviceId: string,
  catalog: AudioDevicesCatalog,
) {
  const inputs = resolveUiInputOptions(catalog.inputs);
  const outputs = resolveUiOutputOptions(catalog.outputs);
  const selectedInput = inputs.find((option) => option.deviceId === inputDeviceId) ?? inputs[0] ?? null;
  const outputCandidates = outputs.filter((option) => selectedInput?.compatibleDeviceIds?.includes(option.deviceId));
  const selectedOutput = outputCandidates.find((option) => option.deviceId === outputDeviceId)
    ?? outputs.find((option) => option.deviceId === outputDeviceId)
    ?? outputCandidates[0]
    ?? outputs[0]
    ?? null;
  const inputCandidates = inputs.filter((option) => selectedOutput?.compatibleDeviceIds?.includes(option.deviceId));
  const normalizedInput = inputCandidates.find((option) => option.deviceId === selectedInput?.deviceId)
    ?? inputs.find((option) => option.deviceId === inputDeviceId)
    ?? inputCandidates[0]
    ?? selectedInput;
  const sharedSampleRate = normalizedInput?.compatibleSampleRates?.find((value) => selectedOutput?.compatibleSampleRates?.includes(value))
    ?? selectedOutput?.defaultSampleRate
    ?? normalizedInput?.defaultSampleRate
    ?? "48000";

  return {
    input: normalizedInput?.deviceId ?? inputDeviceId,
    output: selectedOutput?.deviceId ?? outputDeviceId,
    sampleRate: sharedSampleRate,
  };
}

export default function EarloopUIPrototype() {
  const initialEngineStatus = useMemo(() => getEngineStatus(), []);
  const initialMainState = useMemo(() => getMainState(), []);
  const initialAudioDevicesCatalog = useMemo(() => listAudioDevices(), []);
  const visibleAudioInputs = useMemo(() => resolveUiInputOptions(initialAudioDevicesCatalog.inputs), [initialAudioDevicesCatalog.inputs]);
  const visibleAudioOutputs = useMemo(() => resolveUiOutputOptions(initialAudioDevicesCatalog.outputs), [initialAudioDevicesCatalog.outputs]);
  const initialUiSettings = useMemo(() => loadUiSettings(), []);
  const initialEngineConfig = useMemo(() => getEngineConfig(), []);
  const initialProfiles = useMemo(() => listProfiles(), []);
  const initialSelectedProfile = initialProfiles.some((profile) => profile.id === initialMainState.activeProfileId)
    ? (initialMainState.activeProfileId ?? "")
    : initialProfiles.some((profile) => profile.id === initialUiSettings.lastSelectedProfileId)
      ? initialUiSettings.lastSelectedProfileId
      : initialProfiles.some((profile) => profile.id === initialEngineConfig.defaults.activeProfileId)
        ? initialEngineConfig.defaults.activeProfileId
        : initialProfiles[0]?.id ?? "";

  const [screen, setScreen] = useState<Screen>(initialUiSettings.lastSelectedScreen);
  const [sidebarExpanded, setSidebarExpanded] = useState(initialUiSettings.sidebarExpanded);
  const [showIntroHint, setShowIntroHint] = useState(true);
  const [wasDisabledOnce, setWasDisabledOnce] = useState(false);
  const [listeningTarget, setListeningTarget] = useState<ListeningTarget>("base");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [themeMode, setThemeMode] = useState<ThemeMode>(initialUiSettings.theme);
  const [prefersDarkTheme, setPrefersDarkTheme] = useState(true);
  const [blockSize, setBlockSize] = useState(1024);
  const [gainCompensation, setGainCompensation] = useState(50);
  const [finalChoice, setFinalChoice] = useState<ListeningTarget | null>(null);
  const [savedProfiles, setSavedProfiles] = useState(initialProfiles);
  const [selectedProfile, setSelectedProfile] = useState(initialSelectedProfile);
  const [engineStatusSnapshot, setEngineStatusSnapshot] = useState<EngineStatus>(initialEngineStatus);
  const [mainStateSnapshot, setMainStateSnapshot] = useState<MainRuntimeState>(initialMainState);
  const [sessionBaseProfileId, setSessionBaseProfileId] = useState(initialEngineStatus.sessionDefaultProfileId);
  const [profileDraftName, setProfileDraftName] = useState(initialEngineStatus.defaultProfileName);
  const [saveNotice, setSaveNotice] = useState<string | null>(null);
  const [audioDevicesCatalog] = useState<AudioDevicesCatalog>(initialAudioDevicesCatalog);
  const initialInputDeviceId = resolveInitialDeviceId(
    initialUiSettings.lastSelectedInputDeviceId,
    initialEngineConfig.audio.inputDeviceId,
    visibleAudioInputs,
  );
  const initialOutputDeviceId = resolveInitialDeviceId(
    initialUiSettings.lastSelectedOutputDeviceId,
    initialEngineConfig.audio.outputDeviceId,
    visibleAudioOutputs,
  );
  const initialCompatiblePair = resolveCompatiblePair(
    initialInputDeviceId,
    initialOutputDeviceId,
    initialAudioDevicesCatalog,
  );
  const [devices, setDevices] = useState<AudioDevices>({
    input: initialCompatiblePair.input,
    output: initialCompatiblePair.output,
    sampleRate: initialCompatiblePair.sampleRate || initialEngineConfig.audio.sampleRate,
    channels: initialEngineConfig.audio.channels,
  });
  const [feedback, setFeedback] = useState("none");
  const [showFeedbackJump, setShowFeedbackJump] = useState(false);
  const [showSessionPipelineConfig, setShowSessionPipelineConfig] = useState(false);
  const [paramsModalFor, setParamsModalFor] = useState<PairKey | null>(null);
  const [showSessionIntroDialog, setShowSessionIntroDialog] = useState(false);
  const [hasSeenSessionIntroPrompt, setHasSeenSessionIntroPrompt] = useState(initialUiSettings.hasSeenSessionIntro);
  const [hasSeenAppOnboarding, setHasSeenAppOnboarding] = useState(initialUiSettings.hasSeenAppOnboarding);
  const [appOnboardingVersion, setAppOnboardingVersion] = useState<number | null>(initialUiSettings.appOnboardingVersion);
  const [sessionTutorialMode, setSessionTutorialMode] = useState(false);
  const [sessionPhase, setSessionPhase] = useState<SessionPhase>("idle");
  const [sessionPreviewStatus, setSessionPreviewStatus] = useState<SessionPreviewStatus | null>(null);
  const [startupStep, setStartupStep] = useState(0);
  const [isGeneratingPair, setIsGeneratingPair] = useState(false);
  const [homeWavePreset, setHomeWavePreset] = useState<WavePreset>("default");
  const [isShellMaximized, setIsShellMaximized] = useState(false);
  const [profileEditorName, setProfileEditorName] = useState("");
  const [profileEditorParams, setProfileEditorParams] = useState<PerceptualParams>(initialEngineStatus.defaultProfileParams);
  const [sessionPipelineConfig, setSessionPipelineConfig] = useState<PipelineConfig>(initialEngineStatus.defaultPipelineConfig);
  const [sessionSnapshot, setSessionSnapshot] = useState(() => createSession({
    profiles: initialProfiles,
    pipelineConfig: initialEngineStatus.defaultPipelineConfig,
    sessionBaseProfileId: initialEngineStatus.sessionDefaultProfileId,
  }));

  const refreshRuntimeSnapshots = () => {
    setMainStateSnapshot(getMainState());
    setEngineStatusSnapshot(getEngineStatus());
  };

  const feedbackAnchorRef = useRef<HTMLDivElement | null>(null);
  const feedbackSectionRef = useRef<HTMLDivElement | null>(null);
  const isPlaying = true;
  const shouldShowAppOnboarding = !hasSeenAppOnboarding || appOnboardingVersion !== APP_ONBOARDING_VERSION;
  const appOnboarding = useAppOnboarding({
    enabled: shouldShowAppOnboarding,
    onComplete: () => {
      setHasSeenAppOnboarding(true);
      setAppOnboardingVersion(APP_ONBOARDING_VERSION);
    },
  });
  const handleRestartAppOnboarding = () => {
    setHasSeenAppOnboarding(false);
    setAppOnboardingVersion(null);
    setFinalChoice(null);
    resetSessionFlow();
    setScreen("settings");
    appOnboarding.restart();
  };

  const sessionBaseProfile = sessionSnapshot.sessionBaseProfile;
  const sessionBaseParams = sessionSnapshot.sessionBaseParams;
  const sessionBaseLabel = sessionSnapshot.sessionBaseLabel;
  const pairA = sessionSnapshot.pairA;
  const pairB = sessionSnapshot.pairB;
  const iteration = sessionSnapshot.iteration;
  const progress = sessionSnapshot.progress;
  const pairVersion = sessionSnapshot.pairVersion;
  const modalParams = paramsModalFor === "A" ? pairA.params : paramsModalFor === "B" ? pairB.params : null;
  const managedProfile = useMemo(() => savedProfiles.find((profile) => profile.id === selectedProfile) ?? savedProfiles[0] ?? null, [savedProfiles, selectedProfile]);
  const startupSteps = engineStatusSnapshot.startupSteps;
  const handleDeviceChange = (field: keyof AudioDevices, value: string) => {
    setDevices((current) => {
      if (field === "output") {
        const nextPair = resolveCompatiblePair(current.input, value, audioDevicesCatalog);
        return {
          ...current,
          input: nextPair.input,
          output: nextPair.output,
          sampleRate: nextPair.sampleRate,
        };
      }
      if (field === "input") {
        const nextPair = resolveCompatiblePair(value, current.output, audioDevicesCatalog);
        return {
          ...current,
          input: nextPair.input,
          output: nextPair.output,
          sampleRate: nextPair.sampleRate,
        };
      }
      return {
        ...current,
        [field]: value,
      };
    });
  };

  const handleResetManagedProfile = () => {
    if (!managedProfile) return;
    setProfileEditorName(managedProfile.name);
    setProfileEditorParams(managedProfile.params);
  };

  const updateManagedProfileParam = (key: keyof PerceptualParams, nextValue: number) => {
    setProfileEditorParams((current) => ({ ...current, [key]: Number(nextValue.toFixed(2)) }));
  };

  const handleSaveManagedProfile = () => {
    if (!managedProfile) return;
    const trimmedName = profileEditorName.trim();
    if (!trimmedName) return;
    const nextProfiles = updateProfile({
      profileId: managedProfile.id,
      name: trimmedName,
      params: profileEditorParams,
    });
    setSavedProfiles(nextProfiles);
    refreshRuntimeSnapshots();
  };

  const handleDeleteManagedProfile = () => {
    if (!managedProfile || savedProfiles.length <= 1) return;
    const remainingProfiles = deleteProfile({
      profileId: managedProfile.id,
    });
    setSavedProfiles(remainingProfiles);
    const nextMainState = getMainState();
    setMainStateSnapshot(nextMainState);
    setEngineStatusSnapshot(getEngineStatus());
    setSelectedProfile(nextMainState.activeProfileId ?? remainingProfiles[0]?.id ?? "");
  };

  const handleStartSessionTutorial = () => {
    setSessionTutorialMode(true);
    setShowSessionIntroDialog(false);
  };

  const handleSkipSessionTutorial = () => {
    setSessionTutorialMode(false);
    setShowSessionIntroDialog(false);
  };

  const resetSessionFlow = () => {
    const resetSnapshot = createSession({
      profiles: savedProfiles,
      pipelineConfig: sessionPipelineConfig,
      sessionBaseProfileId,
    });
    setListeningTarget("base");
    setFeedback("none");
    setParamsModalFor(null);
    setFinalChoice(null);
    setSessionPhase("idle");
    setSessionPreviewStatus(null);
    setStartupStep(0);
    setProfileDraftName(engineStatusSnapshot.defaultProfileName);
    setSessionSnapshot(resetSnapshot);
    setSessionPipelineConfig(resetSnapshot.sessionPipelineConfig);
    setShowSessionPipelineConfig(false);
    setSaveNotice(null);
    setShowFeedbackJump(false);
    setIsGeneratingPair(false);
  };

  const handleHomeToggle = () => {
    const nextEnabled = !mainStateSnapshot.processingEnabled;
    const nextMainState = setProcessingEnabled({ enabled: nextEnabled });
    setMainStateSnapshot(nextMainState);
    setEngineStatusSnapshot(getEngineStatus());
    if (!nextEnabled) {
      setWasDisabledOnce(true);
      return;
    }
    if (wasDisabledOnce) setShowIntroHint(false);
  };

  const handleActiveProfileChange = (profileId: string) => {
    const nextMainState = setActiveProfile({ profileId });
    setSelectedProfile(nextMainState.activeProfileId ?? profileId);
    setMainStateSnapshot(nextMainState);
    setEngineStatusSnapshot(getEngineStatus());
  };

  const handlePreviewTarget = (target: ListeningTarget, snapshot = sessionSnapshot) => {
    const previewStatus = previewSessionTarget({
      sessionId: snapshot.sessionId,
      target,
    });
    setSessionPreviewStatus(previewStatus);
    if (previewStatus.applyStatus !== "failed") {
      setListeningTarget(target);
    }
    return previewStatus;
  };

  const startNextPairGeneration = (target?: ListeningTarget) => {
    const selectedTarget = target ?? listeningTarget;
    const nextFeedback = target ? "none" : feedback;
    if (target) {
      handlePreviewTarget(target);
    }
    setIsGeneratingPair(true);
    window.setTimeout(() => {
      const nextSession = generateNextPair({
        profiles: savedProfiles,
        pipelineConfig: sessionPipelineConfig,
        sessionBaseProfileId,
        sessionBaseParams: sessionSnapshot.sessionBaseParams,
        sessionBaseLabel: sessionSnapshot.sessionBaseLabel,
        iteration,
        progress,
        pairVersion,
        selectedTarget,
        feedback: nextFeedback,
      });
      setSessionSnapshot(nextSession);
      const nextPreviewStatus = handlePreviewTarget("base", nextSession);
      if (nextPreviewStatus.applyStatus === "failed") {
        setListeningTarget("base");
      } else {
        setListeningTarget(nextPreviewStatus.target ?? "base");
      }
      setFeedback("none");
      setIsGeneratingPair(false);
    }, 850);
  };

  const handleReject = () => {
    startNextPairGeneration();
  };

  const handleCardDoubleClick = (target: PairKey) => {
    if (isGeneratingPair) return;
    setFeedback("none");
    startNextPairGeneration(target);
  };

  const handleFinish = () => {
    const suggestedName = listeningTarget === "base" ? "Базовый профиль" : `Профиль ${listeningTarget}`;
    setProfileDraftName(suggestedName);
    setSaveNotice(null);
    setFinalChoice(listeningTarget);
  };

  const handleSaveProfile = () => {
    const trimmedName = profileDraftName.trim();
    if (!trimmedName) {
      setSaveNotice("Укажи название профиля");
      return;
    }

    const result = saveProfileFromSession({
      name: trimmedName,
      finalChoice: finalChoice ?? "base",
    });
    setSavedProfiles(result.profiles);
    const nextMainState = getMainState();
    setMainStateSnapshot(nextMainState);
    setEngineStatusSnapshot(getEngineStatus());
    resetSessionFlow();
    setSelectedProfile(nextMainState.activeProfileId ?? result.profileId);
    setScreen("home");
    setSaveNotice(`Профиль «${trimmedName}» сохранён`);
  };

  useEffect(() => {
    if (managedProfile) {
      setProfileEditorName(managedProfile.name);
      setProfileEditorParams(managedProfile.params);
    }
  }, [managedProfile?.id]);

  useEffect(() => {
    if (!mainStateSnapshot.activeProfileId) return;
    if (mainStateSnapshot.activeProfileId === selectedProfile) return;
    if (!savedProfiles.some((profile) => profile.id === mainStateSnapshot.activeProfileId)) return;
    setSelectedProfile(mainStateSnapshot.activeProfileId);
  }, [mainStateSnapshot.activeProfileId, savedProfiles, selectedProfile]);

  useEffect(() => {
    if (sessionBaseProfileId !== engineStatusSnapshot.sessionDefaultProfileId && !savedProfiles.some((profile) => profile.id === sessionBaseProfileId)) {
      setSessionBaseProfileId(engineStatusSnapshot.sessionDefaultProfileId);
    }
  }, [engineStatusSnapshot.sessionDefaultProfileId, savedProfiles, sessionBaseProfileId]);

  useEffect(() => {
    saveUiSettings({
      theme: themeMode,
      sidebarExpanded,
      hasSeenSessionIntro: hasSeenSessionIntroPrompt,
      hasSeenAppOnboarding,
      appOnboardingVersion,
      lastSelectedScreen: screen,
      lastSelectedProfileId: selectedProfile,
      lastSelectedInputDeviceId: devices.input,
      lastSelectedOutputDeviceId: devices.output,
    });
  }, [appOnboardingVersion, devices.input, devices.output, hasSeenAppOnboarding, hasSeenSessionIntroPrompt, screen, selectedProfile, sidebarExpanded, themeMode]);

  useEffect(() => {
    updateEngineConfig({
      config: {
        audio: {
          inputDeviceId: devices.input,
          outputDeviceId: devices.output,
          sampleRate: devices.sampleRate,
          channels: devices.channels,
        },
      },
    });
    refreshRuntimeSnapshots();
  }, [devices.channels, devices.input, devices.output, devices.sampleRate]);

  useEffect(() => {
    const nextPipelineConfig = sessionBaseProfile?.pipelineConfig ?? engineStatusSnapshot.defaultPipelineConfig;
    setSessionPipelineConfig(nextPipelineConfig);
  }, [engineStatusSnapshot.defaultPipelineConfig, sessionBaseProfile?.id]);

  useEffect(() => {
    if (sessionPhase === "ready" || finalChoice !== null) return;
    const nextBaseSnapshot = createSession({
      profiles: savedProfiles,
      pipelineConfig: sessionPipelineConfig,
      sessionBaseProfileId,
    });
    setSessionSnapshot(nextBaseSnapshot);
  }, [finalChoice, savedProfiles, sessionBaseProfileId, sessionPhase, sessionPipelineConfig]);

  useEffect(() => {
    if (screen === "session" && !hasSeenSessionIntroPrompt && !appOnboarding.active) {
      setShowSessionIntroDialog(true);
      setHasSeenSessionIntroPrompt(true);
    }
  }, [appOnboarding.active, screen, hasSeenSessionIntroPrompt]);

  useEffect(() => {
    if (!appOnboarding.active) return;
    if (!appOnboarding.requestedScreen) return;
    if (screen === appOnboarding.requestedScreen) return;
    setScreen(appOnboarding.requestedScreen);
  }, [appOnboarding.active, appOnboarding.requestedScreen, screen]);

  useEffect(() => {
    if (screen !== "session" || sessionPhase !== "starting") return;
    setStartupStep(1);
    const timers = [
      window.setTimeout(() => setStartupStep(2), 700),
      window.setTimeout(() => setStartupStep(3), 1500),
      window.setTimeout(() => {
        const previewStatus = previewSessionTarget({
          sessionId: sessionSnapshot.sessionId,
          target: "base",
        });
        setSessionPreviewStatus(previewStatus);
        setSessionPhase("ready");
        if (previewStatus.applyStatus !== "failed") {
          setListeningTarget(previewStatus.target ?? "base");
          return;
        }
        setListeningTarget("base");
      }, 2300),
    ];
    return () => timers.forEach((timer) => window.clearTimeout(timer));
  }, [screen, sessionPhase, sessionSnapshot.sessionId]);

  useEffect(() => {
    if (screen !== "session" || sessionPhase !== "ready" || finalChoice !== null) {
      setShowFeedbackJump(false);
      return;
    }
    const target = feedbackSectionRef.current;
    if (!target) return;
    const observer = new IntersectionObserver(([entry]) => setShowFeedbackJump(!entry.isIntersecting), { threshold: 0.35 });
    observer.observe(target);
    return () => observer.disconnect();
  }, [screen, sessionPhase, finalChoice]);

  useEffect(() => {
    if (!saveNotice || screen !== "home") return;
    const timer = window.setTimeout(() => setSaveNotice(null), 5000);
    return () => window.clearTimeout(timer);
  }, [saveNotice, screen]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const updateWavePreset = () => {
      const height = window.innerHeight;
      const ratio = window.innerWidth / Math.max(window.innerHeight, 1);
      if (height < 760 || ratio < 1.45) return setHomeWavePreset("compact");
      if (height > 980 && ratio > 1.75) return setHomeWavePreset("expanded");
      setHomeWavePreset("default");
    };
    updateWavePreset();
    window.addEventListener("resize", updateWavePreset);
    return () => window.removeEventListener("resize", updateWavePreset);
  }, []);

  useEffect(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") return;
    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    const applyPreference = (matches: boolean) => setPrefersDarkTheme(matches);
    applyPreference(mediaQuery.matches);
    const handleChange = (event: MediaQueryListEvent) => applyPreference(event.matches);
    if (typeof mediaQuery.addEventListener === "function") {
      mediaQuery.addEventListener("change", handleChange);
      return () => mediaQuery.removeEventListener("change", handleChange);
    }
    mediaQuery.addListener(handleChange);
    return () => mediaQuery.removeListener(handleChange);
  }, []);

  const isLightTheme = themeMode === "light" ? true : themeMode === "dark" ? false : !prefersDarkTheme;
  const shellStyle: CSSProperties = { filter: isLightTheme ? "invert(1) hue-rotate(180deg)" : "none", transition: "filter 260ms ease, background-color 260ms ease" };
  const resultProfileLabel = finalChoice === "A" ? "Вариант A" : finalChoice === "B" ? "Вариант B" : sessionBaseLabel;
  const resultProfileParams = finalChoice === "A" ? pairA.params : finalChoice === "B" ? pairB.params : sessionBaseParams;
  const shouldLockAudioSettings = appOnboarding.shouldLockAudioSettings;

  return (
    <AppOnboardingProvider value={appOnboarding}>
      <div className={`h-screen overflow-hidden text-white ${isLightTheme ? "bg-[#05060a]" : "bg-[#05060a]"}`}>
        <div className="flex h-full w-full flex-col overflow-hidden border border-white/10 bg-black shadow-2xl shadow-black/35 md:rounded-[12px]" style={shellStyle}>
          <div className={`flex h-9 shrink-0 items-center justify-between border-b border-white/6 bg-black/82 px-3 transition-opacity duration-200 ${shouldLockAudioSettings ? "pointer-events-none opacity-25" : ""}`}>
            <div className="text-[11px] font-medium tracking-[0.16em] text-white/62">EarLoop</div>
            <div className="flex items-center">
              {[
                { label: "Свернуть окно", icon: Minus, hoverClassName: "hover:bg-white/10", onClick: undefined },
                {
                  label: isShellMaximized ? "Восстановить окно" : "Развернуть окно",
                  icon: isShellMaximized ? Copy : Square,
                  hoverClassName: "hover:bg-white/10",
                  onClick: () => setIsShellMaximized((value) => !value),
                },
                { label: "Закрыть окно", icon: X, hoverClassName: "hover:bg-red-500/80", onClick: undefined },
              ].map((item) => {
                const Icon = item.icon;
                return (
                  <button
                    key={item.label}
                    type="button"
                    aria-label={item.label}
                    className={`flex h-7 w-8 items-center justify-center rounded-md text-white/58 transition ${item.hoverClassName} hover:text-white`}
                    onClick={item.onClick}
                  >
                    <Icon className={`${item.label === "Свернуть окно" || item.label === "Закрыть окно" ? "h-3.5 w-3.5" : "h-3 w-3"}`} />
                  </button>
                );
              })}
            </div>
          </div>

          <div className="flex min-h-0 flex-1">
            <div className={`transition-opacity duration-200 ${shouldLockAudioSettings ? "pointer-events-none opacity-25" : ""}`}>
              <Sidebar screen={screen} sidebarExpanded={sidebarExpanded} onToggleSidebar={() => setSidebarExpanded((value) => !value)} onSelectScreen={setScreen} />
            </div>

            <div className="relative flex min-w-0 flex-1 flex-col bg-[#05060a]">
              <div className={`relative flex-1 ${screen === "session" ? "px-1.5 py-1.5 md:px-2 md:py-2" : "px-2 py-2 md:px-3 md:py-3"} ${screen === "home" ? "overflow-hidden" : "overflow-auto"}`}>
                {screen === "home" && (
                  <HomeScreen
                    isEnabled={mainStateSnapshot.processingEnabled}
                    isPlaying={isPlaying}
                    homeWavePreset={homeWavePreset}
                    saveNotice={saveNotice}
                    activeProfileName={mainStateSnapshot.activeProfileName}
                    selectedProfile={selectedProfile}
                    savedProfiles={savedProfiles}
                    showIntroHint={showIntroHint}
                    processingStatus={mainStateSnapshot.runtimeProfileStatus?.applyStatus ?? mainStateSnapshot.audioStatus?.status ?? null}
                    processingError={mainStateSnapshot.runtimeProfileStatus?.lastError ?? mainStateSnapshot.audioStatus?.lastError ?? null}
                    onSelectProfile={handleActiveProfileChange}
                    onToggleHome={handleHomeToggle}
                  />
                )}

                {screen === "settings" && (
                  <SettingsScreen
                    devices={devices}
                    showAdvanced={showAdvanced}
                    blockSize={blockSize}
                    gainCompensation={gainCompensation}
                    themeMode={themeMode}
                    audioDevicesCatalog={audioDevicesCatalog}
                    engineAudioStatus={engineStatusSnapshot.audioStatus}
                    runtimeProfileStatus={engineStatusSnapshot.runtimeProfileStatus}
                    pipelineCatalog={engineStatusSnapshot.pipelineCatalog}
                    savedProfiles={savedProfiles}
                    selectedProfile={selectedProfile}
                    managedProfile={managedProfile}
                    profileEditorName={profileEditorName}
                    profileEditorParams={profileEditorParams}
                    onDeviceChange={handleDeviceChange}
                    onShowAdvancedChange={setShowAdvanced}
                    onBlockSizeChange={setBlockSize}
                    onGainCompensationChange={setGainCompensation}
                    onThemeModeChange={setThemeMode}
                    onSelectProfile={handleActiveProfileChange}
                    onProfileEditorNameChange={setProfileEditorName}
                  onProfileParamChange={updateManagedProfileParam}
                  onDeleteProfile={handleDeleteManagedProfile}
                  onResetProfile={handleResetManagedProfile}
                  onSaveProfile={handleSaveManagedProfile}
                  onRestartAppOnboarding={handleRestartAppOnboarding}
                />
              )}

                {screen === "session" && (
                  <div className="flex min-h-full flex-1 flex-col gap-4">
                    {finalChoice !== null ? (
                      <SessionResultScreen
                        finalChoice={finalChoice}
                        profileDraftName={profileDraftName}
                        saveNotice={saveNotice}
                        resultProfileLabel={resultProfileLabel}
                        resultProfileParams={resultProfileParams}
                        pairVersion={pairVersion}
                        iteration={iteration}
                        progress={progress}
                        onProfileDraftNameChange={(value) => {
                          setProfileDraftName(value);
                          if (saveNotice) setSaveNotice(null);
                        }}
                        onSaveProfile={handleSaveProfile}
                        onReturnToSession={() => {
                          setFinalChoice(null);
                          setProfileDraftName(engineStatusSnapshot.defaultProfileName);
                          setSaveNotice(null);
                        }}
                      />
                    ) : sessionPhase !== "ready" ? (
                      <SessionStartScreen
                        pipelineCatalog={engineStatusSnapshot.pipelineCatalog}
                        sessionPipelineConfig={sessionPipelineConfig}
                        sessionTutorialMode={sessionTutorialMode}
                        sessionPhase={sessionPhase}
                        sessionBaseProfileId={sessionBaseProfileId}
                        savedProfiles={savedProfiles}
                        showAdvancedConfig={showSessionPipelineConfig}
                        startupStep={startupStep}
                        startupSteps={startupSteps}
                        onPipelineConfigChange={(key, value) => setSessionPipelineConfig((current) => ({ ...current, [key]: value }))}
                        onSelectSessionBaseProfileId={setSessionBaseProfileId}
                        onStart={() => {
                          const nextSession = startSession({
                            profiles: savedProfiles,
                            pipelineConfig: sessionPipelineConfig,
                            sessionBaseProfileId,
                            iteration,
                            progress,
                            pairVersion,
                          });
                          setSaveNotice(null);
                          setStartupStep(0);
                          setSessionSnapshot(nextSession);
                          setSessionPreviewStatus(null);
                          setSessionPipelineConfig(nextSession.sessionPipelineConfig);
                          setSessionPhase("starting");
                        }}
                        onToggleAdvancedConfig={() => setShowSessionPipelineConfig((current) => !current)}
                      />
                    ) : (
                      <SessionCompareScreen
                        sessionTutorialMode={sessionTutorialMode}
                        iteration={iteration}
                        sessionBaseLabel={sessionBaseLabel}
                        listeningTarget={listeningTarget}
                        isGeneratingPair={isGeneratingPair}
                        pairVersion={pairVersion}
                        pairA={pairA}
                        pairB={pairB}
                        sessionBaseParams={sessionBaseParams}
                        paramsModalFor={paramsModalFor}
                        modalParams={modalParams}
                        feedback={feedback}
                        previewApplyStatus={sessionPreviewStatus?.applyStatus ?? null}
                        previewError={sessionPreviewStatus?.lastError ?? null}
                        feedbackAnchorRef={feedbackAnchorRef}
                        feedbackSectionRef={feedbackSectionRef}
                        onSelectListeningTarget={handlePreviewTarget}
                        onCardDoubleClick={handleCardDoubleClick}
                        onOpenParamsModal={setParamsModalFor}
                        onCloseParamsModal={() => setParamsModalFor(null)}
                        onFinish={handleFinish}
                        onReject={handleReject}
                        onFeedbackChange={setFeedback}
                      />
                    )}

                    {sessionPhase === "ready" && finalChoice === null && showFeedbackJump && (
                      <div className="pointer-events-none sticky bottom-10 z-20 -mt-14 h-0">
                        <div className="flex justify-end pr-1 md:pr-2">
                          <button
                            type="button"
                            aria-label="Перейти к дополнительной обратной связи"
                            className="pointer-events-auto flex h-10 w-10 items-center justify-center rounded-full border border-white/10 bg-black/68 text-white/78 shadow-md backdrop-blur-md transition hover:bg-white/10 hover:text-white"
                            onClick={() => feedbackAnchorRef.current?.scrollIntoView({ behavior: "smooth", block: "start" })}
                          >
                            <ChevronDown className="h-4.5 w-4.5" />
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>

              <SessionIntroDialog open={showSessionIntroDialog} onOpenChange={setShowSessionIntroDialog} onSkip={handleSkipSessionTutorial} onStartTutorial={handleStartSessionTutorial} />
            </div>
          </div>
        </div>

        <OnboardingOverlay />
      </div>
    </AppOnboardingProvider>
  );
}
