import { createContext, useCallback, useContext, useEffect, useLayoutEffect, useRef, useState } from "react";
import type { ReactNode } from "react";

import { appOnboardingSteps } from "@/features/onboarding/app-onboarding";
import type { OnboardingEvent, OnboardingStep, OnboardingTargetId, OnboardingTargetRect } from "@/features/onboarding/types";

type AppOnboardingState = {
  active: boolean;
  currentIndex: number;
  currentStep: OnboardingStep | null;
  requestedScreen: OnboardingStep["screen"] | null;
  currentTargetRect: OnboardingTargetRect | null;
  currentStepCompleted: boolean;
  shouldLockAudioSettings: boolean;
  activeTargetId: OnboardingTargetId | null;
  registerTarget: (targetId: OnboardingTargetId, node: HTMLElement | null) => void;
  setActiveTargetId: (targetId: OnboardingTargetId | null) => void;
  emitEvent: (event: OnboardingEvent) => void;
  next: () => void;
  back: () => void;
  skip: () => void;
  restart: () => void;
};

const AppOnboardingContext = createContext<AppOnboardingState | null>(null);

type UseAppOnboardingOptions = {
  enabled: boolean;
  onComplete: () => void;
};

function toRect(node: HTMLElement): OnboardingTargetRect {
  const rect = node.getBoundingClientRect();
  return {
    top: rect.top,
    left: rect.left,
    width: rect.width,
    height: rect.height,
  };
}

export function useAppOnboarding({ enabled, onComplete }: UseAppOnboardingOptions): AppOnboardingState {
  const [active, setActive] = useState(enabled);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [currentTargetRect, setCurrentTargetRect] = useState<OnboardingTargetRect | null>(null);
  const [currentStepCompleted, setCurrentStepCompleted] = useState(false);
  const [activeTargetId, setActiveTargetId] = useState<OnboardingTargetId | null>(null);
  const [targetRegistryVersion, setTargetRegistryVersion] = useState(0);
  const targetNodesRef = useRef<Partial<Record<OnboardingTargetId, HTMLElement | null>>>({});
  const audioSelectionStateRef = useRef({ inputReady: false, outputReady: false });

  useEffect(() => {
    setActive(enabled);
    if (enabled) {
      setCurrentIndex(0);
      setCurrentStepCompleted(false);
      setActiveTargetId(null);
      audioSelectionStateRef.current = { inputReady: false, outputReady: false };
    }
  }, [enabled]);

  const currentStep = active ? appOnboardingSteps[currentIndex] ?? null : null;
  const requestedScreen = currentStepCompleted ? (currentStep?.screenAfterCompleted ?? currentStep?.screen ?? null) : (currentStep?.screen ?? null);
  const shouldLockAudioSettings = active && currentStep?.id === "audio-devices";

  const finish = () => {
    setActive(false);
    onComplete();
  };

  const next = useCallback(() => {
    if (currentStep?.type === "action" && !currentStepCompleted) return;
    setCurrentIndex((index) => {
      const nextIndex = index + 1;
      if (nextIndex >= appOnboardingSteps.length) {
        window.setTimeout(finish, 0);
        return index;
      }
      return nextIndex;
    });
  }, [currentStep?.type, currentStepCompleted, onComplete]);

  const back = () => {
    setCurrentIndex((index) => Math.max(0, index - 1));
  };

  const skip = () => finish();

  const restart = () => {
    audioSelectionStateRef.current = { inputReady: false, outputReady: false };
    setCurrentStepCompleted(false);
    setActiveTargetId(null);
    setCurrentIndex(0);
    setActive(true);
  };

  const registerTarget = useCallback((targetId: OnboardingTargetId, node: HTMLElement | null) => {
    if (targetNodesRef.current[targetId] === node) return;
    targetNodesRef.current[targetId] = node;
    setTargetRegistryVersion((value) => value + 1);
  }, []);

  const emitEvent = (event: OnboardingEvent) => {
    if (!active || !currentStep) return;

    if (currentStep.id === "audio-devices") {
      if (event === "audio.input.changed") audioSelectionStateRef.current.inputReady = true;
      if (event === "audio.output.changed") audioSelectionStateRef.current.outputReady = true;
      if (audioSelectionStateRef.current.inputReady && audioSelectionStateRef.current.outputReady) {
        setCurrentStepCompleted(true);
      }
      return;
    }

    if (currentStep.completeOn?.includes(event)) {
      setCurrentStepCompleted(true);
    }
  };

  useEffect(() => {
    if (!active || !currentStep) return;
    setCurrentStepCompleted(
      currentStep.id === "audio-devices"
        || (currentStep.type === "info" && !currentStep.completeOn?.length),
    );
    setActiveTargetId(currentStep.targetId ?? null);
  }, [active, currentStep?.id, currentStep?.type, currentStep?.completeOn]);

  useEffect(() => {
    if (!active || !currentStepCompleted || !currentStep?.autoAdvanceOnComplete) return;
    const timer = window.setTimeout(() => next(), 140);
    return () => window.clearTimeout(timer);
  }, [active, currentStep?.autoAdvanceOnComplete, currentStepCompleted, next]);

  useLayoutEffect(() => {
    const resolvedTargetId = currentStep?.id === "session-base-profile"
      ? (activeTargetId ?? currentStep?.targetId)
      : currentStep?.targetId;

    if (!resolvedTargetId) {
      setCurrentTargetRect(null);
      return;
    }

    const node = targetNodesRef.current[resolvedTargetId];
    if (!node) {
      setCurrentTargetRect(null);
      return;
    }

    const updateRect = () => setCurrentTargetRect(toRect(node));
    updateRect();

    const resizeObserver = new ResizeObserver(updateRect);
    resizeObserver.observe(node);
    window.addEventListener("resize", updateRect);
    window.addEventListener("scroll", updateRect, true);

    return () => {
      resizeObserver.disconnect();
      window.removeEventListener("resize", updateRect);
      window.removeEventListener("scroll", updateRect, true);
    };
  }, [activeTargetId, currentStep?.id, currentStep?.targetId, currentIndex, targetRegistryVersion]);

  useEffect(() => {
    if (!currentStep?.targetId || !currentStep.autoScroll) return;
    const node = targetNodesRef.current[currentStep.targetId];
    if (!node) return;
    node.scrollIntoView({
      behavior: "smooth",
      block: currentStep.autoScroll === "smooth-center"
        ? "center"
        : currentStep.autoScroll === "smooth-end"
          ? "end"
          : "start",
      inline: "nearest",
    });
  }, [currentStep?.autoScroll, currentStep?.id, currentStep?.targetId]);

  useEffect(() => {
    if (currentStep?.id !== "session-base-profile") return;

    const handlePointerDown = (event: PointerEvent) => {
      const node = currentStep.targetId ? targetNodesRef.current[currentStep.targetId] : null;
      if (!node) return;
      if (event.target instanceof Node && node.contains(event.target)) return;
      emitEvent("onboarding.step4.dismissed");
    };

    document.addEventListener("pointerdown", handlePointerDown, true);
    return () => document.removeEventListener("pointerdown", handlePointerDown, true);
  }, [currentStep?.id]);

  return {
    active,
    currentIndex,
    currentStep,
    requestedScreen,
    currentTargetRect,
    currentStepCompleted,
    shouldLockAudioSettings,
    activeTargetId,
    registerTarget,
    setActiveTargetId,
    emitEvent,
    next,
    back,
    skip,
    restart,
  };
}

export function AppOnboardingProvider({
  children,
  value,
}: {
  children: ReactNode;
  value: AppOnboardingState;
}) {
  return <AppOnboardingContext.Provider value={value}>{children}</AppOnboardingContext.Provider>;
}

export function useAppOnboardingContext() {
  const context = useContext(AppOnboardingContext);
  if (!context) {
    throw new Error("useAppOnboardingContext must be used within AppOnboardingProvider");
  }
  return context;
}
