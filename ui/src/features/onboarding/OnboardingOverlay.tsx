import { AnimatePresence, motion } from "framer-motion";
import { useLayoutEffect, useMemo, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { appOnboardingSteps } from "@/features/onboarding/app-onboarding";
import type { OnboardingTargetPlacement, OnboardingTargetRect } from "@/features/onboarding/types";
import { useAppOnboardingContext } from "@/features/onboarding/useAppOnboarding";

const SPOTLIGHT_PADDING = 14;
const SAFE_MARGIN = 20;
const CARD_MAX_WIDTH = 380;
const GAP = 24;

type CardSize = {
  width: number;
  height: number;
};

type PositionedPlacement = {
  placement: OnboardingTargetPlacement;
  top: number;
  left: number;
  overlapArea: number;
  fits: boolean;
};

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

function getSpotlightRect(targetRect: OnboardingTargetRect | null) {
  if (!targetRect) return null;
  return {
    top: Math.max(targetRect.top - SPOTLIGHT_PADDING, SAFE_MARGIN - 6),
    left: Math.max(targetRect.left - SPOTLIGHT_PADDING, SAFE_MARGIN - 6),
    width: targetRect.width + SPOTLIGHT_PADDING * 2,
    height: targetRect.height + SPOTLIGHT_PADDING * 2,
  };
}

function getPlacementOrder(preferred: OnboardingTargetPlacement) {
  switch (preferred) {
    case "left":
      return ["left", "right", "bottom", "top", "center"] as const;
    case "right":
      return ["right", "left", "bottom", "top", "center"] as const;
    case "top":
      return ["top", "bottom", "right", "left", "center"] as const;
    case "bottom":
      return ["bottom", "top", "right", "left", "center"] as const;
    case "center":
      return ["center", "bottom", "top", "right", "left"] as const;
    default:
      return ["right", "bottom", "top", "left", "center"] as const;
  }
}

function getOverlapArea(
  cardLeft: number,
  cardTop: number,
  cardWidth: number,
  cardHeight: number,
  targetRect: OnboardingTargetRect,
) {
  const overlapWidth = Math.max(0, Math.min(cardLeft + cardWidth, targetRect.left + targetRect.width) - Math.max(cardLeft, targetRect.left));
  const overlapHeight = Math.max(0, Math.min(cardTop + cardHeight, targetRect.top + targetRect.height) - Math.max(cardTop, targetRect.top));
  return overlapWidth * overlapHeight;
}

function resolveCandidatePosition(
  placement: OnboardingTargetPlacement,
  targetRect: OnboardingTargetRect | null,
  cardSize: CardSize,
  viewportWidth: number,
  viewportHeight: number,
) {
  if (!targetRect || placement === "center") {
    return {
      top: clamp(viewportHeight / 2 - cardSize.height / 2, SAFE_MARGIN, viewportHeight - cardSize.height - SAFE_MARGIN),
      left: clamp(viewportWidth / 2 - cardSize.width / 2, SAFE_MARGIN, viewportWidth - cardSize.width - SAFE_MARGIN),
    };
  }

  const centeredLeft = targetRect.left + targetRect.width / 2 - cardSize.width / 2;
  const centeredTop = targetRect.top + targetRect.height / 2 - cardSize.height / 2;
  const topAligned = targetRect.top + Math.min(18, Math.max(12, targetRect.height * 0.08));
  const shouldTopAlignSideCard = targetRect.height > cardSize.height * 1.25;

  if (placement === "left") {
    return {
      top: shouldTopAlignSideCard ? topAligned : centeredTop,
      left: targetRect.left - cardSize.width - GAP,
    };
  }

  if (placement === "right") {
    return {
      top: shouldTopAlignSideCard ? topAligned : centeredTop,
      left: targetRect.left + targetRect.width + GAP,
    };
  }

  if (placement === "top") {
    return {
      top: targetRect.top - cardSize.height - GAP,
      left: centeredLeft,
    };
  }

  return {
    top: targetRect.top + targetRect.height + GAP,
    left: centeredLeft,
  };
}

function resolveCornerFallback(
  targetRect: OnboardingTargetRect | null,
  cardSize: CardSize,
  viewportWidth: number,
  viewportHeight: number,
) {
  const candidates = [
    { top: SAFE_MARGIN, left: SAFE_MARGIN },
    { top: SAFE_MARGIN, left: viewportWidth - cardSize.width - SAFE_MARGIN },
    { top: viewportHeight - cardSize.height - SAFE_MARGIN, left: SAFE_MARGIN },
    { top: viewportHeight - cardSize.height - SAFE_MARGIN, left: viewportWidth - cardSize.width - SAFE_MARGIN },
  ].map((candidate) => ({
    placement: "center" as const,
    top: clamp(candidate.top, SAFE_MARGIN, viewportHeight - cardSize.height - SAFE_MARGIN),
    left: clamp(candidate.left, SAFE_MARGIN, viewportWidth - cardSize.width - SAFE_MARGIN),
    overlapArea: targetRect ? getOverlapArea(candidate.left, candidate.top, cardSize.width, cardSize.height, targetRect) : 0,
    fits: true,
  }));

  return candidates.sort((a, b) => a.overlapArea - b.overlapArea)[0];
}

function resolveCardPosition(
  targetRect: OnboardingTargetRect | null,
  preferredPlacement: OnboardingTargetPlacement,
  viewportWidth: number,
  viewportHeight: number,
  cardSize: CardSize,
) {
  const placements = getPlacementOrder(preferredPlacement);
  const candidates = placements.map((placement) => {
    const raw = resolveCandidatePosition(placement, targetRect, cardSize, viewportWidth, viewportHeight);
    const top = clamp(raw.top, SAFE_MARGIN, viewportHeight - cardSize.height - SAFE_MARGIN);
    const left = clamp(raw.left, SAFE_MARGIN, viewportWidth - cardSize.width - SAFE_MARGIN);
    const overlapArea = targetRect ? getOverlapArea(left, top, cardSize.width, cardSize.height, targetRect) : 0;
    const fits = raw.top >= SAFE_MARGIN
      && raw.left >= SAFE_MARGIN
      && raw.top + cardSize.height <= viewportHeight - SAFE_MARGIN
      && raw.left + cardSize.width <= viewportWidth - SAFE_MARGIN
      && overlapArea === 0;

    return { placement, top, left, overlapArea, fits };
  });

  const fullyFitting = candidates.find((candidate) => candidate.fits);
  if (fullyFitting) return fullyFitting;

  const nonOverlapping = candidates
    .filter((candidate) => candidate.overlapArea === 0)
    .sort((a, b) => {
      const aDistance = Math.abs(a.top - SAFE_MARGIN) + Math.abs(a.left - SAFE_MARGIN);
      const bDistance = Math.abs(b.top - SAFE_MARGIN) + Math.abs(b.left - SAFE_MARGIN);
      return aDistance - bDistance;
    })[0];
  if (nonOverlapping) return nonOverlapping;

  const cornerFallback = resolveCornerFallback(targetRect, cardSize, viewportWidth, viewportHeight);
  const bestRegular = candidates.sort((a, b) => a.overlapArea - b.overlapArea)[0];
  return cornerFallback.overlapArea < bestRegular.overlapArea ? cornerFallback : bestRegular;
}

function resolveStepSpecificPosition(
  stepId: string | undefined,
  viewportWidth: number,
  viewportHeight: number,
  cardSize: CardSize,
  targetRect: OnboardingTargetRect | null,
) {
  if (stepId === "audio-devices") {
    return {
      placement: "right" as const,
      top: SAFE_MARGIN + 6,
      left: clamp(viewportWidth - cardSize.width - SAFE_MARGIN, SAFE_MARGIN, viewportWidth - cardSize.width - SAFE_MARGIN),
      overlapArea: 0,
      fits: true,
    };
  }

  if (stepId === "session-base-profile") {
    return {
      placement: "right" as const,
      top: clamp(viewportHeight / 2 - cardSize.height / 2 - 36, SAFE_MARGIN + 8, viewportHeight - cardSize.height - SAFE_MARGIN),
      left: clamp(viewportWidth - cardSize.width - 28, SAFE_MARGIN, viewportWidth - cardSize.width - SAFE_MARGIN),
      overlapArea: 0,
      fits: true,
    };
  }

  if (stepId === "session-start" && targetRect) {
    return {
      placement: "top" as const,
      top: clamp(targetRect.top - cardSize.height - 84, SAFE_MARGIN, viewportHeight - cardSize.height - SAFE_MARGIN),
      left: clamp(targetRect.left + targetRect.width / 2 - cardSize.width / 2, SAFE_MARGIN, viewportWidth - cardSize.width - SAFE_MARGIN),
      overlapArea: 0,
      fits: true,
    };
  }

  if (stepId === "compare-previous-choice" && targetRect) {
    return {
      placement: "top" as const,
      top: clamp(targetRect.top - cardSize.height - 72, SAFE_MARGIN, viewportHeight - cardSize.height - SAFE_MARGIN),
      left: clamp(targetRect.left + targetRect.width / 2 - cardSize.width / 2, SAFE_MARGIN, viewportWidth - cardSize.width - SAFE_MARGIN),
      overlapArea: 0,
      fits: true,
    };
  }

  if (stepId === "compare-finish" && targetRect) {
    return {
      placement: "left" as const,
      top: clamp(targetRect.top + targetRect.height / 2 - cardSize.height / 2, SAFE_MARGIN, viewportHeight - cardSize.height - SAFE_MARGIN),
      left: clamp(targetRect.left - cardSize.width - 28, SAFE_MARGIN, viewportWidth - cardSize.width - SAFE_MARGIN),
      overlapArea: 0,
      fits: true,
    };
  }

  return null;
}

export function OnboardingOverlay() {
  const {
    active,
    currentIndex,
    currentStep,
    currentTargetRect,
    currentStepCompleted,
    next,
    back,
    skip,
  } = useAppOnboardingContext();
  const cardRef = useRef<HTMLDivElement | null>(null);
  const [cardSize, setCardSize] = useState<CardSize>({ width: CARD_MAX_WIDTH, height: 248 });

  useLayoutEffect(() => {
    const node = cardRef.current;
    if (!node) return;

    const updateSize = () => {
      const rect = node.getBoundingClientRect();
      setCardSize({
        width: Math.min(rect.width || CARD_MAX_WIDTH, window.innerWidth - SAFE_MARGIN * 2),
        height: rect.height || 248,
      });
    };

    updateSize();
    const observer = new ResizeObserver(updateSize);
    observer.observe(node);
    window.addEventListener("resize", updateSize);

    return () => {
      observer.disconnect();
      window.removeEventListener("resize", updateSize);
    };
  }, [currentStep?.id]);

  const spotlightRect = getSpotlightRect(currentTargetRect);
  const isLastStep = currentIndex === appOnboardingSteps.length - 1;
  const cardPosition = useMemo(() => {
    if (typeof window === "undefined") {
      return { placement: "center" as const, top: SAFE_MARGIN, left: SAFE_MARGIN, overlapArea: 0, fits: true };
    }
    const stepSpecificPosition = resolveStepSpecificPosition(
      currentStep?.id,
      window.innerWidth,
      window.innerHeight,
      cardSize,
      currentTargetRect,
    );
    if (stepSpecificPosition) {
      return stepSpecificPosition;
    }
    return resolveCardPosition(
      currentTargetRect,
      currentStep?.placement ?? "auto",
      window.innerWidth,
      window.innerHeight,
      cardSize,
    );
  }, [cardSize, currentStep?.placement, currentTargetRect]);

  if (typeof window === "undefined") return null;

  return (
    <AnimatePresence mode="wait">
      {active && currentStep && (
        <motion.div
          key={currentStep.id}
          className={`fixed inset-0 z-[120] ${currentStep.presentation === "intro" ? "" : "pointer-events-none"}`}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.28, ease: "easeOut" }}
        >
          {currentStep.presentation === "intro" ? (
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,rgba(58,118,150,0.24),transparent_42%),rgba(3,6,11,0.82)] backdrop-blur-[7px]" />
          ) : spotlightRect ? (
            <>
              <div className="pointer-events-none absolute left-0 right-0 top-0 bg-black/56 backdrop-blur-[3px]" style={{ height: spotlightRect.top }} />
              <div
                className="pointer-events-none absolute bg-black/56 backdrop-blur-[3px]"
                style={{ top: spotlightRect.top, left: 0, width: spotlightRect.left, height: spotlightRect.height }}
              />
              <div
                className="pointer-events-none absolute bg-black/56 backdrop-blur-[3px]"
                style={{
                  top: spotlightRect.top,
                  left: spotlightRect.left + spotlightRect.width,
                  right: 0,
                  height: spotlightRect.height,
                }}
              />
              <div
                className="pointer-events-none absolute bottom-0 left-0 right-0 bg-black/56 backdrop-blur-[3px]"
                style={{ top: spotlightRect.top + spotlightRect.height }}
              />
              <motion.div
                className="pointer-events-none absolute rounded-[28px] border border-cyan-300/70 bg-cyan-300/6 shadow-[0_0_0_1px_rgba(103,232,249,0.24),0_0_42px_rgba(34,211,238,0.12)]"
                style={spotlightRect}
                initial={{ opacity: 0.68, scale: 0.985 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.24, ease: "easeOut" }}
              />
            </>
          ) : (
            <div className="pointer-events-none absolute inset-0 bg-black/56 backdrop-blur-[3px]" />
          )}

          <motion.div
            ref={cardRef}
            className={`absolute text-white shadow-[0_20px_70px_rgba(0,0,0,0.42)] backdrop-blur-xl ${
              currentStep.presentation === "intro"
                ? "left-1/2 top-1/2 w-[min(460px,calc(100vw-32px))] -translate-x-1/2 -translate-y-1/2 rounded-[32px] border border-white/12 bg-[#0e131c]/96 p-7"
                : currentStep.id === "audio-devices"
                  ? "pointer-events-auto w-[min(420px,calc(100vw-40px))] rounded-[28px] border border-white/12 bg-[#10131d]/96 p-5"
                  : "pointer-events-auto w-[min(380px,calc(100vw-32px))] rounded-[28px] border border-white/12 bg-[#10131d]/96 p-5"
            }`}
            style={currentStep.presentation === "intro" ? undefined : { top: cardPosition.top, left: cardPosition.left }}
            initial={{ opacity: 0, y: 18, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 10, scale: 0.985 }}
            transition={{ duration: 0.28, ease: "easeOut" }}
          >
            {currentStep.presentation === "intro" ? (
              <>
                <div className="text-[11px] font-medium tracking-[0.22em] text-white/55">EARLOOP</div>
                <div className="mt-5 text-3xl font-semibold leading-tight text-white">{currentStep.title}</div>
                <div className="mt-4 text-sm leading-7 text-white/72">{currentStep.description}</div>
                <div className="mt-7">
                  <Button type="button" className="w-full rounded-2xl bg-cyan-300 text-slate-950 hover:bg-cyan-200" onClick={next}>
                    {currentStep.nextLabel ?? "Начать"}
                  </Button>
                </div>
                <div className="mt-3 flex justify-center">
                  <button type="button" className="text-sm text-white/56 transition hover:text-white" onClick={skip}>
                    Пропустить
                  </button>
                </div>
              </>
            ) : (
              <>
                <div className="flex items-center justify-between gap-3">
                  <div className="rounded-full border border-cyan-300/30 bg-cyan-300/10 px-3 py-1 text-xs font-medium tracking-[0.18em] text-cyan-100">
                    ШАГ {currentIndex + 1}/{appOnboardingSteps.length}
                  </div>
                  <button type="button" className="text-sm text-white/55 transition hover:text-white" onClick={skip}>
                    Пропустить
                  </button>
                </div>

                <div className="mt-4 space-y-3">
                  <div className="text-xl font-semibold leading-tight text-white">{currentStep.title}</div>
                  <div className="text-sm leading-6 text-white/74">{currentStep.description}</div>
                </div>

                <div className="mt-5 space-y-3">
                  <div className="text-xs leading-5 text-white/45">
                    {currentStep.id === "audio-devices"
                      ? "Убедись, что звук после обработки приходит на указанное устройство вывода."
                      : currentStep.type === "action"
                        ? "Сделай действие в интерфейсе и затем переходи дальше."
                        : "Шаг можно прочитать и продолжить вручную."}
                  </div>
                  <div className="flex items-center justify-end gap-2">
                    {currentStep.canGoBack !== false && currentIndex > 0 && (
                      <Button
                        type="button"
                        variant="outline"
                        className="rounded-2xl border-white/12 bg-white/5 text-white hover:bg-white/10"
                        onClick={back}
                      >
                        Назад
                      </Button>
                    )}
                    <Button
                      type="button"
                      className="rounded-2xl bg-cyan-300 text-slate-950 hover:bg-cyan-200 disabled:bg-white/10 disabled:text-white/40"
                      onClick={next}
                      disabled={!currentStepCompleted}
                    >
                      {currentStep.nextLabel ?? (isLastStep ? "Завершить" : "Дальше")}
                    </Button>
                  </div>
                </div>
              </>
            )}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
