import type { ReactNode } from "react";

import type { Screen } from "@/lib/types/ui";

export type OnboardingStepType = "info" | "action";
export type OnboardingStepPresentation = "spotlight" | "intro";

export type OnboardingTargetPlacement = "auto" | "top" | "right" | "bottom" | "left" | "center";

export type OnboardingTargetId =
  | "sidebar-new-profile"
  | "session-base-profile"
  | "session-base-profile-dropdown"
  | "session-start-button"
  | "compare-cards"
  | "compare-toolbar"
  | "feedback-panel"
  | "finish-button"
  | "result-actions";

export type OnboardingEvent =
  | "navigation.new-profile.clicked"
  | "session.base-profile.interacted"
  | "session.start.clicked"
  | "session.compare.card.clicked"
  | "session.generate-next.clicked"
  | "session.finish.clicked"
  | "onboarding.step4.dismissed";

export type OnboardingStep = {
  id: string;
  screen: Screen;
  type: OnboardingStepType;
  presentation?: OnboardingStepPresentation;
  title: string;
  description: ReactNode;
  targetId?: OnboardingTargetId;
  placement?: OnboardingTargetPlacement;
  nextLabel?: string;
  canGoBack?: boolean;
  autoScroll?: "smooth-center" | "smooth-start" | "smooth-end";
  completeOn?: OnboardingEvent[];
  screenAfterCompleted?: Screen;
  autoAdvanceOnComplete?: boolean;
};

export type OnboardingTargetRect = {
  top: number;
  left: number;
  width: number;
  height: number;
};
