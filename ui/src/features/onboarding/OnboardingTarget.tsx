import { cloneElement, isValidElement, useLayoutEffect, useRef } from "react";
import type { ReactElement, Ref } from "react";

import { useAppOnboardingContext } from "@/features/onboarding/useAppOnboarding";
import type { OnboardingTargetId } from "@/features/onboarding/types";

type OnboardingTargetProps = {
  targetId: OnboardingTargetId;
  children: ReactElement<{ ref?: Ref<HTMLElement> }>;
};

function assignRef<T>(ref: Ref<T> | undefined, value: T) {
  if (!ref) return;
  if (typeof ref === "function") {
    ref(value);
    return;
  }
  (ref as { current: T }).current = value;
}

export function OnboardingTarget({ targetId, children }: OnboardingTargetProps) {
  const { registerTarget } = useAppOnboardingContext();
  const nodeRef = useRef<HTMLElement | null>(null);

  if (!isValidElement(children)) {
    return children;
  }

  useLayoutEffect(() => {
    registerTarget(targetId, nodeRef.current);
    return () => registerTarget(targetId, null);
  }, [registerTarget, targetId]);

  return cloneElement(children, {
    ref: (node: HTMLElement | null) => {
      nodeRef.current = node;
      assignRef(children.props.ref, node);
    },
  });
}
