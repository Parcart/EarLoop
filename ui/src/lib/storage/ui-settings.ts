import type { Screen, ThemeMode } from "@/lib/types/ui";

export type UiSettings = {
  theme: ThemeMode;
  sidebarExpanded: boolean;
  hasSeenSessionIntro: boolean;
  hasSeenAppOnboarding: boolean;
  appOnboardingVersion: number | null;
  lastSelectedScreen: Screen;
  lastSelectedProfileId: string;
  lastSelectedInputDeviceId: string;
  lastSelectedOutputDeviceId: string;
};

const UI_SETTINGS_STORAGE_KEY = "earloop.ui.settings";

const defaultUiSettings: UiSettings = {
  theme: "system",
  sidebarExpanded: false,
  hasSeenSessionIntro: false,
  hasSeenAppOnboarding: false,
  appOnboardingVersion: null,
  lastSelectedScreen: "home",
  lastSelectedProfileId: "",
  lastSelectedInputDeviceId: "CABLE Output (VB-Audio Virtual Cable)",
  lastSelectedOutputDeviceId: "Динамики (Razer Barracuda X 2.4)",
};

function sanitizeUiSettings(value: unknown): UiSettings {
  if (!value || typeof value !== "object") {
    return { ...defaultUiSettings };
  }

  const candidate = value as Partial<UiSettings>;
  return {
    theme: candidate.theme === "light" || candidate.theme === "dark" || candidate.theme === "system"
      ? candidate.theme
      : defaultUiSettings.theme,
    sidebarExpanded: typeof candidate.sidebarExpanded === "boolean" ? candidate.sidebarExpanded : defaultUiSettings.sidebarExpanded,
    hasSeenSessionIntro: typeof candidate.hasSeenSessionIntro === "boolean" ? candidate.hasSeenSessionIntro : defaultUiSettings.hasSeenSessionIntro,
    hasSeenAppOnboarding: typeof candidate.hasSeenAppOnboarding === "boolean" ? candidate.hasSeenAppOnboarding : defaultUiSettings.hasSeenAppOnboarding,
    appOnboardingVersion: typeof candidate.appOnboardingVersion === "number" ? candidate.appOnboardingVersion : defaultUiSettings.appOnboardingVersion,
    lastSelectedScreen: candidate.lastSelectedScreen === "home" || candidate.lastSelectedScreen === "settings" || candidate.lastSelectedScreen === "session"
      ? candidate.lastSelectedScreen
      : defaultUiSettings.lastSelectedScreen,
    lastSelectedProfileId: typeof candidate.lastSelectedProfileId === "string" ? candidate.lastSelectedProfileId : defaultUiSettings.lastSelectedProfileId,
    lastSelectedInputDeviceId: typeof candidate.lastSelectedInputDeviceId === "string" ? candidate.lastSelectedInputDeviceId : defaultUiSettings.lastSelectedInputDeviceId,
    lastSelectedOutputDeviceId: typeof candidate.lastSelectedOutputDeviceId === "string" ? candidate.lastSelectedOutputDeviceId : defaultUiSettings.lastSelectedOutputDeviceId,
  };
}

export function loadUiSettings(): UiSettings {
  if (typeof window === "undefined") {
    return { ...defaultUiSettings };
  }

  try {
    const raw = window.localStorage.getItem(UI_SETTINGS_STORAGE_KEY);
    if (!raw) return { ...defaultUiSettings };
    return sanitizeUiSettings(JSON.parse(raw));
  } catch {
    return { ...defaultUiSettings };
  }
}

export function saveUiSettings(settings: UiSettings): UiSettings {
  const nextSettings = sanitizeUiSettings(settings);
  if (typeof window !== "undefined") {
    window.localStorage.setItem(UI_SETTINGS_STORAGE_KEY, JSON.stringify(nextSettings));
  }
  return nextSettings;
}

export function getDefaultUiSettings(): UiSettings {
  return { ...defaultUiSettings };
}

export function resetAppOnboardingInUiSettings(): UiSettings {
  const current = loadUiSettings();
  return saveUiSettings({
    ...current,
    hasSeenAppOnboarding: false,
    appOnboardingVersion: null,
  });
}
