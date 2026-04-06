import { ChevronLeft, ChevronRight, Home, Settings2, Wand2 } from "lucide-react";

import { OnboardingTarget } from "@/features/onboarding/OnboardingTarget";
import { useAppOnboardingContext } from "@/features/onboarding/useAppOnboarding";
import type { Screen } from "@/lib/types/ui";

type SidebarProps = {
  screen: Screen;
  sidebarExpanded: boolean;
  onToggleSidebar: () => void;
  onSelectScreen: (screen: Screen) => void;
};

const navItems = [
  { id: "home" as const, label: "Главная", icon: Home },
  { id: "session" as const, label: "Новый профиль", icon: Wand2 },
  { id: "settings" as const, label: "Настройки", icon: Settings2 },
];

export function Sidebar({ screen, sidebarExpanded, onToggleSidebar, onSelectScreen }: SidebarProps) {
  const { emitEvent } = useAppOnboardingContext();

  return (
    <aside className={`${sidebarExpanded ? "w-[220px]" : "w-[74px]"} flex shrink-0 flex-col border-r border-white/10 bg-black/95 transition-all duration-300`}>
      <div className="flex items-center justify-center border-b border-white/8 px-2 py-2">
        <button
          type="button"
          aria-label={sidebarExpanded ? "Свернуть боковую панель" : "Развернуть боковую панель"}
          className="flex h-8 w-8 items-center justify-center rounded-xl border border-transparent bg-transparent text-white/62 transition hover:border-white/10 hover:bg-white/6 hover:text-white"
          onClick={onToggleSidebar}
        >
          {sidebarExpanded ? <ChevronLeft className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
        </button>
      </div>

      <div className="flex-1 space-y-2 px-3 py-3">
        {navItems.map((item) => {
          const Icon = item.icon;
          const active = screen === item.id;

          const button = (
            <button
              key={item.id}
              type="button"
              className={`flex w-full items-center gap-3 rounded-2xl px-3 py-3 text-left transition ${active ? "bg-white/8 text-white" : "text-white/65 hover:bg-white/5 hover:text-white"} ${sidebarExpanded ? "justify-start" : "justify-center"}`}
              onClick={() => {
                onSelectScreen(item.id);
                if (item.id === "session") emitEvent("navigation.new-profile.clicked");
              }}
            >
              <Icon className="h-5 w-5 shrink-0" />
              {sidebarExpanded && <span className="font-medium">{item.label}</span>}
            </button>
          );

          if (item.id === "session") {
            return (
              <OnboardingTarget key={item.id} targetId="sidebar-new-profile">
                {button}
              </OnboardingTarget>
            );
          }

          return button;
        })}
      </div>
    </aside>
  );
}
