import { motion } from "framer-motion";
import { ChevronDown, Wand2 } from "lucide-react";

import { OnboardingTarget } from "@/features/onboarding/OnboardingTarget";
import { useAppOnboardingContext } from "@/features/onboarding/useAppOnboarding";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { SESSION_DEFAULT_PROFILE_ID } from "@/lib/mock/profiles";

import type { PipelineCatalog, PipelineConfig, SavedProfile, SessionPhase } from "@/lib/types/ui";

type SessionStartScreenProps = {
  pipelineCatalog: PipelineCatalog;
  sessionPipelineConfig: PipelineConfig;
  sessionTutorialMode: boolean;
  sessionPhase: SessionPhase;
  sessionBaseProfileId: string;
  savedProfiles: SavedProfile[];
  showAdvancedConfig: boolean;
  startupStep: number;
  startupSteps: readonly string[];
  onPipelineConfigChange: (key: keyof PipelineConfig, value: string) => void;
  onSelectSessionBaseProfileId: (value: string) => void;
  onStart: () => void;
  onToggleAdvancedConfig: () => void;
};

export function SessionStartScreen({
  pipelineCatalog,
  sessionPipelineConfig,
  sessionTutorialMode,
  sessionPhase,
  sessionBaseProfileId,
  savedProfiles,
  showAdvancedConfig,
  onPipelineConfigChange,
  onSelectSessionBaseProfileId,
  onStart,
  onToggleAdvancedConfig,
}: SessionStartScreenProps) {
  const { emitEvent, setActiveTargetId } = useAppOnboardingContext();
  const baseProfileSelectWidthClassName = "w-full md:w-[360px]";
  const baseProfileContentWidthClassName = "w-[360px] min-w-[360px] max-w-[360px]";
  const pipelineSelectWidthClassName = "w-full";
  const isStarting = sessionPhase === "starting";

  if (isStarting) {
    return (
      <div className="flex h-full min-h-full w-full flex-1 items-center justify-center">
        <div className="flex h-24 w-24 items-center justify-center rounded-full border border-white/10 bg-white/5">
          <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 1.5, ease: "linear" }}>
            <Wand2 className="h-10 w-10 text-cyan-300" />
          </motion.div>
        </div>
      </div>
    );
  }

  return (
    <Card className="flex h-full min-h-0 w-full flex-col rounded-[28px] border border-white/10 bg-[#0b0c11] shadow-none">
      <CardHeader className="pb-1 text-center md:pb-2">
        <CardTitle className="text-2xl text-white">Создание нового профиля</CardTitle>
        <CardDescription className="text-white/78">Сначала запускаем необходимые компоненты, затем генерируем первые варианты для сравнения.</CardDescription>
      </CardHeader>
      <CardContent className="flex min-h-0 flex-1 px-4 pt-0 pb-3 md:px-6 md:pb-5">
          <div className="mx-auto flex min-h-full w-full max-w-[min(100%,96rem)] flex-col items-center justify-center gap-4 py-4 text-center md:gap-5 md:py-6">
            {sessionTutorialMode && (
              <div className="w-full max-w-5xl rounded-[24px] border border-cyan-300/20 bg-cyan-400/10 p-4 text-left">
                <div className="text-sm font-semibold text-cyan-200">Обучение</div>
                <div className="mt-2 text-sm text-white/78">На этом экране можно начать новую сессию с нуля или взять за основу уже существующий профиль. Если нужен стандартный сценарий, просто выбери базу и нажми «Начать». Блок «Расширенная конфигурация» нужен для более точной настройки пайплайна, поэтому заходить в него лучше только с пониманием того, что именно ты хочешь изменить.</div>
              </div>
            )}

            <div className="flex w-full justify-center pt-1">
              <div className="flex h-24 w-24 items-center justify-center rounded-full border border-white/10 bg-white/5">
                <motion.div animate={{ scale: [1, 1.04, 1] }} transition={{ repeat: Infinity, duration: 2.2, ease: "easeInOut" }}>
                  <Wand2 className="h-10 w-10 text-cyan-300" />
                </motion.div>
              </div>
            </div>

            <div className="flex w-full max-w-4xl flex-col items-center gap-2 text-center">
              <Label className="text-white/75">Базовый профиль для новой сессии</Label>
              <div className={baseProfileSelectWidthClassName}>
                <Select
                  value={sessionBaseProfileId}
                  onOpenChange={(open) => {
                    setActiveTargetId(open ? "session-base-profile-dropdown" : "session-base-profile");
                    emitEvent("session.base-profile.interacted");
                  }}
                  onValueChange={(value) => {
                    onSelectSessionBaseProfileId(value);
                    setActiveTargetId("session-base-profile");
                    emitEvent("session.base-profile.interacted");
                  }}
                >
                  <OnboardingTarget targetId="session-base-profile">
                    <SelectTrigger className={`rounded-2xl border-white/10 bg-white/5 text-white ${baseProfileSelectWidthClassName}`}>
                      <SelectValue className="truncate" />
                    </SelectTrigger>
                  </OnboardingTarget>
                  <SelectContent className={baseProfileContentWidthClassName}>
                    <OnboardingTarget targetId="session-base-profile-dropdown">
                      <div>
                        <SelectItem value={SESSION_DEFAULT_PROFILE_ID}>С нуля</SelectItem>
                        {savedProfiles.map((profile) => (
                          <SelectItem key={profile.id} value={profile.id}>
                            <span className="block truncate">{profile.name}</span>
                          </SelectItem>
                        ))}
                      </div>
                    </OnboardingTarget>
                  </SelectContent>
                </Select>
              </div>
              <p className="max-w-2xl text-sm text-white/55">Следующие варианты будут генерироваться на основе выбранного профиля или с нуля.</p>
            </div>

            <div className="flex w-full max-w-6xl flex-col items-center gap-3 pt-1">
              <Button variant="outline" className="w-full max-w-4xl rounded-2xl border-white/10 bg-white/5 text-white hover:bg-white/10" onClick={onToggleAdvancedConfig}>
                <ChevronDown className={`mr-2 h-4 w-4 transition-transform ${showAdvancedConfig ? "rotate-180" : ""}`} />
                Расширенная конфигурация
              </Button>

              {showAdvancedConfig && (
                <div className="w-full rounded-[24px] border border-white/10 bg-white/5 p-4 text-left xl:p-5">
                  <div className="mb-4 text-sm text-white/68">Выбери конфигурацию pipeline для новой сессии. Она будет сохранена вместе с итоговым профилем.</div>
                  <div className="grid grid-cols-1 gap-4 md:grid-cols-2 2xl:grid-cols-3">
                    <div className="w-full space-y-2">
                      <Label className="text-white/75">Модель</Label>
                      <Select value={sessionPipelineConfig.modelId} onValueChange={(value) => onPipelineConfigChange("modelId", value)}>
                        <SelectTrigger className={`rounded-2xl border-white/10 bg-black/20 text-white ${pipelineSelectWidthClassName}`}><SelectValue className="truncate" /></SelectTrigger>
                        <SelectContent className="min-w-[280px]">
                          {pipelineCatalog.models.map((option) => (
                            <SelectItem key={option.id} value={option.id}>
                              <span className="block truncate">{option.label}</span>
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="w-full space-y-2">
                      <Label className="text-white/75">Эквалайзер</Label>
                      <Select value={sessionPipelineConfig.eqId} onValueChange={(value) => onPipelineConfigChange("eqId", value)}>
                        <SelectTrigger className={`rounded-2xl border-white/10 bg-black/20 text-white ${pipelineSelectWidthClassName}`}><SelectValue className="truncate" /></SelectTrigger>
                        <SelectContent className="min-w-[280px]">
                          {pipelineCatalog.eqs.map((option) => (
                            <SelectItem key={option.id} value={option.id}>
                              <span className="block truncate">{option.label}</span>
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="w-full space-y-2">
                      <Label className="text-white/75">Маппер</Label>
                      <Select value={sessionPipelineConfig.mapperId} onValueChange={(value) => onPipelineConfigChange("mapperId", value)}>
                        <SelectTrigger className={`rounded-2xl border-white/10 bg-black/20 text-white ${pipelineSelectWidthClassName}`}><SelectValue className="truncate" /></SelectTrigger>
                        <SelectContent className="min-w-[280px]">
                          {pipelineCatalog.mappers.map((option) => (
                            <SelectItem key={option.id} value={option.id}>
                              <span className="block truncate">{option.label}</span>
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="w-full space-y-2">
                      <Label className="text-white/75">Генератор кандидатов</Label>
                      <Select value={sessionPipelineConfig.generatorId} onValueChange={(value) => onPipelineConfigChange("generatorId", value)}>
                        <SelectTrigger className={`rounded-2xl border-white/10 bg-black/20 text-white ${pipelineSelectWidthClassName}`}><SelectValue className="truncate" /></SelectTrigger>
                        <SelectContent className="min-w-[280px]">
                          {pipelineCatalog.generators.map((option) => (
                            <SelectItem key={option.id} value={option.id}>
                              <span className="block truncate">{option.label}</span>
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="w-full space-y-2">
                      <Label className="text-white/75">Стратегия</Label>
                      <Select value={sessionPipelineConfig.strategyId} onValueChange={(value) => onPipelineConfigChange("strategyId", value)}>
                        <SelectTrigger className={`rounded-2xl border-white/10 bg-black/20 text-white ${pipelineSelectWidthClassName}`}><SelectValue className="truncate" /></SelectTrigger>
                        <SelectContent className="min-w-[280px]">
                          {pipelineCatalog.strategies.map((option) => (
                            <SelectItem key={option.id} value={option.id}>
                              <span className="block truncate">{option.label}</span>
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                </div>
              )}

              <OnboardingTarget targetId="session-start-button">
                <Button
                  className="w-full max-w-[240px] rounded-2xl bg-cyan-300 px-8 py-[18px] text-base text-slate-950 shadow-[0_14px_36px_rgba(56,189,248,0.18)] transition-transform duration-200 hover:-translate-y-0.5 hover:scale-[1.02] hover:bg-cyan-200 hover:shadow-[0_18px_42px_rgba(56,189,248,0.24)] active:translate-y-px active:scale-[0.985] active:bg-cyan-100"
                  onClick={() => {
                    emitEvent("session.start.clicked");
                    onStart();
                  }}
                >
                  Начать
                </Button>
              </OnboardingTarget>
            </div>
          </div>
      </CardContent>
    </Card>
  );
}
