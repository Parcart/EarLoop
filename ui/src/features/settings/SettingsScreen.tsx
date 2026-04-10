import { Trash2 } from "lucide-react";

import { AudioSetupPanel } from "@/features/audio/AudioSetupPanel";
import { EqChart } from "@/components/shared/EqChart";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { profileParamMeta, sliderClassName } from "@/lib/mock/profiles";
import { curveFromProfileParams } from "@/lib/session/helpers";

import type { AudioDevicesCatalog, EngineStatus } from "@/lib/api/engine.types";
import type { AudioDevices, PerceptualParams, PipelineCatalog, SavedProfile, ThemeMode } from "@/lib/types/ui";

type SettingsScreenProps = {
  devices: AudioDevices;
  showAdvanced: boolean;
  blockSize: number;
  gainCompensation: number;
  themeMode: ThemeMode;
  audioDevicesCatalog: AudioDevicesCatalog;
  engineStatusSnapshot: EngineStatus;
  engineAudioStatus?: EngineStatus["audioStatus"];
  runtimeProfileStatus?: EngineStatus["runtimeProfileStatus"];
  pipelineCatalog: PipelineCatalog;
  savedProfiles: SavedProfile[];
  selectedProfile: string;
  managedProfile: SavedProfile | null;
  profileEditorName: string;
  profileEditorParams: PerceptualParams;
  onDeviceChange: (field: keyof AudioDevices, value: string) => void;
  onShowAdvancedChange: (value: boolean) => void;
  onBlockSizeChange: (value: number) => void;
  onGainCompensationChange: (value: number) => void;
  onThemeModeChange: (value: ThemeMode) => void;
  onSelectProfile: (value: string) => void;
  onProfileEditorNameChange: (value: string) => void;
  onProfileParamChange: (key: keyof PerceptualParams, value: number) => void;
  onDeleteProfile: () => void;
  onResetProfile: () => void;
  onSaveProfile: () => void;
  onRestartAppOnboarding: () => void;
  onRefreshAudioEnvironment: () => void;
};

type AudioStatusConfig = NonNullable<NonNullable<EngineStatus["audioStatus"]>["activeConfig"]>;
const AUDIO_STATUS_SUMMARY: Record<string, string> = {
  applied: "Текущая конфигурация применена в runtime.",
  pending_restart: "Конфигурация подготовлена, но нужен перезапуск аудиомоста.",
  device_unavailable: "Одно из выбранных устройств сейчас недоступно.",
  failed: "Конфигурация не применилась. Подробности ниже.",
  idle: "Конфигурация ещё не применялась.",
};

export function SettingsScreen({
  devices,
  showAdvanced,
  blockSize,
  gainCompensation,
  themeMode,
  audioDevicesCatalog,
  engineStatusSnapshot,
  engineAudioStatus,
  runtimeProfileStatus,
  pipelineCatalog,
  savedProfiles,
  selectedProfile,
  managedProfile,
  profileEditorName,
  profileEditorParams,
  onDeviceChange,
  onShowAdvancedChange,
  onBlockSizeChange,
  onGainCompensationChange,
  onThemeModeChange,
  onSelectProfile,
  onProfileEditorNameChange,
  onProfileParamChange,
  onDeleteProfile,
  onResetProfile,
  onSaveProfile,
  onRestartAppOnboarding,
  onRefreshAudioEnvironment,
}: SettingsScreenProps) {
  const resolvePipelineLabel = (options: PipelineCatalog[keyof PipelineCatalog], id?: string) => {
    if (!id) return null;
    return options.find((option) => option.id === id)?.label ?? id;
  };

  return (
    <div className="mx-auto flex w-full max-w-[1600px] flex-col gap-6 text-left">
      <AudioSetupPanel
        title="Настройки аудиотракта"
        description="Здесь можно переоткрыть первый шаг настройки, выбрать WASAPI input/output и проверить маршрут без возврата в onboarding."
        devices={devices}
        audioDevicesCatalog={audioDevicesCatalog}
        engineStatus={engineStatusSnapshot}
        onDeviceChange={onDeviceChange}
        onRecheck={onRefreshAudioEnvironment}
      />

      <Card
        className="rounded-[28px] border border-white/10 bg-[#0b0c11] shadow-none transition-opacity duration-200"
      >
        <CardHeader>
          <CardTitle className="text-2xl text-white">Интерфейс</CardTitle>
          <CardDescription className="text-white/70">Здесь будут собраны настройки внешнего вида приложения.</CardDescription>
        </CardHeader>
        <CardContent className="pb-5">
          <div className="space-y-3 rounded-2xl border border-white/10 bg-white/5 p-4">
            <div>
              <div className="font-medium text-white/88">Тема</div>
              <div className="text-sm text-white/68">Выбор светлой, тёмной темы или режима как на устройстве.</div>
            </div>
            <RadioGroup value={themeMode} onValueChange={(value) => onThemeModeChange(value as ThemeMode)} className="grid gap-3 md:grid-cols-3">
              {[
                ["light", "Светлая", "Всегда использовать светлую тему"],
                ["dark", "Тёмная", "Всегда использовать тёмную тему"],
                ["system", "Как на устройстве", "Следовать системной теме"],
              ].map(([value, label, description]) => (
                <label key={value} htmlFor={`theme-${value}`} className="flex cursor-pointer items-start gap-3 rounded-2xl border border-white/12 bg-black/25 p-3 text-white transition hover:border-violet-300/35 hover:bg-white/5">
                  <RadioGroupItem value={value} id={`theme-${value}`} className="mt-0.5 border-slate-300 text-violet-300 data-[state=checked]:border-violet-300 data-[state=checked]:bg-violet-300" />
                  <div>
                    <div className="text-sm font-medium text-white">{label}</div>
                    <div className="mt-1 text-xs text-white/60">{description}</div>
                  </div>
                </label>
              ))}
            </RadioGroup>
          </div>
        </CardContent>
      </Card>

      <Card
        className="rounded-[28px] border border-white/10 bg-[#0b0c11] shadow-none transition-opacity duration-200"
      >
        <CardHeader>
          <CardTitle className="text-2xl text-white">Профили</CardTitle>
          <CardDescription className="text-white/70">Выбери профиль, измени его название и perceptual-параметры или удали из библиотеки.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-5 pb-5">
          <div className="space-y-2">
            <Label className="text-white/75">Профиль для редактирования</Label>
            <Select value={selectedProfile} onValueChange={onSelectProfile}>
              <SelectTrigger className="rounded-2xl border-white/10 bg-black/20 text-white"><SelectValue /></SelectTrigger>
              <SelectContent>
                {savedProfiles.map((profile) => (
                  <SelectItem key={profile.id} value={profile.id}>
                    {profile.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {managedProfile && (
            <>
              <div className="space-y-2">
                <Label htmlFor="managed-profile-name" className="text-white/75">Название профиля</Label>
                <Input id="managed-profile-name" value={profileEditorName} onChange={(event) => onProfileEditorNameChange(event.target.value)} className="rounded-2xl border-white/10 bg-black/20 text-white placeholder:text-white/35" />
              </div>

              <EqChart values={curveFromProfileParams(profileEditorParams, 3)} accentShift={7} />

              <div className="grid gap-4 lg:grid-cols-2">
                {profileParamMeta.map((param) => (
                  <div key={param.key} className="space-y-2 rounded-2xl border border-white/10 bg-black/20 p-3">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-white/78">{param.label}</span>
                      <span className="font-medium text-cyan-200">{profileEditorParams[param.key].toFixed(2)}</span>
                    </div>
                    <Slider value={[profileEditorParams[param.key]]} min={-1} max={1} step={0.01} className={sliderClassName} onValueChange={(value) => onProfileParamChange(param.key, value[0])} />
                  </div>
                ))}
              </div>

              <div className="space-y-4 rounded-[24px] border border-white/10 bg-white/5 p-4">
                <div>
                  <div className="text-base font-medium text-white">Конфигурация пайплайна</div>
                  <div className="mt-1 text-sm text-white/60">Сохранённая конфигурация движка для выбранного профиля.</div>
                </div>

                {managedProfile.pipelineConfig ? (
                  <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
                    <div className="rounded-2xl border border-white/10 bg-black/20 p-3">
                      <div className="text-xs uppercase tracking-[0.14em] text-white/45">Модель</div>
                      <div className="mt-2 text-sm text-white/85">{resolvePipelineLabel(pipelineCatalog.models, managedProfile.pipelineConfig.modelId)}</div>
                    </div>
                    <div className="rounded-2xl border border-white/10 bg-black/20 p-3">
                      <div className="text-xs uppercase tracking-[0.14em] text-white/45">Эквалайзер</div>
                      <div className="mt-2 text-sm text-white/85">{resolvePipelineLabel(pipelineCatalog.eqs, managedProfile.pipelineConfig.eqId)}</div>
                    </div>
                    <div className="rounded-2xl border border-white/10 bg-black/20 p-3">
                      <div className="text-xs uppercase tracking-[0.14em] text-white/45">Маппер</div>
                      <div className="mt-2 text-sm text-white/85">{resolvePipelineLabel(pipelineCatalog.mappers, managedProfile.pipelineConfig.mapperId)}</div>
                    </div>
                    {managedProfile.pipelineConfig.generatorId && (
                      <div className="rounded-2xl border border-white/10 bg-black/20 p-3">
                        <div className="text-xs uppercase tracking-[0.14em] text-white/45">Генератор кандидатов</div>
                        <div className="mt-2 text-sm text-white/85">{resolvePipelineLabel(pipelineCatalog.generators, managedProfile.pipelineConfig.generatorId)}</div>
                      </div>
                    )}
                    {managedProfile.pipelineConfig.strategyId && (
                      <div className="rounded-2xl border border-white/10 bg-black/20 p-3">
                        <div className="text-xs uppercase tracking-[0.14em] text-white/45">Стратегия</div>
                        <div className="mt-2 text-sm text-white/85">{resolvePipelineLabel(pipelineCatalog.strategies, managedProfile.pipelineConfig.strategyId)}</div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="rounded-2xl border border-dashed border-white/10 bg-black/15 p-4 text-sm text-white/55">Конфигурация пайплайна не сохранена</div>
                )}
              </div>

              <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                <Button variant="outline" className="rounded-2xl border-red-400/20 bg-red-400/10 text-red-200 hover:bg-red-400/15" onClick={onDeleteProfile} disabled={savedProfiles.length <= 1}><Trash2 className="mr-2 h-4 w-4" /> Удалить профиль</Button>
                <div className="flex flex-col gap-3 sm:flex-row">
                  <Button variant="outline" className="rounded-2xl border-white/10 bg-white/5 text-white hover:bg-white/10" onClick={onResetProfile}>Сбросить</Button>
                  <Button variant="outline" className="rounded-2xl border-white/10 bg-white/5 text-white hover:bg-white/10" onClick={onSaveProfile}>Сохранить</Button>
                </div>
              </div>
            </>
          )}
        </CardContent>
      </Card>

      <Card
        className="rounded-[28px] border border-white/10 bg-[#0b0c11] shadow-none transition-opacity duration-200"
      >
        <CardHeader>
          <CardTitle className="text-2xl text-white">Обучение</CardTitle>
          <CardDescription className="text-white/70">Повторно показывает guided onboarding первого запуска.</CardDescription>
        </CardHeader>
        <CardContent className="pb-5">
          <div className="flex flex-col gap-4 rounded-[24px] border border-white/10 bg-white/5 p-4 md:flex-row md:items-center md:justify-between">
            <div>
              <div className="text-base font-medium text-white">Запустить обучение заново</div>
              <div className="mt-1 max-w-2xl text-sm text-white/60">
                Сценарий снова проведёт по выбору аудиоустройств, созданию профиля, A/B сравнению и сохранению результата.
              </div>
            </div>
            <Button
              className="rounded-2xl bg-cyan-300 text-slate-950 shadow-[0_12px_30px_rgba(56,189,248,0.22)] transition-colors hover:bg-cyan-200 active:bg-cyan-100"
              onClick={onRestartAppOnboarding}
            >
              Запустить обучение
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
