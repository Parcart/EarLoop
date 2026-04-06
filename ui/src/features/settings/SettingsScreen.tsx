import { Trash2 } from "lucide-react";

import { OnboardingTarget } from "@/features/onboarding/OnboardingTarget";
import { useAppOnboardingContext } from "@/features/onboarding/useAppOnboarding";
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
import type { AudioDeviceOption, AudioDevices, PerceptualParams, PipelineCatalog, SavedProfile, ThemeMode } from "@/lib/types/ui";

type SettingsScreenProps = {
  devices: AudioDevices;
  showAdvanced: boolean;
  blockSize: number;
  gainCompensation: number;
  themeMode: ThemeMode;
  audioDevicesCatalog: AudioDevicesCatalog;
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
}: SettingsScreenProps) {
  const { emitEvent, shouldLockAudioSettings } = useAppOnboardingContext();
  const deviceSelectWidthClassName = "w-full md:w-[360px]";
  const deviceSelectContentWidthClassName = "w-[360px] min-w-[360px] max-w-[360px]";
  const isWasapiDevice = (device: AudioDeviceOption) => device.hostapi?.includes("WASAPI") ?? false;
  const isVirtualCableInputDevice = (device: AudioDeviceOption) => device.kind === "input" && device.label.toLowerCase().includes("cable output");
  const hasCompatiblePartners = (device: AudioDeviceOption) => (device.compatibleDeviceIds?.length ?? 0) > 0;
  const inputDevices = audioDevicesCatalog.inputs.filter((device) => isWasapiDevice(device) && isVirtualCableInputDevice(device));
  const outputDevices = audioDevicesCatalog.outputs.filter((device) => isWasapiDevice(device));
  const allResolvedInputDevices = inputDevices.length > 0 ? inputDevices : audioDevicesCatalog.inputs;
  const allResolvedOutputDevices = outputDevices.length > 0 ? outputDevices : audioDevicesCatalog.outputs;
  const selectedInput = allResolvedInputDevices.find((device) => device.deviceId === devices.input) ?? allResolvedInputDevices[0] ?? null;
  const selectedOutput = allResolvedOutputDevices.find((device) => device.deviceId === devices.output) ?? allResolvedOutputDevices[0] ?? null;
  const filteredInputDevices = selectedOutput
    ? allResolvedInputDevices.filter((device) => selectedOutput.compatibleDeviceIds?.includes(device.deviceId))
    : allResolvedInputDevices;
  const filteredOutputDevices = selectedInput
    ? allResolvedOutputDevices.filter((device) => selectedInput.compatibleDeviceIds?.includes(device.deviceId))
    : allResolvedOutputDevices;
  const resolvedInputDevices = filteredInputDevices.length > 0 ? filteredInputDevices : allResolvedInputDevices;
  const resolvedOutputDevices = filteredOutputDevices.length > 0 ? filteredOutputDevices : allResolvedOutputDevices;
  const audioStatusTone = engineAudioStatus?.status === "applied"
    ? "border-emerald-400/20 bg-emerald-400/10 text-emerald-100"
    : engineAudioStatus?.status === "pending_restart"
      ? "border-amber-400/20 bg-amber-400/10 text-amber-100"
      : engineAudioStatus?.status === "device_unavailable" || engineAudioStatus?.status === "failed"
        ? "border-red-400/20 bg-red-400/10 text-red-100"
        : "border-white/10 bg-black/20 text-white/78";
  const audioStatusLabel = engineAudioStatus?.status === "applied"
    ? "Применено"
    : engineAudioStatus?.status === "pending_restart"
      ? "Нужен перезапуск"
      : engineAudioStatus?.status === "device_unavailable"
        ? "Устройство недоступно"
        : engineAudioStatus?.status === "failed"
          ? "Ошибка применения"
          : "Статус неизвестен";
  const formatAppliedAt = (value?: string | null) => {
    if (!value) return "Ещё не применялось";
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? value : date.toLocaleString("ru-RU");
  };
  const renderAudioConfig = (title: string, config: AudioStatusConfig | null | undefined) => (
    <div className="rounded-2xl border border-white/10 bg-black/20 p-3">
      <div className="text-xs uppercase tracking-[0.14em] text-white/45">{title}</div>
      {config ? (
        <div className="mt-3 grid gap-2 text-sm text-white/78">
          <div className="flex items-start justify-between gap-4">
            <span className="text-white/48">Input</span>
            <span className="text-right text-white/88">{resolveDeviceLabel(allResolvedInputDevices, config.inputDeviceId)}</span>
          </div>
          <div className="flex items-start justify-between gap-4">
            <span className="text-white/48">Output</span>
            <span className="text-right text-white/88">{resolveDeviceLabel(allResolvedOutputDevices, config.outputDeviceId)}</span>
          </div>
          <div className="flex items-center justify-between gap-4">
            <span className="text-white/48">Sample rate</span>
            <span className="text-white/88">{config.sampleRate}</span>
          </div>
          <div className="flex items-center justify-between gap-4">
            <span className="text-white/48">Channels</span>
            <span className="text-white/88">{config.channels}</span>
          </div>
        </div>
      ) : (
        <div className="mt-3 text-sm text-white/52">Нет данных</div>
      )}
    </div>
  );
  const formatDeviceLabel = (device: AudioDeviceOption) => (
    device.kind === "input" && device.label.toLowerCase().includes("cable output")
      ? "CABLE Output (VB-Audio Virtual Cable)"
      : device.label
  );
  const resolveDeviceLabel = (options: AudioDeviceOption[], deviceId: string) => {
    const device = options.find((option) => option.deviceId === deviceId)
      ?? [...audioDevicesCatalog.inputs, ...audioDevicesCatalog.outputs].find((option) => option.deviceId === deviceId);
    return device ? formatDeviceLabel(device) : "Не выбрано";
  };
  const selectedInputDevice = [...audioDevicesCatalog.inputs, ...audioDevicesCatalog.outputs].find((option) => option.deviceId === devices.input) ?? null;
  const resolvePipelineLabel = (options: PipelineCatalog[keyof PipelineCatalog], id?: string) => {
    if (!id) return null;
    return options.find((option) => option.id === id)?.label ?? id;
  };
  const visibleDevicesSummary = `Совместимые WASAPI devices: inputs ${allResolvedInputDevices.length}, outputs ${allResolvedOutputDevices.length}`;
  const audioStatusSummary = engineAudioStatus?.lastError ?? AUDIO_STATUS_SUMMARY[engineAudioStatus?.status ?? "idle"] ?? "Конфигурация ещё не применялась.";
  const availableSampleRates = selectedInput?.compatibleSampleRates?.filter((value) => selectedOutput?.compatibleSampleRates?.includes(value)) ?? [];
  const hasVirtualCableInput = audioDevicesCatalog.inputs.some((device) => (
    device.hostapi?.includes("WASAPI") && device.label.toLowerCase().includes("cable output")
  ));

  return (
    <div className="mx-auto flex w-full max-w-[1600px] flex-col gap-6 text-left">
      <OnboardingTarget targetId="audio-setup-card">
        <Card className="rounded-[28px] border border-white/10 bg-[#0b0c11] shadow-none">
          <CardHeader>
            <CardTitle className="text-2xl text-white">Настройки аудиотракта</CardTitle>
            <CardDescription className="text-white/70">Правая рабочая область меняется на окно настройки, боковая панель остаётся на месте.</CardDescription>
          </CardHeader>
          <CardContent className="pb-5">
            <div className="space-y-5">
              <OnboardingTarget targetId="audio-setup-focus">
                <div className="space-y-5 rounded-[24px]">
                  <div className="space-y-2">
                    <Label className="text-white/75">Источник захвата</Label>
                    <div className={deviceSelectWidthClassName}>
                      <Select
                        value={devices.input}
                        onValueChange={(value) => {
                          onDeviceChange("input", value);
                          emitEvent("audio.input.changed");
                        }}
                      >
                        <OnboardingTarget targetId="audio-input-select">
                          <SelectTrigger className={`rounded-2xl border-white/10 bg-white/5 text-white ${deviceSelectWidthClassName}`}>
                            <SelectValue className="truncate" placeholder="Выберите input device" />
                          </SelectTrigger>
                        </OnboardingTarget>
                        <SelectContent
                          className={`${deviceSelectContentWidthClassName} max-h-[320px] overflow-y-auto`}
                          portal={false}
                          position="popper"
                          side="bottom"
                          align="start"
                          sideOffset={8}
                          avoidCollisions={false}
                        >
                          {resolvedInputDevices.length > 0 ? resolvedInputDevices.map((device) => (
                            <SelectItem key={device.deviceId} value={device.deviceId}>
                              <span className="block truncate">{formatDeviceLabel(device)}</span>
                            </SelectItem>
                          )) : (
                            <div className="px-3 py-2 text-sm text-white/55">Устройства захвата недоступны</div>
                          )}
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="text-xs text-white/50">
                      {resolveDeviceLabel(resolvedInputDevices, devices.input)}
                      {selectedInputDevice?.kind === "input" && selectedInputDevice.label.toLowerCase().includes("cable output")
                        ? " · захват сигнала из VB-Cable"
                        : ""}
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label className="text-white/75">Устройство вывода</Label>
                    <div className={deviceSelectWidthClassName}>
                      <Select
                        value={devices.output}
                        onValueChange={(value) => {
                          onDeviceChange("output", value);
                          emitEvent("audio.output.changed");
                        }}
                      >
                        <OnboardingTarget targetId="audio-output-select">
                          <SelectTrigger className={`rounded-2xl border-white/10 bg-white/5 text-white ${deviceSelectWidthClassName}`}>
                            <SelectValue className="truncate" placeholder="Выберите output device" />
                          </SelectTrigger>
                        </OnboardingTarget>
                        <SelectContent
                          className={`${deviceSelectContentWidthClassName} max-h-[320px] overflow-y-auto`}
                          portal={false}
                          position="popper"
                          side="bottom"
                          align="start"
                          sideOffset={8}
                          avoidCollisions={false}
                        >
                          {resolvedOutputDevices.length > 0 ? resolvedOutputDevices.map((device) => (
                            <SelectItem key={device.deviceId} value={device.deviceId}>
                              <span className="block truncate">{formatDeviceLabel(device)}</span>
                            </SelectItem>
                          )) : (
                            <div className="px-3 py-2 text-sm text-white/55">Устройства вывода недоступны</div>
                          )}
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="text-xs text-white/50">{resolveDeviceLabel(resolvedOutputDevices, devices.output)}</div>
                  </div>
                </div>
              </OnboardingTarget>
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label className="text-white/75">Sample rate</Label>
                  <Select value={devices.sampleRate} onValueChange={(value) => onDeviceChange("sampleRate", value)}>
                    <SelectTrigger className="rounded-2xl border-white/10 bg-white/5 text-white"><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {availableSampleRates.length > 0 ? availableSampleRates.map((sampleRate) => (
                        <SelectItem key={sampleRate} value={sampleRate}>{sampleRate}</SelectItem>
                      )) : (
                        <>
                          <SelectItem value="44100">44100</SelectItem>
                          <SelectItem value="48000">48000</SelectItem>
                        </>
                      )}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label className="text-white/75">Channels</Label>
                  <Select value={devices.channels} onValueChange={(value) => onDeviceChange("channels", value)}>
                    <SelectTrigger className="rounded-2xl border-white/10 bg-white/5 text-white"><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="1">1</SelectItem>
                      <SelectItem value="2">2</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="flex items-center justify-between rounded-2xl border border-white/10 bg-white/5 p-4">
                <div>
                  <div className="font-medium text-white/88">Расширенные настройки</div>
                  <div className="text-sm text-white/68">Latency, buffer size, debug logging</div>
                </div>
                <Switch checked={showAdvanced} onCheckedChange={onShowAdvancedChange} className="data-[state=checked]:bg-violet-300 data-[state=unchecked]:bg-white/15" />
              </div>

              {showAdvanced && (
                <div className="space-y-4 rounded-2xl border border-white/10 bg-white/5 p-4">
                  <div className="space-y-2 rounded-2xl border border-white/10 bg-black/20 p-3">
                    <div className="flex items-center justify-between text-sm"><span className="text-white/78">Block size</span><span className="font-medium text-cyan-200">{blockSize}</span></div>
                    <Slider value={[blockSize]} min={128} max={4096} step={128} className={sliderClassName} onValueChange={(value) => onBlockSizeChange(value[0])} />
                  </div>
                  <div className="space-y-2 rounded-2xl border border-white/10 bg-black/20 p-3">
                    <div className="flex items-center justify-between text-sm"><span className="text-white/78">Gain compensation</span><span className="font-medium text-cyan-200">{gainCompensation}</span></div>
                    <Slider value={[gainCompensation]} min={0} max={100} step={1} className={sliderClassName} onValueChange={(value) => onGainCompensationChange(value[0])} />
                  </div>
                </div>
              )}

              <div className="rounded-2xl border border-white/10 bg-black/20 p-3 text-sm text-white/70">
                <div className="font-medium text-white/82">Источник устройств: {audioDevicesCatalog.status}</div>
                <div className="mt-1 text-white/55">
                  {audioDevicesCatalog.error ?? visibleDevicesSummary}
                </div>
              </div>

              {!hasVirtualCableInput && (
                <div className="rounded-2xl border border-amber-400/20 bg-amber-400/10 p-4 text-sm text-amber-100">
                  <div className="font-medium">Virtual cable capture недоступен</div>
                  <div className="mt-2 text-amber-50/90">
                    EarLoop не видит `CABLE Output (VB-Audio Virtual Cable)` в WASAPI capture-источниках.
                  </div>
                  <div className="mt-2 text-amber-50/80">
                    Выбери другой capture source или установи/включи VB-Cable, чтобы захватывать системный звук через виртуальный тракт.
                  </div>
                </div>
              )}

              <div className="space-y-4 rounded-[24px] border border-white/10 bg-white/5 p-4">
                <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
                  <div>
                    <div className="text-base font-medium text-white">Статус аудиодвижка</div>
                    <div className="mt-1 text-sm text-white/60">Компактный статус применения текущей audio-конфигурации.</div>
                  </div>
                  <div className={`inline-flex rounded-full border px-3 py-1 text-xs font-medium ${audioStatusTone}`}>
                    {audioStatusLabel}
                  </div>
                </div>

                <div className="grid gap-3 lg:grid-cols-[minmax(0,1.15fr)_minmax(260px,0.85fr)]">
                  <div className={`rounded-2xl border p-3 ${engineAudioStatus?.lastError ? "border-red-400/20 bg-red-400/8" : "border-white/10 bg-black/20"}`}>
                    <div className="text-xs uppercase tracking-[0.14em] text-white/45">Кратко</div>
                    <div className={`mt-3 text-sm ${engineAudioStatus?.lastError ? "text-red-100" : "text-white/82"}`}>
                      {audioStatusSummary}
                    </div>
                  </div>

                  <div className="rounded-2xl border border-white/10 bg-black/20 p-3">
                    <div className="text-xs uppercase tracking-[0.14em] text-white/45">Последняя попытка применения</div>
                    <div className="mt-3 text-sm text-white/78">{formatAppliedAt(engineAudioStatus?.lastAppliedAt)}</div>
                  </div>
                </div>

                <details className="rounded-2xl border border-white/10 bg-black/20 p-3">
                  <summary className="cursor-pointer list-none text-sm font-medium text-white/78 marker:hidden">
                    Показать детали аудиоконфигурации
                  </summary>
                  <div className="mt-3 grid gap-3 lg:grid-cols-2">
                    {renderAudioConfig("Активная конфигурация", engineAudioStatus?.activeConfig ?? null)}
                    {renderAudioConfig("Желаемая конфигурация", engineAudioStatus?.desiredConfig ?? null)}
                  </div>

                  <div className={`mt-3 rounded-2xl border p-3 ${engineAudioStatus?.lastError ? "border-red-400/20 bg-red-400/8" : "border-white/10 bg-black/20"}`}>
                    <div className="text-xs uppercase tracking-[0.14em] text-white/45">Последняя ошибка</div>
                    <div className={`mt-3 text-sm ${engineAudioStatus?.lastError ? "text-red-100" : "text-white/52"}`}>
                      {engineAudioStatus?.lastError ?? "Ошибок нет"}
                    </div>
                  </div>
                </details>
              </div>

              <div className="space-y-4 rounded-[24px] border border-white/10 bg-white/5 p-4">
                <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
                  <div>
                    <div className="text-base font-medium text-white">Runtime EQ</div>
                    <div className="mt-1 text-sm text-white/60">Состояние активного профиля в audio processing chain.</div>
                  </div>
                  <div className={`inline-flex rounded-full border px-3 py-1 text-xs font-medium ${
                    runtimeProfileStatus?.applyStatus === "applied"
                      ? "border-emerald-400/20 bg-emerald-400/10 text-emerald-100"
                      : runtimeProfileStatus?.applyStatus === "ready"
                        ? "border-amber-400/20 bg-amber-400/10 text-amber-100"
                        : runtimeProfileStatus?.applyStatus === "failed"
                          ? "border-red-400/20 bg-red-400/10 text-red-100"
                          : "border-white/10 bg-black/20 text-white/78"
                  }`}>
                    {runtimeProfileStatus?.applyStatus ?? "not_applied"}
                  </div>
                </div>

                <div className="grid gap-3 md:grid-cols-2">
                  <div className="rounded-2xl border border-white/10 bg-black/20 p-3">
                    <div className="text-xs uppercase tracking-[0.14em] text-white/45">Активный профиль</div>
                    <div className="mt-2 text-sm text-white/88">{runtimeProfileStatus?.activeProfileName ?? "Не выбран"}</div>
                    <div className="mt-1 text-xs text-white/50">{runtimeProfileStatus?.activeProfileId ?? "Нет activeProfileId"}</div>
                  </div>
                  <div className="rounded-2xl border border-white/10 bg-black/20 p-3">
                    <div className="text-xs uppercase tracking-[0.14em] text-white/45">Процессор</div>
                    <div className="mt-2 text-sm text-white/88">{runtimeProfileStatus?.processorMode ?? "passthrough"}</div>
                    <div className="mt-1 text-xs text-white/50">
                      {runtimeProfileStatus?.eqCurveReady
                        ? `${runtimeProfileStatus.eqBandCount ?? 0} bands, preamp ${runtimeProfileStatus.preampDb ?? 0} dB`
                        : "EQ curve пока не рассчитана"}
                    </div>
                  </div>
                </div>

                <div className={`rounded-2xl border p-3 ${runtimeProfileStatus?.lastError ? "border-red-400/20 bg-red-400/8" : "border-white/10 bg-black/20"}`}>
                  <div className="text-xs uppercase tracking-[0.14em] text-white/45">Ошибки runtime EQ</div>
                  <div className={`mt-3 text-sm ${runtimeProfileStatus?.lastError ? "text-red-100" : "text-white/52"}`}>
                    {runtimeProfileStatus?.lastError ?? (runtimeProfileStatus?.appliedToAudio ? "EQ chain синхронизирована с audio runtime" : "EQ representation готова, но live apply ещё не активен")}
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </OnboardingTarget>

      <Card
        className={`rounded-[28px] border border-white/10 bg-[#0b0c11] shadow-none transition-opacity duration-200 ${
          shouldLockAudioSettings ? "pointer-events-none opacity-20 saturate-50" : ""
        }`}
        aria-hidden={shouldLockAudioSettings}
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
        className={`rounded-[28px] border border-white/10 bg-[#0b0c11] shadow-none transition-opacity duration-200 ${
          shouldLockAudioSettings ? "pointer-events-none opacity-20 saturate-50" : ""
        }`}
        aria-hidden={shouldLockAudioSettings}
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
        className={`rounded-[28px] border border-white/10 bg-[#0b0c11] shadow-none transition-opacity duration-200 ${
          shouldLockAudioSettings ? "pointer-events-none opacity-20 saturate-50" : ""
        }`}
        aria-hidden={shouldLockAudioSettings}
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
