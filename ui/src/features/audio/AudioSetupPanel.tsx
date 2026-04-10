import { useEffect, useMemo, useRef, useState } from "react";
import { ChevronDown, Copy, RefreshCw } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  buildAudioDiagnosticsText,
  evaluateAudioSetupEnvironment,
  formatAudioDeviceLabel,
  isSampleRateMismatchReason,
  resolveAudioDeviceLabel,
} from "@/lib/audio/setup";

import type { AudioDevicesCatalog, EngineStatus } from "@/lib/api/engine.types";
import type { AudioDevices } from "@/lib/types/ui";

type AudioSetupPanelProps = {
  title: string;
  description: string;
  devices: AudioDevices;
  audioDevicesCatalog: AudioDevicesCatalog;
  engineStatus?: EngineStatus;
  forceVisible?: boolean;
  continueLabel?: string;
  onDeviceChange: (field: keyof AudioDevices, value: string) => void;
  onRecheck: () => void;
  onContinue?: () => void;
};

function SetupSelectCard({
  label,
  helper,
  value,
  placeholder,
  options,
  onValueChange,
}: {
  label: string;
  helper: string;
  value: string;
  placeholder: string;
  options: Array<{ deviceId: string; label: string }>;
  onValueChange: (value: string) => void;
}) {
  const selectedOption = options.find((option) => option.deviceId === value);

  return (
    <div className="rounded-[26px] border border-white/10 bg-white/[0.03] p-4 text-left">
      <div className="text-left text-sm font-medium text-white">{label}</div>

      <div className="mt-3">
        <Select value={value} onValueChange={onValueChange}>
          <SelectTrigger className="w-full rounded-2xl border-white/10 bg-black/20 px-4 py-3 text-sm text-white">
            <SelectValue placeholder={placeholder}>
              <span
                className="block max-w-full truncate pr-6 text-left"
                title={selectedOption?.label ?? placeholder}
              >
                {selectedOption?.label ?? placeholder}
              </span>
            </SelectValue>
          </SelectTrigger>

          <SelectContent className="max-h-[320px] overflow-y-auto">
            {options.length > 0 ? (
              options.map((option) => (
                <SelectItem key={option.deviceId} value={option.deviceId} className="py-2">
                  <span className="block max-w-[420px] whitespace-normal break-words leading-5">
                    {option.label}
                  </span>
                </SelectItem>
              ))
            ) : (
              <div className="px-3 py-2 text-sm text-white/55">Устройства не найдены</div>
            )}
          </SelectContent>
        </Select>
      </div>

      <div className="mt-3 min-h-[40px] text-xs leading-5 text-white/46">{helper}</div>
    </div>
  );
}

function SetupStatusCard({
  title,
  description,
  buttonLabel,
  onButtonClick,
  isButtonDisabled = false,
  statusText,
  tone,
  minBodyHeightClassName = "min-h-[88px]",
  children,
}: {
  title: string;
  description?: React.ReactNode;
  buttonLabel: string;
  onButtonClick: () => void;
  isButtonDisabled?: boolean;
  statusText: string;
  tone: "success" | "warning" | "error";
  minBodyHeightClassName?: string;
  children?: React.ReactNode;
}) {
  const toneClassName = tone === "success"
    ? "border-emerald-400/20 bg-emerald-400/10 text-emerald-100"
    : tone === "error"
      ? "border-red-400/20 bg-red-400/10 text-red-100"
      : "border-amber-400/20 bg-amber-400/10 text-amber-100";

  return (
    <div className="rounded-[26px] border border-white/10 bg-white/[0.03] p-4 text-left">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1 text-left text-sm font-medium leading-5 text-white">{title}</div>
        <Button
          variant="outline"
          className="shrink-0 rounded-full border-white/10 bg-white/5 px-5 py-2.5 text-sm text-white hover:bg-white/10"
          disabled={isButtonDisabled}
          onClick={onButtonClick}
        >
          {buttonLabel}
        </Button>
      </div>
      <div className={`mt-3 ${minBodyHeightClassName}`}>
        {typeof description === "string" ? (
          <div className="whitespace-pre-line text-xs leading-5 text-white/46">{description}</div>
        ) : (
          description
        )}
      </div>
      {children}
      <div className={`mt-4 rounded-2xl border px-4 py-3 text-center text-sm font-medium ${toneClassName}`}>
        {statusText}
      </div>
    </div>
  );
}

export function AudioSetupPanel({
  title,
  description,
  devices,
  audioDevicesCatalog,
  engineStatus,
  forceVisible = false,
  continueLabel = "Продолжить",
  onDeviceChange,
  onRecheck,
  onContinue,
}: AudioSetupPanelProps) {
  const [bridgeNotice, setBridgeNotice] = useState<string | null>(null);
  const [copyState, setCopyState] = useState<"idle" | "copied" | "failed">("idle");
  const [isDiagnosticsOpen, setIsDiagnosticsOpen] = useState(false);
  const [isBitrateHelpOpen, setIsBitrateHelpOpen] = useState(false);
  const [isVBCableInstallOpen, setIsVBCableInstallOpen] = useState(false);
  const evaluation = useMemo(
    () => evaluateAudioSetupEnvironment(audioDevicesCatalog, devices, engineStatus),
    [audioDevicesCatalog, devices, engineStatus],
  );
  const diagnosticsText = useMemo(
    () => buildAudioDiagnosticsText({
      catalog: audioDevicesCatalog,
      devices,
      engineStatus,
      forceAudioSetupScreen: forceVisible,
    }),
    [audioDevicesCatalog, devices, engineStatus, forceVisible],
  );
  const platformTools = window.earloopDesktop?.platformTools;
  const didMountRef = useRef(false);

  const triggerRefresh = ({ preserveNotice = false }: { preserveNotice?: boolean } = {}) => {
    if (!preserveNotice) {
      setBridgeNotice(null);
    }
    onRecheck();
  };

  useEffect(() => {
    if (!didMountRef.current) {
      didMountRef.current = true;
      return;
    }
    onRecheck();
  }, [devices.channels, devices.input, devices.output, devices.sampleRate]);

  const handleCopyDiagnostics = async () => {
    try {
      await navigator.clipboard.writeText(diagnosticsText);
      setCopyState("copied");
      window.setTimeout(() => setCopyState("idle"), 1800);
    } catch {
      setCopyState("failed");
      window.setTimeout(() => setCopyState("idle"), 1800);
    }
  };

  const handleOpenSoundSettings = async () => {
    if (!platformTools) {
      try {
        const fallbackWindow = window.open("ms-settings:sound", "_blank", "noopener,noreferrer");
        fallbackWindow?.close?.();
        if (!fallbackWindow) {
          window.location.href = "ms-settings:sound";
        }
        setBridgeNotice("Отправлена команда на открытие настроек звука Windows");
      } catch {
        setBridgeNotice("Не удалось открыть настройки звука без desktop bridge");
      }
      triggerRefresh({ preserveNotice: true });
      return;
    }
    const result = await platformTools.openSoundSettings();
    setBridgeNotice(result.ok ? "Открыты настройки звука Windows" : (result.error ?? "Операция не выполнена"));
    triggerRefresh({ preserveNotice: true });
  };

  const handleStartVirtualCableInstall = async () => {
    if (!platformTools) {
      setBridgeNotice("Desktop bridge недоступен");
      return;
    }
    const result = await platformTools.installVirtualCable();
    if (!result.ok) {
      setBridgeNotice(result.error ?? "Не удалось запустить установщик VB-Cable");
      return;
    }
    setBridgeNotice("Установщик VB-Cable запущен");
    setIsVBCableInstallOpen(true);
  };

  const isSampleRateMismatch = isSampleRateMismatchReason(evaluation.routeValidation.reason);
  const routeTone = evaluation.isRouteValid ? "success" : "error";
  const fallbackRouteNotice = evaluation.shouldExplainFallbackActiveRoute ? (
    <div className="mt-3 rounded-2xl border border-white/10 bg-black/20 p-3 text-left">
      <div className="text-[11px] uppercase tracking-[0.14em] text-white/42">Активный маршрут</div>
      <div className="mt-2 text-xs leading-5 text-white/62">
        Сейчас EarLoop продолжает работать на предыдущем маршруте.
        <br />
        Активный маршрут: {evaluation.activeInputLabel ?? "Неизвестно"} → {evaluation.activeOutputLabel ?? "Неизвестно"}
      </div>
    </div>
  ) : null;
  const routeDescription = evaluation.isRouteValid ? (
    <div className="text-left text-xs leading-5 text-white/46">
      <div>Вход: {resolveAudioDeviceLabel(audioDevicesCatalog, devices.input)}</div>
      <div>Выход: {resolveAudioDeviceLabel(audioDevicesCatalog, devices.output)}</div>
      {evaluation.routeModeHint && <div className="mt-1 text-white/38">{evaluation.routeModeHint}</div>}
    </div>
  ) : isSampleRateMismatch ? (
    <div className="text-left">
      <div className="text-xs leading-5 text-white/46">
        Sample rate у источника захвата и устройства вывода не совпадает
        <br />
        Сделайте одинаковый sample rate у Источника захвата и Устройства вывода.
      </div>
      <div className="mt-2 min-h-[24px]">
        <Button
          variant="outline"
          className="rounded-full border-amber-400/20 bg-amber-400/10 px-3 py-1 text-xs text-amber-100 hover:bg-amber-400/15 hover:text-amber-50"
          onClick={() => setIsBitrateHelpOpen(true)}
        >
          Что делать?
        </Button>
      </div>
      {evaluation.routeModeHint && <div className="mt-2 text-xs text-white/38">{evaluation.routeModeHint}</div>}
      {fallbackRouteNotice}
    </div>
  ) : (
    <div className="text-left">
      <div className="text-xs leading-5 text-white/46">Причина: {evaluation.routeValidation.reason}</div>
      {evaluation.routeModeHint && <div className="mt-2 text-xs text-white/38">{evaluation.routeModeHint}</div>}
      {fallbackRouteNotice}
    </div>
  );

  return (
    <>
      <Card className="rounded-[28px] border border-white/10 bg-[#0b0c11] shadow-none">
        <CardHeader className="relative px-6 pb-6 pt-8 md:px-10 md:pt-10">
          <div className="absolute right-6 top-6">
            <div className={`inline-flex rounded-full border px-3 py-1 text-[11px] font-medium ${evaluation.statusToneClassName}`}>
              {forceVisible && evaluation.isRouteValid ? "Показан по debug-флагу" : evaluation.statusLabel}
            </div>
          </div>
          <div className="text-center">
            <CardTitle className="text-3xl font-semibold tracking-tight text-white">{title}</CardTitle>
            <CardDescription className="mx-auto mt-3 max-w-2xl text-sm leading-6 text-white/68">{description}</CardDescription>
          </div>
        </CardHeader>
        <CardContent className="px-6 pb-8 md:px-10 md:pb-10">
          <div className="grid gap-4 md:grid-cols-2">
            <SetupSelectCard
              label="Источник захвата"
              value={devices.input}
              placeholder="Выберите WASAPI устройство"
              helper="Показываются обычные input-источники, виртуальные входы и совместимые loopback-источники. Метка Loopback означает захват системного звука с устройства вывода."
              options={evaluation.resolvedCaptureOptions.map((source) => ({
                deviceId: source.deviceId,
                label: source.displayLabel,
              }))}
              onValueChange={(value) => onDeviceChange("input", value)}
            />

            <SetupStatusCard
              title={evaluation.sourceAssistTitle}
              buttonLabel="Установить VB-Cable"
              onButtonClick={() => void handleStartVirtualCableInstall()}
              isButtonDisabled={!evaluation.shouldOfferVBCableInstall}
              statusText={evaluation.sourceAssistStatusText}
              tone={evaluation.sourceAssistTone}
              minBodyHeightClassName="min-h-[56]"
            >
              <div className="text-xs leading-5 text-white/46">
                {evaluation.sourceAssistDescription}
              </div>
            </SetupStatusCard>

            <SetupSelectCard
              label="Устройство вывода"
              value={devices.output}
              placeholder="Выберите WASAPI output"
              helper="Выберите физическое устройство вывода: наушники, колонки или другой реальный WASAPI output."
              options={evaluation.resolvedOutputOptions.map((device) => ({
                deviceId: device.deviceId,
                label: formatAudioDeviceLabel(device),
              }))}
              onValueChange={(value) => onDeviceChange("output", value)}
            />

            <SetupStatusCard
              title="Проверка маршрута"
              buttonLabel="Открыть настройки звука"
              onButtonClick={() => void handleOpenSoundSettings()}
              statusText={evaluation.isRouteValid ? "Маршрут проверен и запущен" : "Маршрут не готов"}
              tone={routeTone}
              minBodyHeightClassName="min-h-[72px]"
              description={routeDescription}
            />
          </div>

          <div className="mt-5 overflow-hidden rounded-[24px] border border-white/10 bg-white/[0.03]">
            <button
              type="button"
              className="flex w-full items-center justify-between gap-4 px-5 py-4 text-left"
              onClick={() => setIsDiagnosticsOpen((value) => !value)}
            >
              <div>
                <div className="text-sm font-semibold text-white">Диагностика</div>
                <div className="mt-1 text-xs leading-5 text-white/46">
                  Скрыта по умолчанию. Раскройте только если нужно отправить состояние в баг-репорт.
                </div>
              </div>
              <div className="flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-white/72">
                {isDiagnosticsOpen ? "Свернуть" : "Развернуть"}
                <ChevronDown className={`h-3.5 w-3.5 transition-transform ${isDiagnosticsOpen ? "rotate-180" : ""}`} />
              </div>
            </button>

            {isDiagnosticsOpen && (
              <div className="border-t border-white/10 px-5 py-5">
                <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
                  {[
                    ["Тип захвата", evaluation.selectedCaptureSourceTypeLabel ?? "-"],
                    ["Стратегия захвата", evaluation.selectedCaptureStrategyLabel ?? "-"],
                    ["Выбранный источник", evaluation.selectedCaptureLabel ?? "-"],
                    ["Выбранный вывод", resolveAudioDeviceLabel(audioDevicesCatalog, devices.output)],
                    ["Активный тип захвата", evaluation.activeCaptureSourceTypeLabel ?? "-"],
                    ["Активная стратегия", evaluation.activeCaptureStrategyLabel ?? "-"],
                    ["Активный источник", evaluation.activeInputLabel ?? "-"],
                    ["Активный вывод", evaluation.activeOutputLabel ?? "-"],
                    ["Loopback endpoint", evaluation.selectedLoopbackEndpointId ?? evaluation.activeLoopbackEndpointId ?? "-"],
                    ["VB-Cable", evaluation.hasVirtualCable ? "detected" : "not detected"],
                    ["Runtime", evaluation.runtimeState],
                    ["Backend mode", evaluation.backendMode],
                    ["Маршрут", evaluation.isRouteValid ? "valid" : "invalid"],
                    ["Причина", evaluation.routeValidation.reason],
                    ["Logs", evaluation.logsPath],
                  ].map(([label, value]) => (
                    <div key={label} className="rounded-2xl border border-white/10 bg-black/20 p-3">
                      <div className="text-[11px] uppercase tracking-[0.14em] text-white/45">{label}</div>
                      <div className="mt-2 break-words text-sm text-white/84">{value}</div>
                    </div>
                  ))}
                </div>
                <div className="mt-4 flex justify-start">
                  <Button
                    variant="outline"
                    className="rounded-full border-white/10 bg-white/5 px-4 py-2 text-sm text-white hover:bg-white/10"
                    onClick={() => void handleCopyDiagnostics()}
                  >
                    <Copy className="mr-2 h-4 w-4" />
                    Скопировать диагностику
                  </Button>
                </div>
                <div className="mt-3 text-xs text-white/50">
                  {copyState === "copied" ? "Copied" : copyState === "failed" ? "Copy failed" : ""}
                </div>
              </div>
            )}
          </div>

          {bridgeNotice && (
            <div className="mt-4 rounded-2xl border border-white/10 bg-black/20 p-3 text-sm text-white/78">
              {bridgeNotice}
            </div>
          )}

          <div className="mt-6 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <Button
              variant="outline"
              className="rounded-full border-white/10 bg-white/5 px-5 py-3 text-sm text-white hover:bg-white/10"
              onClick={() => triggerRefresh()}
            >
              <RefreshCw className="mr-2 h-4 w-4" />
              Проверить ещё раз
            </Button>
            {onContinue && (
              <Button
                className="rounded-full bg-cyan-300 px-6 py-3 text-sm font-semibold text-slate-950 shadow-[0_12px_30px_rgba(56,189,248,0.22)] transition-colors hover:bg-cyan-200 active:bg-cyan-100"
                disabled={!evaluation.isRouteValid}
                onClick={onContinue}
              >
                {continueLabel}
              </Button>
            )}
          </div>
        </CardContent>
      </Card>

      <Dialog open={isBitrateHelpOpen} onOpenChange={setIsBitrateHelpOpen}>
        <DialogContent showCloseButton={false} className="border-white/10 bg-[#0b0c11] text-white sm:max-w-[560px]">
          <DialogHeader>
            <DialogTitle className="text-white">Как исправить несовпадение битрейта</DialogTitle>
            <DialogDescription className="text-white/70">
              EarLoop не сможет корректно запустить маршрут, пока у Источника захвата и Устройства вывода разный sample rate.
            </DialogDescription>
          </DialogHeader>
          <div className="rounded-2xl border border-white/10 bg-white/5 p-4 text-sm leading-6 text-white/82">
            <div>1. Откройте настройки звука Windows.</div>
            <div>2. Найдите Источник захвата и Устройство вывода. Если используете VB-Cable — ищите его в списке устройств ввода.</div>
            <div>3. Убедитесь, что у обоих устройств установлен одинаковый sample rate (частота дискретизации / битрейт).</div>
            <div>4. После изменения вернитесь в EarLoop и нажмите «Проверить ещё раз».</div>
          </div>
          <DialogFooter className="border-white/10 bg-white/5/40">
            <Button
              variant="outline"
              className="rounded-2xl border-white/10 bg-white/5 text-white hover:bg-white/10"
              onClick={() => void handleOpenSoundSettings()}
            >
              Открыть настройки звука
            </Button>
            <Button
              className="rounded-2xl bg-cyan-300 text-slate-950 hover:bg-cyan-200"
              onClick={() => setIsBitrateHelpOpen(false)}
            >
              Понятно
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={isVBCableInstallOpen} onOpenChange={setIsVBCableInstallOpen}>
        <DialogContent showCloseButton={false} className="border-white/10 bg-[#0b0c11] text-white sm:max-w-[500px]">
          <DialogHeader>
            <DialogTitle className="text-white">Установка VB-Cable</DialogTitle>
            <DialogDescription className="text-white/70">
              После завершения установки нажмите «Продолжить».
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="border-white/10 bg-white/5/40 sm:justify-end">
            <Button
              className="rounded-2xl bg-cyan-300 text-slate-950 hover:bg-cyan-200"
              onClick={() => {
                setIsVBCableInstallOpen(false);
                triggerRefresh();
              }}
            >
              Продолжить
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
