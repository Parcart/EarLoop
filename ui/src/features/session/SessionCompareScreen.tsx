import { AnimatePresence, motion } from "framer-motion";
import { AudioLines, CheckCircle2, CircleHelp, Sparkles } from "lucide-react";
import type { RefObject } from "react";

import { OnboardingTarget } from "@/features/onboarding/OnboardingTarget";
import { useAppOnboardingContext } from "@/features/onboarding/useAppOnboarding";
import { EqChart } from "@/components/shared/EqChart";
import { ParamList } from "@/components/shared/ParamList";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { curveFromParams } from "@/lib/session/helpers";

import type { ListeningTarget, PairData, PairKey, PerceptualParams } from "@/lib/types/ui";

type SessionCompareScreenProps = {
  sessionTutorialMode: boolean;
  iteration: number;
  sessionBaseLabel: string;
  listeningTarget: ListeningTarget;
  isGeneratingPair: boolean;
  pairVersion: number;
  pairA: PairData;
  pairB: PairData;
  sessionBaseParams: PerceptualParams;
  paramsModalFor: PairKey | null;
  modalParams: PerceptualParams | null;
  feedback: string;
  previewApplyStatus: string | null;
  previewError: string | null;
  feedbackAnchorRef: RefObject<HTMLDivElement | null>;
  feedbackSectionRef: RefObject<HTMLDivElement | null>;
  onSelectListeningTarget: (target: ListeningTarget) => void;
  onCardDoubleClick: (target: PairKey) => void;
  onOpenParamsModal: (target: PairKey) => void;
  onCloseParamsModal: () => void;
  onFinish: () => void;
  onReject: () => void;
  onFeedbackChange: (value: string) => void;
};

export function SessionCompareScreen({
  sessionTutorialMode,
  iteration,
  sessionBaseLabel,
  listeningTarget,
  isGeneratingPair,
  pairVersion,
  pairA,
  pairB,
  sessionBaseParams,
  paramsModalFor,
  modalParams,
  feedback,
  previewApplyStatus,
  previewError,
  feedbackAnchorRef,
  feedbackSectionRef,
  onSelectListeningTarget,
  onCardDoubleClick,
  onOpenParamsModal,
  onCloseParamsModal,
  onFinish,
  onReject,
  onFeedbackChange,
}: SessionCompareScreenProps) {
  const { emitEvent } = useAppOnboardingContext();
  const currentListeningLabel = listeningTarget === "base" ? "прошлый выбор" : `вариант ${listeningTarget}`;

  return (
    <Card className="w-full rounded-[28px] border border-white/10 bg-[#0b0c11] shadow-none">
      <CardHeader className="px-4 text-left md:px-5 lg:px-6">
        <div className="space-y-1.5 text-left">
          <CardTitle className="text-2xl text-white">Создание нового профиля</CardTitle>
          <CardDescription className="text-white/78">Выбери карточку для прослушивания. Активная карточка подсвечивается, а при генерации новой пары интерфейс плавно обновляется.</CardDescription>
        </div>
      </CardHeader>
      <CardContent className="relative px-4 pb-5 md:px-5 lg:px-6">
        <div className="space-y-6">

        {sessionTutorialMode && (
          <div className="rounded-[24px] border border-cyan-300/20 bg-cyan-400/10 p-4 text-left text-sm text-white/82">
            <div className="font-semibold text-cyan-200">Подсказка по сессии</div>
            <div className="mt-2 space-y-2">
              <p>Один клик по карточке включает прослушивание варианта. Двойной клик сразу выбирает его и запускает следующую пару с обратной связью «Без дополнительного комментария».</p>
              <p>Если ни один из вариантов не подходит, можно вернуться к кнопке «Слушать прошлый выбор» и оставить базовый профиль активным.</p>
              <p>Ниже расположен раздел «Обратная связь»: в нём можно указать, чего не хватает в звучании, а затем сгенерировать новую пару.</p>
              <p>Если этот раздел не виден на экране, воспользуйся боковой кнопкой со стрелкой вниз — она быстро переместит к нему.</p>
            </div>
          </div>
        )}

        <div className="flex flex-wrap items-center gap-3">
          <Badge className="rounded-full bg-white/10 text-white/85">Итерация {iteration}</Badge>
          <Badge className="rounded-full border border-white/10 bg-white/5 text-white/70">База: {sessionBaseLabel}</Badge>
          <Badge className="rounded-full border border-cyan-300/20 bg-cyan-400/15 text-cyan-200">Сейчас играет: {currentListeningLabel}</Badge>
          {previewApplyStatus && <Badge className={`rounded-full border ${previewApplyStatus === "applied" ? "border-emerald-300/20 bg-emerald-400/15 text-emerald-200" : previewApplyStatus === "failed" ? "border-red-300/20 bg-red-400/15 text-red-200" : "border-white/10 bg-white/5 text-white/70"}`}>Preview: {previewApplyStatus}</Badge>}
          {isGeneratingPair && <Badge className="rounded-full border border-fuchsia-300/20 bg-fuchsia-400/15 text-fuchsia-200">Генерация следующей пары...</Badge>}
        </div>
        {previewError && <div className="text-sm text-red-200/90">{previewError}</div>}

        <OnboardingTarget targetId="compare-cards">
          <div>
            <AnimatePresence mode="wait">
              <motion.div key={`pair-set-${pairVersion}`} initial={{ opacity: 0, y: 18, scale: 0.985 }} animate={{ opacity: 1, y: 0, scale: 1 }} exit={{ opacity: 0, y: -18, scale: 0.985 }} transition={{ duration: 0.45, ease: "easeOut" }} className="grid gap-3 md:grid-cols-2 xl:gap-4 2xl:gap-5">
                {[
                  { title: "Вариант A", data: pairA, active: listeningTarget === "A", onClick: () => !isGeneratingPair && onSelectListeningTarget("A"), onDoubleClick: () => onCardDoubleClick("A") },
                  { title: "Вариант B", data: pairB, active: listeningTarget === "B", onClick: () => !isGeneratingPair && onSelectListeningTarget("B"), onDoubleClick: () => onCardDoubleClick("B") },
                ].map((item, index) => (
                  <motion.button
                    key={`${item.title}-${pairVersion}`}
                    type="button"
                    onClick={() => {
                      emitEvent("session.compare.card.clicked");
                      item.onClick();
                    }}
                    onDoubleClick={item.onDoubleClick}
                    className="block h-full w-full text-left"
                    initial={{ opacity: 0, y: 24, filter: "blur(10px)" }}
                    animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
                    transition={{ delay: index * 0.08, duration: 0.5, ease: "easeOut" }}
                  >
                    <Card className={`relative h-full overflow-hidden rounded-[28px] border bg-white/5 shadow-none transition-all hover:border-white/25 ${item.active ? "border-cyan-300 shadow-[0_0_0_1px_rgba(103,232,249,0.35)]" : "border-white/10"}`}>
                      {isGeneratingPair && <motion.div className="absolute inset-0 bg-[linear-gradient(120deg,transparent_10%,rgba(255,255,255,0.08)_45%,transparent_80%)]" initial={{ x: "-120%" }} animate={{ x: "120%" }} transition={{ repeat: Infinity, duration: 1.1, ease: "linear" }} />}
                      <CardHeader>
                        <div className="flex items-center justify-between gap-3">
                          <CardTitle className="text-xl text-white">{item.title}</CardTitle>
                          <div className="flex items-center gap-2">
                            <Badge className="rounded-full bg-white/10 text-white/75">score {item.data.score.toFixed(2)}</Badge>
                            <Button
                              type="button"
                              size="icon"
                              variant="outline"
                              className="h-8 w-8 rounded-full border-white/10 bg-white/5 text-white/80 hover:bg-white/10 hover:text-white"
                              onClick={(event) => {
                                event.stopPropagation();
                                onOpenParamsModal(item.data.id);
                              }}
                            >
                              <CircleHelp className="h-4 w-4" />
                            </Button>
                          </div>
                        </div>
                      </CardHeader>
                      <CardContent className="space-y-4 pt-1">
                        <EqChart values={curveFromParams(item.data.id, pairVersion + 1, sessionBaseParams)} accentShift={pairVersion + index} />
                      </CardContent>
                    </Card>
                  </motion.button>
                ))}
              </motion.div>
            </AnimatePresence>
          </div>
        </OnboardingTarget>

        <OnboardingTarget targetId="compare-toolbar">
          <div className="grid gap-3 rounded-[24px] border border-white/10 bg-white/5 p-4 md:grid-cols-2">
            <Button variant={listeningTarget === "base" ? "default" : "outline"} className={listeningTarget === "base" ? "rounded-2xl bg-cyan-300 text-slate-950 hover:bg-cyan-200" : "rounded-2xl border-white/10 bg-white/5 text-white hover:bg-white/10"} onClick={() => onSelectListeningTarget("base")}><AudioLines className="mr-2 h-4 w-4" /> Слушать прошлый выбор</Button>
            <OnboardingTarget targetId="finish-button">
              <Button
                variant="outline"
                className="rounded-2xl border-white/10 bg-white/5 text-white hover:bg-white/10"
                onClick={() => {
                  emitEvent("session.finish.clicked");
                  onFinish();
                }}
              >
                <CheckCircle2 className="mr-2 h-4 w-4" /> Завершить
              </Button>
            </OnboardingTarget>
          </div>
        </OnboardingTarget>

        <div ref={feedbackAnchorRef} className="flex items-center gap-3">
          <div className="h-px flex-1 bg-white/10" />
          <span className="text-xs font-semibold uppercase tracking-[0.22em] text-white/40">{"Якорь :>"}</span>
          <div className="h-px flex-1 bg-white/10" />
        </div>

        <Dialog open={paramsModalFor !== null} onOpenChange={(open) => !open && onCloseParamsModal()}>
          <DialogContent className="border-white/10 bg-[#0b0c11] text-white sm:max-w-[520px]">
            <DialogHeader>
              <DialogTitle className="text-white">Параметры {paramsModalFor === "A" ? "варианта A" : paramsModalFor === "B" ? "варианта B" : "профиля"}</DialogTitle>
              <DialogDescription className="text-white/70">Детальное значение perceptual-параметров выбранной карточки.</DialogDescription>
            </DialogHeader>
            {modalParams && <ParamList params={modalParams} />}
          </DialogContent>
        </Dialog>

        <OnboardingTarget targetId="feedback-panel">
          <div ref={feedbackSectionRef}>
          <Card className="rounded-[28px] border border-dashed border-white/10 bg-white/5 shadow-none">
            <CardHeader className="text-left">
              <CardTitle className="text-lg text-white">Обратная связь</CardTitle>
              <CardDescription className="text-white/78">Отметь, чего не хватает в текущей паре. При генерации следующей пары текущий выбранный вариант станет новой базой для кнопки «Слушать».</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-6 lg:grid-cols-2">
              <div className="space-y-4">
                <RadioGroup value={feedback} onValueChange={onFeedbackChange}>
                  {[
                    ["none", "Без дополнительного комментария"],
                    ["more_bass", "Хочется больше баса"],
                    ["less_harsh", "Слишком резко"],
                    ["more_air", "Хочется больше воздуха"],
                    ["warmer", "Сделать теплее"],
                  ].map(([value, label]) => (
                    <label key={value} htmlFor={value} className="flex cursor-pointer items-center space-x-3 rounded-2xl border border-white/12 bg-black/25 p-3 text-white transition hover:border-cyan-300/35 hover:bg-white/5">
                      <RadioGroupItem value={value} id={value} className="border-slate-300 text-cyan-300 data-[state=checked]:border-cyan-300 data-[state=checked]:bg-cyan-300" />
                      <span className="text-sm font-medium text-white">{label}</span>
                    </label>
                  ))}
                </RadioGroup>
              </div>
              <div className="space-y-3">
                <Button
                  className="w-full rounded-2xl border-white/10 bg-white/5 text-white hover:bg-white/10"
                  variant="outline"
                  onClick={() => {
                    emitEvent("session.generate-next.clicked");
                    onReject();
                  }}
                  disabled={isGeneratingPair}
                >
                  <Sparkles className="mr-2 h-4 w-4" /> {isGeneratingPair ? "Генерация..." : "Сгенерировать следующую пару"}
                </Button>
                <div className="rounded-2xl bg-black/20 p-4 text-sm text-white/80">Активное прослушивание: <span className="font-medium text-white">{currentListeningLabel}</span><br />Причина: <span className="font-medium text-cyan-200">{feedback === "none" ? "не указана" : feedback}</span></div>
              </div>
            </CardContent>
          </Card>
          </div>
        </OnboardingTarget>
        </div>
      </CardContent>
    </Card>
  );
}
