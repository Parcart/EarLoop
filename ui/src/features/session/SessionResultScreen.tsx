import { OnboardingTarget } from "@/features/onboarding/OnboardingTarget";
import { EqChart } from "@/components/shared/EqChart";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { curveFromProfileParams } from "@/lib/session/helpers";

import type { ListeningTarget, PerceptualParams } from "@/lib/types/ui";

type SessionResultScreenProps = {
  finalChoice: ListeningTarget;
  profileDraftName: string;
  saveNotice: string | null;
  resultProfileLabel: string;
  resultProfileParams: PerceptualParams;
  pairVersion: number;
  iteration: number;
  progress: number;
  onProfileDraftNameChange: (value: string) => void;
  onSaveProfile: () => void;
  onReturnToSession: () => void;
  onCancelSession: () => void;
};

export function SessionResultScreen({
  finalChoice,
  profileDraftName,
  saveNotice,
  resultProfileLabel,
  resultProfileParams,
  pairVersion,
  iteration,
  progress,
  onProfileDraftNameChange,
  onSaveProfile,
  onReturnToSession,
  onCancelSession,
}: SessionResultScreenProps) {
  return (
    <Card className="rounded-[28px] border border-white/10 bg-[#0b0c11] shadow-none">
      <CardHeader>
        <CardTitle className="text-2xl text-white">Результат</CardTitle>
      </CardHeader>
      <CardContent className="grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
        <div className="space-y-4">
          <OnboardingTarget targetId="result-actions">
            <div className="rounded-[24px] border border-white/10 bg-white/5 p-4">
              <Label htmlFor="profile-name" className="mb-2 block text-sm text-white/78">Название профиля</Label>
              <div className="flex flex-col gap-3 md:flex-row">
                <Input id="profile-name" value={profileDraftName} onChange={(event) => onProfileDraftNameChange(event.target.value)} placeholder="Например, Ночной бас" className="rounded-2xl border-white/10 bg-black/20 text-white placeholder:text-white/35" />
                <Button className="rounded-2xl bg-cyan-300 text-slate-950 shadow-[0_12px_30px_rgba(56,189,248,0.22)] transition-colors hover:bg-cyan-200 active:bg-cyan-100 md:min-w-[180px]" onClick={onSaveProfile}>Сохранить профиль</Button>
              </div>
              <div className="mt-3">
                <div className="flex flex-col gap-3 md:flex-row">
                  <Button variant="outline" className="rounded-2xl border-white/10 bg-white/5 text-white hover:bg-white/10" onClick={onReturnToSession}>Вернуться</Button>
                  <Button variant="outline" className="rounded-2xl border-white/10 bg-white/5 text-white hover:bg-white/10" onClick={onCancelSession}>Отмена</Button>
                </div>
              </div>
              {saveNotice && <p className="mt-3 text-sm text-cyan-200">{saveNotice}</p>}
            </div>
          </OnboardingTarget>

          <div className="rounded-[28px] border border-white/10 bg-white/5 p-5">
            <div className="mb-3 flex items-center justify-between">
              <div>
                <div className="text-sm text-white/50">Выбранный профиль</div>
                <div className="text-xl font-semibold text-white">{resultProfileLabel}</div>
              </div>
              <Badge className="rounded-full bg-white/10 text-white/80">готово к сохранению</Badge>
            </div>
            <EqChart values={curveFromProfileParams(resultProfileParams, pairVersion + 1)} accentShift={pairVersion + 11} />
          </div>

          <div className="grid gap-3 md:grid-cols-1">
            <Button variant="outline" className="rounded-2xl border-white/10 bg-white/5 text-white hover:bg-white/10">Экспорт JSON</Button>
          </div>
        </div>

        <div className="space-y-4">
          <Card className="rounded-[28px] border border-white/10 bg-white/5 shadow-none">
            <CardHeader><CardTitle className="text-lg text-white">Краткая сводка</CardTitle></CardHeader>
            <CardContent className="space-y-3 text-sm text-white/60">
              <div className="flex items-center justify-between rounded-2xl bg-black/20 p-3"><span>Итераций</span><span className="font-medium text-white">{iteration}</span></div>
              <div className="flex items-center justify-between rounded-2xl bg-black/20 p-3"><span>Прогресс</span><span className="font-medium text-white">{progress}%</span></div>
              <div className="flex items-center justify-between rounded-2xl bg-black/20 p-3"><span>Последний выбор</span><span className="font-medium text-white">{resultProfileLabel}</span></div>
            </CardContent>
          </Card>
        </div>
      </CardContent>
    </Card>
  );
}
