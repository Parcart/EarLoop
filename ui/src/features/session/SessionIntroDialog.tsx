import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";

type SessionIntroDialogProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSkip: () => void;
  onStartTutorial: () => void;
};

export function SessionIntroDialog({ open, onOpenChange, onSkip, onStartTutorial }: SessionIntroDialogProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="border-white/10 bg-[#0b0c11] text-white sm:max-w-[520px]">
        <DialogHeader>
          <DialogTitle className="text-white">Пройти обучение по созданию профиля?</DialogTitle>
          <DialogDescription className="text-white/70">Мы можем кратко показать, как работает сценарий A/B-сравнения и быстрый переход к следующей паре.</DialogDescription>
        </DialogHeader>
        <div className="flex flex-col gap-3 sm:flex-row sm:justify-end">
          <Button variant="outline" className="rounded-2xl border-white/10 bg-white/5 text-white hover:bg-white/10" onClick={onSkip}>Пропустить</Button>
          <Button className="rounded-2xl" onClick={onStartTutorial}>Пройти обучение</Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
