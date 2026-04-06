import { motion } from "framer-motion";
import { AudioLines } from "lucide-react";

import { AnimatedWave } from "@/components/shared/AnimatedWave";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

import type { SavedProfile, WavePreset } from "@/lib/types/ui";

type HomeScreenProps = {
  isEnabled: boolean;
  isPlaying: boolean;
  homeWavePreset: WavePreset;
  saveNotice: string | null;
  activeProfileName: string | null;
  selectedProfile: string;
  savedProfiles: SavedProfile[];
  showIntroHint: boolean;
  processingStatus: string | null;
  processingError: string | null;
  onSelectProfile: (value: string) => void;
  onToggleHome: () => void;
};

export function HomeScreen({
  isEnabled,
  isPlaying,
  homeWavePreset,
  saveNotice,
  activeProfileName,
  selectedProfile,
  savedProfiles,
  showIntroHint,
  processingStatus,
  processingError,
  onSelectProfile,
  onToggleHome,
}: HomeScreenProps) {
  const statusTone = processingStatus === "applied"
    ? "bg-emerald-400/12 text-emerald-200"
    : processingStatus === "bypass"
      ? "bg-white/8 text-white/70"
      : processingStatus === "failed"
        ? "bg-red-500/14 text-red-200"
        : "bg-amber-400/12 text-amber-200";

  return (
    <div className="h-full min-h-0">
      <Card className="relative h-full min-h-0 overflow-hidden rounded-[28px] border border-white/10 bg-black/70 shadow-none">
        <AnimatedWave enabled={isEnabled} isPlaying={isPlaying} preset={homeWavePreset} />
        <CardContent className="relative z-10 flex h-full min-h-0 flex-col items-center justify-center gap-6 px-5 py-6 md:px-8 md:py-8">
          {saveNotice && <div className="w-full max-w-[min(100%,72rem)] rounded-2xl border border-cyan-300/20 bg-cyan-400/10 px-4 py-3 text-center text-sm text-cyan-200 backdrop-blur-md">{saveNotice}</div>}

          <div className="flex max-w-full flex-wrap items-center justify-center gap-3 rounded-full border border-white/10 bg-white/6 px-3 py-2 backdrop-blur-md">
            {processingStatus && <Badge className={`rounded-full border-0 ${statusTone}`}>{processingStatus}</Badge>}
            <Select value={selectedProfile} onValueChange={onSelectProfile}>
              <SelectTrigger className="h-10 w-[min(22rem,calc(100vw-9rem))] min-w-[200px] rounded-full border-0 bg-transparent px-4 text-center text-white shadow-none ring-0 focus:ring-0">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {savedProfiles.map((profile) => (
                  <SelectItem key={profile.id} value={profile.id}>
                    {profile.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <button type="button" onClick={onToggleHome} className="group flex flex-col items-center gap-5 text-center">
            <div className={`flex h-28 w-28 items-center justify-center rounded-full border backdrop-blur-md ${isEnabled ? "border-white/20 bg-white/10 text-white shadow-[0_0_0_10px_rgba(255,255,255,0.03),0_0_48px_rgba(103,232,249,0.08)]" : "border-white/10 bg-white/5 text-white/45"}`}>
              <AudioLines className="h-12 w-12" />
            </div>

            {showIntroHint && (
              <motion.p initial={false} animate={{ opacity: isEnabled ? 0.72 : 0.42, y: isEnabled ? 0 : 2 }} transition={{ duration: 0.35, ease: "easeInOut" }} className="max-w-2xl text-sm text-white md:text-base" style={{ filter: "none" }}>
                Клик по логотипу включает и выключает обработку.
              </motion.p>
            )}
          </button>

          {processingError && (
            <div className="max-w-2xl text-center">
              <p className="text-sm text-red-200/90">{processingError}</p>
            </div>
          )}

          <div className="absolute bottom-6 right-6 rounded-full bg-white/10 px-3 py-1 text-sm font-semibold text-white/70 backdrop-blur-md">{__APP_VERSION__}</div>
        </CardContent>
      </Card>
    </div>
  );
}
