import { motion } from "framer-motion";

import type { WavePreset } from "@/lib/types/ui";

type AnimatedWaveProps = {
  enabled: boolean;
  isPlaying: boolean;
  preset: WavePreset;
};

export function AnimatedWave({ enabled, isPlaying, preset }: AnimatedWaveProps) {
  const activeOpacity = enabled ? 1 : 0.44;
  const accentOpacity = enabled ? 0.75 : 0.18;
  const scalePulse = isPlaying ? [1, 1.03, 0.98, 1.04, 1] : [1, 1.01, 1];
  const presetScale = preset === "compact" ? 0.84 : preset === "expanded" ? 1.16 : 1;
  const stageSize = preset === "compact" ? 760 : preset === "expanded" ? 980 : 860;

  return (
    <div className="absolute inset-0 overflow-hidden rounded-[32px] bg-[#05060a]">
      <motion.div className="absolute inset-0" animate={{ opacity: enabled ? 1 : 0.96 }} transition={{ duration: 0.45, ease: "easeInOut" }}>
        <motion.div
          className="absolute left-1/2 top-1/2"
          initial={false}
          animate={{
            scale: presetScale * (enabled ? 1 : 0.98),
            filter: enabled ? "grayscale(0) saturate(1.2) brightness(1)" : "grayscale(1) saturate(0.08) brightness(0.74)",
          }}
          transition={{ duration: 0.5, ease: "easeInOut" }}
          style={{ width: stageSize, height: stageSize, x: "-50%", y: "-50%", transformOrigin: "center center" }}
        >
          <motion.div className="absolute left-[18%] top-[10%] h-[50%] w-[42%] rounded-full bg-fuchsia-500/75 blur-3xl" animate={{ x: [0, 24, -8, 0], y: [0, -18, 14, 0], scale: scalePulse, opacity: activeOpacity }} transition={{ repeat: Infinity, duration: isPlaying ? 7 : 10, ease: "easeInOut" }} style={{ filter: "blur(55px)" }} />
          <motion.div className="absolute left-[30%] top-[8%] h-[64%] w-[40%] rounded-[42%] bg-cyan-400/85 blur-3xl" animate={{ x: [0, -18, 14, 0], y: [0, 24, -12, 0], scale: isPlaying ? [1, 1.08, 0.95, 1] : [1, 1.02, 1], opacity: activeOpacity }} transition={{ repeat: Infinity, duration: isPlaying ? 6.4 : 9.8, ease: "easeInOut" }} style={{ filter: "blur(48px)" }} />
          <motion.div className="absolute left-[24%] top-[52%] h-[30%] w-[34%] rounded-full bg-blue-600/70 blur-3xl" animate={{ x: [0, 18, -6, 0], y: [0, -24, 16, 0], scale: isPlaying ? [1, 0.95, 1.08, 1] : [1, 1.02, 1], opacity: activeOpacity }} transition={{ repeat: Infinity, duration: isPlaying ? 5.8 : 9.2, ease: "easeInOut" }} style={{ filter: "blur(46px)" }} />
          <motion.div className="absolute right-[16%] top-[18%] h-[46%] w-[18%] rounded-full bg-pink-500/45 blur-3xl" animate={{ rotate: [0, 18, -10, 0], scaleY: isPlaying ? [1, 1.15, 0.92, 1.1, 1] : [1, 1.03, 1], opacity: accentOpacity }} transition={{ repeat: Infinity, duration: isPlaying ? 4.5 : 8.5, ease: "easeInOut" }} style={{ filter: "blur(42px)" }} />
        </motion.div>
      </motion.div>

      <motion.div className="absolute inset-0 bg-[radial-gradient(circle_at_center,transparent_0%,transparent_45%,rgba(0,0,0,0.42)_100%)]" animate={{ opacity: enabled ? 1 : 0.95 }} transition={{ duration: 0.45, ease: "easeInOut" }} />
      <motion.div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(255,255,255,0.12)_0%,rgba(203,213,225,0.08)_28%,rgba(148,163,184,0.03)_50%,rgba(255,255,255,0)_66%)]" animate={{ opacity: enabled ? 0 : 0.42 }} transition={{ duration: 0.45, ease: "easeInOut" }} />
    </div>
  );
}
