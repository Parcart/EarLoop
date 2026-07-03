import { motion } from "framer-motion";

type EqChartProps = {
  values: number[];
  frequencies?: number[];
  accentShift?: number;
};

function formatFrequency(hz: number) {
  return hz >= 1000 ? `${Number((hz / 1000).toFixed(1)).toString()}k` : `${Math.round(hz)}`;
}

export function EqChart({ values, frequencies, accentShift = 0 }: EqChartProps) {
  const maxAbs = Math.max(10, ...values.map((value) => Math.abs(value)));
  const width = 1000;
  const height = 220;
  const paddingX = 24;
  const zeroY = height / 2;
  const innerWidth = width - paddingX * 2;
  const resolvedFrequencies = frequencies?.length === values.length ? frequencies : undefined;
  const minFreq = resolvedFrequencies?.[0] ?? 20;
  const maxFreq = resolvedFrequencies?.[resolvedFrequencies.length - 1] ?? 20000;
  const logMin = Math.log10(minFreq);
  const logMax = Math.log10(maxFreq);
  const xFor = (idx: number) => {
    if (!resolvedFrequencies || resolvedFrequencies.length < 2) {
      return paddingX + (idx / Math.max(1, values.length - 1)) * innerWidth;
    }
    const freq = Math.max(minFreq, Math.min(maxFreq, resolvedFrequencies[idx]));
    return paddingX + ((Math.log10(freq) - logMin) / (logMax - logMin)) * innerWidth;
  };

  const points = values
    .map((v, idx) => {
      const x = xFor(idx);
      const y = zeroY - (v / maxAbs) * (height * 0.34);
      return `${x},${y}`;
    })
    .join(" ");

  const gradientId = `eqLine-${accentShift}`;
  const filterId = `eqGlow-${accentShift}`;

  return (
    <div className="w-full rounded-[22px] border border-white/10 bg-black/25 p-3 backdrop-blur-sm">
      <div className="relative h-44 w-full overflow-hidden rounded-[16px] bg-[linear-gradient(180deg,rgba(255,255,255,0.02),rgba(255,255,255,0.01))]">
        <div className="absolute inset-x-0 top-1/2 h-px bg-white/10" />
        <div className="absolute inset-0 bg-[linear-gradient(180deg,rgba(255,255,255,0.02),transparent)]" />
        <svg viewBox={`0 0 ${width} ${height}`} className="absolute inset-0 h-full w-full" preserveAspectRatio="none">
          <defs>
            <filter id={filterId} x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="6" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
            <linearGradient id={gradientId} x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor={accentShift % 2 === 0 ? "#67e8f9" : "#a5f3fc"} />
              <stop offset="55%" stopColor={accentShift % 2 === 0 ? "#c4b5fd" : "#f0abfc"} />
              <stop offset="100%" stopColor={accentShift % 2 === 0 ? "#a78bfa" : "#c084fc"} />
            </linearGradient>
          </defs>
          <motion.polyline
            points={points}
            fill="none"
            stroke={`url(#${gradientId})`}
            strokeWidth="5"
            strokeLinecap="round"
            strokeLinejoin="round"
            filter={`url(#${filterId})`}
            initial={{ pathLength: 0, opacity: 0.5 }}
            animate={{ pathLength: 1, opacity: 1 }}
            transition={{ duration: 0.65, ease: "easeOut" }}
          />
        </svg>
      </div>
      <div className="mt-3 grid grid-cols-12 gap-2 px-1 text-[10px] text-white/45">
        {(resolvedFrequencies ?? [25, 40, 63, 100, 160, 250, 400, 630, 1000, 2500, 6300, 16000])
          .filter((_, index, arr) => arr.length <= 12 || index % Math.ceil(arr.length / 12) === 0)
          .slice(0, 12)
          .map((hz) => (
          <span key={hz} className="truncate text-center">
            {formatFrequency(hz)}
          </span>
        ))}
      </div>
    </div>
  );
}
