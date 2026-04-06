import type { PairData, PairKey, PerceptualParams } from "@/lib/types/ui";
import { defaultProfileParams, eqBands } from "@/lib/mock/profiles";

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function pseudoRandom(seed: number) {
  const x = Math.sin(seed * 12.9898) * 43758.5453;
  return x - Math.floor(x);
}

export function buildPair(seed: number, key: PairKey, baseParams: PerceptualParams = defaultProfileParams): PairData {
  const keyOffset = key === "A" ? 0.71 : 1.37;
  const value = (n: number, scale = 1) => (pseudoRandom(seed + keyOffset + n) * 2 - 1) * scale;

  const params: PerceptualParams = {
    bass: Number(clamp(baseParams.bass + value(0.11, 0.42), -1, 1).toFixed(2)),
    tilt: Number(clamp(baseParams.tilt + value(0.29, 0.24), -1, 1).toFixed(2)),
    presence: Number(clamp(baseParams.presence + value(0.43, 0.38), -1, 1).toFixed(2)),
    air: Number(clamp(baseParams.air + value(0.67, 0.32), -1, 1).toFixed(2)),
    lowmid: Number(clamp(baseParams.lowmid + value(0.91, 0.28), -1, 1).toFixed(2)),
    sparkle: Number(clamp(baseParams.sparkle + value(1.17, 0.35), -1, 1).toFixed(2)),
  };

  const scoreRaw =
    params.bass * 0.42 +
    params.presence * 0.36 +
    params.sparkle * 0.28 -
    Math.abs(params.tilt) * 0.18 -
    Math.abs(params.lowmid) * 0.08 +
    (key === "A" ? 0.24 : -0.06);

  return { id: key, params, score: Number(scoreRaw.toFixed(2)) };
}

export function curveFromProfileParams(params: PerceptualParams, seed = 0) {
  const { bass, tilt, presence, air, lowmid, sparkle } = params;

  return eqBands.map((freq, i) => {
    const t = i / (eqBands.length - 1);
    const lowShape = Math.exp(-Math.pow((t - 0.12) / 0.16, 2)) * bass * 6;
    const lowmidShape = Math.exp(-Math.pow((t - 0.33) / 0.14, 2)) * lowmid * 4.5;
    const presenceShape = Math.exp(-Math.pow((t - 0.66) / 0.13, 2)) * presence * 4.8;
    const airShape = Math.exp(-Math.pow((t - 0.87) / 0.15, 2)) * air * 4.2;
    const sparkleShape = Math.exp(-Math.pow((t - 0.95) / 0.11, 2)) * sparkle * 3.8;
    const tiltShape = (t - 0.5) * tilt * 8;
    const ripple = Math.sin((freq / 16000) * Math.PI * 6 + seed * 0.3) * 0.35;
    return clamp(lowShape + lowmidShape + presenceShape + airShape + sparkleShape + tiltShape + ripple, -10, 10);
  });
}

export function curveFromParams(kind: PairKey, seed: number, baseParams: PerceptualParams = defaultProfileParams) {
  return curveFromProfileParams(buildPair(seed, kind, baseParams).params, seed);
}
