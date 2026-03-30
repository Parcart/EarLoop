import argparse
import csv
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd

from src.earloop.audio import ABPlayer

# --------------------------
# Данные профилей
# --------------------------
FEATURES = ["bass", "tilt", "presence", "air", "lowmid", "sparkle"]


@dataclass
class Profile:
    profile_id: int
    archetype: str
    preamp_db: float
    feats: np.ndarray          # shape [6]
    freqs_hz: np.ndarray       # shape [23]
    gains_db: np.ndarray       # shape [23]


def read_profiles_csv(path: Path):
    """
    Читаем profiles_23band.csv.
    Стараемся не зависеть от pandas (чтобы было проще запускать).
    """
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        # собираем полосы по колонкам вида g_XXX
        gcols = [c for c in cols if c.startswith("g_")]
        # частоты вытаскиваем из названий g_*
        # (они округлённые, но соответствие точных частот хранится в meta.json — для прототипа этого достаточно)
        freqs = np.array([float(c.split("_", 1)[1]) for c in gcols], dtype=float)

        profs = []
        for row in reader:
            pid = int(row["profile_id"])
            arch = row.get("archetype", "")
            preamp = float(row.get("preamp_db", 0.0))

            feats = np.array([float(row[k]) for k in FEATURES], dtype=float)
            gains = np.array([float(row[c]) for c in gcols], dtype=float)

            profs.append(Profile(
                profile_id=pid,
                archetype=arch,
                preamp_db=preamp,
                feats=feats,
                freqs_hz=freqs,
                gains_db=gains,
            ))

    return profs, gcols


# --------------------------
# Online baseline preference model (логрег в виде веса w)
# --------------------------
class OnlinePreferenceModel:
    """
    Минимальный онлайн-baseline:
    P(A>B) = sigmoid( w · (xA - xB) )

    Обновление w: SGD по логлоссу + L2-рег.
    """
    def __init__(self, n_feats: int, lr: float = 0.15, l2: float = 0.01, seed: int = 42):
        self.w = np.zeros((n_feats,), dtype=float)
        self.lr = float(lr)
        self.l2 = float(l2)
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _sigmoid(z: float) -> float:
        # стабильная сигмоида
        if z >= 0:
            ez = np.exp(-z)
            return 1.0 / (1.0 + ez)
        ez = np.exp(z)
        return ez / (1.0 + ez)

    def prob_a_wins(self, xa: np.ndarray, xb: np.ndarray) -> float:
        x = xa - xb
        return self._sigmoid(float(np.dot(self.w, x)))

    def score(self, x: np.ndarray) -> float:
        return float(np.dot(self.w, x))

    def update(self, xa: np.ndarray, xb: np.ndarray, y_a_wins: int):
        """
        Один шаг SGD.
        y_a_wins: 1 если A лучше, 0 если B лучше
        """
        x = xa - xb
        p = self._sigmoid(float(np.dot(self.w, x)))
        y = float(y_a_wins)

        # градиент логлосса: (p - y) * x
        grad = (p - y) * x

        # L2-регуляризация
        grad = grad + self.l2 * self.w

        self.w -= self.lr * grad


# --------------------------
# Выбор кандидатов A/B
# --------------------------
def pick_pair(profiles: list[Profile], model: OnlinePreferenceModel, explore_prob: float = 0.25):
    """
    Простой выбор пары:
    - A = текущий лучший по score(w·x)
    - B = либо случайный из топ-K, либо чисто случайный (исследование)
    """
    X = np.stack([p.feats for p in profiles], axis=0)
    scores = X @ model.w

    best_idx = int(np.argmax(scores))
    A = profiles[best_idx]

    n = len(profiles)
    rng = model.rng
    if rng.random() < explore_prob:
        # исследование: случайный профайл
        j = int(rng.integers(0, n))
        if j == best_idx:
            j = (j + 1) % n
        B = profiles[j]
    else:
        # эксплуатация: берём из топ-20, чтобы B был "похожим конкурентом"
        K = min(20, n)
        top_idx = np.argsort(scores)[-K:]
        j = int(rng.choice(top_idx))
        if j == best_idx:
            j = int(rng.choice(top_idx))
        B = profiles[j]

    return A, B


def save_preset_csv(out_path: Path, freqs_hz: np.ndarray, gains_db: np.ndarray):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fc_hz", "gain_db"])
        for f_hz, g_db in zip(freqs_hz, gains_db):
            w.writerow([float(f_hz), float(g_db)])
    print(f"[save] preset saved: {out_path}")


def print_pair(A: Profile, B: Profile, model: OnlinePreferenceModel):
    p = model.prob_a_wins(A.feats, B.feats)
    print("\n=== Текущая пара ===")
    print(f"A: id={A.profile_id} | arch={A.archetype} | feats={dict(zip(FEATURES, np.round(A.feats, 3)))}")
    print(f"B: id={B.profile_id} | arch={B.archetype} | feats={dict(zip(FEATURES, np.round(B.feats, 3)))}")
    print(f"Модель: P(A победит B) ≈ {p:.3f}")
    print("Слушать: 1=A, 2=B | Выбрать: j=A лучше, k=B лучше | n=новая пара | s=сохранить лучший | q=выход\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", type=str, default="test.wav", help="WAV файл для прослушивания")
    ap.add_argument("--profiles", type=str, default="datasets/synth_quick_V1/profiles_23band.csv", help="CSV с профилями")
    ap.add_argument("--out_presets", type=str, default="exports_listening/presets_user", help="Куда сохранять выбранные пресеты")
    ap.add_argument("--lr", type=float, default=0.15, help="Шаг обучения (SGD)")
    ap.add_argument("--l2", type=float, default=0.01, help="L2-регуляризация")
    ap.add_argument("--explore", type=float, default=0.25, help="Вероятность исследования (рандомный B)")
    ap.add_argument("--block", type=int, default=1024, help="Blocksize для sounddevice")
    args = ap.parse_args()

    wav_path = Path(args.wav)
    if not wav_path.exists():
        raise FileNotFoundError(f"Не найден WAV: {wav_path}")

    profiles_path = Path(args.profiles)
    if not profiles_path.exists():
        raise FileNotFoundError(f"Не найден profiles CSV: {profiles_path}")

    profiles, _ = read_profiles_csv(profiles_path)
    if len(profiles) < 2:
        raise RuntimeError("Нужно хотя бы 2 профиля в profiles_23band.csv")

    model = OnlinePreferenceModel(n_feats=len(FEATURES), lr=args.lr, l2=args.l2)

    # первая пара
    A, B = pick_pair(profiles, model, explore_prob=args.explore)

    player = ABPlayer(wav_path, A, B, preamp_db=-6.0, block=args.block)

    def loop_keys():
        nonlocal A, B, player, model
        out_dir = Path(args.out_presets)

        print_pair(A, B, model)

        while True:
            ch = sys.stdin.read(1)
            if not ch:
                continue

            ch = ch.lower()

            if ch == "1":
                player.set_which("A")
                print("[listen] A")
            elif ch == "2":
                player.set_which("B")
                print("[listen] B")

            elif ch == "j":
                # A лучше
                model.update(A.feats, B.feats, y_a_wins=1)
                print("[vote] выбрано: A лучше B | обновили модель")
            elif ch == "k":
                # B лучше
                model.update(A.feats, B.feats, y_a_wins=0)
                print("[vote] выбрано: B лучше A | обновили модель")

            elif ch == "n":
                A, B = pick_pair(profiles, model, explore_prob=args.explore)
                player.set_pair(A, B)
                print_pair(A, B, model)

            elif ch == "s":
                # сохраняем "текущий лучший" по score
                X = np.stack([p.feats for p in profiles], axis=0)
                scores = X @ model.w
                best = profiles[int(np.argmax(scores))]
                out_path = out_dir / f"user_preset_{best.profile_id}_23.csv"
                save_preset_csv(out_path, best.freqs_hz, best.gains_db)

            elif ch == "q":
                print("Выход.")
                # остановим stream через исключение/выход
                os._exit(0)

    import os
    t = threading.Thread(target=loop_keys, daemon=True)
    t.start()

    print("Управление:")
    print("  1 / 2  — слушать A / B")
    print("  j / k  — выбрать A лучше / B лучше (обучение)")
    print("  n      — новая пара")
    print("  s      — сохранить лучший пресет (CSV fc_hz,gain_db)")
    print("  q      — выход\n")

    with sd.OutputStream(
        samplerate=player.fs,
        blocksize=player.block,
        channels=2,
        dtype="float32",
        callback=player.callback,
    ):
        # держим главный поток живым
        while True:
            sd.sleep(250)


if __name__ == "__main__":
    main()
