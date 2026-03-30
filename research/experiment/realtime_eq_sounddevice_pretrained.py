

import argparse
import csv
import json
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd

# ВНИМАНИЕ: импорт как у тебя сейчас в проекте
from src.earloop.audio import ABPlayer

try:
    import joblib  # pip install joblib (обычно уже есть через sklearn)
except Exception:
    joblib = None


# --------------------------
# Данные профилей
# --------------------------
DEFAULT_FEATURES = ["bass", "tilt", "presence", "air", "lowmid", "sparkle"]


@dataclass
class Profile:
    profile_id: int
    archetype: str
    preamp_db: float
    feats: np.ndarray          # shape [F]
    freqs_hz: np.ndarray       # shape [23]
    gains_db: np.ndarray       # shape [23]


def read_profiles_csv(path: Path, features: list[str]):
    """
    Читаем profiles_23band.csv (без pandas).
    Важно: feats собираем по списку features (порядок фиксирован).
    """
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []

        # полосы по колонкам вида g_XXX
        gcols = [c for c in cols if c.startswith("g_")]
        if not gcols:
            raise RuntimeError("В profiles CSV не найдены колонки g_*")
        # частоты вытаскиваем из названий g_*
        freqs = np.array([float(c.split("_", 1)[1]) for c in gcols], dtype=float)

        # проверяем, что в CSV есть нужные признаки
        missing = [k for k in features if k not in cols]
        if missing:
            raise RuntimeError(f"В profiles CSV нет колонок признаков: {missing}")

        profs: list[Profile] = []
        for row in reader:
            pid = int(row["profile_id"])
            arch = row.get("archetype", "")
            preamp = float(row.get("preamp_db", 0.0))

            feats = np.array([float(row[k]) for k in features], dtype=float)
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
# Preference model: w + online update
# --------------------------
class OnlinePreferenceModel:
    """
    Онлайн baseline:
    P(A>B) = sigmoid( w · (xA - xB) )

    Обновление w: SGD по логлоссу + L2-рег.
    """
    def __init__(
        self,
        n_feats: int,
        lr: float = 0.15,
        l2: float = 0.01,
        seed: int = 42,
        w_init: np.ndarray | None = None,
        b_init: float = 0.0,
    ):
        self.w = np.zeros((n_feats,), dtype=float) if w_init is None else np.array(w_init, dtype=float).copy()
        self.b = float(b_init)  # интерсепт
        self.lr = float(lr)
        self.l2 = float(l2)
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _sigmoid(z: float) -> float:
        if z >= 0:
            ez = np.exp(-z)
            return 1.0 / (1.0 + ez)
        ez = np.exp(z)
        return ez / (1.0 + ez)

    def prob_a_wins(self, xa: np.ndarray, xb: np.ndarray) -> float:
        x = xa - xb
        return self._sigmoid(float(np.dot(self.w, x) + self.b))

    def score(self, x: np.ndarray) -> float:
        return float(np.dot(self.w, x))

    def update(self, xa: np.ndarray, xb: np.ndarray, y_a_wins: int):
        x = xa - xb
        p = self._sigmoid(float(np.dot(self.w, x) + self.b))
        y = float(y_a_wins)

        grad_w = (p - y) * x
        grad_w = grad_w + self.l2 * self.w
        self.w -= self.lr * grad_w

        grad_b = (p - y)
        self.b -= self.lr * grad_b


def load_pretrained_w(model_path: Path):
    """
    Загружаем .joblib, ожидаем bundle формата:
      {"model": sklearn_model, "features": [...], ...}
    Возвращаем (features, w_init, b_init, raw_bundle)
    """
    if joblib is None:
        raise RuntimeError("Не найден joblib. Установи: pip install joblib")

    bundle = joblib.load(model_path)

    if isinstance(bundle, dict) and "model" in bundle:
        clf = bundle["model"]
        features = bundle.get("features", DEFAULT_FEATURES)
    else:
        clf = bundle
        features = DEFAULT_FEATURES

    if not hasattr(clf, "coef_"):
        raise RuntimeError("Загруженная модель не содержит coef_. Это точно LogisticRegression/SGDClassifier?")

    w = np.array(clf.coef_).reshape(-1)
    b = float(np.array(getattr(clf, "intercept_", [0.0])).reshape(-1)[0])

    return list(features), w, b, bundle


# --------------------------
# Выбор кандидатов A/B
# --------------------------
def pick_pair(profiles: list[Profile], model: OnlinePreferenceModel, explore_prob: float = 0.25):
    X = np.stack([p.feats for p in profiles], axis=0)
    scores = X @ model.w

    best_idx = int(np.argmax(scores))
    A = profiles[best_idx]

    n = len(profiles)
    rng = model.rng
    if rng.random() < explore_prob:
        j = int(rng.integers(0, n))
        if j == best_idx:
            j = (j + 1) % n
        B = profiles[j]
    else:
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


def save_user_vector(out_path: Path, features: list[str], model: OnlinePreferenceModel):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "features": features,
        "w": [float(x) for x in model.w],
        "b": float(model.b),
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] user vector saved: {out_path}")


def print_pair(A: Profile, B: Profile, model: OnlinePreferenceModel, features: list[str]):
    p = model.prob_a_wins(A.feats, B.feats)
    print("\n=== Текущая пара ===")
    print(f"A: id={A.profile_id} | arch={A.archetype} | feats={dict(zip(features, np.round(A.feats, 3)))}")
    print(f"B: id={B.profile_id} | arch={B.archetype} | feats={dict(zip(features, np.round(B.feats, 3)))}")
    print(f"Модель: P(A победит B) ≈ {p:.3f}")
    print("Слушать: 1=A, 2=B | Выбрать: j=A лучше, k=B лучше | n=новая пара | s=сохранить лучший | w=сохранить вектор | q=выход\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", type=str, default="test.wav", help="WAV файл для прослушивания")
    ap.add_argument("--profiles", type=str, default="datasets/synth_quick_V1/profiles_23band.csv", help="CSV с профилями")
    ap.add_argument(
        "--model_path",
        type=str,
        default=r"models/baseline_pairwise_logreg.joblib",
        help="Путь к сохранённой модели .joblib",
    )
    ap.add_argument("--out_presets", type=str, default="exports_listening/presets_user", help="Куда сохранять выбранные пресеты")
    ap.add_argument("--out_uservec", type=str, default="exports_listening/user_vectors", help="Куда сохранять user-вектор (JSON)")
    ap.add_argument("--lr", type=float, default=0.08, help="Шаг обучения (SGD) — обычно меньше, чем с нуля")
    ap.add_argument("--l2", type=float, default=0.01, help="L2-регуляризация")
    ap.add_argument("--explore", type=float, default=0.25, help="Вероятность исследования (рандомный B)")
    ap.add_argument("--block", type=int, default=1024, help="Blocksize для sounddevice")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    wav_path = Path(args.wav)
    if not wav_path.exists():
        raise FileNotFoundError(f"Не найден WAV: {wav_path}")

    profiles_path = Path(args.profiles)
    if not profiles_path.exists():
        raise FileNotFoundError(f"Не найден profiles CSV: {profiles_path}")

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Не найден model_path: {model_path}")

    # 1) грузим pretrained веса
    features, w0, b0, _ = load_pretrained_w(model_path)
    print("[model] loaded:", model_path)
    print("[model] features:", features)
    print("[model] init w shape:", w0.shape, "| b:", b0)

    # 2) грузим профили с тем же порядком фичей
    profiles, _ = read_profiles_csv(profiles_path, features=features)
    if len(profiles) < 2:
        raise RuntimeError("Нужно хотя бы 2 профиля в profiles_23band.csv")

    # 3) онлайн модель, инициализируем w из pretrained
    model = OnlinePreferenceModel(
        n_feats=len(features),
        lr=args.lr,
        l2=args.l2,
        seed=args.seed,
        w_init=w0,
        b_init=b0,
    )

    # первая пара
    A, B = pick_pair(profiles, model, explore_prob=args.explore)

    player = ABPlayer(wav_path, A, B, preamp_db=-6.0, block=args.block)

    def loop_keys():
        nonlocal A, B
        out_dir = Path(args.out_presets)
        out_uv_dir = Path(args.out_uservec)

        print_pair(A, B, model, features)

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
                model.update(A.feats, B.feats, y_a_wins=1)
                print("[vote] выбрано: A лучше B | обновили user-вектор")
            elif ch == "k":
                model.update(A.feats, B.feats, y_a_wins=0)
                print("[vote] выбрано: B лучше A | обновили user-вектор")

            elif ch == "n":
                A, B = pick_pair(profiles, model, explore_prob=args.explore)
                player.set_pair(A, B)
                print_pair(A, B, model, features)

            elif ch == "s":
                X = np.stack([p.feats for p in profiles], axis=0)
                scores = X @ model.w
                best = profiles[int(np.argmax(scores))]
                out_path = out_dir / f"user_preset_{best.profile_id}_23.csv"
                save_preset_csv(out_path, best.freqs_hz, best.gains_db)

            elif ch == "w":
                out_path = out_uv_dir / f"user_vector_seed{args.seed}.json"
                save_user_vector(out_path, features, model)

            elif ch == "q":
                print("Выход.")
                os._exit(0)

    t = threading.Thread(target=loop_keys, daemon=True)
    t.start()

    print("Управление:")
    print("  1 / 2  — слушать A / B")
    print("  j / k  — выбрать A лучше / B лучше (обучение user-вектора)")
    print("  n      — новая пара")
    print("  s      — сохранить лучший пресет (CSV fc_hz,gain_db)")
    print("  w      — сохранить user-вектор (JSON)")
    print("  q      — выход\n")

    with sd.OutputStream(
        samplerate=player.fs,
        blocksize=player.block,
        channels=2,
        dtype="float32",
        callback=player.callback,
    ):
        while True:
            sd.sleep(250)


if __name__ == "__main__":
    main()
