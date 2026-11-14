r"""Где рвётся цепь наведения на запах в ТЕЛЕ.

Замер (`BANC_MOTOR.md` §12) показал: муха не идёт на источник ни с
какими весами декодера, медиана конечной дистанции около ста
миллиметров во всех условиях, включая контроль с переставленными
антеннами. Значит дело не в адресации нисходящих. Эта проба ищет, на
каком звене теряется сигнал, и не гадает: каждое звено сравнивается со
следующим числом.

Цепь, по звеньям:

    пеленг на источник -> асимметрия запаха на антеннах
      -> асимметрия ORN -> асимметрия DNa02 -> асимметрия команды
      -> угловая скорость -> изменение пеленга

Если первое звено уже нулевое, дело в сенсорике (антенны разнесены на
треть миллиметра при шлейфе шириной двенадцать). Если первое есть, а
третье нет — сигнал теряется в сети. И так далее: рвётся ровно там, где
корреляция падает.

Пеленг считается знаковым и в осях мухи: положительный — источник
слева. Корреляции считаются по шагам после разогрева, по Пирсону, и
рядом печатается размах по эпизодам: одна корреляция на одном эпизоде —
не замер.

    cd D:\fly-brain\lab
    set PYTHONIOENCODING=utf-8
    ..\.venv\Scripts\python.exe probe\odour_chain.py --эпизодов 3 --секунд 10
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from flybrain import data as D  # noqa: E402

ОТЧЁТ = Path(__file__).parent / "odour_chain.json"
CONTROL_MS = 20.0
TAU_S = 0.5


def сгладить(v, tau_s=0.5, dt_s=CONTROL_MS / 1000.0):
    """Два звена экспоненциального сглаживания.

    Нужны они вот зачем. Походка работает на 12 Гц, то есть период
    четыре шага управления, и угловая скорость на шаге — это прежде
    всего качка корпуса, а не ответ на команду. Корреляция по сырым
    шагам меряет её, а не цепь; именно на этом уже попалась камера
    (`worker.py`, КАМЕРА_TAU). Поэтому каждое звено считается и по
    сырым рядам, и по сглаженным, и смотреть надо на вторые.
    """
    a = 1.0 - np.exp(-dt_s / tau_s)
    v = np.asarray(v, float)
    out = v.copy()
    for _ in range(2):
        s = out[0]
        для = np.empty_like(out)
        for i, x in enumerate(out):
            s += a * (x - s)
            для[i] = s
        out = для
    return out


def кор(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def эпизод(neurons, weights, device, seed, секунд, без_запаха=False):
    """`без_запаха` уносит источники за километр: вход падает в ноль.

    Это решающий контроль ко всему разбору. Если при нулевом входе
    `DNa02` всё равно даёт свои 57 Гц слева против 1,3 справа, то её
    асимметрия — собственное свойство модели, а не ответ на запах, и
    тогда декодер весь эпизод рулит по постоянному перекосу.
    """
    from flybrain import body as B

    cfg = B.BodyConfig(duration_s=секунд, control_ms=CONTROL_MS, seed=seed,
                       device=device, readout="ALPN", motor="dn_linear")
    src = ([[1.0e6, 1.0e6, 1.5], [1.0e6, -1.0e6, 1.5]]
           if без_запаха else None)
    t0 = time.perf_counter()
    run = B.run_episode(cfg=cfg, neurons=neurons, weights=weights,
                        sources=src, progress=False)
    recs = run.records
    src = np.asarray(run.sources, float)[:, :2]
    skip = min(int(round(3 * TAU_S / (CONTROL_MS / 1000.0))), len(recs) // 3)

    x = np.array([r["x"] for r in recs], float)
    y = np.array([r["y"] for r in recs], float)
    th = np.array([r["theta"] for r in recs], float)
    # Ближайший источник НА КАЖДОМ шаге: их два, и «тот самый» меняется.
    d = np.linalg.norm(src[None, :, :] - np.stack([x, y], 1)[:, None, :],
                       axis=2)
    ближ = d.argmin(axis=1)
    дист = d[np.arange(len(x)), ближ]
    цель = src[ближ]
    # Пеленг в осях мухи: + слева.
    bx, by = цель[:, 0] - x, цель[:, 1] - y
    пеленг = np.arctan2(by * np.cos(th) - bx * np.sin(th),
                        bx * np.cos(th) + by * np.sin(th))
    ключ = {k: k for k in ("odour_l", "odour_r", "orn_l", "orn_r",
                           "amp_l", "amp_r")}
    зв = {к: np.array([r.get(к, np.nan) for r in recs], float) for к in ключ}
    дн_l = np.array([r.get("readout_DNa02_l", np.nan) for r in recs], float)
    дн_r = np.array([r.get("readout_DNa02_r", np.nan) for r in recs], float)
    зап_ас = зв["odour_l"] - зв["odour_r"]
    orn_ас = зв["orn_l"] - зв["orn_r"]
    дн_ас = дн_l - дн_r
    ком_ас = зв["amp_l"] - зв["amp_r"]
    dth = np.diff(np.unwrap(th), prepend=np.unwrap(th)[0]) / (CONTROL_MS / 1000.0)

    s = slice(skip, None)
    зап_отн = зап_ас[s] / np.maximum(зв["odour_l"][s] + зв["odour_r"][s], 1e-12)
    из = {
        "зерно": seed,
        "дистанция: старт %.1f, наименьшая %.1f, конец %.1f" % (
            дист[0], дист.min(), дист[-1]): None,
        "дистанция старт, мм": float(дист[0]),
        "дистанция наименьшая, мм": float(дист.min()),
        "дистанция конец, мм": float(дист[-1]),
        "шаг до наименьшей": int(дист.argmin()),
        "шагов": len(recs),
        "запах слева, среднее": float(np.nanmean(зв["odour_l"][s])),
        "запах справа, среднее": float(np.nanmean(зв["odour_r"][s])),
        "асимметрия запаха, относительная медиана": float(
            np.nanmedian(np.abs(зап_отн))),
        "DNa02 слева, Гц": float(np.nanmean(дн_l[s])),
        "DNa02 справа, Гц": float(np.nanmean(дн_r[s])),
        "корреляции сглаженные": {
            "пеленг -> асимметрия запаха": кор(сгладить(пеленг[s]),
                                               сгладить(зап_ас[s])),
            "асимметрия запаха -> асимметрия ORN": кор(сгладить(зап_ас[s]),
                                                       сгладить(orn_ас[s])),
            "асимметрия ORN -> асимметрия DNa02": кор(сгладить(orn_ас[s]),
                                                      сгладить(дн_ас[s])),
            "асимметрия DNa02 -> асимметрия команды": кор(сгладить(дн_ас[s]),
                                                          сгладить(ком_ас[s])),
            "асимметрия команды -> угловая скорость": кор(сгладить(ком_ас[s]),
                                                          сгладить(dth[s])),
            "пеленг -> угловая скорость (вся цепь)": кор(сгладить(пеленг[s]),
                                                         сгладить(dth[s])),
        },
        "корреляции": {
            "пеленг -> асимметрия запаха": кор(пеленг[s], зап_ас[s]),
            "асимметрия запаха -> асимметрия ORN": кор(зап_ас[s], orn_ас[s]),
            "асимметрия ORN -> асимметрия DNa02": кор(orn_ас[s], дн_ас[s]),
            "асимметрия DNa02 -> асимметрия команды": кор(дн_ас[s], ком_ас[s]),
            "асимметрия команды -> угловая скорость": кор(ком_ас[s], dth[s]),
            "пеленг -> угловая скорость (вся цепь)": кор(пеленг[s], dth[s]),
        },
    }
    из.pop("дистанция: старт %.1f, наименьшая %.1f, конец %.1f" % (
        дист[0], дист.min(), дист[-1]), None)
    из["стена, с"] = round(time.perf_counter() - t0, 1)
    return из


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--эпизодов", type=int, default=3)
    ap.add_argument("--секунд", type=float, default=10.0)
    ap.add_argument("--без-запаха", action="store_true",
                    dest="без_запаха",
                    help="контроль: источники унесены, вход нулевой")
    args = ap.parse_args()
    import torch

    neurons = D.load_neurons()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = D.load_weights(device=device)
    из = []
    for i in range(int(getattr(args, "эпизодов"))):
        e = эпизод(neurons, weights, device, 300 + i,
                   float(getattr(args, "секунд")),
                   без_запаха=bool(getattr(args, "без_запаха")))
        из.append(e)
        print("зерно %d: дистанция %.1f -> наименьшая %.1f (шаг %d) -> %.1f, "
              "запах Л %.5f П %.5f, DNa02 %.1f/%.1f Гц (%.0f с)"
              % (e["зерно"], e["дистанция старт, мм"],
                 e["дистанция наименьшая, мм"], e["шаг до наименьшей"],
                 e["дистанция конец, мм"], e["запах слева, среднее"],
                 e["запах справа, среднее"], e["DNa02 слева, Гц"],
                 e["DNa02 справа, Гц"], e["стена, с"]), flush=True)

    print("\nзвено за звеном, корреляция по шагам (медиана и размах по "
          "эпизодам):")
    for имя in из[0]["корреляции"]:
        v = np.array([e["корреляции"][имя] for e in из], float)
        print("  %-42s %+.3f   (%+.3f…%+.3f)"
              % (имя, np.nanmedian(v), np.nanmin(v), np.nanmax(v)))
    отн = np.array([e["асимметрия запаха, относительная медиана"]
                    for e in из], float)
    print("\nотносительная асимметрия запаха между антеннами: медиана "
          "%.4f %%" % (100.0 * np.nanmedian(отн)))
    ОТЧЁТ.write_text(json.dumps({"эпизоды": из}, ensure_ascii=False, indent=1),
                     encoding="utf-8")
    print("отчёт: %s" % ОТЧЁТ.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
