"""Обучение декодера варианта Б. Отдельный файл весов с sha256.

Что здесь обучается и на чём
----------------------------
Вектор нормированных на собственный шум асимметрий шести прямых зондов
(DNp01, DNp04, DNa01, DNa02, DNp09, MDN) → скалярный сигнал поворота.
Гребневая регрессия, цель — ИЗВЕСТНАЯ асимметрия обонятельного входа,
которым гоняется сеть. То есть это декодер стимула из популяции в
обычном нейробиологическом смысле: «что нисходящие знают о том, какая
антенна в шлейфе».

Тела в обучении нет нарочно. Во-первых, физика стоит 21,6 с настенных на
секунду модельного времени, и набрать на ней обучающую выборку значит
считать часы. Во-вторых — и это важнее — вход декодера не зависит от
тела: он зависит от того, чем накачана сеть. Обучать на теле значило бы
примешать в веса то, куда муха успела повернуться, а это уже не декодер
популяции.

Чего у этого обучения НЕТ, и это надо сказать
---------------------------------------------
Опубликованной выборки «активность нисходящих → углы суставов» не
существует: ни в FlyWire, ни в наборах NeuroMechFly, ни в трёх
репозиториях апстрима. Поэтому обучить декодер на ИЗМЕРЕННОМ моторном
выходе нельзя — обучать можно только на том, что мы сами же и подали на
вход. Цель обучения тут не внешняя истина, а наш собственный стимул, и
хорошая точность такого декодера НЕ доказывает, что команда идёт из
сети. Доказывает это встречный замер с маской (`probe/motor_check.py`).

Проверка обязательна и устроена как положено: веса ищутся на одних
зёрнах, качество считается на ДРУГОМ, отложенном. Иначе r² измерял бы
способность запомнить, а не предсказать.

    PYTHONIOENCODING=utf-8 FLYBRAIN_OUT=<свой каталог> \\
        .venv/Scripts/python.exe lab/probe/motor_train.py
"""

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

from probe.motor_check import CONTROL_MS, Rig, TAU_S  # noqa: E402

OUT = Path(__file__).parent
# Куда писать веса. По умолчанию — прежнее имя, чтобы вызов без
# аргументов делал то же, что делал. `--out` появился затем, что
# `motor_decoder_v1.npz` лежит в истории как ПРИМЕР негодного файла
# (в нём нет `meta_norm`, см. `MOTOR.md` §4а), и затирать его
# переобучением значит потерять тот самый артефакт, на котором показан
# отказ. Переобученные веса кладутся рядом, под своим именем.
WEIGHTS = OUT / "motor_decoder_v1.npz"

# Расписание накачки. Два периода, чтобы веса не выучили один-единственный
# ритм: 2,5 с — как в `cli lateral`, 1,1 с — заведомо несоизмеримый с ним.
BASE_HZ, DEPTH = 75.0, 0.85
PERIODS = (2.5, 1.1)
TRAIN_SEEDS = (0, 1, 2, 3)
TEST_SEED = 17
STEPS = 300
RIDGE = 1e-2
# Нормировка входа: та же, что по умолчанию у варианта А. Менять её
# после обучения нельзя — веса подогнаны под масштаб.
NORM, PROBE_REF_HZ = "rate", 100.0


def collect(neurons, weights, device, seed, period_s):
    """Один прогон: ряд векторов z и ряд известной асимметрии входа."""
    from flybrain import motor

    rig = Rig(neurons, weights, device, seed=seed)
    dec = motor.make_decoder("dn_linear",
                             shape=motor.DecoderShape(dn_tau_s=TAU_S),
                             norm=NORM, probe_ref_hz=PROBE_REF_HZ)
    order = list(dec.turn_weights)

    rig.drive(orn_l=BASE_HZ, orn_r=BASE_HZ)
    warm = max(5, int(round(3000.0 * TAU_S / CONTROL_MS)))
    for _ in range(warm):
        dec.observe(rig.step(), rig.dt_s)
    dec.calibrate()
    dec.reset_state()

    X, y = [], []
    dt = rig.dt_s
    for i in range(STEPS):
        t = i * dt
        m = DEPTH * math.sin(2 * math.pi * t / period_s)
        rig.drive(orn_l=BASE_HZ * (1.0 + m), orn_r=BASE_HZ * (1.0 - m))
        rd = rig.step()
        dec.command(rd, dt)
        X.append([dec.z_asym(k) for k in order])
        y.append(m)  # асимметрия входа: (l-r)/(l+r) = m
    k0 = int(round(TAU_S / dt))   # переходный процесс фильтра не в выборку
    return np.array(X[k0:], float), np.array(y[k0:], float), order


def ridge(X, y, lam):
    """Гребневая регрессия со свободным членом. Своя, без sklearn."""
    n, d = X.shape
    A = np.hstack([X, np.ones((n, 1))])
    G = A.T @ A
    # Свободный член не штрафуется: штрафовать его значит сдвигать
    # предсказание к нулю тем сильнее, чем больше lam, а это не то, чем
    # гребень должен заниматься.
    reg = lam * n * np.eye(d + 1)
    reg[-1, -1] = 0.0
    sol = np.linalg.solve(G + reg, A.T @ y)
    return sol[:-1], float(sol[-1])


def r2(y, yhat):
    ss = float(np.sum((y - yhat) ** 2))
    tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss / tot if tot > 0 else float("nan")


def main(argv=None):
    import torch

    from flybrain import data as D

    argv = list(sys.argv[1:] if argv is None else argv)
    global WEIGHTS
    report = OUT / "motor_train.json"
    if "--out" in argv:
        WEIGHTS = Path(argv[argv.index("--out") + 1])
        if not WEIGHTS.is_absolute():
            WEIGHTS = OUT / WEIGHTS
        report = WEIGHTS.with_name(WEIGHTS.stem + ".json")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    neurons = D.load_neurons()
    weights = D.load_weights(device=device)
    print("устройство:", device)

    t0 = time.perf_counter()
    Xs, ys, order = [], [], None
    for seed in TRAIN_SEEDS:
        for period in PERIODS:
            X, y, order = collect(neurons, weights, device, seed, period)
            Xs.append(X)
            ys.append(y)
            print("  обучающий прогон: зерно %d, период %.1f с, %d строк "
                  "(%.0f с)" % (seed, period, len(y),
                                time.perf_counter() - t0))
    X = np.vstack(Xs)
    y = np.concatenate(ys)

    Xt, yt, _ = collect(neurons, weights, device, TEST_SEED, PERIODS[0])
    print("  отложенный прогон: зерно %d, %d строк" % (TEST_SEED, len(yt)))

    w, b = ridge(X, y, RIDGE)
    r2_train = r2(y, X @ w + b)
    r2_test = r2(yt, Xt @ w + b)
    print("\n  зонды: %s" % ", ".join(order))
    print("  веса:  %s" % ", ".join("%+.4f" % v for v in w))
    print("  смещение %+.4f" % b)
    print("  r² на обучении %.4f, на ОТЛОЖЕННОМ прогоне %.4f"
          % (r2_train, r2_test))

    meta = {
        "meta_probe_order": np.array(order),
        "meta_train_seeds": np.array(TRAIN_SEEDS),
        "meta_test_seed": np.array([TEST_SEED]),
        "meta_periods_s": np.array(PERIODS),
        "meta_base_hz": np.array([BASE_HZ]),
        "meta_depth": np.array([DEPTH]),
        "meta_ridge": np.array([RIDGE]),
        "meta_rows_train": np.array([len(y)]),
        "meta_rows_test": np.array([len(yt)]),
        "meta_r2_train": np.array([round(r2_train, 6)]),
        "meta_r2_test": np.array([round(r2_test, 6)]),
        "meta_control_ms": np.array([CONTROL_MS]),
        "meta_tau_s": np.array([TAU_S]),
        "meta_target": np.array(["асимметрия обонятельного входа (l-r)/(l+r)"]),
        # Нормировка входа — часть обучения, а не настройка применения:
        # веса подогнаны под этот масштаб `z`. Поэтому она едет в файл и
        # оттуда читается, а не задаётся на месте.
        "meta_norm": np.array([NORM]),
        "meta_probe_ref_hz": np.array([PROBE_REF_HZ]),
    }
    np.savez(WEIGHTS, w=w, b=np.array([b]), probe_order=np.array(order),
             **meta)
    import hashlib

    sha = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
    print("\n  файл весов: %s\n  sha256: %s" % (WEIGHTS, sha))
    report.write_text(json.dumps({
        "зонды": order,
        "веса": [round(float(v), 6) for v in w],
        "смещение": round(b, 6),
        "r2 обучение": round(r2_train, 6),
        "r2 отложенный": round(r2_test, 6),
        "строк обучения": int(len(y)),
        "строк проверки": int(len(yt)),
        "зёрна обучения": list(TRAIN_SEEDS),
        "зерно проверки": TEST_SEED,
        "периоды, с": list(PERIODS),
        "гребень": RIDGE,
        "нормировка": NORM,
        "probe_ref_hz": PROBE_REF_HZ,
        "файл": str(WEIGHTS),
        "sha256": sha,
        "по стене, с": round(time.perf_counter() - t0, 1),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
