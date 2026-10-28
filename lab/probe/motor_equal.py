"""Сверка: прежний путь обязан считаться ПОБИТОВО как раньше.

Этап 2 вынул «придуманную часть» из `body.py` в отдельный модуль
`flybrain.motor` и подвесил к ней прямые зонды нисходящих. Это правка
того самого кода, по которому посчитаны все эталонные прогоны, и
обещание «при выключенном новом пути считается как прежде» без замера
ничего не стоит.

Проверяется так: тот же эпизод, то же семя, два разных дерева кода —
`lab_base` (состояние до правки, развёрнутое `git archive HEAD`) и `lab`
(сейчас), — и отпечатки траектории, сенсоров, моторных команд,
активности и растра сравниваются побайтово.

Запуск (из своего рабочего дерева):

    PYTHONIOENCODING=utf-8 FLYBRAIN_OUT=<свой каталог> \
        .venv/Scripts/python.exe lab/probe/motor_equal.py base
    PYTHONIOENCODING=utf-8 FLYBRAIN_OUT=<свой каталог> \
        .venv/Scripts/python.exe lab/probe/motor_equal.py now
    ... и затем `cmp`, который сравнивает два снимка.

Два процесса, а не один, нарочно: `flybrain` — пакет, и два его
несовпадающих экземпляра в одном процессе ужились бы только подменой
`sys.modules`, то есть сверка проверяла бы подмену, а не код.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
OUT = HERE.parent

# `lab` или `lab_base` выбирается ДО импорта flybrain: оба дерева лежат
# рядом в рабочем каталоге, и путь решает, какой из них будет пакетом.
MODE = sys.argv[1] if len(sys.argv) > 1 else "now"
if MODE == "base":
    LAB = HERE.parents[2] / "lab_base"
else:
    LAB = HERE.parents[1]
sys.path.insert(0, str(LAB))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402


def fp(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


COLS = ("x", "y", "theta", "odour_l", "odour_r", "dn_l", "dn_r",
        "asym", "z", "amp_l", "amp_r", "dist", "grad_cos", "spikes")


def episode(duration, seed, **kw):
    import torch

    from flybrain import body as B, data as D

    device = "cuda" if torch.cuda.is_available() else "cpu"
    neurons = D.load_neurons()
    weights = D.load_weights(device=device)
    cfg = B.BodyConfig(duration_s=duration, control_ms=20.0, seed=seed,
                       device=device, readout="ALPN", **kw)
    t0 = time.perf_counter()
    run = B.run_episode(cfg=cfg, neurons=neurons, weights=weights,
                        progress=False)
    wall = time.perf_counter() - t0
    recs = run.records
    table = np.array([[r[c] for c in COLS] for r in recs], dtype=np.float64)
    return {
        "дерево": MODE,
        "по стене, с": round(wall, 2),
        "шагов": len(recs),
        "траектория": fp(table[:, :3]),
        "сенсоры": fp(table[:, 3:5]),
        "моторные": fp(table[:, 5:11]),
        "всё по шагам": fp(table),
        "активность": fp(run.counts),
        "растр": fp(run.bins),
        "метрики": {k: v for k, v in run.metrics.items()
                    if not str(k).startswith(("cmd_", "r_", "stim_"))},
        "калибровка": {k: v for k, v in run.calib.items() if k != "зонды"},
        "конец": [round(float(recs[-1][c]), 6) for c in ("x", "y", "theta")],
    }


def main():
    if MODE == "cmp":
        a = json.loads((OUT / "motor_equal-base.json").read_text("utf-8"))
        b = json.loads((OUT / "motor_equal-now.json").read_text("utf-8"))
        fails = []
        for key in ("шагов", "траектория", "сенсоры", "моторные",
                    "всё по шагам", "активность", "растр", "конец",
                    "метрики", "калибровка"):
            ok = a[key] == b[key]
            print(("  ok     " if ok else "  ПРОВАЛ ") + str(key)
                  + ("" if ok else "\n      до:    %s\n      после: %s"
                     % (a[key], b[key])))
            if not ok:
                fails.append(key)
        print("\nпровалов %d" % len(fails))
        return 1 if fails else 0

    print("дерево:", LAB)
    snap = episode(2.0, 0)
    (OUT / ("motor_equal-%s.json" % MODE)).write_text(
        json.dumps(snap, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in snap.items():
        print("  %-14s %s" % (k, v))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
