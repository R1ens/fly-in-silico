"""Сверка тела: пропуск лишнего пересчёта не должен менять траекторию.

Аудит физики нашёл, что `Fly.pre_step` пишет `ctrl` через биндинг
dm_control, тот помечает состояние «грязным», и первое же чтение
наблюдений зовёт полный `physics.forward()` — пятая часть шага
управления уходит на повторный расчёт того же состояния.

Пропустить его заманчиво, но не очевидно безопасно: в модели 179
сенсоров, и если хоть один зависит от `ctrl`, наблюдение станет
устаревшим и траектория поедет. Проверяется это не рассуждением, а
здесь: два эпизода с одним сидом, обычный и быстрый, сравниваются
побайтово по траектории, моторным командам, сенсорам и активности.

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\body_equal.py
"""

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

OUT = Path(__file__).parent
fails = []
rep = {"проверки": {}, "замеры": {}}


def check(name, ok, info=""):
    print(("  ok     " if ok else "  ПРОВАЛ ") + name
          + (("  " + str(info)) if info else ""))
    rep["проверки"][name] = {"пройдено": bool(ok), "замер": str(info)}
    if not ok:
        fails.append(name)


def fp(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def episode(fast, duration, neurons, weights, device):
    from flybrain import body as B

    cfg = B.BodyConfig(duration_s=duration, control_ms=20.0, seed=0,
                       device=device, readout="ALPN", fast_physics=fast)
    t0 = time.perf_counter()
    run = B.run_episode(cfg=cfg, neurons=neurons, weights=weights,
                        progress=False)
    wall = time.perf_counter() - t0
    recs = run.records
    cols = ("x", "y", "theta", "odour_l", "odour_r", "dn_l", "dn_r",
            "amp_l", "amp_r", "dist", "grad_cos", "spikes")
    table = np.array([[r[c] for c in cols] for r in recs], dtype=np.float64)
    return {
        "по стене, с": round(wall, 2),
        "шагов": len(recs),
        "траектория": fp(table[:, :3]),
        "сенсоры": fp(table[:, 3:5]),
        "моторные": fp(table[:, 5:9]),
        "всё по шагам": fp(table),
        "активность": fp(run.counts),
        "растр": fp(run.bins),
        "метрики": run.metrics,
        "калибровка": run.calib,
        "конец": [round(float(recs[-1]["x"]), 6),
                  round(float(recs[-1]["y"]), 6),
                  round(float(recs[-1]["theta"]), 6)] if recs else None,
        "_таблица": table,
        "_столбцы": cols,
    }


def main():
    import torch

    from flybrain import data as D

    device = "cuda" if torch.cuda.is_available() else "cpu"
    neurons = D.load_neurons()
    weights = D.load_weights(device=device)
    print("устройство:", device)

    # Эпизод подлиннее: расхождение в одном бите на первом шаге за сто
    # шагов управления разойдётся так, что его нельзя будет не заметить.
    duration = 2.0
    print("обычный эпизод …", flush=True)
    slow = episode(False, duration, neurons, weights, device)
    print("   %.2f с" % slow["по стене, с"], flush=True)
    print("быстрый эпизод …", flush=True)
    fast = episode(True, duration, neurons, weights, device)
    print("   %.2f с" % fast["по стене, с"], flush=True)

    rep["обычный"], rep["быстрый"] = slow, fast
    rep["замеры"] = {
        "обычный, с": slow["по стене, с"],
        "быстрый, с": fast["по стене, с"],
        "ускорение": round(slow["по стене, с"]
                           / max(fast["по стене, с"], 1e-9), 3),
        "шагов": slow["шагов"],
    }

    check("шагов столько же", slow["шагов"] == fast["шагов"],
          "%d и %d" % (slow["шагов"], fast["шагов"]))
    for key in ("траектория", "сенсоры", "моторные", "всё по шагам",
                "активность", "растр"):
        check("отпечаток «%s» совпал" % key, slow[key] == fast[key],
              "%s против %s" % (slow[key], fast[key]))
    check("конечная поза совпала", slow["конец"] == fast["конец"],
          "%s против %s" % (slow["конец"], fast["конец"]))
    same_m = all(abs(float(slow["метрики"][k]) - float(fast["метрики"][k]))
                 < 1e-12 for k in slow["метрики"])
    check("метрики совпали", same_m,
          "grad_alignment %s против %s"
          % (slow["метрики"].get("grad_alignment"),
             fast["метрики"].get("grad_alignment")))

    # Где именно разошлось. Отличать «один сенсор отстал на шаг» от
    # «наблюдение сломано» важно: первое — накопление хаоса, второе —
    # ошибка в коде. Поэтому печатается первый расходящийся шаг и по
    # какому столбцу, а не только итоговое несовпадение отпечатков.
    ta, tb = slow.pop("_таблица"), fast.pop("_таблица")
    cols = slow.pop("_столбцы")
    fast.pop("_столбцы")
    n = min(len(ta), len(tb))
    diff = np.abs(ta[:n] - tb[:n])
    bad = np.argwhere(diff > 0)
    first = {}
    if len(bad):
        step = int(bad[0][0])
        row = diff[step]
        first = {
            "шаг": step,
            "столбцы": {cols[j]: {"обычный": float(ta[step, j]),
                                  "быстрый": float(tb[step, j]),
                                  "разница": float(row[j])}
                        for j in range(len(cols)) if row[j] > 0},
        }
        print("\nпервое расхождение на шаге %d из %d:" % (step, n))
        for k, v in first["столбцы"].items():
            print("   %-9s %.6g -> %.6g   (на %.3g)"
                  % (k, v["обычный"], v["быстрый"], v["разница"]))
        worst = {cols[j]: float(diff[:, j].max())
                 for j in range(len(cols))}
        first["наибольшее за эпизод"] = worst
        print("наибольшее расхождение за эпизод:")
        for k, v in sorted(worst.items(), key=lambda x: -x[1]):
            print("   %-9s %.6g" % (k, v))
    rep["первое расхождение"] = first

    print("\nскорость: %.2f с -> %.2f с, в %.2f раза"
          % (slow["по стене, с"], fast["по стене, с"],
             rep["замеры"]["ускорение"]))

    (OUT / "body_equal.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nпроверок %d, провалов %d" % (len(rep["проверки"]), len(fails)))
    for f in fails:
        print("  - " + f)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
