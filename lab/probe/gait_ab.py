"""Быстрый контроллер походки против стокового: равенство и отношение.

Два эпизода с одним семенем в ОДНОМ процессе, вперемешку, кругами:
`legacy` — `HybridTurningController` из пакета, `fast` — вендоренная
копия шага (`flybrain.gait`). Сравниваются отпечатки (траектория,
сенсоры, моторные команды, активность, растр) и отношение времени.

Почему кругами и почему пары. На этой машине одновременно считают
несколько задач, и межпроцессное сравнение времени ненадёжно: разброс
между запусками больше измеряемого. Поэтому меряется только то, что
прошло рядом: отношение внутри круга, и в итоге — медиана отношений,
их разброс и число пар. Абсолютные секунды тут справочные.

    cd D:\\fly-brain\\lab
    set PYTHONIOENCODING=utf-8
    ..\\.venv\\Scripts\\python.exe probe\\gait_ab.py --duration 2.0 --rounds 3
"""

import argparse
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
rep = {"проверки": {}, "круги": []}


def check(name, ok, info=""):
    print(("  ok     " if ok else "  ПРОВАЛ ") + name
          + (("  " + str(info)) if info else ""))
    rep["проверки"][name] = {"пройдено": bool(ok), "замер": str(info)}
    if not ok:
        fails.append(name)


def fp(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


COLS = ("x", "y", "theta", "odour_l", "odour_r", "dn_l", "dn_r",
        "asym", "z", "amp_l", "amp_r", "dist", "grad_cos", "spikes")


def episode(gait, duration, neurons, weights, device):
    from flybrain import body as B

    cfg = B.BodyConfig(duration_s=duration, control_ms=20.0, seed=0,
                       device=device, readout="ALPN", gait=gait)
    t0 = time.perf_counter()
    run = B.run_episode(cfg=cfg, neurons=neurons, weights=weights,
                        progress=False)
    wall = time.perf_counter() - t0
    recs = run.records
    table = np.array([[r[c] for c in COLS] for r in recs], dtype=np.float64)
    return {
        "по стене, с": round(wall, 3),
        "цикл, с": run.timing["sim_s"],
        "шагов": len(recs),
        "траектория": fp(table[:, :3]),
        "сенсоры": fp(table[:, 3:5]),
        "моторные": fp(table[:, 5:11]),
        "всё по шагам": fp(table),
        "активность": fp(run.counts),
        "растр": fp(run.bins),
        "метрики": run.metrics,
        "конец": [round(float(recs[-1]["x"]), 6),
                  round(float(recs[-1]["y"]), 6),
                  round(float(recs[-1]["theta"]), 6)] if recs else None,
        "_t": table,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=2.0)
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args()

    import torch

    from flybrain import data as D

    neurons = D.load_neurons()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = D.load_weights(device=device)

    rounds = []
    for r in range(a.rounds):
        # Порядок внутри круга чередуется: если машину занимает что-то
        # медленно разгоняющееся, постоянный порядок дарил бы второму
        # эпизоду систематическое преимущество.
        order = ("legacy", "fast") if r % 2 == 0 else ("fast", "legacy")
        got = {}
        for g in order:
            print("круг %d: %s" % (r + 1, g))
            got[g] = episode(g, a.duration, neurons, weights, device)
        rounds.append(got)
        rep["круги"].append({
            g: {k: v for k, v in got[g].items() if not k.startswith("_")}
            for g in got})
        print("  legacy %.2f с, fast %.2f с, отношение %.3f"
              % (got["legacy"]["цикл, с"], got["fast"]["цикл, с"],
                 got["legacy"]["цикл, с"] / got["fast"]["цикл, с"]))

    # --- равенство ---
    base = rounds[0]["legacy"]
    for key in ("траектория", "сенсоры", "моторные", "всё по шагам",
                "активность", "растр", "конец"):
        vals = {g: rounds[i][g][key] for i in range(len(rounds))
                for g in ("legacy", "fast")}
        ok = len({json.dumps(v, ensure_ascii=False)
                  for v in vals.values()}) == 1
        check("отпечаток совпал: %s" % key, ok,
              json.dumps(base[key], ensure_ascii=False))

    # Первое расхождение по шагам: если отпечатки разошлись, надо
    # сказать где, а не только что.
    t0 = rounds[0]["legacy"]["_t"]
    t1 = rounds[0]["fast"]["_t"]
    if t0.shape == t1.shape and not np.array_equal(t0, t1):
        d = np.argwhere(t0 != t1)
        i, j = d[0]
        rep["первое расхождение"] = {
            "шаг": int(i), "столбец": COLS[j],
            "legacy": float(t0[i, j]), "fast": float(t1[i, j])}
        print("  первое расхождение: шаг %d, %s: %r против %r"
              % (i, COLS[j], float(t0[i, j]), float(t1[i, j])))

    # --- отношение ---
    ratios = [rd["legacy"]["цикл, с"] / rd["fast"]["цикл, с"]
              for rd in rounds]
    walls = [(rd["legacy"]["цикл, с"], rd["fast"]["цикл, с"])
             for rd in rounds]
    rep["отношение"] = {
        "пар": len(ratios),
        "по кругам": [round(x, 4) for x in ratios],
        "медиана": round(float(np.median(ratios)), 4),
        "минимум": round(float(np.min(ratios)), 4),
        "максимум": round(float(np.max(ratios)), 4),
        "шаг управления, мс": {
            "legacy": round(float(np.median([w[0] for w in walls]))
                            * 1000.0 / rounds[0]["legacy"]["шагов"], 2),
            "fast": round(float(np.median([w[1] for w in walls]))
                          * 1000.0 / rounds[0]["fast"]["шагов"], 2)},
    }
    print()
    print("отношение legacy/fast: медиана %.3f, разброс %.3f…%.3f, пар %d"
          % (rep["отношение"]["медиана"], rep["отношение"]["минимум"],
             rep["отношение"]["максимум"], len(ratios)))
    print("шаг управления: legacy %.2f мс, fast %.2f мс"
          % (rep["отношение"]["шаг управления, мс"]["legacy"],
             rep["отношение"]["шаг управления, мс"]["fast"]))

    rep["провалов"] = len(fails)
    p = OUT / "gait_ab.json"
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    print("отчёт: %s" % p)
    if fails:
        print("ПРОВАЛЫ: " + ", ".join(fails))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
