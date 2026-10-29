"""Полный встроенный контур до правок и после, попарно в одном процессе.

Зачем так. Главное число брифа — 73,93 мс «мозг: LIF, синхронизация,
учёт» на шаге управления в 20 мс — снимается `probe/bodyfinal_timing.py`
за один запуск. Сравнивать два запуска на этой машине нельзя: рядом
считают другие прогоны, и разброс между запусками больше того, что мы
измеряем. Поэтому прежний пакет `flybrain` достаётся из Git и
поднимается РЯДОМ с нынешним под именем `flybrain_old`, а эпизоды
гоняются вперемежку: старый, новый, старый, новый…

Относительные импорты внутри пакета это позволяют: ни один модуль не
зовёт `flybrain` по имени, все берут соседей через точку, так что копия
каталога под другим именем — полноценный второй пакет. Данные (таблица
нейронов и матрица связей) загружаются один раз и отдаются обоим.

Меряется ровно то, что мерит проба владельца: от конца предыдущего шага
управления до первого шага физики — мозг, синхронизация и учёт.

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\lifperf_contour.py
"""

import argparse
import importlib
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

OUT = Path(__file__).parent
ROOT = Path(__file__).resolve().parents[2]
T_EPISODE = 0.4            # модельного времени на эпизод, с
CHANGED = ("body.py", "loop.py", "motor.py")


def old_package(ref="HEAD", changed=CHANGED):
    """Копия пакета с правленными файлами из ``ref``.

    Какие файлы откатывать — задаётся снаружи: так каждая правка мерится
    отдельно, а не все вместе. Остальные модули берутся нынешние, то есть
    различие между двумя пакетами ровно одно.
    """
    tmp = Path(tempfile.mkdtemp(prefix="flybrain-old-"))
    dst = tmp / "flybrain_old"
    shutil.copytree(ROOT / "lab" / "flybrain", dst,
                    ignore=shutil.ignore_patterns("__pycache__"))
    for name in changed:
        got = subprocess.run(
            ["git", "show", "%s:lab/flybrain/%s" % (ref, name)],
            cwd=str(ROOT), capture_output=True, text=True,
            encoding="utf-8", errors="replace")
        if got.returncode != 0:
            raise SystemExit("не достал %s из %s: %s"
                             % (name, ref, got.stderr[:300]))
        (dst / name).write_text(got.stdout, encoding="utf-8")
    sys.path.insert(0, str(tmp))
    return importlib.import_module("flybrain_old")


def brain_ms(pkg, neurons, weights, device):
    """Один эпизод. Возвращает средние мс мозговой части на шаг управления."""
    st = {"prev_end": None, "phys_first": None, "rows": [], "tail": [],
          "phys": [], "phys_tot": 0.0, "phys_n": 0, "body": None}

    def on_body(body):
        orig = body.step

        def phys(action):
            t0 = time.perf_counter()
            if st["phys_first"] is None:
                st["phys_first"] = t0
            out = orig(action)
            t1 = time.perf_counter()
            st["phys_tot"] += (t1 - t0) * 1000.0
            st["phys_n"] += 1
            st["phys_last"] = t1
            return out

        body.step = phys
        st["body"] = body

    def on_step(step, rec, row):
        now = time.perf_counter()
        if st["prev_end"] is not None and st["phys_first"] is not None:
            st["rows"].append((now - st["prev_end"]) * 1000.0)
            st["tail"].append((st["phys_first"] - st["prev_end"]) * 1000.0)
        st["prev_end"] = time.perf_counter()
        st["phys_first"] = None

    cfg = pkg.body.BodyConfig(duration_s=T_EPISODE, device=device)
    run = pkg.body.run_episode(cfg=cfg, neurons=neurons, weights=weights,
                               progress=False, on_step=on_step,
                               on_body=on_body)
    return {
        "шаг управления всего, мс": float(np.mean(st["rows"])),
        "мозг (LIF+синхр+учёт), мс": float(np.mean(st["tail"])),
        "физика, мс на шаг управления": st["phys_tot"] / max(
            len(st["rows"]) + 1, 1),
        "шагов": len(st["rows"]),
        "отпечатки": {
            "траектория": [round(r["x"], 3) for r in run.records[-3:]],
            "спайков на последнем шаге": run.records[-1]["spikes"],
            "dn_l": run.records[-1]["dn_l"], "dn_r": run.records[-1]["dn_r"],
            "конечная поза": [round(float(v), 4) for v in
                              (run.records[-1]["x"], run.records[-1]["y"],
                               run.records[-1]["theta"])],
            "спайков всего": int(run.counts.sum()),
            "растр": int(run.bins.sum()),
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--ref", default="HEAD")
    ap.add_argument("--tag", default="контур")
    ap.add_argument("--files", default=",".join(CHANGED),
                    help="какие файлы пакета откатить к --ref, через запятую")
    args = ap.parse_args()
    changed = tuple(s.strip() for s in args.files.split(",") if s.strip())

    import torch

    import flybrain
    from flybrain import data as D

    device = "cuda" if torch.cuda.is_available() else "cpu"
    old = old_package(args.ref, changed)
    print("прежний пакет: %d файл(ов) из %s (%s)"
          % (len(changed), args.ref, ", ".join(changed)))

    neurons = D.load_neurons()
    weights = D.load_weights(device=device)
    print("данные загружены, нейронов %d" % len(neurons))

    # Оба пакета должны видеть одни и те же подмодули по имени.
    importlib.import_module("flybrain.body")
    importlib.import_module("flybrain_old.body")

    rep = {"метка": args.tag, "устройство": device, "пар": args.pairs,
           "эпизод, с модельного": T_EPISODE, "прежний из": args.ref,
           "откачено к прежнему": list(changed), "круги": []}
    for i in range(args.pairs):
        print("\n--- пара %d: прежний" % (i + 1))
        a = brain_ms(old, neurons, weights, device)
        print("--- пара %d: нынешний" % (i + 1))
        b = brain_ms(flybrain, neurons, weights, device)
        rep["круги"].append({"прежний": a, "нынешний": b,
                             "мозг a/b": round(a["мозг (LIF+синхр+учёт), мс"]
                                               / b["мозг (LIF+синхр+учёт), мс"],
                                               4)})
        print("    мозг: прежний %.2f мс, нынешний %.2f мс, в %.4f раза"
              % (a["мозг (LIF+синхр+учёт), мс"],
                 b["мозг (LIF+синхр+учёт), мс"],
                 rep["круги"][-1]["мозг a/b"]))
        same = a["отпечатки"] == b["отпечатки"]
        rep["круги"][-1]["отпечатки совпали"] = bool(same)
        if not same:
            print("    ОТПЕЧАТКИ РАЗОШЛИСЬ:")
            for k in a["отпечатки"]:
                if a["отпечатки"][k] != b["отпечатки"][k]:
                    print("      %s: %s -> %s"
                          % (k, a["отпечатки"][k], b["отпечатки"][k]))
        else:
            print("    отпечатки эпизода совпали")

    rr = [c["мозг a/b"] for c in rep["круги"]]
    ma = [c["прежний"]["мозг (LIF+синхр+учёт), мс"] for c in rep["круги"]]
    mb = [c["нынешний"]["мозг (LIF+синхр+учёт), мс"] for c in rep["круги"]]
    rep["мозг: прежний, мс"] = round(statistics.median(ma), 2)
    rep["мозг: нынешний, мс"] = round(statistics.median(mb), 2)
    rep["мозг: a/b медиана"] = round(statistics.median(rr), 4)
    rep["мозг: a/b разброс"] = [round(min(rr), 4), round(max(rr), 4)]
    rep["мозг: выигрыш, мс"] = round(statistics.median(ma)
                                     - statistics.median(mb), 2)
    rep["все отпечатки совпали"] = all(c["отпечатки совпали"]
                                       for c in rep["круги"])
    print("\nИТОГ по %d парам:" % args.pairs)
    print("  мозг: %.2f -> %.2f мс, в %.4f раза (%.4f…%.4f)"
          % (rep["мозг: прежний, мс"], rep["мозг: нынешний, мс"],
             rep["мозг: a/b медиана"], rep["мозг: a/b разброс"][0],
             rep["мозг: a/b разброс"][1]))
    print("  выигрыш %.2f мс на шаг управления" % rep["мозг: выигрыш, мс"])
    print("  отпечатки совпали во всех парах: %s"
          % rep["все отпечатки совпали"])

    path = OUT / ("lifperf_contour_%s.json" % args.tag)
    path.write_text(json.dumps(rep, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print("отчёт:", path)
    return 0 if rep["все отпечатки совпали"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
