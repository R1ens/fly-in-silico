"""Честное сравнение шага LIF: до правки и после, в одном процессе.

Два замера в разных запусках сравнивать нельзя: видеокарта на этой
машине занята и рабочим столом тоже, и разброс между запусками больше
того, что мы хотим измерить. Поэтому оба варианта поднимаются рядом и
гоняются вперемежку, чередуясь блоками, а в итог идёт медиана.

Старый вариант берётся из Git — из коммита, которым зафиксирована v1.
Так сравнение опирается не на память о том, как было, а на файл.

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\lif_ab.py
"""

import importlib.util
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe._isolate  # noqa: E402,F401

OUT = Path(__file__).parent
REF = "v1"                       # тег, в котором лежит прежний шаг


def load_old_sim():
    """Поднять sim.py из тега v1 как отдельный модуль."""
    root = Path(__file__).resolve().parents[2]
    src = subprocess.run(["git", "show", "%s:lab/flybrain/sim.py" % REF],
                         cwd=str(root), capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    if src.returncode != 0:
        raise SystemExit("не достал sim.py из %s: %s" % (REF, src.stderr))
    tmp = Path(tempfile.mkdtemp(prefix="flybrain-old-"))
    path = tmp / "sim_old.py"
    # Модуль живёт внутри пакета: правим относительные импорты на пакетные,
    # чтобы он подтянул те же данные и те же константы.
    text = src.stdout.replace("from . import data as D",
                              "from flybrain import data as D")
    text = text.replace("from .regions", "from flybrain.regions")
    path.write_text(text, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("sim_old", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sim_old"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    import torch

    from flybrain import data as D, sim

    device = "cuda" if torch.cuda.is_available() else "cpu"
    old = load_old_sim()
    print("старый шаг взят из тега", REF)

    neurons = D.load_neurons()
    weights = D.load_weights(device=device)
    n = len(neurons)
    idx = D.select(neurons, "class:gustatory+side:left")
    free = torch.as_tensor(idx, device=device)

    def build(mod):
        e = mod.build_engine(weights=weights, neurons=neurons, device=device,
                             batch=1, seed=0, no_refractory=free)
        r = torch.zeros(1, n, device=device)
        r[:, idx] = 200.0
        return e, r

    e_old, r_old = build(old)
    e_new, r_new = build(sim)

    def block(eng, rates, k=1500):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(k):
            eng.step(rates)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0 / k

    # прогрев обоих
    for _ in range(3):
        block(e_old, r_old, 50)
        block(e_new, r_new, 50)

    olds, news = [], []
    rounds = 7
    for i in range(rounds):
        olds.append(block(e_old, r_old))
        news.append(block(e_new, r_new))
        print("  круг %d: было %.4f мс, стало %.4f мс"
              % (i + 1, olds[-1], news[-1]))

    mo, mn = statistics.median(olds), statistics.median(news)
    rep = {
        "устройство": device,
        "кругов": rounds,
        "шаг v1, мс": {"медиана": round(mo, 4), "мин": round(min(olds), 4),
                       "макс": round(max(olds), 4)},
        "шаг сейчас, мс": {"медиана": round(mn, 4), "мин": round(min(news), 4),
                           "макс": round(max(news), 4)},
        "ускорение шага": round(mo / mn, 3),
    }
    print("\nмедиана: было %.4f мс, стало %.4f мс — в %.2f раза"
          % (mo, mn, mo / mn))

    # --- то же для цикла прогона целиком ---
    def whole(mod, t=0.3):
        cfg = mod.RunConfig(t_run_s=t, bin_ms=10.0, n_trials=1, seed=0,
                            device=device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = mod.run([mod.Stimulus(idx, 200.0, name="s")], config=cfg,
                      neurons=neurons, weights=weights, progress=False)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0), res

    wo, wn = [], []
    for _ in range(5):
        a, res_o = whole(old)
        b, res_n = whole(sim)
        wo.append(a)
        wn.append(b)
    same = (int(res_o.counts.sum()) == int(res_n.counts.sum())
            and bool((res_o.bins == res_n.bins).all()))
    rep["прогон 0,3 с"] = {
        "v1, с": round(statistics.median(wo), 3),
        "сейчас, с": round(statistics.median(wn), 3),
        "ускорение": round(statistics.median(wo) / statistics.median(wn), 3),
        "результат совпал": bool(same),
    }
    print("прогон 0,3 с: было %.2f с, стало %.2f с — в %.2f раза; "
          "результат совпал: %s"
          % (statistics.median(wo), statistics.median(wn),
             statistics.median(wo) / statistics.median(wn), same))

    (OUT / "lif_ab.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nотчёт:", OUT / "lif_ab.json")
    return 0 if rep["прогон 0,3 с"]["результат совпал"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
