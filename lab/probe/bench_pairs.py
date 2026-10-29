"""Перемежающиеся пары: сколько стоит фон и сколько надо, чтобы поверить.

Отношение двух вариантов на этой машине нельзя снимать двумя прогонами
подряд: фон гуляет, и в одном измерении загрузка видеокарты держалась на
96 % ещё до старта стенда. Поэтому варианты гоняются кругами A, B, A, B,
… и отношение считается ВНУТРИ пары.

У пар два встречных замера, и без них сама оснастка ничего не стоит:

* **A/A — порог доверия.** Один и тот же вариант против себя же. Медиана
  отношений обязана лечь около единицы, а размах по парам — это и есть
  тот порог, ниже которого «ускорение» неотличимо от фона. Число, которое
  надо знать раньше, чем обсуждать чьё-то ускорение;
* **A/B с заранее известной разницей — зубы.** Движок LIF с CUDA-графами
  против него же без графов: по `SPEED.md` §1 это 2,6…2,95 раза. Если
  оснастка не видит разницу такого размера, она не увидит никакую, и
  говорить о порогах бессмысленно.

Три предмета:

* ``lif``  — блок из 200 шагов движка: графы против без графов и A/A;
* ``loop`` — полный замкнутый шаг управления: A/A, то есть порог доверия
  для **официального числа** стенда;
* ``fast`` — полный контур против него же с `fast_physics`. Ускорение
  заранее известно (1,44 раза по `SPEED.md` §3), а равенства нет, и это
  пара, на которой видно, что стенд мерит скорость и выносит категорию
  отдельно: быстрее — да, равнозначно — нет.

    cd D:\\fly-brain\\lab
    set PYTHONIOENCODING=utf-8
    ..\\.venv\\Scripts\\python.exe probe\\bench_pairs.py --what lif --пар 8
    ..\\.venv\\Scripts\\python.exe probe\\bench_pairs.py --what loop --пар 5
"""

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

from probe import bench_stand as BS  # noqa: E402

BENCH = Path(__file__).resolve().parents[1] / "bench"
BENCH.mkdir(parents=True, exist_ok=True)
COLD = 3


def part_lif(ctx, n_pairs, blocks=5):
    """Блок из 200 шагов движка: графы против без графов, и A/A рядом."""
    import torch

    from flybrain import sim as SIM

    dev, neurons, weights = ctx["device"], ctx["neurons"], ctx["weights"]
    lif_per = int(round(BS.CONTROL_MS / SIM.DT))
    rates = torch.zeros(1, len(neurons), device=dev)
    rep = {"что мерится": "мс на блок из %d шагов движка LIF (шаг "
                          "управления 20 мс)" % lif_per,
           "блоков в одном замере": blocks}

    def build(graphs):
        t = time.perf_counter()
        eng = SIM.build_engine(weights=weights, neurons=neurons, device=dev,
                              seed=0, graphs=graphs)
        return eng, time.perf_counter() - t

    g1, t_g1 = build(True)
    g2, t_g2 = build(True)
    ref, t_ref = build(False)
    rep["сборка движка, с"] = {"с графами": round(t_g1, 2),
                               "второй с графами": round(t_g2, 2),
                               "без графов": round(t_ref, 2),
                               "графов захвачено": len(g1._graphs)
                               if g1._graphs is not None else 0}

    def unit(eng):
        ms = []
        for _ in range(blocks):
            if dev == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(lif_per):
                eng.step(rates)
            if dev == "cuda":
                torch.cuda.synchronize()
            ms.append((time.perf_counter() - t) * 1000.0)
        return statistics.median(ms)

    print(" A/A: графы против таких же графов (порог доверия)", flush=True)
    rep["A/A: графы против графов"] = BS.pairs(
        lambda: unit(g1), lambda: unit(g2), n=n_pairs,
        name_a="графы 1, мс", name_b="графы 2, мс")
    print(" A/B: графы против без графов (заранее известные 2,6…2,95)",
          flush=True)
    rep["A/B: без графов против графов"] = BS.pairs(
        lambda: unit(ref), lambda: unit(g1), n=n_pairs,
        name_a="без графов, мс", name_b="графы, мс")
    aa = rep["A/A: графы против графов"].get("отношение", {})
    ab = rep["A/B: без графов против графов"].get("отношение", {})
    rep["итог"] = {
        "порог доверия (размах A/A), % ": aa.get("размах, % медианы"),
        "A/A медиана отношения": aa.get("медиана"),
        "A/B медиана отношения": ab.get("медиана"),
        "оснастка видит заранее известную разницу":
            bool(ab.get("медиана") and ab["медиана"] > 2.0),
        "A/A легло около единицы":
            bool(aa.get("медиана") and abs(aa["медиана"] - 1.0) < 0.05),
        "что это значит": (
            "ускорение полного шага меньше размаха A/A (%s %%) отличить от "
            "фона нельзя: его надо снимать парами и называть вместе с "
            "числом пар" % aa.get("размах, % медианы")),
    }
    del g1, g2, ref
    return rep


def part_loop(ctx, n_pairs, steps, fast=False):
    """Полный замкнутый шаг: A/A — порог доверия официального числа."""
    from flybrain import body as B

    dev, neurons, weights = ctx["device"], ctx["neurons"], ctx["weights"]

    def unit(fast_physics=False):
        cfg = B.BodyConfig(duration_s=steps * BS.CONTROL_MS / 1000.0,
                           device=dev, seed=0, readout="ALPN",
                           fast_physics=fast_physics)
        rows, state = [], {"prev": None}

        def on_step(step, rec, bins_row):
            now = time.perf_counter()
            if state["prev"] is not None:
                rows.append((now - state["prev"]) * 1000.0)
            state["prev"] = now

        run = B.run_episode(cfg=cfg, neurons=neurons, weights=weights,
                            progress=False, on_step=on_step)
        keep = rows[COLD:] or rows
        out = statistics.median(keep)
        fin = (round(run.records[-1]["x"], 3), round(run.records[-1]["y"], 3),
               round(run.records[-1]["theta"], 4))
        prints.append({"fast_physics": fast_physics, "конечная поза": fin,
                       "отпечаток траектории": BS.fp(np.array(
                           [[r["x"], r["y"]] for r in run.records],
                           dtype=np.float64)),
                       "отпечаток активности": BS.fp(run.counts),
                       "медиана шага, мс": round(out, 2)})
        del run
        return out

    prints = []
    rep = {"что мерится": "медиана разогретого шага управления в полном "
                          "контуре, мс (%d шагов на замер, первые %d "
                          "холодных отброшены)" % (steps, COLD)}
    if not fast:
        print(" A/A: полный контур против себя же (порог доверия)", flush=True)
        rep["A/A: полный контур"] = BS.pairs(
            lambda: unit(False), lambda: unit(False), n=n_pairs,
            name_a="контур 1, мс", name_b="контур 2, мс")
        aa = rep["A/A: полный контур"].get("отношение", {})
        rep["итог"] = {
            "A/A медиана отношения": aa.get("медиана"),
            "размах по парам, % медианы": aa.get("размах, % медианы"),
            "СО по парам": aa.get("СО по парам"),
            "что это значит": (
                "два одинаковых прогона полного контура расходятся на %s %% "
                "по размаху. Заявленное ускорение меньше этого — не "
                "ускорение, а фон." % aa.get("размах, % медианы")),
        }
    else:
        print(" A/B: обычный контур против fast_physics "
              "(заранее известные 1,44)", flush=True)
        rep["A/B: обычный против fast_physics"] = BS.pairs(
            lambda: unit(False), lambda: unit(True), n=n_pairs,
            name_a="обычный, мс", name_b="fast_physics, мс")
        ab = rep["A/B: обычный против fast_physics"].get("отношение", {})
        # Равнозначность проверяется отпечатками, а не отчётом автора.
        reg = [p for p in prints if not p["fast_physics"]]
        fst = [p for p in prints if p["fast_physics"]]
        same_traj = len({p["отпечаток траектории"] for p in reg}) == 1
        rep["равнозначность"] = {
            "обычный повторяем побитово": same_traj,
            "отпечаток траектории обычного":
                reg[0]["отпечаток траектории"] if reg else None,
            "отпечаток траектории fast_physics":
                fst[0]["отпечаток траектории"] if fst else None,
            "траектории совпали":
                bool(reg and fst and reg[0]["отпечаток траектории"]
                     == fst[0]["отпечаток траектории"]),
            "конечная поза обычного": reg[0]["конечная поза"] if reg else None,
            "конечная поза fast_physics":
                fst[0]["конечная поза"] if fst else None,
            "категория": "C — экспериментальное приближение: быстрее, но "
                         "это правка модели, а не ускорение; равнозначным "
                         "не называется",
        }
        rep["итог"] = {
            "A/B медиана отношения": ab.get("медиана"),
            "размах по парам, % медианы": ab.get("размах, % медианы"),
        }
    rep["отпечатки прогонов"] = prints
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--what", default="lif", choices=("lif", "loop", "fast"))
    ap.add_argument("--пар", type=int, default=5)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--tag", default="пары")
    args = ap.parse_args()
    n_pairs = getattr(args, "пар")

    import torch

    from flybrain import data as D

    gpu = BS.Gpu()
    if gpu.ok:
        gpu.start()
    cpu = BS.Cpu()
    m = cpu.mark()
    t0 = time.perf_counter()
    print("снимаю фон до замера (стенд простаивает)…", flush=True)
    time.sleep(6.0)
    bg = {"процессор": cpu.since(m),
          "видеокарта": gpu.window(t0, time.perf_counter())}
    print("  фон: видеокарта %s %%, машина %s %% ядер"
          % (bg["видеокарта"].get("загрузка видеокарты, % медиана"),
             bg["процессор"].get("вся машина, %")), flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    neurons = D.load_neurons()
    weights = D.load_weights(device=dev)
    ctx = {"device": dev, "neurons": neurons, "weights": weights, "gpu": gpu}

    rep = {"метка": getattr(args, "tag"), "предмет": args.what,
           "когда": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "машина": {"видеокарта": torch.cuda.get_device_name(0)
                      if torch.cuda.is_available() else None,
                      "ядер": os.cpu_count(),
                      "нейронов": int(len(neurons)),
                      "связей": int(weights._nnz())},
           "фон до замера (стенд простаивал 6 с)": bg}
    t1 = time.perf_counter()
    if args.what == "lif":
        rep["пары"] = part_lif(ctx, n_pairs)
    elif args.what == "loop":
        rep["пары"] = part_loop(ctx, n_pairs, args.steps, fast=False)
    else:
        rep["пары"] = part_loop(ctx, n_pairs, args.steps, fast=True)
    t2 = time.perf_counter()
    rep["фон в окне пар"] = {"процессор": cpu.since(m),
                             "видеокарта": gpu.window(t1, t2)}
    rep["заняло, с"] = round(t2 - t1, 1)
    BS.save(BENCH / ("bench_pairs_%s_%s.json"
                     % (args.what, getattr(args, "tag"))), rep)
    gpu.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
