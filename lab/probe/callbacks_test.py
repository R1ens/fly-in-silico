"""Проверка колбэков прогресса и остановки в симуляторе.

Смотрим на то, ради чего колбэки и добавлены: что окно видит ход длинной
задачи, что «стоп» срабатывает за время одного шага, а не в конце, и что
оборванная серия честно отдаёт фактическое число эпизодов.

Отдельно сверяем, что вызов без новых параметров даёт то же, что раньше:
структура строк `lateral_probe` сравнивается с сохранённым
`out/lateral-probe/lateral.json`.

Запуск: cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\callbacks_test.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flybrain import body, data as D, loop                      # noqa: E402

CONTROL_MS = 20.0


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    fails = []

    # Матрица связей грузится 4,4 с, таблица нейронов 0,11 с: берём один раз
    # на весь тест, иначе загрузки съедят больше времени, чем сами замеры.
    section("загрузка")
    t0 = time.perf_counter()
    neurons = D.load_neurons()
    t_neurons = time.perf_counter() - t0
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    weights = D.load_weights(device=device)
    t_weights = time.perf_counter() - t0
    print(f"нейроны {len(neurons)} за {t_neurons:.2f} с, "
          f"матрица за {t_weights:.2f} с, устройство {device}")

    # --- 1. ход замера ---
    section("1. lateral_probe + on_step")
    seen = []
    stamps = []

    def on_step(step, t_ms, n_steps):
        seen.append((step, t_ms, n_steps))
        stamps.append(time.perf_counter())

    t0 = time.perf_counter()
    rows = loop.lateral_probe(neurons=neurons, weights=weights, device=device,
                              duration_s=1.0, progress=False, on_step=on_step)
    t_probe = time.perf_counter() - t0
    gaps = np.diff(stamps) * 1000.0 if len(stamps) > 1 else np.array([0.0])
    print(f"колбэк позвался {len(seen)} раз за {t_probe:.2f} с, "
          f"в среднем раз в {gaps.mean():.1f} мс "
          f"(мин {gaps.min():.1f}, макс {gaps.max():.1f})")
    print(f"до первого вызова {stamps[0] - t0:.2f} с: это сборка движка и "
          f"63 выборки популяций, до них «хода» ещё нет")
    print(f"первый вызов: {seen[0]}, последний: {seen[-1]}")
    if len(seen) != 50:
        fails.append(f"ждали 50 вызовов на 1 с при шаге 20 мс, пришло {len(seen)}")
    if seen[0][2] != 50 or seen[-1][1] != 49 * CONTROL_MS:
        fails.append(f"колбэк врёт про время или число шагов: {seen[0]}, {seen[-1]}")
    if any("partial" in r for r in rows):
        fails.append("целый замер помечен как неполный")

    # --- 2. остановка замера ---
    section("2. lateral_probe + should_stop через 1 с")
    # Секунду считаем от первого шага, а не от вызова: до первого шага идёт
    # сборка движка, и «стоп», нажатый в это время, всё равно ждёт её конца.
    t_first = [None]
    t_stop = [None]
    n_done = [0]

    def on_step2(step, t_ms, n_steps):
        if t_first[0] is None:
            t_first[0] = time.perf_counter()
        n_done[0] = step + 1

    def should_stop():
        if t_first[0] is None or time.perf_counter() - t_first[0] < 1.0:
            return False
        if t_stop[0] is None:
            t_stop[0] = time.perf_counter()
        return True

    t0 = time.perf_counter()
    cut = loop.lateral_probe(neurons=neurons, weights=weights, device=device,
                             duration_s=10.0, progress=False,
                             on_step=on_step2, should_stop=should_stop)
    t_cut = time.perf_counter() - t0
    lag_ms = (time.perf_counter() - t_stop[0]) * 1000.0
    print(f"замер на 10 с (500 шагов) оборван на шаге {n_done[0]}: "
          f"всего {t_cut:.2f} с, из них {t_first[0] - t0:.2f} с сборка, "
          f"{t_stop[0] - t_first[0]:.2f} с шаги")
    print(f"от «стоп» до возврата {lag_ms:.0f} мс — это подсчёт корреляций "
          f"по накопленному")
    print(f"строк {len(cut)}, первая: "
          + json.dumps(cut[0], ensure_ascii=False))
    if lag_ms > 1000.0:
        fails.append(f"от «стоп» до возврата {lag_ms:.0f} мс")
    if not 1 <= n_done[0] <= 60:
        fails.append(f"оборвали на шаге {n_done[0]} из 500 — «стоп» "
                     "сработал не за секунду")
    if not cut or not all(r.get("partial") for r in cut):
        fails.append("нет признака неполноты в строках оборванного замера")
    elif cut[0]["measured_s"] >= 10.0:
        fails.append(f"measured_s = {cut[0]['measured_s']}, а замер оборван")
    else:
        print(f"фактически измерено {cut[0]['measured_s']} с из 10.0")
    if len(cut) != len(rows):
        fails.append(f"строк {len(cut)}, а в целом замере {len(rows)}")

    # --- 3. серия на плоской арене ---
    section("3. loop.bench(n_runs=2, duration_s=1.0) + on_episode")
    got = []
    res = loop.bench(n_runs=2, duration_s=1.0, neurons=neurons,
                     weights=weights, device=device,
                     on_episode=lambda c, i, n, m: got.append((c, i, n, m)))
    for c, i, n, m in got:
        print(f"  on_episode: {c} [{i}/{n}] градиент {m['grad_alignment']:+.4f}, "
              f"шагов {m['steps']}")
    if len(got) != 4:
        fails.append(f"ждали 4 эпизода, колбэк позвался {len(got)} раз")
    if got and sorted({c for c, _, _, _ in got}) != ["норма", "перестановка антенн"]:
        fails.append(f"условия пришли не те: {[c for c, _, _, _ in got]}")
    if any(r["n_runs"] != 2 for r in res.values()):
        fails.append(f"n_runs целой серии: {[r['n_runs'] for r in res.values()]}")

    # --- 4. серия в теле ---
    section("4. body.bench(n_runs=2, duration_s=0.8) + on_episode")
    got_b = []
    res_b = body.bench(n_runs=2, duration_s=0.8, neurons=neurons,
                       weights=weights, device=device, progress=False,
                       on_episode=lambda c, i, n, m: got_b.append((c, i, n, m)))
    for c, i, n, m in got_b:
        print(f"  on_episode: {c} [{i}/{n}] градиент {m['grad_alignment']:+.4f}, "
              f"до еды {m['dist_start_mm']:.1f} -> {m['dist_end_mm']:.1f} мм")
    if len(got_b) != 4:
        fails.append(f"ждали 4 эпизода в теле, колбэк позвался {len(got_b)} раз")
    if any(r["n_runs"] != 2 for r in res_b.values()):
        fails.append(f"n_runs целой серии в теле: "
                     f"{[r['n_runs'] for r in res_b.values()]}")

    # --- 5. остановка серии после первого эпизода ---
    section("5. серия со «стоп» после первого эпизода")
    for name, fn, kw in (("loop.bench", loop.bench, {"duration_s": 1.0}),
                         ("body.bench", body.bench,
                          {"duration_s": 0.8, "progress": False})):
        done = []
        t0 = time.perf_counter()
        r = fn(n_runs=4, neurons=neurons, weights=weights, device=device,
               on_episode=lambda c, i, n, m: done.append((c, i)),
               should_stop=lambda: len(done) >= 1, **kw)
        dt = time.perf_counter() - t0
        print(f"  {name}: эпизодов {len(done)} за {dt:.1f} с, "
              f"в результате {len(r)} условий, "
              f"n_runs = {[v['n_runs'] for v in r.values()]}")
        print(f"    sd = {[v['grad_alignment_sd'] for v in r.values()]}, "
              f"runs = {[v['runs'] for v in r.values()]}")
        if len(done) != 1:
            fails.append(f"{name}: после «стоп» прошло {len(done)} эпизодов")
        if len(r) != 1:
            fails.append(f"{name}: оборванная серия отдала {len(r)} условий")
        for v in r.values():
            if v["n_runs"] != len(v["runs"]) or v["n_runs"] != 1:
                fails.append(f"{name}: n_runs = {v['n_runs']}, "
                             f"а эпизодов {len(v['runs'])}")

    # --- 6. вызов без новых параметров ---
    section("6. вызов без новых параметров")
    saved_path = D.OUT / "lateral-probe" / "lateral.json"
    if saved_path.exists():
        saved = json.loads(saved_path.read_text(encoding="utf-8"))
        keys_old = sorted(saved[0])
        keys_new = sorted(rows[0])
        print(f"сохранённый замер: {len(saved)} строк, ключи {keys_old}")
        print(f"свежий замер:      {len(rows)} строк, ключи {keys_new}")
        if keys_old != keys_new:
            fails.append(f"ключи строк разошлись: {keys_old} против {keys_new}")
        if len(saved) != len(rows):
            fails.append(f"строк {len(rows)}, в сохранённом {len(saved)}")
        if {r["population"] for r in saved} != {r["population"] for r in rows}:
            fails.append("набор популяций разошёлся с сохранённым")
        # Длительности разные (10 с против 1 с), поэтому сверяем не числа, а
        # то, что знак сигнала у заведомо сильных популяций не перевернулся.
        new_r = {r["population"]: r["r"] for r in rows}
        for s in saved[:5]:
            print(f"  {s['population']:24s} сохранено r = {s['r']:+.3f}, "
                  f"сейчас {new_r.get(s['population'], float('nan')):+.3f}")
    else:
        fails.append(f"нет сохранённого замера {saved_path}")

    section("итог")
    if fails:
        for f in fails:
            print(f"[ПРОВАЛ] {f}")
        return 1
    print("[ок] все проверки прошли")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
