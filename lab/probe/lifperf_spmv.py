"""Одно ядро, в которое упирается шаг: произведение матрицы на спайки.

Из замера `probe/lifperf.py`: шаг движка упирается не в драйвер, а в
видеокарту (постановка шага в очередь 0,115 мс против 0,2945 мс
исполнения), и на видеокарте 86 % шага — одно ядро `csrmv`. Значит всё
остальное можно разбирать сколько угодно, а шаг от этого не сдвинется.

Здесь перебираются дороги к тому же произведению. Сверять их на равенство
НЕ НУЖНО по построению, и это доказано отдельно: значения матрицы целые
(от −2405 до 1897), спайк ровно 0.0 или 1.0, максимум суммы модулей по
строке 69 948 против предела точности float32 в 2^24. Сумма целых во
float32 точна при любом порядке сложения, пока частичные суммы не выходят
за предел, — поэтому любая перестановка слагаемых даёт тот же результат
до бита. Проба всё равно сравнивает байты: доказательство доказательством,
а замер замером.

Машина занята, поэтому сначала мерится сам фон (блок из 200 шагов
движка), и все отношения считаются попарно.

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\lifperf_spmv.py
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

from probe.lifperf import BRAIN_STEPS, build, med, pairs, sync  # noqa: E402

OUT = Path(__file__).parent
BASE_STEP_MS = 0.3203      # чистый шаг LIF базовой линии, спокойная машина


def main():                                                   # noqa: C901
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="сейчас")
    ap.add_argument("--rounds", type=int, default=21)
    args = ap.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = build(device)
    eng, n = env["engine"], env["n"]
    rates = env["rates"]
    R = args.rounds
    rep = {"метка": args.tag, "устройство": device, "нейронов": int(n)}

    # --------------------------------------------------- фон на этой машине
    for _ in range(50):
        eng.step(rates)
    sync()
    vals = []
    for _ in range(5):
        sync()
        t0 = time.perf_counter()
        for _ in range(BRAIN_STEPS):
            eng.step(rates)
        sync()
        vals.append((time.perf_counter() - t0) * 1000.0 / BRAIN_STEPS)
    step_ms = statistics.median(vals)
    rep["шаг движка сейчас, мс"] = round(step_ms, 4)
    rep["базовая линия (спокойная машина), мс"] = BASE_STEP_MS
    rep["фон, раз медленнее базовой линии"] = round(step_ms / BASE_STEP_MS, 2)
    print("шаг движка сейчас %.4f мс против %.4f базовой линии — фон ×%.2f"
          % (step_ms, BASE_STEP_MS, step_ms / BASE_STEP_MS))

    csr = eng.w_csr
    crow, col, val = csr.crow_indices(), csr.col_indices(), csr.values()
    rep["тип индексов движка"] = str(col.dtype)
    csr64 = torch.sparse_csr_tensor(crow.to(torch.int64), col.to(torch.int64),
                                    val, size=csr.shape, device=device)
    csr32 = torch.sparse_csr_tensor(crow.to(torch.int32), col.to(torch.int32),
                                    val, size=csr.shape, device=device)

    sp2 = (eng.spikes * eng.m_out)              # (1, n)
    sp1 = sp2[0].contiguous()                   # (n,)
    col2 = sp2.t().contiguous()                 # (n, 1)

    ways = {
        "mm, int64, (n,1) с транспонированием":
            lambda: torch.sparse.mm(csr64, sp2.t().contiguous()).t(),
        "mm, int32, (n,1) с транспонированием":
            lambda: torch.sparse.mm(csr32, sp2.t().contiguous()).t(),
        "mm, int64, готовый столбец":
            lambda: torch.sparse.mm(csr64, col2),
        "mm, int32, готовый столбец":
            lambda: torch.sparse.mm(csr32, col2),
        "mv, int64, вектор (n,)": lambda: torch.mv(csr64, sp1),
        "mv, int32, вектор (n,)": lambda: torch.mv(csr32, sp1),
        "matmul @, int64, вектор (n,)": lambda: csr64 @ sp1,
        "matmul @, int32, вектор (n,)": lambda: csr32 @ sp1,
    }

    # --------------------------------------------------- равенство побитово
    base = torch.sparse.mm(csr64, sp2.t().contiguous()).t().reshape(-1)
    rep["равенство побитово"] = {}
    for name, fn in ways.items():
        try:
            got = fn().reshape(-1)
            rep["равенство побитово"][name] = bool(np.array_equal(
                base.cpu().numpy().view(np.uint8),
                got.cpu().numpy().view(np.uint8)))
        except Exception as exc:                              # noqa: BLE001
            rep["равенство побитово"][name] = "не вышло: " + str(exc)[:120]
    print("\nравенство байтов с нынешней дорогой:")
    for k, v in rep["равенство побитово"].items():
        print("   %-44s %s" % (k, v))

    # ------------------------------------------------------ время, попарно
    print("\nвремя, попарно против нынешней дороги:")
    cur = ways["mm, int64, (n,1) с транспонированием"]
    rep["время"] = {}
    for name, fn in ways.items():
        if rep["равенство побитово"].get(name) is not True:
            continue
        p = pairs(cur, fn, rounds=R, reps=20)
        rep["время"][name] = p
        print("   %-44s %7.4f мс  в %.3f раза (%.3f…%.3f, %d пар)"
              % (name, p["b, мс"], p["a/b медиана"], p["a/b мин"],
                 p["a/b макс"], p["пар"]))

    best = max((v["a/b медиана"], k) for k, v in rep["время"].items())
    rep["лучшая дорога"] = best[1]
    rep["лучшее ускорение ядра"] = best[0]
    rep["стоит менять"] = bool(best[0] > 1.05)
    print("\n   лучшая: %s, в %.3f раза -> %s"
          % (best[1], best[0],
             "стоит менять" if rep["стоит менять"]
             else "не стоит: в пределах разброса"))

    # ----------------------------------- сколько байтов и какая это полоса
    for nm, idx_bytes in (("int64", 8), ("int32", 4)):
        mb = (crow.numel() * idx_bytes + col.numel() * idx_bytes
              + val.numel() * 4) / 1e6
        key = "mm, %s, (n,1) с транспонированием" % nm
        got = rep["время"].get(key)
        if got:
            band = mb / 1e3 / (got["b, мс"] / 1e3)
            rep["полоса, ГБ/с (%s)" % nm] = round(band, 1)
            print("   %s: %.1f МБ за %.4f мс = %.0f ГБ/с"
                  % (nm, mb, got["b, мс"], band))
    rep["полоса видеокарты, ГБ/с (паспорт RTX 4090)"] = 1008

    path = OUT / ("lifperf_spmv_%s.json" % args.tag)
    path.write_text(json.dumps(rep, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print("\nотчёт:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
