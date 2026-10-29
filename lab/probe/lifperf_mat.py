"""Матрица связей: сколько байтов читает шаг и можно ли читать меньше.

Шаг движка упирается в видеокарту, а на видеокарте — в одно ядро:
`csrmv` по матрице связей. Поэтому вопрос один: сколько байтов эта
матрица весит и какая их часть нужна.

Два замера, и оба со сверкой результата, а не с обещанием.

**Точность суммы.** Все значения матрицы — ЦЕЛЫЕ (число синапсов со
знаком по нейротрансмиттеру), а спайк — ровно 0.0 или 1.0. Значит
произведение целое, и сумма целых во float32 точна при любом порядке
сложения, пока частичные суммы не выходят за 2^24 = 16 777 216. Это
проверяется здесь по максимуму суммы модулей в строке: если он меньше
2^24, то порядок суммирования на результат не влияет ВОВСЕ, и
перестановки слагаемых (другой тип индексов, пропуск нулей) остаются
побитовыми, а не «почти».

**Индексы.** Столбцовые индексы лежат в int64: 15 091 983 × 8 байт =
120,7 МБ на шаг против 60,4 МБ значений. В int32 они укладываются
(нейронов 138 639, связей 15 млн — обе величины меньше 2^31), и тогда
матрица худеет на треть.

**Событийное умножение.** Спайкуют десятки нейронов из 138 639, то есть
csrmv читает все 15 млн связей, чтобы умножить почти всё на ноль. Здесь
мерится ПОТОЛОК: сколько стоило бы сложить только строки спайкующих.

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\lifperf_mat.py
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

from probe.lifperf import build, med, pairs, sync  # noqa: E402

OUT = Path(__file__).parent


def main():                                                   # noqa: C901
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="сейчас")
    ap.add_argument("--rounds", type=int, default=15)
    args = ap.parse_args()

    import torch

    from flybrain import data as D

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = build(device)
    eng, n = env["engine"], env["n"]
    rates = env["rates"]
    R = args.rounds
    rep = {"метка": args.tag, "устройство": device, "нейронов": int(n)}

    csr = eng.w_csr
    crow, col, val = csr.crow_indices(), csr.col_indices(), csr.values()
    nnz = int(val.numel())
    bts = (crow.numel() * crow.element_size() + col.numel()
           * col.element_size() + val.numel() * val.element_size())
    rep["связей"] = nnz
    rep["матрица, МБ"] = round(bts / 1e6, 1)
    rep["тип индексов"] = str(col.dtype)
    rep["столбцовые индексы, МБ"] = round(
        col.numel() * col.element_size() / 1e6, 1)
    rep["значения, МБ"] = round(val.numel() * val.element_size() / 1e6, 1)
    print("связей %d, матрица %.1f МБ (индексы %.1f + значения %.1f)"
          % (nnz, bts / 1e6, rep["столбцовые индексы, МБ"],
             rep["значения, МБ"]))

    # ------------------------------------------- 1. точность суммы по строке
    print("\n1. точность суммы: все ли значения целые и влезают ли суммы")
    rep["все значения целые"] = bool(torch.all(val == val.round()).item())
    rep["значение мин"] = float(val.min())
    rep["значение макс"] = float(val.max())
    absrow = torch.zeros(n + 1, device=device, dtype=torch.float64)
    a64 = val.abs().to(torch.float64)
    # сумма модулей по строке: по границам crow
    cs = torch.cat((torch.zeros(1, device=device, dtype=torch.float64),
                    torch.cumsum(a64, 0)))
    rowsum = cs[crow[1:]] - cs[crow[:-1]]
    rep["макс сумма модулей в строке"] = float(rowsum.max())
    rep["предел точности float32 (2^24)"] = 16777216
    rep["порядок сложения на результат не влияет"] = bool(
        rep["все значения целые"]
        and rep["макс сумма модулей в строке"] < 16777216)
    print("   все значения целые: %s, от %.0f до %.0f"
          % (rep["все значения целые"], rep["значение мин"],
             rep["значение макс"]))
    print("   макс сумма модулей в строке: %.0f против предела 16 777 216"
          % rep["макс сумма модулей в строке"])
    print("   -> порядок сложения %s"
          % ("НЕ ВЛИЯЕТ на результат"
             if rep["порядок сложения на результат не влияет"]
             else "ВЛИЯЕТ, перестановки небезопасны"))
    del absrow, a64, cs, rowsum
    torch.cuda.empty_cache()

    # --------------------------------------------- 2. индексы int32
    print("\n2. индексы int32 против int64")
    try:
        # Обе матрицы собираются ЗДЕСЬ, из одних и тех же значений, а не
        # берутся как лежат: движок после правки держит int32, и без
        # явной сборки сравнивать было бы нечего с чем.
        csr64 = torch.sparse_csr_tensor(
            crow.to(torch.int64), col.to(torch.int64), val,
            size=csr.shape, device=device)
        csr32 = torch.sparse_csr_tensor(
            crow.to(torch.int32), col.to(torch.int32), val,
            size=csr.shape, device=device)
        vec = (eng.spikes * eng.m_out).t().contiguous()
        o64 = torch.sparse.mm(csr64, vec)
        o32 = torch.sparse.mm(csr32, vec)
        same = bool(np.array_equal(o64.cpu().numpy().view(np.uint8),
                                   o32.cpu().numpy().view(np.uint8)))
        rep["int32: побитово то же"] = same
        rep["int32: матрица, МБ"] = round(
            (crow.numel() * 4 + col.numel() * 4
             + val.numel() * val.element_size()) / 1e6, 1)
        rep["int64: матрица, МБ"] = round(
            (crow.numel() * 8 + col.numel() * 8
             + val.numel() * val.element_size()) / 1e6, 1)
        rep["int32"] = pairs(lambda: torch.sparse.mm(csr64, vec),
                             lambda: torch.sparse.mm(csr32, vec),
                             rounds=R, reps=20)
        p = rep["int32"]
        print("   побитово то же: %s" % same)
        print("   матрица худеет с %.1f до %.1f МБ"
              % (rep["int64: матрица, МБ"], rep["int32: матрица, МБ"]))
        print("   int64 %7.4f мс, int32 %7.4f мс, в %.3f раза "
              "(%.3f…%.3f, %d пар)"
              % (p["a, мс"], p["b, мс"], p["a/b медиана"], p["a/b мин"],
                 p["a/b макс"], p["пар"]))
        # и весь шаг движка: граф на int32 придётся перезахватить, поэтому
        # здесь мерится только ядро; шаг целиком — после правки в sim.py
    except Exception as exc:                                  # noqa: BLE001
        rep["int32"] = {"не вышло": str(exc)[:200]}
        print("   не вышло:", str(exc)[:200])
        csr32 = None

    # ---------------------------------------- 3. событийное умножение: потолок
    print("\n3. событийное умножение: потолок выигрыша")
    # Матрица по ПРЕсинаптическому: строки W^T — исходящие связи нейрона.
    with D._quiet_sparse():
        wt = csr.t().to_sparse_csr() if hasattr(csr, "t") else None
    if wt is not None:
        crow_t, col_t, val_t = wt.crow_indices(), wt.col_indices(), wt.values()

        # прогоняем движок, чтобы спайки были настоящие
        for _ in range(50):
            eng.step(rates)
        sp = (eng.spikes * eng.m_out)[0]
        act = torch.nonzero(sp > 0).squeeze(1)
        rep["спайкующих нейронов"] = int(act.numel())
        # сколько связей у них
        deg = (crow_t[1:] - crow_t[:-1])
        rep["связей у спайкующих"] = int(deg[act].sum().item())
        rep["доля связей"] = round(rep["связей у спайкующих"] / nnz, 6)
        print("   спайкуют %d нейронов, у них %d связей из %d (%.4f %%)"
              % (rep["спайкующих нейронов"], rep["связей у спайкующих"],
                 nnz, 100.0 * rep["доля связей"]))

        def event_mm():
            a = torch.nonzero(sp > 0).squeeze(1)
            starts, ends = crow_t[a], crow_t[a + 1]
            cnt = ends - starts
            tot = int(cnt.sum().item())
            off = torch.repeat_interleave(starts, cnt)
            base = torch.arange(tot, device=device) \
                - torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
            pos = off + base
            out = torch.zeros(n, device=device)
            out.index_add_(0, col_t[pos], val_t[pos])
            return out

        ref = torch.sparse.mm(csr, (eng.spikes * eng.m_out).t().contiguous())
        got = event_mm()
        d = (ref.squeeze(1) - got).abs().max().item()
        rep["событийное: макс расхождение"] = float(d)
        rep["событийное: побитово то же"] = bool(
            np.array_equal(ref.squeeze(1).cpu().numpy().view(np.uint8),
                           got.cpu().numpy().view(np.uint8)))
        rep["событийное"] = pairs(
            lambda: torch.sparse.mm(csr,
                                    (eng.spikes * eng.m_out).t().contiguous()),
            event_mm, rounds=R, reps=10)
        p = rep["событийное"]
        print("   макс расхождение %.6g, побитово то же: %s"
              % (d, rep["событийное: побитово то же"]))
        print("   csrmv %7.4f мс, событийно %7.4f мс, в %.2f раза "
              "(%.2f…%.2f, %d пар)"
              % (p["a, мс"], p["b, мс"], p["a/b медиана"], p["a/b мин"],
                 p["a/b макс"], p["пар"]))
        # отдельно: сколько стоит сама nonzero (она синхронизирует и
        # даёт ДИНАМИЧЕСКУЮ форму, то есть в CUDA-граф не пишется)
        rep["nonzero, мс"] = med(lambda: torch.nonzero(sp > 0),
                                 rounds=R, reps=20)["мс"]
        print("   из них torch.nonzero: %.4f мс (и форма динамическая — "
              "в граф не пишется)" % rep["nonzero, мс"])

    path = OUT / ("lifperf_mat_%s.json" % args.tag)
    path.write_text(json.dumps(rep, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print("\nотчёт:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
