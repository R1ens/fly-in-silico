"""Попарные A/B по каждому кандидату в обвязке мозгового блока.

Машина занята, поэтому ни одно одиночное время здесь ничего не значит.
Каждый кандидат мерится ПОПАРНО: в одном круге подряд идут нынешний
вариант и предлагаемый, отношение берётся по кругу, в итог — медиана
отношений, разброс и число пар. Так посторонняя нагрузка попадает в оба
варианта, а не в один.

Кандидаты (все обязаны давать те же числа, и равенство тут же
проверяется сравнением результатов, а не обещанием):

1. накопление `acc += step()[0]` ВНУТРИ блока из 200 шагов, а не в
   отдельном замере: в отрыве ядро ждёт места в очереди, а внутри блока
   видеокарта и так занята;
2. запись частот `rates[0, idx] = v` против `rates.index_fill_`;
3. пять синхронизаций учёта против одной;
4. копия строки растра на хост: обычная память против закреплённой.

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\lifperf_ab.py
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

from probe.lifperf import BLOCK_S, BRAIN_STEPS, build, pairs, sync  # noqa: E402

OUT = Path(__file__).parent


def main():                                                   # noqa: C901
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="сейчас")
    ap.add_argument("--rounds", type=int, default=15)
    args = ap.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = build(device)
    eng, n = env["engine"], env["n"]
    rates, readout = env["rates"], env["readout"]
    t_ml, t_mr = env["t_ml"], env["t_mr"]
    n_ml, n_mr = env["n_ml"], env["n_mr"]
    R = args.rounds
    rep = {"метка": args.tag, "устройство": device, "пар в круге": R,
           "нейронов": int(n), "графов": len(eng._graphs or [])}
    print("устройство: %s, нейронов %d, графов %d"
          % (device, n, rep["графов"]))

    # ---------------------------------------- 1. накопление внутри блока
    print("\n1. накопление acc внутри блока из %d шагов" % BRAIN_STEPS)

    acc_p = torch.zeros(n, device=device)

    def with_acc():
        acc_p.zero_()
        for _ in range(BRAIN_STEPS):
            acc_p.add_(eng.step(rates)[0])

    def without_acc():
        for _ in range(BRAIN_STEPS):
            eng.step(rates)

    rep["накопление в блоке"] = pairs(with_acc, without_acc, rounds=R)
    p = rep["накопление в блоке"]
    rep["цена накопления на шаг управления, мс"] = round(
        p["a, мс"] - p["b, мс"], 3)
    print("   с накоплением %8.2f мс, без %8.2f мс, отношение %.4f "
          "(%.4f…%.4f, %d пар)"
          % (p["a, мс"], p["b, мс"], p["a/b медиана"], p["a/b мин"],
             p["a/b макс"], p["пар"]))
    print("   -> накопление стоит %.3f мс на шаг управления"
          % rep["цена накопления на шаг управления, мс"])

    # --------------------------------------------- 2. запись частот входа
    print("\n2. запись частот входа: срезом против index_fill_")
    i_l = env["t_sens"]["odour_left"]
    i_r = env["t_sens"]["odour_right"]
    i_t = env["t_sens"]["taste"]
    RL, RR, TT = 137.3, 91.7, 0.0

    def old_drive():
        rates.zero_()
        rates[0, i_l] = RL
        rates[0, i_r] = RR
        if TT > 0:
            rates[0, i_t] = TT

    def new_drive():
        rates.zero_()
        rates.index_fill_(1, i_l, RL)
        rates.index_fill_(1, i_r, RR)
        if TT > 0:
            rates.index_fill_(1, i_t, TT)

    old_drive()
    a = rates.clone()
    new_drive()
    b = rates.clone()
    rep["запись частот: равенство побитово"] = bool(
        np.array_equal(a.cpu().numpy().view(np.uint8),
                       b.cpu().numpy().view(np.uint8)))
    rep["запись частот"] = pairs(old_drive, new_drive, rounds=R, reps=10)
    p = rep["запись частот"]
    print("   побитово то же: %s" % rep["запись частот: равенство побитово"])
    print("   срезом %7.4f мс, index_fill_ %7.4f мс, в %.2f раза "
          "(%.2f…%.2f, %d пар)"
          % (p["a, мс"], p["b, мс"], p["a/b медиана"], p["a/b мин"],
             p["a/b макс"], p["пар"]))

    # -------------------------------------------- 3. пять съёмов или один
    print("\n3. учёт: пять синхронизаций против одной")
    acc_ref = torch.zeros(n, device=device)
    for _ in range(BRAIN_STEPS):
        acc_ref += eng.step(rates)[0]
    bins = torch.zeros(4, n, dtype=torch.int16, device=device)
    counts = torch.zeros(n, dtype=torch.int64, device=device)

    def old_account():
        dl = float(acc_ref[t_ml].sum()) / (n_ml * BLOCK_S)
        dr = float(acc_ref[t_mr].sum()) / (n_mr * BLOCK_S)
        rd = readout.read(acc_ref, BLOCK_S)
        bins[0] = acc_ref.to(torch.int16)
        counts.add_(acc_ref.to(torch.int64))
        s = int(acc_ref.sum())
        return dl, dr, rd, s

    flat = readout._flat

    def new_account():
        bins[0] = acc_ref.to(torch.int16)
        counts.add_(acc_ref.to(torch.int64))
        both = torch.cat((acc_ref[flat], acc_ref.sum().reshape(1)))
        host = both.cpu().numpy()
        vals = host[:-1].astype(np.float64)
        rd = {}
        for name, aa, bb, cc in readout._slices:
            nl, nr = readout._n[name]
            rd[name] = (float(vals[aa:bb].sum()) / (nl * BLOCK_S),
                        float(vals[bb:cc].sum()) / (nr * BLOCK_S))
        dl, dr = rd["__readout__"]
        s = int(host[-1])
        return dl, dr, rd, s

    oa, na = old_account(), new_account()
    same = (oa[0] == na[0] and oa[1] == na[1] and oa[3] == na[3]
            and all(oa[2][k] == na[2][k] for k in oa[2]))
    rep["учёт: числа совпали"] = bool(same)
    rep["учёт: пример"] = {"было": [oa[0], oa[1], oa[3]],
                           "стало": [na[0], na[1], na[3]]}
    rep["учёт"] = pairs(old_account, new_account, rounds=R, reps=10)
    p = rep["учёт"]
    print("   числа совпали: %s  (%s против %s)"
          % (same, oa[0], na[0]))
    print("   пять съёмов %7.4f мс, один %7.4f мс, в %.2f раза "
          "(%.2f…%.2f, %d пар)"
          % (p["a, мс"], p["b, мс"], p["a/b медиана"], p["a/b мин"],
             p["a/b макс"], p["пар"]))

    # ----------------------------------------- 4. строка растра на хост
    print("\n4. строка растра на хост: обычная память против закреплённой")
    pin = torch.empty(n, dtype=torch.int16, pin_memory=True)

    def old_row():
        return bins[0].cpu().numpy()

    def new_row():
        pin.copy_(bins[0], non_blocking=True)
        torch.cuda.current_stream().synchronize()
        return pin.numpy().copy()

    rep["строка на хост: равенство"] = bool(
        np.array_equal(old_row(), new_row()))
    rep["строка на хост"] = pairs(old_row, new_row, rounds=R, reps=20)
    p = rep["строка на хост"]
    print("   те же байты: %s" % rep["строка на хост: равенство"])
    print("   обычная %7.4f мс, закреплённая %7.4f мс, в %.2f раза "
          "(%.2f…%.2f, %d пар)"
          % (p["a, мс"], p["b, мс"], p["a/b медиана"], p["a/b мин"],
             p["a/b макс"], p["пар"]))

    # ------------------------------- 5. весь мозговой блок: было / станет
    print("\n5. мозговой блок целиком: нынешний против собранного из правок")

    def block_old():
        old_drive()
        acc = torch.zeros(n, device=device)
        for _ in range(BRAIN_STEPS):
            acc += eng.step(rates)[0]
        dl = float(acc[t_ml].sum()) / (n_ml * BLOCK_S)
        dr = float(acc[t_mr].sum()) / (n_mr * BLOCK_S)
        rd = readout.read(acc, BLOCK_S)
        bins[0] = acc.to(torch.int16)
        counts.add_(acc.to(torch.int64))
        s = int(acc.sum())
        row = bins[0].cpu().numpy()
        return dl, dr, rd, s, row

    def block_new():
        new_drive()
        acc_p.zero_()
        for _ in range(BRAIN_STEPS):
            acc_p.add_(eng.step(rates)[0])
        bins[0] = acc_p.to(torch.int16)
        counts.add_(acc_p.to(torch.int64))
        both = torch.cat((acc_p[flat], acc_p.sum().reshape(1)))
        host = both.cpu().numpy()
        vals = host[:-1].astype(np.float64)
        rd = {}
        for name, aa, bb, cc in readout._slices:
            nl, nr = readout._n[name]
            rd[name] = (float(vals[aa:bb].sum()) / (nl * BLOCK_S),
                        float(vals[bb:cc].sum()) / (nr * BLOCK_S))
        dl, dr = rd["__readout__"]
        s = int(host[-1])
        pin.copy_(bins[0], non_blocking=True)
        torch.cuda.current_stream().synchronize()
        return dl, dr, rd, s, pin.numpy().copy()

    rep["мозговой блок"] = pairs(block_old, block_new, rounds=R)
    p = rep["мозговой блок"]
    rep["выигрыш на шаге управления, мс"] = round(p["a, мс"] - p["b, мс"], 3)
    print("   нынешний %8.2f мс, с правками %8.2f мс, в %.4f раза "
          "(%.4f…%.4f, %d пар)"
          % (p["a, мс"], p["b, мс"], p["a/b медиана"], p["a/b мин"],
             p["a/b макс"], p["пар"]))
    print("   -> %.2f мс на шаг управления"
          % rep["выигрыш на шаге управления, мс"])

    path = OUT / ("lifperf_ab_%s.json" % args.tag)
    path.write_text(json.dumps(rep, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print("\nотчёт:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
