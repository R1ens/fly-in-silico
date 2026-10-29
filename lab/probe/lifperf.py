"""Профиль мозговой части шага управления: чистый счёт отдельно от обвязки.

Вопрос один: из чего складываются 73,93 мс «мозг: LIF, синхронизация,
учёт» на шаге управления в 20 мс, и сколько из них вообще можно убрать,
не меняя ни одного числа.

Чем эта проба отличается от `probe/profile_lif.py`: та мерит ядро в
отрыве (и мерила его ДО графов), а здесь воспроизводится ровно тот блок,
который гоняет `body.run_episode` на каждом шаге управления — движок,
накопление, два съёма, измеритель зондов, растр, счётчики и копия строки
на хост. Тело MuJoCo не поднимается: оно в мозговую часть не входит, а
стоит тридцать секунд сборки.

Главный встречный замер, без которого разбор ничего не значит: время
ОЧЕРЕДИ против времени ИСПОЛНЕНИЯ. Блок ставится в очередь без
ожидания — если на это уходит заметно меньше, чем на его исполнение, то
шаг упирается в видеокарту, и убирать обращения к драйверу
бессмысленно. Если наоборот — упирается в драйвер, и тогда счёт ядер и
есть цель. Очередь мерится на трёх длинах блока: на длинном блоке она
упирается в заполнение аппаратной очереди и показывает не стоимость
постановки, а ожидание места.

**Машина занята.** Одновременно считают другие прогоны, поэтому любое
одиночное число здесь — с посторонней нагрузкой. Отношения считаются
попарно: в одном круге подряд меряются оба варианта, отношение берётся
по кругу, а в итог идёт медиана отношений, разброс и число пар.

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\lifperf.py
"""

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

OUT = Path(__file__).parent
BRAIN_STEPS = 200          # шагов движка на шаг управления (20 мс / 0,1 мс)
CONTROL_MS = 20.0
BLOCK_S = BRAIN_STEPS * 0.1 / 1000.0


def sync():
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def med(fn, rounds=11, warm=3, reps=1):
    """Медиана по кругам, мс на ОДИН вызов. Синхронизация до и после круга.

    ``reps`` — сколько раз вызвать внутри круга. Для мелких действий это
    обязательно: на занятой машине одиночное крошечное ядро ждёт места в
    аппаратной очереди около полумиллисекунды, и замер показывал бы это
    ожидание, а не стоимость действия.
    """
    for _ in range(warm):
        fn()
    sync()
    vals = []
    for _ in range(rounds):
        sync()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        sync()
        vals.append((time.perf_counter() - t0) * 1000.0 / reps)
    return {"мс": round(statistics.median(vals), 4),
            "мин": round(min(vals), 4), "макс": round(max(vals), 4),
            "кругов": rounds, "повторов в круге": reps}


def pairs(a, b, rounds=11, warm=2, reps=1):
    """Два варианта попарно в одном круге. Возвращает отношение a/b.

    Это единственный способ что-то сравнить на занятой машине: оба
    варианта видят одну и ту же постороннюю нагрузку, потому что стоят
    рядом во времени.
    """
    for _ in range(warm):
        a()
        b()
    sync()
    ra, rb, rr = [], [], []
    for _ in range(rounds):
        sync()
        t0 = time.perf_counter()
        for _ in range(reps):
            a()
        sync()
        t1 = time.perf_counter()
        for _ in range(reps):
            b()
        sync()
        t2 = time.perf_counter()
        va = (t1 - t0) * 1000.0 / reps
        vb = (t2 - t1) * 1000.0 / reps
        ra.append(va)
        rb.append(vb)
        rr.append(va / vb if vb else float("nan"))
    return {"a, мс": round(statistics.median(ra), 4),
            "b, мс": round(statistics.median(rb), 4),
            "a/b медиана": round(statistics.median(rr), 4),
            "a/b мин": round(min(rr), 4), "a/b макс": round(max(rr), 4),
            "пар": rounds}


def fp(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def build(device):
    """Движок и окружение ровно как у `body.run_episode`."""
    import torch

    from flybrain import data as D, motor, sim, world

    neurons = D.load_neurons()
    weights = D.load_weights(device=device)
    n = len(neurons)

    cands = world.probe_candidates(neurons)
    cands = {**cands, **world.dn_probe_candidates()}
    ql, qr = cands["ALPN"]
    motor_l, motor_r = D.select(neurons, ql), D.select(neurons, qr)
    sensor_idx = {
        "odour_left": D.select(neurons, world.SENSORS["odour_left"]),
        "odour_right": D.select(neurons, world.SENSORS["odour_right"]),
        "taste": D.select(neurons, world.SENSORS["taste"]),
    }
    driven = np.unique(np.concatenate(list(sensor_idx.values())))
    engine = sim.build_engine(weights=weights, neurons=neurons, device=device,
                              seed=0, no_refractory=driven)
    readout = motor.DnReadout(neurons, device, pairs={"__readout__": (ql, qr)},
                              strict=False)
    t_sens = {k: torch.as_tensor(v, device=device)
              for k, v in sensor_idx.items()}
    rates = torch.zeros(1, n, device=device)
    rates[0, t_sens["odour_left"]] = 150.0
    rates[0, t_sens["odour_right"]] = 150.0
    return dict(n=n, engine=engine, readout=readout, rates=rates,
                t_sens=t_sens, neurons=neurons,
                t_ml=torch.as_tensor(motor_l, device=device),
                t_mr=torch.as_tensor(motor_r, device=device),
                n_ml=len(motor_l), n_mr=len(motor_r))


def main():                                                   # noqa: C901
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="сейчас")
    ap.add_argument("--rounds", type=int, default=11)
    ap.add_argument("--kernels", action="store_true",
                    help="снять счёт ядер профилировщиком torch")
    args = ap.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rep = {"метка": args.tag, "устройство": device, "torch": torch.__version__,
           "шагов движка на шаг управления": BRAIN_STEPS}
    print("устройство:", device, "| torch", torch.__version__)

    env = build(device)
    eng, n = env["engine"], env["n"]
    rates = env["rates"]
    rep["нейронов"] = int(n)
    rep["графы захвачены"] = eng._graphs is not None
    rep["графов"] = 0 if eng._graphs is None else len(eng._graphs)
    print("нейронов %d, графов %s" % (n, rep["графов"]))

    R = args.rounds

    # ---------------------------------------------- 1. движок: счёт и очередь
    print("\n1. движок: счёт против очереди")

    def block(k=BRAIN_STEPS):
        for _ in range(k):
            eng.step(rates)

    rep["блок движка (200 шагов), мс"] = med(block, rounds=R)
    blk = rep["блок движка (200 шагов), мс"]["мс"]
    rep["шаг движка, мс"] = round(blk / BRAIN_STEPS, 4)

    rep["очередь, мс на шаг"] = {}
    for k in (10, 50, 200):
        for _ in range(2):
            block(k)
        sync()
        qv = []
        for _ in range(R):
            sync()
            t0 = time.perf_counter()
            block(k)
            qv.append((time.perf_counter() - t0) * 1000.0 / k)
            sync()
        rep["очередь, мс на шаг"][k] = round(statistics.median(qv), 4)
    q10 = rep["очередь, мс на шаг"][10]
    rep["постановка шага в очередь (по блоку 10), мс"] = q10
    rep["доля постановки в шаге"] = round(q10 / rep["шаг движка, мс"], 4)
    rep["упирается в"] = ("драйвер (постановка)"
                          if q10 > 0.75 * rep["шаг движка, мс"]
                          else "видеокарту (счёт)")
    print("   блок 200 шагов    %8.2f мс  (%.4f мс на шаг)"
          % (blk, blk / BRAIN_STEPS))
    for k, v in rep["очередь, мс на шаг"].items():
        print("   очередь блока %-3s %8.4f мс на шаг" % (k, v))
    print("   -> упирается в %s" % rep["упирается в"])

    # --------------------------------------- 2. из чего состоит шаг движка
    print("\n2. части шага движка, мс на один шаг")
    parts = {}
    if eng._graphs is not None:
        gl = eng._graphs

        def replay1():
            gl[0].replay()
        parts["повтор графа (без пуассона)"] = med(replay1, rounds=R, reps=50)

    parts["пуассоновский вход"] = med(
        lambda: eng._poisson(rates), rounds=R, reps=50)
    if eng._vstim is not None:
        parts["пуассон + копия в буфер графа"] = med(
            lambda: eng._vstim.copy_(eng._poisson(rates)), rounds=R, reps=50)
    parts["разрежённое умножение (15 млн связей)"] = med(
        lambda: torch.sparse.mm(eng.w_csr,
                                (eng.spikes * eng.m_out).t().contiguous()),
        rounds=R, reps=50)
    acc_t = torch.zeros(n, device=device)
    parts["накопление acc += spikes[0]"] = med(
        lambda: acc_t.add_(eng.spikes[0]), rounds=R, reps=50)
    rep["части шага, мс"] = parts
    for k, v in parts.items():
        print("   %-44s %8.4f мс  (%.0f %% шага)"
              % (k, v["мс"], 100.0 * v["мс"] / rep["шаг движка, мс"]))

    # ------------------------------------------------- 3. счёт ядер и работа
    if args.kernels:
        print("\n3. ядра на шаг движка")
        try:
            from torch.profiler import ProfilerActivity, profile

            with profile(activities=[ProfilerActivity.CPU,
                                     ProfilerActivity.CUDA]) as prof:
                for _ in range(20):
                    eng.step(rates)
                sync()
            evts = [e for e in prof.key_averages() if e.device_time_total > 0]
            dev = sum(e.device_time_total for e in evts) / 20.0 / 1000.0
            launches = sum(e.count for e in evts) / 20.0
            rep["ядер на шаг"] = round(launches, 2)
            rep["работы на видеокарте, мс/шаг"] = round(dev, 4)
            print("   ядер на шаг %.1f, работы на видеокарте %.4f мс"
                  % (launches, dev))
            top = sorted(evts, key=lambda e: -e.device_time_total)[:6]
            rep["самые дорогие ядра"] = [
                {"ядро": e.key[:58],
                 "мс на шаг": round(e.device_time_total / 20 / 1000.0, 4),
                 "вызовов на шаг": round(e.count / 20.0, 2)} for e in top]
            for e in rep["самые дорогие ядра"]:
                print("     %-56s %7.4f мс × %.2f"
                      % (e["ядро"][:56], e["мс на шаг"], e["вызовов на шаг"]))
        except Exception as exc:                              # noqa: BLE001
            rep["профиль ядер"] = "не снялся: " + str(exc)[:160]
            print("   не снялся:", str(exc)[:160])

    # ------------------------------------------- 4. обвязка шага управления
    print("\n4. обвязка: что стоит учёт на шаге управления")
    readout = env["readout"]
    t_ml, t_mr = env["t_ml"], env["t_mr"]
    n_ml, n_mr = env["n_ml"], env["n_mr"]
    bins = torch.zeros(4, n, dtype=torch.int16, device=device)
    counts = torch.zeros(n, dtype=torch.int64, device=device)
    acc_ref = torch.zeros(n, device=device)
    for _ in range(BRAIN_STEPS):
        acc_ref += eng.spikes[0]

    over = {}
    over["заведение acc (torch.zeros)"] = med(
        lambda: torch.zeros(n, device=device), rounds=R, reps=50)
    over["rates.zero_() + три записи по индексам"] = med(
        lambda: (rates.zero_(),
                 rates.__setitem__((0, env["t_sens"]["odour_left"]), 150.0),
                 rates.__setitem__((0, env["t_sens"]["odour_right"]), 150.0),
                 rates.__setitem__((0, env["t_sens"]["taste"]), 0.0)),
        rounds=R, reps=20)
    over["СИНХР float(acc[t_ml].sum()) — съём слева"] = med(
        lambda: float(acc_ref[t_ml].sum()) / (n_ml * BLOCK_S),
        rounds=R, reps=20)
    over["СИНХР float(acc[t_mr].sum()) — съём справа"] = med(
        lambda: float(acc_ref[t_mr].sum()) / (n_mr * BLOCK_S),
        rounds=R, reps=20)
    over["СИНХР readout.read — все зонды одним съёмом"] = med(
        lambda: readout.read(acc_ref, BLOCK_S), rounds=R, reps=20)
    over["bins[row] = acc.to(int16)"] = med(
        lambda: bins.__setitem__(0, acc_ref.to(torch.int16)),
        rounds=R, reps=50)
    over["counts += acc.to(int64)"] = med(
        lambda: counts.add_(acc_ref.to(torch.int64)), rounds=R, reps=50)
    over["СИНХР int(acc.sum()) — спайки в запись"] = med(
        lambda: int(acc_ref.sum()), rounds=R, reps=20)
    over["СИНХР bins[row].cpu() — строка на хост (277 КБ)"] = med(
        lambda: bins[0].cpu().numpy(), rounds=R, reps=20)
    rep["обвязка шага управления, мс"] = over
    rep["обвязка всего, мс"] = round(sum(v["мс"] for v in over.values()), 3)
    rep["из них синхронизаций"] = sum(1 for k in over if k.startswith("СИНХР"))
    rep["обвязка: синхронизации, мс"] = round(
        sum(v["мс"] for k, v in over.items() if k.startswith("СИНХР")), 3)
    for k, v in over.items():
        print("   %-52s %7.4f мс" % (k, v["мс"]))
    print("   %-52s %7.3f мс (из них %d синхронизаций на %.3f мс)"
          % ("ИТОГО обвязки", rep["обвязка всего, мс"],
             rep["из них синхронизаций"], rep["обвязка: синхронизации, мс"]))

    # -------------------------------------- 5. мозговой блок против счёта
    print("\n5. мозговой блок против чистого счёта, попарно")

    def brain_block():
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

    rep["блок мозга / чистый счёт"] = pairs(brain_block, block, rounds=R)
    pb = rep["блок мозга / чистый счёт"]
    rep["обвязка по вычитанию, мс"] = round(pb["a, мс"] - pb["b, мс"], 3)
    print("   блок мозга %8.2f мс, чистый счёт %8.2f мс, отношение %.4f "
          "(%.4f…%.4f по %d парам)"
          % (pb["a, мс"], pb["b, мс"], pb["a/b медиана"], pb["a/b мин"],
             pb["a/b макс"], pb["пар"]))
    print("   обвязка по вычитанию: %.2f мс" % rep["обвязка по вычитанию, мс"])

    # -------------------------------------------- 6. насколько редки спайки
    print("\n6. разрежённость спайков (чего стоило бы событийное умножение)")
    live = []
    for _ in range(50):
        eng.step(rates)
        live.append(int(eng.spikes.sum().item()))
    rep["спайков на шаг"] = {"медиана": int(statistics.median(live)),
                             "мин": int(min(live)), "макс": int(max(live))}
    frac = statistics.median(live) / n
    rep["доля активных нейронов"] = round(frac, 6)
    rep["связей всего"] = int(eng.w_csr.values().numel())
    rep["связей у спайкующих, оценка"] = int(rep["связей всего"] * frac)
    print("   спайков на шаг: медиана %d из %d (%.3f %%)"
          % (rep["спайков на шаг"]["медиана"], n, 100.0 * frac))
    print("   значит событийное умножение читало бы ~%d связей из %d"
          % (rep["связей у спайкующих, оценка"], rep["связей всего"]))

    # -------------------------------------------------- 7. счётчики и подписи
    #
    # Отпечатки не ловят «ускорились, потому что перестали считать»:
    # пропущенный шаг меняет и отпечаток, но столько же меняет и правка
    # арифметики, а это разные беды. Поэтому рядом с отпечатками стоят
    # счётчики: вызовов движка на шаг управления обязано быть ровно 200,
    # и взятий генератора — столько же, потому что `_poisson` зовётся по
    # разу на шаг и ровно одним `torch.bernoulli`. Подделать такую
    # подпись нельзя: счётчик `calls` растёт внутри самого шага, а взятия
    # считаются подменой `torch.bernoulli` на время замера.
    print("\n7. счётчики")
    eng.calls = 0
    gen0 = eng.gen.get_state().cpu().numpy().copy()
    draws = {"n": 0}
    real_bernoulli = torch.bernoulli

    def counted(*a, **kw):
        draws["n"] += 1
        return real_bernoulli(*a, **kw)

    torch.bernoulli = counted
    try:
        acc = torch.zeros(n, device=device)
        for _ in range(BRAIN_STEPS):
            acc += eng.step(rates)[0]
        sync()
    finally:
        torch.bernoulli = real_bernoulli
    rep["взятий генератора на шаг управления"] = int(draws["n"])
    rep["взятий генератора обязано быть"] = BRAIN_STEPS
    rep["счётчики сошлись"] = bool(int(eng.calls) == BRAIN_STEPS
                                   and int(draws["n"]) == BRAIN_STEPS)
    print("   взятий генератора: %d (обязано быть %d)"
          % (draws["n"], BRAIN_STEPS))
    rep["вызовов движка на шаг управления"] = int(eng.calls)
    rep["состояние генератора до"] = fp(gen0)
    rep["состояние генератора после 200 шагов"] = fp(
        eng.gen.get_state().cpu().numpy())
    rep["генератор сдвинулся"] = bool(
        rep["состояние генератора до"]
        != rep["состояние генератора после 200 шагов"])
    print("   вызовов движка: %d (обязано быть %d)"
          % (eng.calls, BRAIN_STEPS))
    print("   генератор: %s -> %s"
          % (rep["состояние генератора до"],
             rep["состояние генератора после 200 шагов"]))

    path = OUT / ("lifperf_%s.json" % args.tag)
    path.write_text(json.dumps(rep, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print("\nотчёт:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
