"""Профиль шага LIF: куда уходит время внутри `sim.Engine.step`.

Вопрос, на который отвечает эта проба, один: шаг упирается в работу или
в накладные расходы на запуск ядер. От ответа зависит, что вообще имеет
смысл оптимизировать.

Признак «упирается в запуски»: время шага почти не растёт, когда партия
становится больше. Тогда видеокарта простаивает между крошечными ядрами,
и выигрыш даёт не быстрая математика, а меньшее их число — объединение
операций или запись всего шага в CUDA-граф.

Ничего в проекте не меняется: всё считается на своей копии движка.

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\profile_lif.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

OUT = Path(__file__).parent


def timed(fn, n=200, warm=20):
    """Среднее время вызова, мс. Синхронизация честная, до и после."""
    import torch

    for _ in range(warm):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / n


def main():                                                   # noqa: C901
    import torch

    from flybrain import data as D, sim

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rep = {"устройство": device, "torch": torch.__version__}
    print("устройство:", device)

    neurons = D.load_neurons()
    weights = D.load_weights(device=device)
    n = len(neurons)
    rep["нейронов"] = int(n)
    rep["связей"] = int(weights._nnz()) if weights.is_sparse else None
    print("нейронов %d, связей %s" % (n, rep["связей"]))

    idx = D.select(neurons, "class:gustatory+side:left")

    # ---------------------------------------------------- шаг целиком
    print("\nшаг целиком при разных размерах партии:")
    rep["партии"] = {}
    for batch in (1, 2, 4, 8, 16):
        eng = sim.build_engine(weights=weights, neurons=neurons,
                               device=device, batch=batch, seed=0,
                               no_refractory=torch.as_tensor(idx,
                                                             device=device))
        rates = torch.zeros(batch, n, device=device)
        rates[:, idx] = 200.0
        ms = timed(lambda: eng.step(rates), n=300)
        rep["партии"][batch] = round(ms, 4)
        print("  партия %-3d %7.3f мс/шаг  (%.3f мс на прогон в партии)"
              % (batch, ms, ms / batch))
        del eng
        torch.cuda.empty_cache() if device == "cuda" else None

    one = rep["партии"][1]
    sixteen = rep["партии"][16]
    rep["рост от партии 1 к 16"] = round(sixteen / one, 3)
    launch_bound = sixteen < one * 4.0
    rep["упирается в запуски ядер"] = bool(launch_bound)
    print("\n  шаг вырос в %.2f раза при партии в 16 раз больше -> %s"
          % (sixteen / one,
             "упирается в ЗАПУСКИ ядер" if launch_bound
             else "упирается в РАБОТУ"))

    # ------------------------------------------- разбор по операциям
    print("\nпо операциям (партия 1):")
    eng = sim.build_engine(weights=weights, neurons=neurons, device=device,
                           batch=1, seed=0,
                           no_refractory=torch.as_tensor(idx, device=device))
    rates = torch.zeros(1, n, device=device)
    rates[:, idx] = 200.0
    sp = eng.spikes
    parts = {
        "пуассоновский вход (bernoulli)":
            lambda: torch.bernoulli(rates * eng.prob_scale, generator=eng.gen),
        "транспонирование спайков":
            lambda: eng.spikes.t().contiguous(),
        "разрежённое умножение (15 млн связей)":
            lambda: torch.sparse.mm(eng.w_csr, sp.t().contiguous()),
        "счётчик рефрактерности (where)":
            lambda: torch.where(eng.spikes > 0,
                                torch.zeros_like(eng.refrac),
                                eng.refrac + 1.0),
        "ворота рефрактерности (ge+float)":
            lambda: (eng.refrac >= eng.refrac_steps.unsqueeze(0)).float(),
        "чтение кольца задержек":
            lambda: eng.delay_buffer[:, eng.ptr, :],
        "запись в кольцо задержек":
            lambda: eng.delay_buffer[:, eng.ptr, :].copy_(eng.conductance),
        "мембрана: два сложения":
            lambda: eng.v + eng.tau_mem_f * (eng.conductance
                                             - (eng.v - eng.v_rest)),
        "порог (gt+float)": lambda: (eng.v > eng.v_th).float(),
        "сумма спайков по партии": lambda: eng.spikes.sum(0),
        "перенос суммы на хост": lambda: eng.spikes.sum(0).cpu().numpy(),
    }
    rep["операции, мс"] = {}
    for name, fn in parts.items():
        ms = timed(fn, n=300)
        rep["операции, мс"][name] = round(ms, 4)
        print("  %-42s %7.4f мс  %5.1f %% шага"
              % (name, ms, 100.0 * ms / one))

    # ------------------------------------------------ счёт запусков ядер
    try:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU,
                                 ProfilerActivity.CUDA]) as prof:
            for _ in range(20):
                eng.step(rates)
            torch.cuda.synchronize() if device == "cuda" else None
        evts = [e for e in prof.key_averages() if e.device_time_total > 0]
        total_dev = sum(e.device_time_total for e in evts) / 20.0 / 1000.0
        launches = sum(e.count for e in evts) / 20.0
        rep["ядер на шаг"] = round(launches, 1)
        rep["время на видеокарте, мс"] = round(total_dev, 4)
        rep["простой видеокарты, мс"] = round(one - total_dev, 4)
        print("\n  ядер на шаг: %.0f" % launches)
        print("  из них работы на видеокарте: %.3f мс из %.3f мс шага"
              % (total_dev, one))
        print("  то есть простаивает: %.3f мс (%.0f %%)"
              % (one - total_dev, 100.0 * (one - total_dev) / one))
        top = sorted(evts, key=lambda e: -e.device_time_total)[:8]
        rep["самые дорогие ядра"] = [
            {"ядро": e.key, "мс на шаг": round(e.device_time_total / 20
                                               / 1000.0, 4),
             "вызовов на шаг": round(e.count / 20.0, 1)} for e in top]
        print("\n  самые дорогие ядра:")
        for e in rep["самые дорогие ядра"]:
            print("    %-44s %7.4f мс  × %.1f"
                  % (e["ядро"][:44], e["мс на шаг"], e["вызовов на шаг"]))
    except Exception as exc:                                  # noqa: BLE001
        rep["профиль ядер"] = "не снялся: " + str(exc)[:120]
        print("  профиль ядер не снялся:", str(exc)[:120])

    # ----------------------------------------------- можно ли в CUDA-граф
    # Шаг статичен по формам: те же тензоры, те же ядра, тот же порядок.
    # Такое пишется в граф и запускается одним вызовом — математика та же,
    # меняется только число обращений к драйверу. Мешать может генератор
    # случайных чисел: его состояние должно быть захватываемым.
    rep["cuda-граф"] = {}
    if device == "cuda":
        try:
            eng2 = sim.build_engine(
                weights=weights, neurons=neurons, device=device, batch=1,
                seed=0, no_refractory=torch.as_tensor(idx, device=device))
            static_rates = rates.clone()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(5):
                    eng2.step(static_rates)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = eng2.step(static_rates)
            ms = timed(lambda: g.replay(), n=300)
            rep["cuda-граф"] = {"получилось": True,
                                "мс на шаг": round(ms, 4),
                                "ускорение": round(one / ms, 2)}
            print("\n  CUDA-граф: %.3f мс против %.3f мс — в %.2f раза"
                  % (ms, one, one / ms))
        except Exception as exc:                              # noqa: BLE001
            rep["cuda-граф"] = {"получилось": False,
                                "почему": str(exc)[:200]}
            print("\n  CUDA-граф не снялся:", str(exc)[:160])

    path = OUT / "profile_lif.json"
    path.write_text(json.dumps(rep, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print("\nотчёт:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
