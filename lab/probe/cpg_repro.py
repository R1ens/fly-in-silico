"""НЕЗАВИСИМОЕ воспроизведение ядра генератора ритма из опубликованной работы.

Свой код, свои решатель и проверки; из скачанного читаются ТОЛЬКО таблицы
(матрица связей и таблица нейронов). Ни одна строка опубликованного кода не
исполняется.

Уравнение взято из текста статьи/репозитория (`rate_equation_half_tanh`):

    total_input = I(t) + Wr @ R
    activation  = max( fr_cap * tanh( (a/fr_cap) * (total_input - threshold) ), 0 )
    dR/dt       = (activation - R) / tau
    Wr          = exc_mult * max(W.T, 0) + inh_mult * min(W.T, 0)

Запуск (каталог опубликованного репозитория передаётся АРГУМЕНТОМ, из него
ничего не исполняется, только читаются `.csv`):

    python -I -X utf8 lab/probe/cpg_repro.py <каталог Pugliese_2026>

Данные в дерево не кладутся: нужны два файла из `data/manc t1 connectome
data/` опубликованного репозитория (`https://github.com/smpuglie/Pugliese_2026`,
код MIT по README, данные CC BY 4.0). Разбор — `lab/flybrain/CPG_MOTOR.md` §3.

Эта проба УМЕЕТ ПРОВАЛИТЬСЯ, и это проверено: перемешивание связности ядра
убирает колебание на 4 зёрнах из 5 (§3.4 отчёта). Пороги объявлены в коде
НИЖЕ константой, до первого прогона: OSC_THRESHOLD = 0.5 — порог самой
публикации.

Видеокарта не нужна: всё на процессоре, numpy, около двух минут.
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
PUB = Path(sys.argv[1])

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

D = PUB / "data" / "manc t1 connectome data"
tab = pd.read_csv(D / "wTable_20250813_DNtoMN_unsorted_withModules.csv",
                  low_memory=False)
W = pd.read_csv(D / "W_20250813_DNtoMN_unsorted.csv", index_col=0).to_numpy(
    dtype=np.float64)
n = W.shape[0]
print(f"матрица связей: {W.shape}, таблица: {len(tab)} нейронов")
assert W.shape[0] == W.shape[1] == len(tab), "размеры не сошлись"
print(f"ненулевых рёбер: {int((W != 0).sum())}, "
      f"возбуждающих {int((W > 0).sum())}, тормозных {int((W < 0).sum())}")

# ---- объявленные ДО замера параметры (configs/neuron_params/default.yaml)
P = dict(tau=0.02, a=1.0, threshold=7.5, fr_cap=200.0,
         exc_mult=0.03, inh_mult=0.03)
P_STD = dict(tau=0.002, a=0.1, threshold=0.6, fr_cap=10.0)
T, DT_SAVE = 2.0, 0.001
PULSE = (0.02, 1.999)
CORE = [31, 277, 617, 1167]          # DNg100 + IN17A001 + INXXX466 + IN16B036
STIM_IDX, STIM_I = 31, 250.0         # из DNg100_Stim_CoreCPG.yaml
OSC_THRESHOLD = 0.5                  # их же порог

mn_mask = tab["class"].astype("string").str.contains(
    "motor", case=False, na=False).to_numpy()
print(f"мотонейронов: {int(mn_mask.sum())}")
for i in CORE:
    print(f"  ядро idx {i}: {tab.iloc[i]['type']} "
          f"({tab.iloc[i]['class']}, nt={tab.iloc[i]['predictedNt']})")


def reweight(Wm, exc, inh):
    Wt = Wm.T
    return exc * np.maximum(Wt, 0.0) + inh * np.minimum(Wt, 0.0)


def run(Wr, inputs, tau, a, thr, cap, dt=2e-5):
    """RK4 с мелким шагом; выдаёт R на сетке DT_SAVE."""
    steps = int(round(T / dt))
    every = int(round(DT_SAVE / dt))
    R = np.zeros(Wr.shape[0])
    out = []

    def f(t, R):
        I = inputs if (PULSE[0] <= t <= PULSE[1]) else 0.0 * inputs
        ti = I + Wr @ R
        act = np.maximum(cap * np.tanh((a / cap) * (ti - thr)), 0.0)
        return (act - R) / tau

    for s in range(steps):
        t = s * dt
        k1 = f(t, R)
        k2 = f(t + dt / 2, R + dt / 2 * k1)
        k3 = f(t + dt / 2, R + dt / 2 * k2)
        k4 = f(t + dt, R + dt * k3)
        R = R + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if s % every == 0:
            out.append(R.copy())
    return np.array(out)


def osc_score(x):
    """Оценка колебательности: автокорреляция, нормированная на эталонную
    синусоиду той же частоты — как в опубликованной работе."""
    x = np.asarray(x, dtype=np.float64)
    rng = x.max() - x.min()
    if rng <= 1e-6:
        return 0.0, 0.0
    z = 2 * (x - x.min()) / rng - 1

    def raw(v):
        ac = np.correlate(v, v, mode="full")[len(v) - 1:]
        if ac[0] <= 0:
            return 0.0, 0.0
        ac = ac / ac[0]
        # первый внутренний максимум после первого спуска ниже нуля
        neg = np.where(ac < 0)[0]
        if not len(neg):
            return 0.0, 0.0
        k = neg[0]
        if k >= len(ac) - 2:
            return 0.0, 0.0
        j = k + int(np.argmax(ac[k:]))
        return float(ac[j]), (1.0 / j if j else 0.0)

    r, freq = raw(z)
    if r <= 1e-6 or freq <= 0:
        return 0.0, 0.0
    t = np.arange(len(z))
    ref = max(raw(np.sin(2 * np.pi * freq * t))[0],
              raw(np.cos(2 * np.pi * freq * t))[0])
    return (r / ref if ref > 1e-6 else 0.0), freq / DT_SAVE


# ---------------------------------------------- урезанная сеть, как keepOnly
keep = np.zeros(n, dtype=bool)
keep[CORE] = True
keep = keep | mn_mask          # keepOnly ИЛИ мотонейроны (так в их коде)
print(f"\nв урезанной сети: {int(keep.sum())} нейронов "
      f"({len(CORE)} ядра + {int(mn_mask.sum())} мотонейронов)")

# Сужение до оставленных индексов ТОЧНО равно обнулению строк и столбцов:
# у отключённого нейрона total_input = 0, а activation при threshold > 0
# равна max(cap*tanh(-a*thr/cap), 0) = 0, значит R -> 0 и вклада он не даёт.
# Проверяется отдельно ниже на малом прогоне.
kidx = np.where(keep)[0]
Wk = W[np.ix_(kidx, kidx)].copy()
mn_k = mn_mask[kidx]
core_k = [int(np.where(kidx == i)[0][0]) for i in CORE]
stim_k = int(np.where(kidx == STIM_IDX)[0][0])
print(f"  сужено до {Wk.shape[0]} нейронов; ядро -> {core_k}, "
      f"раздражаемый -> {stim_k}")

print("\n" + "=" * 78)
print("ПОРОГИ ОБЪЯВЛЕНЫ ДО ЗАМЕРА: колебанием считается оценка >= 0,5")
print("на мотонейронах, частота в коридоре 5...25 Гц (ходьба мухи)")
print("=" * 78)

def measure(Wuse, mnm, cores, stim, tau=None, a=None, thr=None, cap=None,
            dt=5e-5):
    m = Wuse.shape[0]
    Wr = reweight(Wuse, P["exc_mult"], P["inh_mult"])
    inputs = np.zeros(m)
    inputs[stim] = STIM_I
    R = run(Wr, inputs,
            P["tau"] if tau is None else tau,
            P["a"] if a is None else a,
            P["threshold"] if thr is None else thr,
            P["fr_cap"] if cap is None else cap, dt=dt)
    tail = R[int(0.25 / DT_SAVE):]          # отбросить разгон
    active = (tail.max(axis=0) > 1.0)
    sc, fq = [], []
    for i in np.where(active & mnm)[0]:
        s, fr = osc_score(tail[:, i])
        sc.append(s)
        fq.append(fr)
    return tail, active, np.array(sc), np.array(fq)


tail, active, scores, freqs = measure(Wk, mn_k, core_k, stim_k)
print("\n--- ядро + мотонейроны (keepOnly), эталонные параметры ---")
print(f"  активных нейронов всего: {int(active.sum())}, "
      f"из них мотонейронов: {int((active & mn_k).sum())}")
print("  средняя частота ядра, Гц: "
      f"{ {tab.iloc[i]['type']: round(float(tail[:, j].mean()), 2) for i, j in zip(CORE, core_k)} }")
if len(scores):
    print(f"  оценка колебаний мотонейронов: медиана {np.median(scores):.4f}, "
          f"макс {scores.max():.4f}")
    print(f"  мотонейронов выше порога 0,5: "
          f"{int((scores >= OSC_THRESHOLD).sum())} из {len(scores)}")
    good = freqs[scores >= OSC_THRESHOLD]
    if len(good):
        print(f"  их частоты, Гц: медиана {np.median(good):.2f}, "
              f"размах {good.min():.2f}...{good.max():.2f}")
else:
    print("  активных мотонейронов нет — колебания мерить нечем")

# ------------------------------- встречный замер: перемешанная связность
print("\n" + "=" * 78)
print("ВСТРЕЧНЫЙ ЗАМЕР: та же проба на перемешанной связности ядра")
print("(если она тоже даёт колебания, проверка ничего не стоит)")
print("=" * 78)
rng = np.random.default_rng(0)
hits = []
for seed in range(5):
    Wsh = Wk.copy()
    sub = np.ix_(core_k, core_k)
    blk = Wsh[sub].copy()
    flat = blk.flatten()
    rng.shuffle(flat)
    Wsh[sub] = flat.reshape(blk.shape)
    _, act_s, sc_s, _ = measure(Wsh, mn_k, core_k, stim_k)
    nhit = int((sc_s >= OSC_THRESHOLD).sum()) if len(sc_s) else 0
    hits.append(nhit)
    print(f"  зерно {seed}: активных МН {int((act_s & mn_k).sum())}, "
          f"колеблющихся {nhit}")
print(f"  перемешанная связность ядра: колеблющихся МН {hits}")

# ------------------- разброс по случайным параметрам, как у публикации
print("\n" + "=" * 78)
print("Доля параметризаций, дающих колебание (их способ: случайные параметры)")
print("=" * 78)
m = Wk.shape[0]
rng = np.random.default_rng(1)
nrep, osc_reps = 24, 0
fr_list = []
for r in range(nrep):
    tau = np.abs(rng.normal(P["tau"], P_STD["tau"], m))
    a = np.abs(rng.normal(P["a"], P_STD["a"], m))
    thr = rng.normal(P["threshold"], P_STD["threshold"], m)
    cap = np.abs(rng.normal(P["fr_cap"], P_STD["fr_cap"], m))
    _, _, sc_r, fq_r = measure(Wk, mn_k, core_k, stim_k,
                               tau=tau, a=a, thr=thr, cap=cap)
    nhit = int((sc_r >= OSC_THRESHOLD).sum()) if len(sc_r) else 0
    if nhit:
        osc_reps += 1
        fr_list += list(fq_r[sc_r >= OSC_THRESHOLD])
print(f"  параметризаций с колебанием: {osc_reps} из {nrep} "
      f"= {100*osc_reps/nrep:.1f} %")
if fr_list:
    fr_list = np.array(fr_list)
    print(f"  частоты колеблющихся МН, Гц: медиана {np.median(fr_list):.2f}, "
          f"размах {fr_list.min():.2f}...{fr_list.max():.2f}")
