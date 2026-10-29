"""Встречный замер четырёх правок обвязки MuJoCo.

Машина занята другими прогонами, поэтому межпроцессное сравнение
времени здесь не годится: разброс между запусками больше того, что надо
измерить. Мерится так:

* **один процесс, перемежающиеся круги.** Тела с разными ручками
  собраны все сразу и шагаются по очереди блоками. Все правки
  численно равнозначны (`probe/mj_equal.py`), поэтому тела идут по ОДНОЙ
  траектории и в каждом круге находятся в одинаковом состоянии — то
  есть сравниваются не два разных эпизода, а одна и та же работа
  (совпадение `qpos` проверяется в конце и печатается);
* **отношения попарно**, по кругам: для каждого круга своё отношение,
  в итог идут медиана, крайние значения и число пар;
* абсолютные числа печатаются, но итогом не считаются: доля реального
  времени на занятой машине не измеряется.

Три части:

1. сами наблюдения — та пара вызовов, которая приходится на шаг физики;
2. тело без мозга — чистое время шага физики и вклад каждой ручки;
3. эпизод целиком (`body.run_episode`) короткими кругами — сколько из
   этого видно на шаге управления вместе с мозгом.

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\mj_bench.py
    ..\\.venv\\Scripts\\python.exe probe\\mj_bench.py --только тело
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

OUT = Path(__file__).parent
ROUNDS = 21          # кругов в перемежении
BLOCK = 200          # шагов физики в блоке одного круга
OBS_ROUNDS = 15      # кругов на отдельном замере самих наблюдений
OBS_BLOCK = 300      # вызовов наблюдения в блоке
EP_ROUNDS = 4        # кругов на эпизодах
EP_T = 1.0           # модельное время короткого эпизода, с

OFF = dict(acc_forward=False, fast_obs=False, rule_obs=False,
           fast_gain=False)
KNOBS = {
    "прежняя дорога": dict(OFF),
    "только пересчёт со ступени ускорений": dict(OFF, acc_forward=True),
    "только дешёвое наблюдение": dict(OFF, fast_obs=True),
    "дешёвое наблюдение + правила": dict(OFF, fast_obs=True, rule_obs=True),
    "только фазовый множитель": dict(OFF, fast_gain=True),
    "все четыре": dict(acc_forward=True, fast_obs=True, rule_obs=True,
                       fast_gain=True),
}


def stats(ratios):
    a = np.asarray(ratios, float)
    return {"медиана": round(float(np.median(a)), 3),
            "наименьшее": round(float(a.min()), 3),
            "наибольшее": round(float(a.max()), 3),
            "пар": int(a.size)}


def bench_body():
    from flybrain import body as B

    rep = {"кругов": ROUNDS, "шагов физики в блоке": BLOCK}
    bodies, action = {}, np.array([1.0, 1.0])
    for name, kw in KNOBS.items():
        cfg = B.BodyConfig(duration_s=1.0, control_ms=20.0, seed=0, **kw)
        sim, _ = B._build_body(cfg, B.DEFAULT_SOURCES, None, head_stab=None)
        sim.reset(seed=0)
        bodies[name] = sim
    # разогрев: одинаковый для всех, и состояния после него совпадают
    for sim in bodies.values():
        for _ in range(100):
            sim.step(action)

    rounds = {name: [] for name in KNOBS}
    for r in range(ROUNDS):
        for name, sim in bodies.items():
            t = time.perf_counter()
            for _ in range(BLOCK):
                sim.step(action)
            rounds[name].append((time.perf_counter() - t) / BLOCK * 1e6)

    # Тела шли по одной траектории — проверим это, а не поверим.
    ref = bodies["прежняя дорога"]
    same = {}
    for name, sim in bodies.items():
        same[name] = bool(np.array_equal(ref.physics.data.qpos,
                                         sim.physics.data.qpos))
    rep["состояния совпали с прежней дорогой"] = same

    base = np.asarray(rounds["прежняя дорога"], float)
    rep["мкс на шаг физики (медиана по кругам)"] = {
        name: round(float(np.median(v)), 1) for name, v in rounds.items()}
    rep["отношение к прежней дороге, попарно по кругам"] = {
        name: stats(base / np.asarray(v, float))
        for name, v in rounds.items() if name != "прежняя дорога"}
    rep["разброс внутри круга (крайние/медиана)"] = {
        name: [round(float(np.min(v) / np.median(v)), 3),
               round(float(np.max(v) / np.median(v)), 3)]
        for name, v in rounds.items()}
    for name, sim in bodies.items():
        st = getattr(sim.physics, "_fb_accfwd", None)
        if st is not None:
            rep.setdefault("пересчётов прямой динамики", {})[name] = {
                "дешёвых": st["дешёвых"], "полных": st["полных"]}
        sim.close()

    print("\nтело без мозга, %d кругов по %d шагов физики:" % (ROUNDS, BLOCK))
    for name in KNOBS:
        v = rep["мкс на шаг физики (медиана по кругам)"][name]
        r = rep["отношение к прежней дороге, попарно по кругам"].get(name)
        print("  %-34s %7.1f мкс  %s" % (name, v,
              "" if r is None else
              "x%.3f  (%.3f…%.3f по %d парам)"
              % (r["медиана"], r["наименьшее"], r["наибольшее"], r["пар"])))
    print("  состояния совпали: " + json.dumps(same, ensure_ascii=False))
    return rep


def bench_obs():
    """Сами наблюдения, перемежающимися кругами на одном теле.

    Состояние модели при этом стоит на месте, поэтому три дороги считают
    ровно одно и то же, и шум машины ложится на все три одинаково.
    """
    from flygym_gymnasium import Fly as PlainFly
    from flybrain import body as B

    cfg = B.BodyConfig(duration_s=1.0, control_ms=20.0, seed=0)
    sim, _ = B._build_body(cfg, B.DEFAULT_SOURCES, None, head_stab=None)
    sim.reset(seed=0)
    for _ in range(200):
        sim.step(np.array([1.0, 1.0]))
    fly = sim.fly

    # Так это и идёт в настоящем цикле: полное наблюдение в `post_step`
    # ВСЕГДА считает контактные силы заново (и обновляет запомненное), а
    # наблюдение правил перед следующим шагом берёт их запомненными —
    # состояние с тех пор не менялось. Здесь ровно эта пара и мерится, по
    # вызову каждого; номер состояния между ними не двигается, как и в
    # цикле между `post_step` и съёмом правил.
    ways = {
        "прежнее наблюдение (FlyGym)": lambda: PlainFly.get_observation(
            fly, sim),
        "дешёвое наблюдение (считает контакты)":
            lambda: fly.get_observation(sim),
        "наблюдение правил (контакты запомнены)":
            lambda: fly.rule_observation(sim),
    }
    rounds = {k: [] for k in ways}
    for _ in range(3):
        for fn in ways.values():
            for _ in range(50):
                fn()
    for r in range(OBS_ROUNDS):
        for name, fn in ways.items():
            t = time.perf_counter()
            for _ in range(OBS_BLOCK):
                fn()
            rounds[name].append(
                (time.perf_counter() - t) / OBS_BLOCK * 1e6)
    base = np.asarray(rounds["прежнее наблюдение (FlyGym)"], float)
    rep = {
        "кругов": OBS_ROUNDS, "вызовов в блоке": OBS_BLOCK,
        "мкс на вызов (медиана)": {k: round(float(np.median(v)), 1)
                                   for k, v in rounds.items()},
        "отношение к прежнему, попарно по кругам": {
            k: stats(base / np.asarray(v, float))
            for k, v in rounds.items() if k != "прежнее наблюдение (FlyGym)"},
    }
    print("\nсами наблюдения, %d кругов по %d вызовов:"
          % (OBS_ROUNDS, OBS_BLOCK))
    for k in ways:
        v = rep["мкс на вызов (медиана)"][k]
        r = rep["отношение к прежнему, попарно по кругам"].get(k)
        print("  %-30s %7.1f мкс  %s" % (k, v, "" if r is None else
              "x%.3f  (%.3f…%.3f по %d парам)"
              % (r["медиана"], r["наименьшее"], r["наибольшее"], r["пар"])))
    sim.close()
    return rep


def bench_episode():
    import torch

    from flybrain import body as B, data as D

    device = "cuda" if torch.cuda.is_available() else "cpu"
    neurons = D.load_neurons()
    weights = D.load_weights(device=device)
    pair = {
        "прежняя дорога": KNOBS["прежняя дорога"],
        "все четыре": KNOBS["все четыре"],
    }
    # Мерится `timing["sim_s"]` — время САМОГО цикла шагов, без сборки
    # тела, перебора популяций, захвата графов CUDA и разогрева. На
    # коротком эпизоде подготовка занимает больше самого счёта (замер:
    # 47 с из 47 на 0,4 с модельного), и отношение по стене мерило бы
    # подготовку, а не шаг.
    rounds = {k: [] for k in pair}
    walls = {k: [] for k in pair}
    prints = {k: [] for k in pair}
    steps = {k: [] for k in pair}
    for r in range(EP_ROUNDS):
        for name, kw in pair.items():
            cfg = B.BodyConfig(duration_s=EP_T, control_ms=20.0, seed=0,
                               device=device, readout="ALPN", **kw)
            t = time.perf_counter()
            run = B.run_episode(cfg=cfg, neurons=neurons, weights=weights,
                                progress=False)
            walls[name].append(time.perf_counter() - t)
            rounds[name].append(float(run.timing["sim_s"]))
            steps[name].append(int(run.timing["steps_done"]))
            recs = run.records
            prints[name].append([round(float(recs[-1]["x"]), 6),
                                 round(float(recs[-1]["y"]), 6),
                                 round(float(recs[-1]["theta"]), 6)])
            print("  круг %d  %-18s цикл %6.2f с, по стене %6.2f с, "
                  "шаг %6.1f мс  поза %s"
                  % (r, name, rounds[name][-1], walls[name][-1],
                     rounds[name][-1] / steps[name][-1] * 1000.0,
                     prints[name][-1]), flush=True)
    a = np.asarray(rounds["прежняя дорога"], float)
    b = np.asarray(rounds["все четыре"], float)
    rep = {
        "кругов": EP_ROUNDS,
        "модельного времени в эпизоде, с": EP_T,
        "устройство": device,
        "шагов управления": steps["прежняя дорога"][0],
        "цикл шагов (timing[sim_s]), с (медиана)": {
            k: round(float(np.median(v)), 2) for k, v in rounds.items()},
        "цикл шагов, с (все круги)": {k: [round(x, 2) for x in v]
                                      for k, v in rounds.items()},
        "мс на шаг управления (медиана)": {
            k: round(float(np.median(v)) / steps[k][0] * 1000.0, 1)
            for k, v in rounds.items()},
        "по стене целиком, с (медиана)": {
            k: round(float(np.median(v)), 2) for k, v in walls.items()},
        "отношение цикла попарно по кругам": stats(a / b),
        "конечная поза совпала во всех кругах":
            all(p == prints["прежняя дорога"][0]
                for p in prints["прежняя дорога"] + prints["все четыре"]),
        "конечная поза": prints["прежняя дорога"][0],
    }
    r = rep["отношение цикла попарно по кругам"]
    print("\nэпизод целиком (мозг + тело), %d кругов по %.1f с модельного:"
          % (EP_ROUNDS, EP_T))
    print("  шаг управления: %.1f мс -> %.1f мс,  x%.3f "
          "(%.3f…%.3f по %d парам)"
          % (rep["мс на шаг управления (медиана)"]["прежняя дорога"],
             rep["мс на шаг управления (медиана)"]["все четыре"],
             r["медиана"], r["наименьшее"], r["наибольшее"], r["пар"]))
    print("  конечная поза одна и та же во всех кругах: %s  %s"
          % (rep["конечная поза совпала во всех кругах"],
             rep["конечная поза"]))
    return rep


def main():
    only = "тело" if "--только" in sys.argv and "тело" in sys.argv else None
    rep = {"сами наблюдения": bench_obs(), "тело без мозга": bench_body()}
    if only is None:
        rep["эпизод целиком"] = bench_episode()
    (OUT / "mj_bench.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nотчёт: probe/mj_bench.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
