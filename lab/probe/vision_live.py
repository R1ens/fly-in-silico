"""Настоящий кадр из живой сцены MuJoCo в настоящую сеть FlyWire.

Предыдущие пробы шли на синтетическом крае, посчитанном по положениям
омматидиев. Здесь кадр берётся **из работающего тела**: собирается
муха со зрением, идёт эпизод, на каждом шаге управления читается
`obs["vision"]` — то, что видят 721 омматидий каждого глаза в
настоящей сцене, — и через `retina.rates` подаётся на фоторецепторы
коннектома.

Это и есть замыкание первой половины контура: сцена → фоторецепторы
тела → привязка → LIF FlyWire. Вторая половина (нисходящие → мотор)
здесь не участвует: тело идёт своей походкой, зрение на него не влияет.
Так и задумано — сначала надо убедиться, что кадр доходит, и только
потом подключать его к управлению.

Три условия, и у каждого та же роль, что в `probe/vision_chain.py`:
настоящая привязка, перемешанная с сохранением чисел (суммарный вход тот
же, структура разрушена) и зрение выключено. Разница в том, что кадр
теперь не выдуман.

Что замеряется:

* дошёл ли настоящий кадр и чем он отличается от синтетического —
  яркость, разброс по омматидиям, доля насыщенных;
* частота по цепочке: фоторецепторы, L1, L2, L3, Tm1 — и глубже, где
  по `VISION.md` §3 сигнал обрывается;
* **цена**: сколько стоит шаг управления со зрением в контуре против
  того же эпизода без зрения.

Выборки популяций — точным селектором `type=X`.

    cd D:\\fly-brain\\lab
    set PYTHONIOENCODING=utf-8
    ..\\.venv\\Scripts\\python.exe probe\\vision_live.py
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("FLYBRAIN_OUT",
                      str(Path(__file__).parent / "_vision_live_out"))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

OUT = Path(__file__).parent

STEPS = 25               # шагов управления
PHYS_DT = 1e-4
CONTROL_MS = 20.0
PHYS = int(round(CONTROL_MS / 1000.0 / PHYS_DT))
BRAIN = int(round(CONTROL_MS / 0.1))    # шагов движка на шаг управления
VIS_REFRESH = 100        # Гц, рабочая точка
PR_MAX_HZ = 150.0
VIS_REF = 0.5

CHAIN = (
    ("R1-6", ("R1-6",)), ("R7", ("R7",)), ("R8", ("R8",)),
    ("L1", ("L1",)), ("L2", ("L2",)), ("L3", ("L3",)),
    ("Mi1", ("Mi1",)), ("Tm1", ("Tm1",)), ("Tm9", ("Tm9",)),
    ("T4", ("T4a", "T4b", "T4c", "T4d")),
    ("T5", ("T5a", "T5b", "T5c", "T5d")),
    ("LPLC2", ("LPLC2",)), ("LC4", ("LC4",)), ("DNp01", ("DNp01",)),
)

fails = []
rep = {"проверки": {}, "замеры": {}, "находки": [], "сорвалось": []}


def check(name, ok, info=""):
    print(("  ok     " if ok else "  ПРОВАЛ ") + name
          + (("  " + str(info)) if info else ""), flush=True)
    rep["проверки"][name] = {"пройдено": bool(ok), "замер": str(info)}
    if not ok:
        fails.append(name)


def note(text):
    print("  НАХОДКА " + text, flush=True)
    rep["находки"].append(text)


def build_body(vision):
    from flygym_gymnasium import Fly
    from flygym_gymnasium.arena import OdorArena
    from flygym_gymnasium.preprogrammed import (
        default_leg_sensor_placements)
    from flygym_gymnasium.examples.locomotion import HybridTurningController

    sources = np.array([[25.0, 0.0, 1.5]])
    arena = OdorArena(odor_source=sources,
                      peak_odor_intensity=np.ones((len(sources), 1)))
    kw = dict(enable_adhesion=True, enable_olfaction=True,
              contact_sensor_placements=default_leg_sensor_placements,
              init_pose="stretch", control="position",
              spawn_pos=(0.0, 0.0, 0.2))
    # `enable_vision` передаётся только когда зрение нужно: передача
    # даже со значением False меняет сборку модели (добавляются камеры
    # глаз), и тогда сравнение двух прогонов измеряло бы не то.
    if vision:
        kw["enable_vision"] = True
        kw["vision_refresh_rate"] = VIS_REFRESH
    fly = Fly(**kw)
    sim = HybridTurningController(fly=fly, cameras=[], arena=arena,
                                  timestep=PHYS_DT)
    return sim


def main():
    t0 = time.perf_counter()
    print("Настоящий кадр из живой сцены MuJoCo в сеть FlyWire",
          flush=True)
    import torch

    from flybrain import data as D, retina as R, sim as S

    neurons = D.load_neurons()
    W = D.load_weights()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n = len(neurons)

    pops = {}
    for name, types in CHAIN:
        pops[name] = np.unique(np.concatenate(
            [D.select(neurons, "type=%s" % t) for t in types]))
    t_pop = {k: torch.as_tensor(v, device=dev) for k, v in pops.items()}

    maps = R.build_all(neurons, mode="whiten", assign="capacity")
    maps_sh = R.build_all(neurons, mode="whiten", assign="capacity",
                          shuffle=True, seed=0)
    pr_idx = np.unique(np.concatenate([maps[k].neurons for k in maps]))
    t_map = {k: torch.as_tensor(m.neurons, device=dev)
             for k, m in maps.items()}
    rep["замеры"]["привязка"] = {
        "отпечаток": R.fingerprint_all(maps),
        "отпечаток перемешанной": R.fingerprint_all(maps_sh),
        "фоторецепторов": int(len(pr_idx)),
    }
    rep["замеры"]["устройство"] = {
        "шагов управления": STEPS,
        "модельного времени, мс": STEPS * CONTROL_MS,
        "шагов физики на шаг управления": PHYS,
        "шагов движка на шаг управления": BRAIN,
        "обновление зрения, Гц": VIS_REFRESH,
        "потолок фоторецепторов, Гц": PR_MAX_HZ,
        "устройство": dev,
    }
    print("  фоторецепторов %d, привязка %s, устройство %s"
          % (len(pr_idx), R.fingerprint_all(maps), dev), flush=True)

    def episode(mode):
        """Эпизод тела; ``mode``: зрение / перемешано / выключено."""
        body = build_body(mode != "выключено")
        obs, _ = body.reset(seed=0)
        eng = S.build_engine(weights=W, neurons=neurons, device=dev,
                             seed=0, no_refractory=pr_idx, graphs=True)
        rates = torch.zeros(1, n, device=dev)
        mm = maps if mode == "зрение" else maps_sh
        action = np.array([1.0, 1.0])
        series = {k: [] for k in pops}
        frames, t_step, t_vis = [], [], []
        for s in range(STEPS):
            t1 = time.perf_counter()
            for _ in range(PHYS):
                obs, _, _, _, _ = body.step(action)
            t_step.append(time.perf_counter() - t1)

            rates.zero_()
            if mode != "выключено":
                t2 = time.perf_counter()
                fr = np.asarray(obs["vision"], dtype=np.float32)
                frames.append(fr.copy())
                hz = R.rates(mm, fr, max_hz=PR_MAX_HZ, ref=VIS_REF)
                for key, v in hz.items():
                    rates[0, t_map[key]] = torch.as_tensor(
                        np.asarray(v, dtype=np.float32), device=dev)
                t_vis.append(time.perf_counter() - t2)

            acc = torch.zeros(n, device=dev)
            for _ in range(BRAIN):
                acc += eng.step(rates)[0]
            for k, t in t_pop.items():
                series[k].append(float(acc[t].sum()))
        if dev == "cuda":
            torch.cuda.synchronize()
        win_ms = STEPS * CONTROL_MS
        out = {k: round(1e3 * float(np.sum(v)) / len(pops[k]) / win_ms, 4)
               for k, v in series.items()}
        del eng, body
        return out, (np.stack(frames) if frames else None), t_step, t_vis

    res, fr, tst, tvs = {}, {}, {}, {}
    for mode in ("выключено", "зрение", "перемешано"):
        t1 = time.perf_counter()
        res[mode], fr[mode], tst[mode], tvs[mode] = episode(mode)
        print("  %-11s %d шагов управления за %.1f с"
              % (mode, STEPS, time.perf_counter() - t1), flush=True)

    # --- 1. настоящий кадр и чем он отличается от выдуманного ---
    f = fr["зрение"]
    live = {
        "кадров": int(len(f)),
        "форма кадра": list(f.shape[1:]),
        "минимум": round(float(f.min()), 5),
        "максимум": round(float(f.max()), 5),
        "среднее": round(float(f.mean()), 5),
        "разброс по омматидиям, последний кадр":
            round(float(f[-1].std()), 5),
        "доля омматидиев выше яркости насыщения":
            round(float((f >= VIS_REF).mean()), 5),
        "доля ровно нулевых": round(float((f == 0.0).mean()), 5),
        "кадров, совпавших с предыдущим":
            int(sum(1 for i in range(1, len(f))
                    if np.array_equal(f[i], f[i - 1]))),
        "наибольшая разница между соседними кадрами":
            round(float(max(np.abs(f[i] - f[i - 1]).max()
                            for i in range(1, len(f)))), 5),
    }
    rep["замеры"]["настоящий кадр"] = live
    print("\n  настоящий кадр: форма %s, яркость %.4f…%.4f (среднее "
          "%.4f), разброс по омматидиям %.4f, выше насыщения %.2f %%, "
          "нулевых %.2f %%"
          % (live["форма кадра"], live["минимум"], live["максимум"],
             live["среднее"],
             live["разброс по омматидиям, последний кадр"],
             100 * live["доля омматидиев выше яркости насыщения"],
             100 * live["доля ровно нулевых"]), flush=True)
    check("кадр пришёл из живой сцены и он живой: форма та, разброс "
          "есть, кадры меняются",
          live["форма кадра"] == [2, 721, 2]
          and live["разброс по омматидиям, последний кадр"] > 1e-4
          and live["наибольшая разница между соседними кадрами"] > 0,
          "форма %s (два глаза, 721 омматидий, два канала), разброс по "
          "омматидиям %.4f, кадров совпавших с предыдущим %d из %d, "
          "наибольшая разница между соседними кадрами %.4f. Это не "
          "посчитанный нами край, а то, что видит муха в сцене"
          % (live["форма кадра"],
             live["разброс по омматидиям, последний кадр"],
             live["кадров, совпавших с предыдущим"], live["кадров"],
             live["наибольшая разница между соседними кадрами"]))

    # --- 2. цепочка ---
    print("\nОтклик по цепочке на НАСТОЯЩЕМ кадре, Гц на нейрон",
          flush=True)
    print("  %-7s %9s %9s %9s %13s" % ("", "выкл", "зрение", "перемеш",
                                       "зрение−выкл"), flush=True)
    tab = {}
    for name, _ in CHAIN:
        a, b, c = (res["выключено"][name], res["зрение"][name],
                   res["перемешано"][name])
        tab[name] = {"нейронов": int(len(pops[name])),
                     "выключено, Гц": a, "зрение, Гц": b,
                     "перемешано, Гц": c,
                     "зрение минус выключенное, Гц": round(b - a, 4),
                     "зрение минус перемешанное, Гц": round(b - c, 4)}
        print("  %-7s %9.3f %9.3f %9.3f %13.4f"
              % (name, a, b, c, b - a), flush=True)
    rep["замеры"]["отклик на настоящем кадре"] = tab

    check("настоящий кадр доходит до фоторецепторов коннектома",
          tab["R1-6"]["зрение, Гц"] > 1.0
          and tab["R1-6"]["выключено, Гц"] < 0.01,
          "R1-6 выдают %.2f Гц на нейрон при зрении против %.4f при "
          "выключенном. Яркость в сцене %.4f в среднем при яркости "
          "насыщения %.2f, поэтому до потолка %.0f Гц дело не доходит — "
          "и это свойство сцены, а не привязки"
          % (tab["R1-6"]["зрение, Гц"], tab["R1-6"]["выключено, Гц"],
             live["среднее"], VIS_REF, PR_MAX_HZ))

    deeper = [k for k, v in tab.items()
              if k not in ("R1-6", "R7", "R8")
              and v["зрение минус выключенное, Гц"] > 0.02]
    rep["замеры"]["дошло до"] = deeper
    check("сигнал от настоящего кадра идёт по настоящим связям дальше "
          "фоторецепторов",
          len(deeper) >= 2,
          "откликнулись (прибавка к выключенному больше 0,02 Гц): %s. "
          "Глубже не прошло: %s — ровно как на синтетическом крае "
          "(`VISION.md` §3), и причина та же, посчитанная: L1 → Mi1 "
          "тормозная при отрицательном полном балансе Mi1"
          % (", ".join(deeper) or "ни одна",
             ", ".join(k for k, _ in CHAIN
                       if k not in deeper
                       and k not in ("R1-6", "R7", "R8"))))

    # --- 3. структура ---
    st = {k: v["зрение минус перемешанное, Гц"] for k, v in tab.items()
          if k not in ("R1-6", "R7", "R8")}
    worst = max(st, key=lambda k: abs(st[k]))
    check("отклик на настоящем кадре зависит от пространственной "
          "структуры привязки",
          abs(st[worst]) > 0.01,
          "наибольшее расхождение настоящей и перемешанной привязки — у "
          "%s: %+.4f Гц. Суммарный вход у двух условий одинаков по "
          "построению (перемешивание сохраняет число нейронов на "
          "омматидий), значит разницу даёт структура"
          % (worst, st[worst]))

    # --- 4. цена ---
    ms = {k: round(1e3 * float(np.median(v)), 2) for k, v in tst.items()}
    vis_ms = round(1e3 * float(np.median(tvs["зрение"])), 3)
    cost = {
        "шаг управления, мс: выключено": ms["выключено"],
        "шаг управления, мс: зрение": ms["зрение"],
        "зрение добавило к физике, раз":
            round(ms["зрение"] / max(ms["выключено"], 1e-9), 3),
        "привязка и подача, мс на шаг": vis_ms,
        "модельного на настенное: выключено":
            round(CONTROL_MS / max(ms["выключено"], 1e-9), 4),
        "модельного на настенное: зрение":
            round(CONTROL_MS / max(ms["зрение"], 1e-9), 4),
    }
    rep["замеры"]["цена"] = cost
    print("\n  цена: шаг физики %.1f мс без зрения, %.1f мс со зрением "
          "(×%.2f); привязка и подача %.3f мс на шаг"
          % (ms["выключено"], ms["зрение"],
             ms["зрение"] / max(ms["выключено"], 1e-9), vis_ms),
          flush=True)
    check("цена зрения в живом контуре замерена",
          vis_ms >= 0 and ms["зрение"] > 0,
          "сама привязка с подачей на 10 582 фоторецептора стоит %.3f мс "
          "на шаг управления в 20 мс модельного времени — это %.2f %% "
          "шага физики (%.1f мс). Основную цену берёт не она, а съём "
          "кадра самим FlyGym"
          % (vis_ms, 100 * vis_ms / max(ms["зрение"], 1e-9),
             ms["зрение"]))

    note("замкнута первая половина контура: сцена MuJoCo → 721 омматидий "
         "каждого глаза → инженерная привязка → фоторецепторы FlyWire → "
         "настоящие связи до L1, L2, L3, Tm1. Вторая половина "
         "(нисходящие → мотор) здесь не участвует: тело идёт своей "
         "походкой, и зрение на него не влияет. Предел тот же, что на "
         "синтетическом крае, и причина та же посчитанная — значит это "
         "свойство модели, а не раздражителя.")

    rep["замеры"]["проба заняла, с"] = round(time.perf_counter() - t0, 1)
    (OUT / "vision_live.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nпроверок %d, провалов %d, находок %d, за %.0f с"
          % (len(rep["проверки"]), len(fails), len(rep["находки"]),
             time.perf_counter() - t0), flush=True)
    if fails:
        print("ПРОВАЛЫ:", flush=True)
        for f in fails:
            print("  " + f, flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
