"""Что будет, если считать контроллер походки реже физики.

Это НЕ предложение. Многочастотное расписание проверяется здесь затем,
чтобы отказ от него стоял на числе, а не на осторожности. Правило брифа:
снижение частоты обновления научно равнозначным не считается никогда, и
каждая частота проверяется отдельно.

Что мерится. Тело FlyGym без мозга: нисходящая команда задана числом и
постоянна, поэтому различие между прогонами даёт РОВНО частота
контроллера, а не разошедшаяся активность. Контроллер (интегрирование
CPG, правила втягивания и спотыкания, целевые углы, прилипание) считается
раз в ``k`` шагов физики; между пересчётами углы и прилипание держатся
прежними, а физика идёт с полным шагом. При ``k=1`` это в точности
нынешний контроллер — это и есть эталон.

Два раздражителя, и оба нужны. Прямая ходьба (симметричная команда)
меняется медленно, и по ней любое разрежение выглядит безобидным.
Резкий поворот (несимметричная команда, переключаемая на середине
прогона) — то, на чём разрежение видно: команда меняется ступенькой, а
контроллер видит её только на своём шаге.

Третий случай, который здесь НЕ проверен и так и назван: неровная земля
и спотыкание. Правила спотыкания и втягивания — фильтры с временами
около миллисекунды (ставки 800/700 и 2200/1800 в секунду при шаге
2e-4 с), то есть при ``k`` больше пяти их собственная динамика считается
грубее собственного времени. На плоской арене они почти не включаются,
и этот замер про них ничего не говорит.

    cd D:\\fly-brain\\lab
    set PYTHONIOENCODING=utf-8
    ..\\.venv\\Scripts\\python.exe probe\\ctrl_rate.py
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

OUT = Path(__file__).parent
DT = 2e-4


def build(gait="fast"):
    from flygym_gymnasium import Fly
    from flygym_gymnasium.arena import OdorArena
    from flygym_gymnasium.preprogrammed import default_leg_sensor_placements

    from flybrain import gait as G

    fly = Fly(enable_adhesion=True, enable_olfaction=True,
              contact_sensor_placements=default_leg_sensor_placements,
              init_pose="stretch", control="position",
              spawn_pos=(0.0, 0.0, 0.2))
    arena = OdorArena(odor_source=np.array([[25.0, 12.0, 1.5]]),
                      peak_odor_intensity=np.ones((1, 1)))
    return G.build(gait=gait, fly=fly, cameras=[], arena=arena, timestep=DT)


def episode(every, n_steps, action_of, seed=0):
    """Прогон, где контроллер считается раз в ``every`` шагов физики."""
    from flygym_gymnasium.simulation import SingleFlySimulation

    body = build()
    obs, _ = body.reset(seed=seed)
    held = None
    track = []
    t0 = time.perf_counter()
    for i in range(n_steps):
        act = action_of(i)
        if i % every == 0:
            # Полный шаг контроллера: он сам шагнёт физику один раз.
            # Интегрирование CPG внутри идёт с его собственным
            # `timestep`, то есть при every>1 фаза отстаёт — это и есть
            # то самое «реже», которое проверяется.
            obs, _, term, trunc, info = body.step(act)
            held = {"joints": info["joints"], "adhesion": info["adhesion"]}
        else:
            obs, _, term, trunc, info = SingleFlySimulation.step(body, held)
            # Наблюдение, которое держит быстрый контроллер, относится к
            # этому шагу: его поставил сам шаг физики.
            body._obs_cache = obs
        if i % 50 == 0:
            p = np.asarray(obs["fly"])[0]
            f = np.asarray(obs["fly_orientation"])
            track.append([float(p[0]), float(p[1]), float(p[2]),
                          math.atan2(float(f[1]), float(f[0]))])
        if term or trunc:
            break
    wall = time.perf_counter() - t0
    p = np.asarray(obs["fly"])[0]
    f = np.asarray(obs["fly_orientation"])
    body.close()
    return {
        "каждый": every,
        "по стене, с": round(wall, 3),
        "конец": [round(float(p[0]), 5), round(float(p[1]), 5),
                  round(float(p[2]), 5)],
        "курс": round(math.atan2(float(f[1]), float(f[0])), 5),
        "фазы": [round(float(v), 6) for v in body.cpg_network.curr_phases],
        "_track": np.array(track, dtype=np.float64),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", type=float, default=0.6,
                    help="модельное время прогона, с")
    ap.add_argument("--every", default="1,2,5,10,25,50,100")
    a = ap.parse_args()

    n = int(round(a.time / DT))
    everys = [int(v) for v in a.every.split(",")]

    stimuli = {
        "прямая ходьба": lambda i: np.array([1.0, 1.0]),
        # Ступенька на середине: команда меняется мгновенно, и реже
        # считающий контроллер узнаёт о ней позже.
        "резкий поворот": lambda i: (np.array([1.2, 0.4]) if i < n // 2
                                     else np.array([0.4, 1.2])),
    }

    rep = {"модельное время, с": a.time, "шагов физики": n,
           "шаг физики, с": DT, "раздражители": {}}
    for name, fn in stimuli.items():
        print("=== %s ===" % name)
        rows = []
        ref = None
        for e in everys:
            r = episode(e, n, fn)
            if ref is None:
                ref = r
            tr, rt = r["_track"], ref["_track"]
            m = min(len(tr), len(rt))
            dxy = np.linalg.norm(tr[:m, :2] - rt[:m, :2], axis=1)
            dth = np.abs(np.arctan2(np.sin(tr[:m, 3] - rt[:m, 3]),
                                    np.cos(tr[:m, 3] - rt[:m, 3])))
            row = {
                "каждый": e,
                "частота контроллера, Гц": round(1.0 / (DT * e), 1),
                "по стене, с": r["по стене, с"],
                "ускорение к k=1": round(ref["по стене, с"]
                                         / r["по стене, с"], 3),
                "конец": r["конец"],
                "смещение конца, мм": round(float(
                    np.linalg.norm(np.array(r["конец"][:2])
                                   - np.array(ref["конец"][:2]))), 4),
                "расхождение пути, макс мм": round(float(dxy.max()), 4),
                "расхождение курса, макс рад": round(float(dth.max()), 5),
                "расхождение фаз CPG, макс рад": round(float(np.max(
                    np.abs(np.array(r["фазы"])
                           - np.array(ref["фазы"])))), 5),
            }
            rows.append(row)
            print("  k=%-4d %6.1f Гц  конец %s  смещение %7.4f мм  "
                  "курс %7.5f рад  ускорение %.3f"
                  % (e, row["частота контроллера, Гц"], r["конец"][:2],
                     row["смещение конца, мм"],
                     row["расхождение курса, макс рад"],
                     row["ускорение к k=1"]))
        rep["раздражители"][name] = rows

    # Собственные времена подсистем — числом, а не на словах.
    rep["собственные времена"] = {
        "шаг физики, мс": DT * 1000,
        "фаза CPG за шаг физики, рад": round(2 * math.pi * 12 * DT, 6),
        "период шага ноги при 12 Гц, мс": round(1000.0 / 12, 2),
        "поправка втягивания: время нарастания, мс":
            round(1000.0 / 800, 3),
        "поправка спотыкания: время нарастания, мс":
            round(1000.0 / 2200, 3),
        "пояснение": (
            "фаза ноги за один шаг физики проходит 0,0151 рад; окно "
            "прилипания переключается по порогу фазы, поэтому при k "
            "шагах удержания миг переключения сдвигается на k×0,0151 "
            "рад, то есть на k×0,24 % периода шага. Поправки правил "
            "нарастают за единицы миллисекунд, то есть за единицы "
            "десятков шагов физики: при k≥5 их собственная динамика "
            "считается грубее собственного времени."),
    }
    p = OUT / "ctrl_rate.json"
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    print("\nотчёт: %s" % p)


if __name__ == "__main__":
    raise SystemExit(main())
