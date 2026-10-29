"""Отношение быстрого и стокового контроллера — короткими блоками.

Зачем отдельно от `probe/gait_ab.py`. Там сравниваются целые эпизоды по
две секунды, и на занятой машине это не работает: пока считается один
эпизод, загрузка успевает смениться, и отношение в круге оказывается про
машину, а не про код (замерено: четыре пары дали разброс 0,993…1,499).

Здесь то же сравнение сделано **короткими перемежающимися блоками**. В
одном процессе собираются ДВА тела — одно со стоковым контроллером, одно
с быстрым, — с одним семенем и одной и той же заданной числом командой.
Оба контроллера побитово равны (`probe/gait_equal.py`), поэтому их
состояния идут шаг в шаг, и физика в них делает ровно одну и ту же
работу: различие во времени — это различие контроллера. Это проверяется
здесь же: фазы CPG, амплитуды и счётчики поправок обоих тел сверяются
побитово после каждого блока.

Мозга нет нарочно: он добавил бы к каждому шагу управления 200 вызовов
движка LIF и синхронизацию с видеокартой, то есть ровно тот шум, от
которого здесь уходят.

Что печатается: отношение **времени контроллера** (его шаг минус шаг
физики) и отношение **полного шага тела**, попарно по кругам, с медианой,
разбросом и числом пар.

    cd D:\\fly-brain\\lab
    set PYTHONIOENCODING=utf-8
    ..\\.venv\\Scripts\\python.exe probe\\gait_bench.py --rounds 12 --block 50
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

OUT = Path(__file__).parent
DT = 2e-4
fails = []


def build(gait):
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
    body = G.build(gait=gait, fly=fly, cameras=[], arena=arena, timestep=DT)
    body.reset(seed=0)
    return body


class Clock:
    """Накопитель времени шага физики внутри шага контроллера."""

    def __init__(self):
        self.inner = 0.0


def patch_inner(clock):
    """Обёртка на `SingleFlySimulation.step` — одна на оба тела."""
    from flygym_gymnasium.simulation import SingleFlySimulation

    inner = SingleFlySimulation.__dict__["step"]
    pc = time.perf_counter

    def step(self, action):
        t = pc()
        try:
            return inner(self, action)
        finally:
            clock.inner += pc() - t

    SingleFlySimulation.step = step


def same(a, b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return a.shape == b.shape and a.dtype == b.dtype \
        and a.tobytes() == b.tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--block", type=int, default=50,
                    help="шагов физики в блоке")
    ap.add_argument("--warm", type=int, default=200,
                    help="шагов физики на разогрев каждого тела")
    a = ap.parse_args()

    clock = Clock()
    patch_inner(clock)

    bodies = {"legacy": build("legacy"), "fast": build("fast")}
    act = np.array([1.2, 0.4])

    for g, b in bodies.items():
        for _ in range(a.warm):
            b.step(act)

    rows = []
    for r in range(a.rounds):
        order = ("legacy", "fast") if r % 2 == 0 else ("fast", "legacy")
        got = {}
        for g in order:
            body = bodies[g]
            clock.inner = 0.0
            t0 = time.perf_counter()
            for _ in range(a.block):
                body.step(act)
            total = time.perf_counter() - t0
            got[g] = (total, clock.inner)
        # Состояния обязаны идти шаг в шаг: иначе физика в двух телах
        # делает разную работу и отношение времени ни о чём.
        L, F = bodies["legacy"], bodies["fast"]
        for name, x, y in (
                ("фазы CPG", L.cpg_network.curr_phases,
                 F.cpg_network.curr_phases),
                ("амплитуды CPG", L.cpg_network.curr_magnitudes,
                 F.cpg_network.curr_magnitudes),
                ("поправка втягивания", L.retraction_correction,
                 F.retraction_correction),
                ("поправка спотыкания", L.stumbling_correction,
                 F.stumbling_correction),
                ("счётчик удержания", L.retraction_persistence_counter,
                 F.retraction_persistence_counter)):
            if not same(x, y):
                fails.append("круг %d: %s разошлись" % (r + 1, name))
        per = {}
        for g in ("legacy", "fast"):
            total, inner = got[g]
            per[g] = {
                "шаг тела, мс на шаг управления":
                    total / a.block * 100.0 * 1000.0,
                "контроллер, мс на шаг управления":
                    (total - inner) / a.block * 100.0 * 1000.0,
            }
        rows.append(per)
        print("круг %2d: контроллер legacy %6.2f мс, fast %6.2f мс "
              "(×%.2f); шаг тела %6.2f против %6.2f (×%.3f)"
              % (r + 1,
                 per["legacy"]["контроллер, мс на шаг управления"],
                 per["fast"]["контроллер, мс на шаг управления"],
                 per["legacy"]["контроллер, мс на шаг управления"]
                 / max(per["fast"]["контроллер, мс на шаг управления"],
                       1e-9),
                 per["legacy"]["шаг тела, мс на шаг управления"],
                 per["fast"]["шаг тела, мс на шаг управления"],
                 per["legacy"]["шаг тела, мс на шаг управления"]
                 / max(per["fast"]["шаг тела, мс на шаг управления"],
                       1e-9)))

    def stat(key):
        rl = [p["legacy"][key] for p in rows]
        rf = [p["fast"][key] for p in rows]
        rat = [x / max(y, 1e-9) for x, y in zip(rl, rf)]
        return {"пар": len(rat),
                "медиана отношения": round(float(np.median(rat)), 4),
                "минимум": round(float(np.min(rat)), 4),
                "максимум": round(float(np.max(rat)), 4),
                "по кругам": [round(x, 3) for x in rat],
                "legacy, медиана мс": round(float(np.median(rl)), 2),
                "fast, медиана мс": round(float(np.median(rf)), 2),
                "legacy, разброс мс": [round(float(np.min(rl)), 2),
                                       round(float(np.max(rl)), 2)],
                "fast, разброс мс": [round(float(np.min(rf)), 2),
                                     round(float(np.max(rf)), 2)]}

    rep = {"блок, шагов физики": a.block, "кругов": a.rounds,
           "разогрев, шагов": a.warm,
           "команда": act.tolist(),
           "контроллер": stat("контроллер, мс на шаг управления"),
           "шаг тела": stat("шаг тела, мс на шаг управления"),
           "расхождения состояний": fails}
    print()
    for k in ("контроллер", "шаг тела"):
        d = rep[k]
        print("%s: медиана ×%.3f, разброс %.3f…%.3f, пар %d; "
              "legacy %.2f мс (%.2f…%.2f), fast %.2f мс (%.2f…%.2f)"
              % (k, d["медиана отношения"], d["минимум"], d["максимум"],
                 d["пар"], d["legacy, медиана мс"],
                 d["legacy, разброс мс"][0], d["legacy, разброс мс"][1],
                 d["fast, медиана мс"], d["fast, разброс мс"][0],
                 d["fast, разброс мс"][1]))
    print("состояния обоих тел шли шаг в шаг: %s"
          % ("да" if not fails else fails[:3]))
    p = OUT / "gait_bench.json"
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    print("отчёт: %s" % p)
    for b in bodies.values():
        b.close()
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
