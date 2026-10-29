"""Где именно уходит время на шаге физики MuJoCo.

Тело собирается ровно так, как это делает `body._build_body`, и шагается
постоянной командой — мозг не участвует, чтобы время физики не мешалось
со временем LIF. Замеры навешиваются обёртками снаружи, слоями:

* `HybridTurningController.step` целиком (это то, что зовёт `body.py`);
* из него: CPG и правила (до `SingleFlySimulation.step`);
* `Fly.get_observation` — и ОТДЕЛЬНО считается, сколько раз его зовут на
  один шаг физики;
* `arena.step`, `Fly.pre_step`, `Physics.step`, `Fly.post_step`;
* `Physics.forward` — тот самый повторный пересчёт прямой динамики,
  который вызывает «грязный» флаг от записи `ctrl`;
* внутри `Physics.step` — сами `mj_step2`/`mj_step1`;
* голая физика рядом: `mj_step` на том же состоянии, чтобы знать, сколько
  стоит шаг без обвязки.

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\mj_profile.py
"""

import cProfile
import json
import pstats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

OUT = Path(__file__).parent
N_CTRL = 12          # шагов управления
N_PHYS = 100         # шагов физики на шаг управления (control_ms=20, dt=2e-4)
WARM = 2             # шагов управления на разогрев, в среднее не идут


def build():
    from flybrain import body as B

    cfg = B.BodyConfig(duration_s=1.0, control_ms=20.0, seed=0)
    sources = B.DEFAULT_SOURCES
    body, _ = B._build_body(cfg, sources, None, head_stab=None)
    obs, _ = body.reset(seed=cfg.seed)
    return cfg, body, obs


class Tally:
    def __init__(self):
        self.t = {}
        self.n = {}

    def add(self, key, dt):
        self.t[key] = self.t.get(key, 0.0) + dt
        self.n[key] = self.n.get(key, 0) + 1

    def reset(self):
        self.t.clear()
        self.n.clear()


def wrap(tally, obj, name, key, cls=None):
    """Обернуть метод замером. `cls` — если патчим класс, а не объект."""
    target = cls if cls is not None else obj
    orig = getattr(target, name)
    if cls is not None:
        def f(self, *a, **kw):
            t0 = time.perf_counter()
            try:
                return orig(self, *a, **kw)
            finally:
                tally.add(key, time.perf_counter() - t0)
    else:
        def f(*a, **kw):
            t0 = time.perf_counter()
            try:
                return orig(*a, **kw)
            finally:
                tally.add(key, time.perf_counter() - t0)
    setattr(target, name, f)
    return orig


def main():
    import mujoco
    from dm_control.mujoco import engine as DM_ENGINE
    from dm_control.mjcf import physics as DM_PHYS
    from flygym_gymnasium.fly import Fly
    from flygym_gymnasium.simulation import Simulation
    from flygym_gymnasium.examples.locomotion.turning_controller import (
        HybridTurningController)

    rep = {}
    cfg, body, obs = build()
    model = body.physics.model.ptr
    data = body.physics.data.ptr
    rep["модель"] = {
        "nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu),
        "nbody": int(model.nbody), "ngeom": int(model.ngeom),
        "nsensor": int(model.nsensor), "nsensordata": int(model.nsensordata),
        "integrator": int(model.opt.integrator),
        "timestep": float(model.opt.timestep),
        "legacy_step": bool(body.physics.legacy_step),
    }
    print("модель: " + json.dumps(rep["модель"], ensure_ascii=False))

    tally = Tally()
    # слои сверху вниз
    wrap(tally, None, "step", "1 HybridTurningController.step",
         cls=HybridTurningController)
    wrap(tally, None, "step", "2 Simulation.step", cls=Simulation)
    wrap(tally, None, "get_observation", "3 Fly.get_observation", cls=Fly)
    wrap(tally, None, "pre_step", "4 Fly.pre_step", cls=Fly)
    wrap(tally, None, "post_step", "5 Fly.post_step", cls=Fly)
    wrap(tally, None, "step", "6 Physics.step", cls=DM_ENGINE.Physics)
    wrap(tally, None, "forward", "7 Physics.forward (mj_forward)",
         cls=DM_PHYS.Physics)
    arena = body.arena
    wrap(tally, arena, "step", "8 arena.step")
    wrap(tally, arena, "get_olfaction", "9 arena.get_olfaction")

    action = np.array([1.0, 1.0])
    # разогрев
    for _ in range(WARM * N_PHYS):
        body.step(action)
    tally.reset()

    t0 = time.perf_counter()
    for _ in range(N_CTRL * N_PHYS):
        body.step(action)
    wall = time.perf_counter() - t0

    n_phys = N_CTRL * N_PHYS
    rep["шагов физики в замере"] = n_phys
    rep["по стене всего, с"] = round(wall, 3)
    rep["на шаг физики, мкс"] = round(wall / n_phys * 1e6, 1)
    rep["на шаг управления (100 шагов физики), мс"] = round(
        wall / N_CTRL * 1000.0, 2)
    layers = {}
    for k in sorted(tally.t):
        layers[k] = {
            "всего, мс": round(tally.t[k] * 1000.0, 1),
            "вызовов": tally.n[k],
            "на вызов, мкс": round(tally.t[k] / tally.n[k] * 1e6, 1),
            "вызовов на шаг физики": round(tally.n[k] / n_phys, 2),
            "доля шага физики": round(tally.t[k] / wall, 4),
        }
    rep["слои"] = layers
    print("\nслои (вложенные, время включает вложенные вызовы):")
    for k, v in layers.items():
        print("  %-36s %8.1f мс  x%-6d %8.1f мкс/вызов  %5.1f%%  (%.2f/шаг)"
              % (k, v["всего, мс"], v["вызовов"], v["на вызов, мкс"],
                 v["доля шага физики"] * 100, v["вызовов на шаг физики"]))

    # ------------------------------------------------- голая физика рядом
    # Снимаем состояние, гоняем голый mj_step/mj_forward на той же модели,
    # возвращаем состояние обратно: сравнивать надо на одном и том же.
    size = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION)
    saved = np.zeros(size)
    mujoco.mj_getState(model, data, saved,
                       mujoco.mjtState.mjSTATE_INTEGRATION)

    def bare(fn, n, label):
        mujoco.mj_setState(model, data, saved,
                           mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(model, data)
        for _ in range(20):
            fn()
        mujoco.mj_setState(model, data, saved,
                           mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(model, data)
        t = time.perf_counter()
        for _ in range(n):
            fn()
        dt = (time.perf_counter() - t) / n * 1e6
        print("  %-34s %8.1f мкс" % (label, dt))
        return round(dt, 1)

    print("\nголая физика на той же модели (мкс на вызов):")
    bare_rep = {}
    bare_rep["mj_step"] = bare(
        lambda: mujoco.mj_step(model, data, 1), 400, "mj_step")
    bare_rep["mj_step2+mj_step1 (как dm_control)"] = bare(
        lambda: (mujoco.mj_step2(model, data),
                 mujoco.mj_step1(model, data)), 400,
        "mj_step2+mj_step1")
    bare_rep["mj_forward"] = bare(
        lambda: mujoco.mj_forward(model, data), 400, "mj_forward")
    bare_rep["mj_forwardSkip(VEL)"] = bare(
        lambda: mujoco.mj_forwardSkip(
            model, data, mujoco.mjtStage.mjSTAGE_VEL, 0), 400,
        "mj_forwardSkip(STAGE_VEL)")
    bare_rep["mj_step2+mj_step1+mj_forward"] = bare(
        lambda: (mujoco.mj_step2(model, data),
                 mujoco.mj_step1(model, data),
                 mujoco.mj_forward(model, data)), 300,
        "step2+step1+forward (наш путь)")
    bare_rep["mj_step2+mj_step1+mj_forwardSkip"] = bare(
        lambda: (mujoco.mj_step2(model, data),
                 mujoco.mj_step1(model, data),
                 mujoco.mj_forwardSkip(
                     model, data, mujoco.mjtStage.mjSTAGE_VEL, 0)), 300,
        "step2+step1+forwardSkip")
    mujoco.mj_setState(model, data, saved,
                       mujoco.mjtState.mjSTATE_INTEGRATION)
    mujoco.mj_forward(model, data)
    rep["голая физика, мкс"] = bare_rep
    rep["обвязка на шаг физики, мкс"] = round(
        rep["на шаг физики, мкс"]
        - bare_rep["mj_step2+mj_step1+mj_forward"], 1)

    # ------------------------------------------------- cProfile
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(3 * N_PHYS):
        body.step(action)
    pr.disable()
    st = pstats.Stats(pr)
    st.sort_stats("tottime")
    rows = []
    for (fn, line, name), (cc, nc, tt, ct, _) in st.stats.items():
        rows.append({"имя": "%s:%d(%s)" % (Path(fn).name, line, name),
                     "вызовов": nc,
                     "своё, мс": round(tt * 1000.0, 1),
                     "с вложенными, мс": round(ct * 1000.0, 1)})
    rows.sort(key=lambda r: -r["своё, мс"])
    rep["cProfile (300 шагов физики), верх по своему времени"] = rows[:35]
    print("\ncProfile, 300 шагов физики, верх по своему времени:")
    for r in rows[:28]:
        print("  %8.1f мс  %8.1f мс  x%-7d %s"
              % (r["своё, мс"], r["с вложенными, мс"], r["вызовов"], r["имя"]))

    body.close()
    (OUT / "mj_profile.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nотчёт: probe/mj_profile.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
