r"""Три случая одним способом: официальный FlyGym 2.x, наша модель, наша обвязка.

Внешняя документация FlyGym 2.x называет порядка 74 микросекунд на шаг
физики. У нас замерено 637 мкс на шаг той же физики (63,7 мс на сто
шагов, `PERF_V3.md` §2). Разница в 8,6 раза, и прежде чем что-то из
неё выводить, надо убедиться, что сравнивается одно и то же.

Проба меряет три случая **одним и тем же способом** — медиана по
кругам, сто шагов физики в круге, рабочая точка с настоящими
контактами:

* **A** — модель FlyGym 2.1.1 в её собственной среде. Запускается
  отдельным окружением `.venv-fg`, где стоит сам FlyGym 2.1.1;
* **B** — НАША модель, выгруженная в XML, голым `mj_step`;
* **C** — наша же модель через нашу обвязку FlyGym 1.x: шаг
  контроллера целиком и отдельно шаг физики внутри него.

Вместе с временем печатается **опись модели**: тел, степеней свободы,
геометрий, сеток, заданных пар контактов, шаг интегрирования,
контактов в рабочей точке. Без неё сравнение времён ничего не значит:
две мухи могут отличаться вдвое по числу проверок столкновений, и тогда
разница во времени — свойство моделей, а не кода.

    rem случай A, отдельное окружение
    ..\..\..\..\.venv-fg\Scripts\python.exe lab\probe\feas_bench.py --случай A
    rem случаи B и C, рабочее окружение
    ..\..\..\..\.venv\Scripts\python.exe lab\probe\feas_bench.py --случай B
    ..\..\..\..\.venv\Scripts\python.exe lab\probe\feas_bench.py --случай C
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
КОРЕНЬ = Path("D:/fly-brain")
МОДЕЛЬ = КОРЕНЬ / ".venv-mj-model"
SUB = 100
КРУГОВ = 9
ОСАДКА = 3000       # шагов, чтобы муха встала на ноги в случае A

ПОЛЯ = ("qpos", "qvel", "act", "ctrl", "qacc_warmstart",
        "qfrc_applied", "mocap_pos", "mocap_quat")


def опись(m, d=None):
    out = {
        "тел": int(m.nbody), "обобщённых координат": int(m.nq),
        "степеней свободы": int(m.nv), "приводов": int(m.nu),
        "геометрий": int(m.ngeom), "сеток": int(m.nmesh),
        "заданных пар": int(m.npair), "исключений": int(m.nexclude),
        "датчиков": int(m.nsensor),
        "шаг, с": float(m.opt.timestep),
        "решатель": int(m.opt.solver), "итераций": int(m.opt.iterations),
        "допуск": float(m.opt.tolerance),
    }
    if d is not None:
        out["контактов"] = int(d.ncon)
    return out


def мерить(шаг, встать, mujoco, m, d):
    """Медиана по кругам. Круг — сто шагов физики из одной точки."""
    for _ in range(3):
        встать()
        mujoco.mj_step(m, d, nstep=SUB)
    ts = []
    for _ in range(КРУГОВ):
        встать()
        t = time.perf_counter()
        mujoco.mj_step(m, d, nstep=SUB)
        ts.append((time.perf_counter() - t) * 1000.0)
    ts.sort()
    return {"мс на 100 шагов": round(ts[len(ts) // 2], 3),
            "мкс на шаг": round(ts[len(ts) // 2] * 10.0, 1),
            "наименьшая": round(ts[0], 3), "наибольшая": round(ts[-1], 3)}


# ------------------------------------------------------------------ A

def случай_A(сочленить=True):
    import importlib.metadata as md

    import mujoco
    import numpy as np
    from flygym import Simulation
    from flygym.compose import FlatGroundWorld, NeuroMechFly
    from flygym.utils.math import Rotation3D

    рез = {"случай": "A — FlyGym 2.1.1, %s"
           % ("сочленённая муха" if сочленить else "как есть из коробки"),
           "сочленена": bool(сочленить),
           "версии": {p: md.version(p) for p in
                      ("flygym", "mujoco", "numpy", "numba", "scipy")}}
    from flygym.anatomy import (ActuatedDOFPreset, AxisOrder, JointPreset,
                                Skeleton)
    from flygym.compose import KinematicPosePreset

    t0 = time.perf_counter()
    world = FlatGroundWorld()
    fly = NeuroMechFly()
    # Муха СОБИРАЕТСЯ СОЧЛЕНЁННОЙ, и это обязательно. По умолчанию
    # `NeuroMechFly()` в 2.x — это иерархия сеток без единого сустава:
    # замерено 8 тел, 6 степеней свободы, 0 приводов, 0 контактов и
    # 3,3 мкс на шаг. Сравнивать такое с нашей мухой (93 степени
    # свободы, 48 приводов, 18 контактов) — значит сравнивать твёрдое
    # тело с ногами.
    if сочленить:
        skel = Skeleton(axis_order=AxisOrder.YAW_PITCH_ROLL,
                        joint_preset=JointPreset.LEGS_ACTIVE_ONLY)
        fly.add_joints(skel, neutral_pose=KinematicPosePreset.NEUTRAL)
        dofs = skel.get_actuated_dofs_from_preset(
            ActuatedDOFPreset.LEGS_ACTIVE_ONLY)
        fly.add_actuators(dofs, "position",
                          neutral_input=KinematicPosePreset.NEUTRAL)
        fly.add_leg_adhesion()
    world.add_fly(fly, spawn_position=(0.0, 0.0, 0.5),
                  spawn_rotation=Rotation3D("quat", [1.0, 0.0, 0.0, 0.0]))
    sim = Simulation(world)
    m, d = sim.mj_model, sim.mj_data
    рез["сборка, с"] = round(time.perf_counter() - t0, 2)

    # Осадка: муха должна стоять на ногах, иначе контактов нет и шаг
    # дешевле, чем в работе. Та же ловушка, что в `mj_versions.py`,
    # только с другой стороны.
    mujoco.mj_step(m, d, nstep=ОСАДКА)
    поля = {k: np.array(getattr(d, k), copy=True) for k in ПОЛЯ
            if hasattr(d, k)}
    t_ = float(d.time)

    def встать():
        mujoco.mj_resetData(m, d)
        for k, v in поля.items():
            getattr(d, k)[:] = v
        d.time = t_

    встать()
    mujoco.mj_forward(m, d)
    рез["модель"] = опись(m, d)
    рез["замер"] = мерить(SUB, встать, mujoco, m, d)

    # Та же модель, но с НАШИМИ численными настройками: допуск 1e-12
    # вместо 1e-8, предел итераций 1000 вместо 100, шаг 2e-4 вместо
    # 1e-4. Так отделяется «их модель дешевле» от «их настройки мягче».
    if сочленить:
        m.opt.tolerance = 1e-12
        m.opt.iterations = 1000
        m.opt.timestep = 2e-4
        рез["с нашими настройками"] = {
            "опись": опись(m, d),
            "замер": мерить(SUB, встать, mujoco, m, d),
        }
    return рез


# ------------------------------------------------------------------ B

def случай_B():
    import importlib.metadata as md

    import mujoco
    import numpy as np

    xml, state = МОДЕЛЬ / "fly.xml", МОДЕЛЬ / "state.npz"
    if not (xml.exists() and state.exists()):
        return {"случай": "B", "отказ":
                "нет выгруженной модели; сперва probe/model_export.py"}
    рез = {"случай": "B — наша модель, голый mj_step",
           "версии": {p: md.version(p) for p in ("mujoco", "numpy")}}
    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    z = np.load(state)
    поля = {k: z[k] for k in ПОЛЯ if k in z.files}
    t_ = float(z["время"][0]) if "время" in z.files else 0.0

    def встать():
        mujoco.mj_resetData(m, d)
        for k, v in поля.items():
            getattr(d, k)[:] = v
        d.time = t_

    встать()
    mujoco.mj_forward(m, d)
    рез["модель"] = опись(m, d)
    рез["замер"] = мерить(SUB, встать, mujoco, m, d)
    return рез


# ------------------------------------------------------------------ C

def случай_D():
    """Наша муха БЕЗ самостолкновений: чем платят 2316 заданных пар.

    Это не кандидат и не предложение. Самостолкновения — часть модели:
    без них нога проходит сквозь ногу. Замер нужен ровно для одного —
    приписать разницу с FlyGym 2.x (55 пар против 2316) её настоящей
    причине, а не версии движка и не обвязке.
    """
    import importlib.metadata as md

    sys.path.insert(0, str(HERE.parent))
    os.environ.setdefault("FLYBRAIN_OUT", str(HERE / "_feas_out"))
    import probe._isolate  # noqa: F401

    import mujoco
    import numpy as np

    from flybrain import body as B

    было = B._fly_class

    def подменённый(*a_, **k_):
        База = было(*a_, **k_)

        class БезСамостолкновений(База):
            def __init__(self, *aa, **kk):
                kk.setdefault("self_collisions", "none")
                super().__init__(*aa, **kk)

        return БезСамостолкновений

    B._fly_class = подменённый
    try:
        рез = {"случай": "D — наша модель БЕЗ самостолкновений",
               "версии": {p_: md.version(p_) for p_ in
                          ("flygym-gymnasium", "mujoco", "numpy")}}
        cfg = B.BodyConfig(duration_s=0.4, control_ms=20.0, seed=0,
                           device="cuda", readout="ALPN", vision=False)
        sim, _ = B._build_body(cfg, [[25.0, 0.0, 1.5], [-25.0, 0.0, 1.5]])
    finally:
        B._fly_class = было
    m, d = sim.physics.model.ptr, sim.physics.data.ptr
    action = np.array([0.9, 0.9], dtype=float)
    for _ in range(SUB * 2):
        sim.step(action)
    поля = {k: np.array(getattr(d, k), copy=True) for k in ПОЛЯ
            if hasattr(d, k)}
    t_ = float(d.time)

    def встать():
        mujoco.mj_resetData(m, d)
        for k, v in поля.items():
            getattr(d, k)[:] = v
        d.time = t_

    встать()
    mujoco.mj_forward(m, d)
    рез["модель"] = опись(m, d)
    рез["голый mj_step"] = мерить(SUB, встать, mujoco, m, d)
    return рез


def случай_C():
    import importlib.metadata as md

    sys.path.insert(0, str(HERE.parent))
    os.environ.setdefault("FLYBRAIN_OUT", str(HERE / "_feas_out"))
    import probe._isolate  # noqa: F401

    import mujoco
    import numpy as np

    from flybrain import body as B

    рез = {"случай": "C — наша модель через нашу обвязку FlyGym 1.x",
           "версии": {p: md.version(p) for p in
                      ("flygym-gymnasium", "mujoco", "dm-control", "numpy")}}
    cfg = B.BodyConfig(duration_s=0.4, control_ms=20.0, seed=0,
                       device="cuda", readout="ALPN",
                       vision=False, vision_hz=100.0)
    t0 = time.perf_counter()
    sim, _ = B._build_body(cfg, [[25.0, 0.0, 1.5], [-25.0, 0.0, 1.5]])
    рез["сборка, с"] = round(time.perf_counter() - t0, 2)
    m, d = sim.physics.model.ptr, sim.physics.data.ptr
    action = np.array([0.9, 0.9], dtype=float)
    for _ in range(SUB * 2):
        sim.step(action)
    поля = {k: np.array(getattr(d, k), copy=True) for k in ПОЛЯ
            if hasattr(d, k)}
    t_ = float(d.time)

    def встать():
        mujoco.mj_resetData(m, d)
        for k, v in поля.items():
            getattr(d, k)[:] = v
        d.time = t_

    встать()
    mujoco.mj_forward(m, d)
    рез["модель"] = опись(m, d)
    рез["голый mj_step"] = мерить(SUB, встать, mujoco, m, d)

    # Шаг обвязки: контроллер походки, наблюдение, dm_control. Зрение
    # выключено нарочно — иначе в число попал бы рендер глаз, а он к
    # физике отношения не имеет.
    def круг():
        t = time.perf_counter()
        for _ in range(SUB):
            sim.step(action)
        return (time.perf_counter() - t) * 1000.0

    for _ in range(2):
        круг()
    ts = sorted(круг() for _ in range(КРУГОВ))
    рез["шаг через обвязку"] = {
        "мс на 100 шагов": round(ts[len(ts) // 2], 3),
        "мкс на шаг": round(ts[len(ts) // 2] * 10.0, 1),
        "наименьшая": round(ts[0], 3), "наибольшая": round(ts[-1], 3)}
    return рез


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--случай", required=True,
                    choices=["A", "A0", "B", "C", "D"])
    args = ap.parse_args()
    случай = getattr(args, "случай")
    рез = {"A": случай_A, "A0": lambda: случай_A(False),
           "B": случай_B, "C": случай_C, "D": случай_D}[случай]()
    путь = HERE / ("feas_%s.json" % случай)
    путь.write_text(json.dumps(рез, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print(json.dumps(рез, ensure_ascii=False, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
