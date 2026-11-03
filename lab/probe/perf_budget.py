r"""Бюджет задержки по составляющим: куда на самом деле уходят 87,72 мс.

Прежняя кампания замерила блок целиком (`PERF_RT.md` §1): шаг управления
со зрением 258,3 мс, блок «физика + контроллер» 125,03 мс, «один счёт
MuJoCo» 87,72 мс. Из последнего числа следовал вывод, что 0,2× не
берётся разрядом A, потому что сама физика дороже всего бюджета.

Эта проба спрашивает другое: **а сколько из 87,72 мс — это настоящий
решатель MuJoCo, и сколько — обвязка вокруг него.** Вопрос не
риторический. На один шаг управления приходится **сто шагов физики**, и
на каждом из них в Python делается:

* обновление параметров CPG и `cpg_network.step()`;
* **полное наблюдение** `get_observation()` для правил контроллера;
* правила втягивания и спотыкания, цикл **по шести ногам** с поиском
  предзаписанного шага и склейкой углов;
* `arena.step()`, `fly.pre_step()`;
* сам `physics.step()`;
* `fly.post_step()`, а он зовёт **ещё одно полное наблюдение**.

То есть полных наблюдений за шаг управления двести, а наружу уходит
одно — последнее. Если цена лежит здесь, она снимается, не трогая
физику ни на бит.

Как меряется. Каждый отрезок — сто повторений своей работы, отрезки
идут **по кругу** и много раундов, берётся медиана по раундам: так
разогрев и дыхание машины попадают во все отрезки одинаково. Состояние
модели снимается и возвращается вокруг каждого отрезка
(`mj_copyData`), иначе отрезки мерили бы разные состояния. Отдельным
отрезком меряется **пустой цикл** — это пол измерения, и всё, что его
не превышает, считается неотличимым от нуля.

    cd D:\fly-brain\lab
    set PYTHONIOENCODING=utf-8
    ..\.venv\Scripts\python.exe probe\perf_budget.py --зрение да
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("FLYBRAIN_OUT",
                      str(Path(__file__).parent / "_perf_budget_out"))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

OUT = Path(__file__).parent
SUB = 100            # шагов физики на шаг управления
ROUNDS = 7           # раундов по кругу

rep = {"проверки": {}, "замеры": {}, "находки": []}
fails = []


def check(name, ok, info=""):
    print(("  ok     " if ok else "  ПРОВАЛ ") + name
          + (("  " + str(info)) if info else ""), flush=True)
    rep["проверки"][name] = {"пройдено": bool(ok), "замер": str(info)}
    if not ok:
        fails.append(name)


def note(text):
    print("  НАХОДКА " + text, flush=True)
    rep["находки"].append(text)


def main():                                                   # noqa: C901
    ap = argparse.ArgumentParser()
    ap.add_argument("--зрение", default="да")
    args = ap.parse_args()
    vision = str(getattr(args, "зрение")).lower() in ("да", "yes", "1")

    import mujoco

    from flybrain import body as B, data as D

    print("собираю тело, зрение %s" % ("включено" if vision else "выключено"),
          flush=True)
    cfg = B.BodyConfig(duration_s=0.4, control_ms=20.0, seed=0,
                       device="cuda", readout="ALPN",
                       vision=vision, vision_hz=100.0)
    t0 = time.perf_counter()
    # Источники запаха те же, что у песочницы: без них арена не
    # собирается, а с чужим числом источников менялся бы и запах.
    sources = [[25.0, 0.0, 1.5], [-25.0, 0.0, 1.5]]
    sim, _cams = B._build_body(cfg, sources)
    build_s = time.perf_counter() - t0
    fly = sim.fly
    physics = sim.physics
    m, d = physics.model.ptr, physics.data.ptr
    rep["замеры"]["сборка тела, с"] = round(build_s, 2)
    rep["замеры"]["шаг физики, с"] = float(m.opt.timestep)
    rep["замеры"]["тел в модели"] = int(m.nbody)
    rep["замеры"]["степеней свободы"] = int(m.nv)
    rep["замеры"]["приводов"] = int(m.nu)
    print("  модель: %d тел, %d степеней свободы, %d приводов, шаг %.1e с"
          % (m.nbody, m.nv, m.nu, m.opt.timestep), flush=True)
    check("шаг физики тот же, что в продукте",
          abs(float(m.opt.timestep) - cfg.physics_dt) < 1e-12,
          "%.1e против %.1e" % (m.opt.timestep, cfg.physics_dt))

    # Разогрев: без него первые отрезки платят за JIT numba внутри
    # FlyGym и за первое касание страниц памяти.
    action = np.array([0.9, 0.9], dtype=float)
    for _ in range(SUB * 2):
        sim.step(action)

    # Низкоуровневое действие — то самое, что контроллер походки
    # складывает из углов предзаписанного шага и признаков прилипания.
    # Своё придумывать нельзя: `fly.pre_step` ждёт словарь с ключами
    # "joints" и "adhesion", и подсунуть туда команду поворота значит
    # мерить не ту работу. Поэтому оно ПЕРЕХВАТЫВАЕТСЯ у контроллера на
    # одном настоящем шаге.
    поймано = {}
    настоящий = fly.pre_step

    def ловушка(a, s_):
        поймано["действие"] = a
        return настоящий(a, s_)

    fly.pre_step = ловушка
    sim.step(action)
    fly.pre_step = настоящий
    low = поймано.get("действие")
    if low is None:
        print("низкоуровневое действие не перехвачено", flush=True)
        return
    rep["замеры"]["ключи низкоуровневого действия"] = sorted(low)

    # --- снимок состояния вокруг каждого отрезка --------------------
    #
    # Состояние снимается ровно теми полями, что и в `replay.py`:
    # `mjSTATE_INTEGRATION` плюс разогрев решателя, показания датчиков и
    # ускорения. Там этот набор подобран не на глаз, а замером — без
    # `qacc_warmstart` поза через сто шагов расходилась на 0,17 мм, а
    # без `sensordata` через двадцать один шаг расходились спайки.
    # Здесь точность не так важна, как там, но брать другой набор
    # значило бы заводить второй, непроверенный.
    _size = mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_INTEGRATION)
    _buf = np.zeros(_size, dtype=np.float64)

    def save():
        mujoco.mj_getState(m, d, _buf, mujoco.mjtState.mjSTATE_INTEGRATION)
        return {
            "состояние": np.array(_buf, copy=True),
            "разогрев": np.array(d.qacc_warmstart, copy=True),
            "датчики": np.array(d.sensordata, copy=True),
            "ускорения": np.array(d.qacc, copy=True),
            "фазы": np.array(sim.cpg_network.curr_phases, copy=True),
            "амплитуды": np.array(sim.cpg_network.curr_magnitudes, copy=True),
            "втягивание": np.array(sim.retraction_correction, copy=True),
            "спотыкание": np.array(sim.stumbling_correction, copy=True),
            "время": float(sim.curr_time),
        }

    def load(st):
        mujoco.mj_setState(m, d, st["состояние"],
                           mujoco.mjtState.mjSTATE_INTEGRATION)
        d.qacc_warmstart[:] = st["разогрев"]
        d.sensordata[:] = st["датчики"]
        d.qacc[:] = st["ускорения"]
        sim.cpg_network.curr_phases[:] = st["фазы"]
        sim.cpg_network.curr_magnitudes[:] = st["амплитуды"]
        sim.retraction_correction[:] = st["втягивание"]
        sim.stumbling_correction[:] = st["спотыкание"]
        sim.curr_time = st["время"]

    # --- что меряем -------------------------------------------------
    def L_пусто():
        for _ in range(SUB):
            pass

    def L_полный():
        for _ in range(SUB):
            sim.step(action)

    def L_dm():
        for _ in range(SUB):
            physics.step()

    def L_mj_по_одному():
        for _ in range(SUB):
            mujoco.mj_step(m, d)

    def L_mj_пачкой():
        mujoco.mj_step(m, d, nstep=SUB)

    def L_наблюдение():
        for _ in range(SUB):
            fly.get_observation(sim)

    def L_pre():
        for _ in range(SUB):
            fly.pre_step(low, sim)

    def L_arena():
        for _ in range(SUB):
            sim.arena.step(dt=sim.timestep, physics=physics)

    def L_cpg():
        for _ in range(SUB):
            sim.cpg_network.step()

    legs = [
        ("пустой цикл", L_пусто),
        ("полный шаг (эталон)", L_полный),
        ("physics.step() dm_control", L_dm),
        ("mj_step по одному", L_mj_по_одному),
        ("mj_step пачкой nstep=100", L_mj_пачкой),
        ("get_observation", L_наблюдение),
        ("fly.pre_step", L_pre),
        ("arena.step", L_arena),
        ("cpg_network.step", L_cpg),
    ]
    if vision:
        def L_vision():
            for _ in range(SUB):
                fly._update_vision(sim)
        legs.append(("_update_vision", L_vision))

    # --- круги ------------------------------------------------------
    times = {имя: [] for имя, _ in legs}
    for r in range(ROUNDS):
        for имя, fn in legs:
            st = save()
            t = time.perf_counter()
            fn()
            times[имя].append((time.perf_counter() - t) * 1000.0)
            load(st)
        print("  круг %d из %d" % (r + 1, ROUNDS), flush=True)

    пол = float(np.median(times["пустой цикл"]))
    эталон = float(np.median(times["полный шаг (эталон)"]))
    строки = []
    for имя, _ in legs:
        v = np.asarray(times[имя])
        строки.append({
            "что": имя,
            "медиана, мс": round(float(np.median(v)), 3),
            "наименьшая, мс": round(float(v.min()), 3),
            "наибольшая, мс": round(float(v.max()), 3),
            "доля эталона, %": round(100.0 * float(np.median(v)) / эталон, 1),
        })
    rep["замеры"]["отрезки, 100 шагов физики"] = строки
    print("\n  %-28s %9s %9s %7s" % ("что", "медиана", "размах", "доля"),
          flush=True)
    for s in строки:
        print("  %-28s %8.3f  %5.1f…%-5.1f %6.1f %%"
              % (s["что"], s["медиана, мс"], s["наименьшая, мс"],
                 s["наибольшая, мс"], s["доля эталона, %"]), flush=True)

    мед = {s["что"]: s["медиана, мс"] for s in строки}
    check("пустой цикл неотличим от нуля", пол < 0.05,
          "%.4f мс на 100 оборотов" % пол)
    check("эталонный шаг сходится с прежним замером кампании",
          60.0 < эталон < 400.0, "%.1f мс против 125,03 мс в PERF_RT §1"
          % эталон)
    # ГЛАВНЫЙ вопрос пробы.
    доля_mj = 100.0 * мед["mj_step пачкой nstep=100"] / эталон
    rep["замеры"]["доля настоящего решателя, %"] = round(доля_mj, 1)
    note("настоящий решатель MuJoCo — %.1f %% эталонного шага (%.2f мс "
         "из %.2f)" % (доля_mj, мед["mj_step пачкой nstep=100"], эталон))
    check("решатель MuJoCo — не весь шаг", доля_mj < 95.0,
          "%.1f %%" % доля_mj)
    note("два полных наблюдения на шаг физики: get_observation %.2f мс на "
         "сто вызовов, то есть %.2f мс на двести"
         % (мед["get_observation"], 2 * мед["get_observation"]))
    rep["замеры"]["обвязка поверх решателя, мс"] = round(
        эталон - мед["mj_step пачкой nstep=100"], 3)

    (OUT / ("perf_budget_%s.json" % ("зрение" if vision else "без"))).write_text(
        json.dumps(rep, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")
    print("\nпровалов: %d" % len(fails), flush=True)
    for name in fails:
        print("  - " + name, flush=True)


if __name__ == "__main__":
    main()
    sys.exit(1 if fails else 0)
