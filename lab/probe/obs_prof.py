"""Из чего состоит наблюдение `Fly.get_observation` и что из него читают.

Наблюдение собирается на КАЖДОМ шаге физики, то есть сто раз на шаг
управления, и в замере `probe/ctrl_prof.py` это оказалось сравнимо с
самим `mj_step`. Здесь оно разобрано на части: сколько стоит каждая и
кто её читает.

Разбор сделан копией метода с таймерами (в пробе, не в библиотеке), и
копия сверяется с настоящим методом побитово по всем полям — иначе
разложение относилось бы не к тому коду.

Мозг тут не нужен: команда телу задана числом, и время наблюдения от неё
не зависит.

    cd D:\\fly-brain\\lab
    set PYTHONIOENCODING=utf-8
    ..\\.venv\\Scripts\\python.exe probe\\obs_prof.py
"""

import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

OUT = Path(__file__).parent
DT = 2e-4
ACC = OrderedDict()

# Кто читает каждую часть — выяснено поиском по дереву и по пакету.
WHO = {
    "сенсоры суставов": "никто: `obs['joints']` в проекте не читается. "
                        "Читал бы только `motor.HeadStabilizer` через "
                        "`Fly._last_observation`, и только при "
                        "включённой стабилизации головы",
    "поза и углы (scipy Rotation)": "`obs['fly'][0]` — `body._pose` и "
                                    "правило втягивания; `last_obs['rot']` "
                                    "— камера видео (YawOnlyCamera)",
    "силы контактов: именованная выборка cfrc_ext":
        "правило спотыкания контроллера (и `last_obs` для видео)",
    "силы контактов: перебор контактов MuJoCo":
        "правило спотыкания: поправка на прилипание",
    "силы контактов: вычет силы прилипания":
        "правило спотыкания: поправка на прилипание",
    "положения контактов": "только отрисовка контактов в видео "
                           "(`draw_contacts`), выключена",
    "концы ног": "правило втягивания контроллера",
    "осевые векторы": "`obs['fly_orientation']` — правило спотыкания и "
                      "`body._pose`; остальные два — только "
                      "`info['flip']`",
    "приведение к float32": "все читатели",
    "запах": "`body.read_odour`",
}


def tic():
    return time.perf_counter()


def add(name, dt):
    row = ACC.get(name)
    if row is None:
        row = ACC[name] = [0.0, 0]
    row[0] += dt
    row[1] += 1


def timed_observation(fly, sim):
    """Копия `Fly.get_observation` с таймерами. Арифметика не менялась."""
    from scipy.spatial.transform import Rotation as R

    physics = sim.physics

    t = tic()
    joint_obs = np.zeros((3, len(fly.monitored_joints)))
    a_sd = physics.bind(fly._actuated_joint_sensors).sensordata
    n_sd = physics.bind(fly._non_actuated_joint_sensors).sensordata
    for i, joint in enumerate(fly.actuated_joints):
        base = i * 3
        joint_obs[:3, i] = a_sd[base:base + 3]
    for i in range(len(fly.monitored_joints) - len(fly.actuated_joints)):
        base = i * 5
        tq = np.linalg.norm(n_sd[base + 1:base + 5])
        joint_obs[:3, len(fly.actuated_joints) + i] = [
            *n_sd[base:base + 2], tq]
    joint_obs[2, :] *= 1e-9
    joint_obs = joint_obs[:, fly._monitored_joint_order]
    add("сенсоры суставов", tic() - t)

    t = tic()
    cart_pos = physics.bind(fly._body_sensors[0]).sensordata
    cart_vel = physics.bind(fly._body_sensors[1]).sensordata
    quat = physics.bind(fly._body_sensors[2]).sensordata
    ang_pos = R.from_quat(quat[[1, 2, 3, 0]]).as_euler("ZYX")
    ang_vel = physics.bind(fly._body_sensors[3]).sensordata
    fly_pos = np.array([cart_pos, cart_vel, ang_pos, ang_vel])
    fly.last_obs["rot"] = ang_pos
    fly.last_obs["pos"] = cart_pos
    add("поза и углы (scipy Rotation)", tic() - t)

    t = tic()
    contact_forces = physics.named.data.cfrc_ext[
        fly.contact_sensor_placements][:, 3:].copy()
    add("силы контактов: именованная выборка cfrc_ext", tic() - t)
    t = tic()
    if fly.enable_adhesion:
        contactid_normal = {}
        fly._active_adhesion = np.zeros(fly.n_legs, dtype=bool)
        for contact in physics.data.contact:
            id_ = np.where(fly._adhesion_actuator_geom_id == contact.geom1)
            if len(id_[0]) > 0 and contact.exclude == 0:
                csid = fly._adhesion_bodies_with_contact_sensors[id_][0]
                contactid_normal.setdefault(csid, []).append(
                    contact.frame[:3])
                fly._active_adhesion[id_] = True
            id_ = np.where(fly._adhesion_actuator_geom_id == contact.geom2)
            if len(id_[0]) > 0 and contact.exclude == 0:
                csid = fly._adhesion_bodies_with_contact_sensors[id_][0]
                contactid_normal.setdefault(csid, []).append(
                    contact.frame[:3])
                fly._active_adhesion[id_] = True
        add("силы контактов: перебор контактов MuJoCo", tic() - t)
        t = tic()
        for csid, normal in contactid_normal.items():
            aid = fly._adhesion_bodies_with_contact_sensors == csid
            if fly._last_adhesion[aid] > 0:
                if len(np.shape(normal)) > 1:
                    normal = np.mean(normal, axis=0)
                contact_forces[csid, :] -= fly.adhesion_force * normal
    fly.last_obs["contact_forces"] = contact_forces
    add("силы контактов: вычет силы прилипания", tic() - t)

    t = tic()
    fly.last_obs["contact_pos"] = (
        physics.named.data.xpos[fly.contact_sensor_placements].copy().T)
    add("положения контактов", tic() - t)

    t = tic()
    ee_pos = physics.bind(fly._end_effector_sensors).sensordata.copy()
    ee_pos = ee_pos.reshape((fly.n_legs, 3))
    add("концы ног", tic() - t)

    t = tic()
    orientation_vec = physics.bind(fly._body_sensors[4]).sensordata.copy()
    cardinal = [physics.bind(fly._body_sensors[4]).sensordata.copy(),
                physics.bind(fly._body_sensors[5]).sensordata.copy(),
                physics.bind(fly._body_sensors[6]).sensordata.copy()]
    add("осевые векторы", tic() - t)

    t = tic()
    obs = {
        "joints": joint_obs.astype(np.float32),
        "fly": fly_pos.astype(np.float32),
        "contact_forces": contact_forces.astype(np.float32),
        "end_effectors": ee_pos.astype(np.float32),
        "fly_orientation": orientation_vec.astype(np.float32),
        "cardinal_vectors": np.array(cardinal).astype(np.float32),
    }
    add("приведение к float32", tic() - t)

    t = tic()
    if fly.enable_olfaction:
        ant = physics.bind(fly._antennae_sensors).sensordata
        obs["odor_intensity"] = sim.arena.get_olfaction(
            ant.reshape(4, 3)).astype(np.float32)
    add("запах", tic() - t)
    return obs


def main():
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
    body = G.build(gait="fast", fly=fly, cameras=[], arena=arena,
                   timestep=DT)
    body.reset(seed=0)
    act = np.array([1.0, 1.0])

    n_phys = 2000                     # 20 шагов управления
    # Сверка копии с настоящим методом: по каждому полю побитово.
    bad = []
    t_real = 0.0
    for i in range(n_phys):
        body.step(act)
        t = tic()
        real = fly.get_observation(body)
        t_real += tic() - t
        mine = timed_observation(fly, body)
        if i % 100 == 0:
            for k in real:
                a, b = real[k], mine.get(k)
                if b is None or a.shape != b.shape or a.dtype != b.dtype \
                        or a.tobytes() != b.tobytes():
                    bad.append((i, k))

    per_ctrl = 100.0 / n_phys * 1000.0     # мс на шаг управления
    rep = {"шагов физики": n_phys,
           "расхождений копии с методом": len(bad),
           "наблюдение целиком, мс на шаг управления":
               round(t_real * per_ctrl, 3),
           "части, мс на шаг управления": {}}
    total = 0.0
    for name, (s, c) in ACC.items():
        v = s * per_ctrl
        total += v
        rep["части, мс на шаг управления"][name] = {
            "мс": round(v, 3), "кто читает": WHO.get(name, "?")}
    rep["сумма частей, мс"] = round(total, 3)

    print("копия сошлась с методом: %s" % ("да" if not bad else bad[:5]))
    print("наблюдение целиком: %.2f мс на шаг управления (сумма частей "
          "%.2f)" % (rep["наблюдение целиком, мс на шаг управления"], total))
    for name, d in rep["части, мс на шаг управления"].items():
        print("  %-32s %7.3f мс" % (name, d["мс"]))
        print("      читает: %s" % d["кто читает"])
    p = OUT / "obs_prof.json"
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    print("отчёт: %s" % p)
    body.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
