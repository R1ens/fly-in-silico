"""Из чего складываются 900 мкс `Fly.get_observation`.

Куски кода наблюдения вырезаны один в один из `flygym_gymnasium/fly.py`
и мерятся по отдельности на собранном теле, без шага физики: состояние
стоит на месте, поэтому числа сравнимы между кусками.

Отдельно мерится дешёвая замена каждому куску: целочисленные индексы
вместо именованных, прямое чтение `data.sensordata` вместо биндинга.
Равенство значений проверяется тут же.

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\mj_obs_parts.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

OUT = Path(__file__).parent
N = 300


def timeit(fn, n=N, warm=30):
    for _ in range(warm):
        fn()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n * 1e6


def main():
    import mujoco
    from scipy.spatial.transform import Rotation as R
    from flybrain import body as B

    cfg = B.BodyConfig(duration_s=1.0, control_ms=20.0, seed=0)
    sim, _ = B._build_body(cfg, B.DEFAULT_SOURCES, None, head_stab=None)
    sim.reset(seed=0)
    action = np.array([1.0, 1.0])
    for _ in range(200):
        sim.step(action)
    fly = sim.fly
    physics = sim.physics
    data = physics.data
    rep = {}

    parts = {}

    # --- 1. суставные датчики
    def p_joints():
        joint_obs = np.zeros((3, len(fly.monitored_joints)))
        a = physics.bind(fly._actuated_joint_sensors).sensordata
        na = physics.bind(fly._non_actuated_joint_sensors).sensordata
        for i, joint in enumerate(fly.actuated_joints):
            base_idx = i * 3
            joint_obs[:3, i] = a[base_idx: base_idx + 3]
        for i in range(len(fly.monitored_joints) - len(fly.actuated_joints)):
            base_idx = i * 5
            jt = np.linalg.norm(na[base_idx + 1: base_idx + 5])
            joint_obs[:3, len(fly.actuated_joints) + i] = [
                *na[base_idx: base_idx + 2], jt]
        joint_obs[2, :] *= 1e-9
        return joint_obs[:, fly._monitored_joint_order]

    parts["1 суставные датчики (42 сустава, цикл на Python)"] = timeit(p_joints)

    # --- 2. поза и ориентация + scipy
    def p_pose():
        cart_pos = physics.bind(fly._body_sensors[0]).sensordata
        cart_vel = physics.bind(fly._body_sensors[1]).sensordata
        quat = physics.bind(fly._body_sensors[2]).sensordata
        ang_pos = R.from_quat(quat[[1, 2, 3, 0]]).as_euler("ZYX")
        ang_vel = physics.bind(fly._body_sensors[3]).sensordata
        return np.array([cart_pos, cart_vel, ang_pos, ang_vel])

    parts["2 поза и ориентация (4 биндинга + scipy)"] = timeit(p_pose)

    def p_scipy():
        quat = physics.bind(fly._body_sensors[2]).sensordata
        return R.from_quat(quat[[1, 2, 3, 0]]).as_euler("ZYX")

    parts["2a из них только scipy quat->euler"] = timeit(p_scipy)

    # --- 3. cfrc_ext по именам
    names = fly.contact_sensor_placements

    def p_cfrc_named():
        return physics.named.data.cfrc_ext[names][:, 3:].copy()

    parts["3 cfrc_ext по именам (named.data[список имён])"] = timeit(p_cfrc_named)

    body_ids = np.array([physics.model.name2id(nm, "body") for nm in names])

    def p_cfrc_ids():
        return data.cfrc_ext[body_ids][:, 3:].copy()

    parts["3a cfrc_ext по целым индексам"] = timeit(p_cfrc_ids)
    same_cfrc = np.array_equal(p_cfrc_named(), p_cfrc_ids())

    # --- 4. поправка на прилипание: цикл по контактам
    def p_adhesion():
        contactid_normal = {}
        active = np.zeros(fly.n_legs, dtype=bool)
        for contact in data.contact:
            id_ = np.where(fly._adhesion_actuator_geom_id == contact.geom1)
            if len(id_[0]) > 0 and contact.exclude == 0:
                csid = fly._adhesion_bodies_with_contact_sensors[id_][0]
                contactid_normal.setdefault(csid, []).append(contact.frame[:3])
                active[id_] = True
            id_ = np.where(fly._adhesion_actuator_geom_id == contact.geom2)
            if len(id_[0]) > 0 and contact.exclude == 0:
                csid = fly._adhesion_bodies_with_contact_sensors[id_][0]
                contactid_normal.setdefault(csid, []).append(contact.frame[:3])
                active[id_] = True
        return contactid_normal, active

    parts["4 поправка на прилипание (цикл по контактам)"] = timeit(p_adhesion)
    rep["контактов в этом состоянии"] = int(data.ncon)

    # --- 5. xpos по именам (last_obs["contact_pos"])
    def p_xpos_named():
        return physics.named.data.xpos[names].copy().T

    parts["5 xpos по именам"] = timeit(p_xpos_named)

    def p_xpos_ids():
        return data.xpos[body_ids].copy().T

    parts["5a xpos по целым индексам"] = timeit(p_xpos_ids)
    same_xpos = np.array_equal(p_xpos_named(), p_xpos_ids())

    # --- 6. концы ног, ориентация, кардинальные векторы
    def p_ee():
        ee = physics.bind(fly._end_effector_sensors).sensordata.copy()
        return ee.reshape((fly.n_legs, 3))

    parts["6 концы ног (1 биндинг)"] = timeit(p_ee)

    def p_cardinal():
        return [physics.bind(fly._body_sensors[4]).sensordata.copy(),
                physics.bind(fly._body_sensors[5]).sensordata.copy(),
                physics.bind(fly._body_sensors[6]).sensordata.copy()]

    parts["7 кардинальные векторы (3 биндинга)"] = timeit(p_cardinal)

    # --- 8. обоняние
    def p_olf():
        ap = physics.bind(fly._antennae_sensors).sensordata
        return sim.arena.get_olfaction(ap.reshape(4, 3))

    parts["8 обоняние (биндинг + арена)"] = timeit(p_olf)

    # --- 9. один биндинг-чтение и его замена
    bnd = physics.bind(fly._end_effector_sensors)
    arr, idx = bnd._get_cached_array_and_index("sensordata")

    def p_bind_read():
        return physics.bind(fly._end_effector_sensors).sensordata

    def p_direct_read():
        return arr[idx]

    parts["9 одно чтение через биндинг"] = timeit(p_bind_read, 2000, 200)
    parts["9a то же прямым индексом"] = timeit(p_direct_read, 2000, 200)
    same_bind = np.array_equal(np.asarray(p_bind_read()), p_direct_read())

    # --- 10. всё наблюдение целиком
    parts["10 ВСЁ get_observation"] = timeit(lambda: fly.get_observation(sim))

    rep["куски, мкс"] = {k: round(v, 1) for k, v in parts.items()}
    rep["значения совпали"] = {
        "cfrc_ext по именам = по индексам": bool(same_cfrc),
        "xpos по именам = по индексам": bool(same_xpos),
        "биндинг = прямой индекс": bool(same_bind),
    }
    print("куски наблюдения, мкс на вызов:")
    for k, v in parts.items():
        print("  %-52s %8.1f" % (k, v))
    print("\nсовпадение значений: "
          + json.dumps(rep["значения совпали"], ensure_ascii=False))
    sim.close()
    (OUT / "mj_obs_parts.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("отчёт: probe/mj_obs_parts.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
