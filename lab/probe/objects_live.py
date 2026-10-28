"""Зрительные объекты через проверенный протокол команд живых условий.

Пул объектов (`flybrain/objects.py`) уже проверен пробой
`probe/objects.py`: он виден сетчатке, нависание следует геометрии,
физику не задевает. Здесь проверяется другое — что им можно **управлять
на ходу тем же путём**, которым проект уже управляет источниками
запаха: ручка в `live.BODY_KNOBS`, проверка до записи, применение на
границе шага, запись в журнал с номером команды и модельным временем.

Своего протокола команд здесь нет нарочно. Ручка `objects` устроена как
ручка `sources`, и это не экономия, а условие: тот путь уже проверен и
умеет отказывать до правки модели, а не посреди неё.

Что проверяется:

1. пул **находится** в собранной модели по именам тел, и его нет, если
   его не вписывали. Встречный замер: эпизод без пула обязан показывать
   ноль слотов и отказывать на правке;
2. правка доходит до сцены: после `apply` объект виден сетчатке. Это и
   есть подтверждение применения — не ответ «принято», а настоящая
   перемена в кадре;
3. **журнал**: у правки есть номер команды, модельное время, что было и
   что стало. Правка, которой нет в журнале, не считается сделанной;
4. **отказы до записи**: неверное число слотов, значение за пределом,
   лишнее поле. После отказа модель обязана остаться прежней — иначе
   отказ хуже, чем его отсутствие.

    cd D:\\fly-brain\\lab
    set PYTHONIOENCODING=utf-8
    ..\\.venv\\Scripts\\python.exe probe\\objects_live.py
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("FLYBRAIN_OUT",
                      str(Path(__file__).parent / "_objects_live_out"))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

OUT = Path(__file__).parent
PHYS = 200
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


def build(pool=True):
    from flygym_gymnasium import Fly
    from flygym_gymnasium.arena import OdorArena
    from flygym_gymnasium.preprogrammed import (
        default_leg_sensor_placements)
    from flygym_gymnasium.examples.locomotion import HybridTurningController

    from flybrain import objects as O

    ar = OdorArena(odor_source=np.array([[25.0, 0.0, 1.5]]),
                   peak_odor_intensity=np.ones((1, 1)))
    if pool:
        O.inject(ar)
    fly = Fly(enable_adhesion=True, enable_olfaction=True,
              enable_vision=True, vision_refresh_rate=100,
              contact_sensor_placements=default_leg_sensor_placements,
              init_pose="stretch", control="position",
              spawn_pos=(0.0, 0.0, 0.2))
    return HybridTurningController(fly=fly, cameras=[], arena=ar,
                                   timestep=1e-4)


def still(sim, steps=4):
    """Постоять и снять последний кадр сетчатки: сравнивать надо позы."""
    obs = None
    frame = None
    for _ in range(steps):
        for _ in range(PHYS):
            obs, _, _, _, _ = sim.step(np.zeros(2))
        frame = np.asarray(obs["vision"], dtype=np.float32).copy()
    return obs, frame


def main():
    t0 = time.perf_counter()
    print("Зрительные объекты через живые условия", flush=True)
    from flybrain import body as B, data as D, live as L

    neurons = D.load_neurons()
    cfg = B.BodyConfig(duration_s=1.0, seed=0)

    # --- 1. пул находится, и без пула его нет ---
    print("\n1. Обнаружение пула", flush=True)
    sim_no = build(pool=False)
    sim_no.reset(seed=0)
    st_no = L.BodyLive()
    st_no.attach(sim_no, np.array([[25.0, 0.0, 1.5]]), cfg, neurons, "cpu")
    n_no = st_no.object_slots()
    refused_no = None
    try:
        st_no.apply({"objects": [{"on": True}]}, cmd_id="x1", t_ms=0.0)
    except Exception as e:                            # noqa: BLE001
        refused_no = str(e)
    del sim_no

    sim = build(pool=True)
    sim.reset(seed=0)
    st = L.BodyLive()
    st.attach(sim, np.array([[25.0, 0.0, 1.5]]), cfg, neurons, "cpu")
    n_yes = st.object_slots()
    rep["замеры"]["обнаружение пула"] = {
        "слотов без пула": n_no, "слотов с пулом": n_yes,
        "отказ без пула": refused_no,
    }
    print("   слотов без пула %d, с пулом %d" % (n_no, n_yes), flush=True)
    check("пул находится в собранной модели по именам тел, а без пула "
          "ручка отказывает и объясняет почему",
          n_no == 0 and n_yes == 6 and refused_no
          and "компиляции" in refused_no,
          "без пула слотов %d и правка отклонена словами «%s»; с пулом "
          "слотов %d. Подпись привязки при этом не менялась: пул ищется "
          "по именам тел, как уже ищутся mocap-строки маркеров запаха"
          % (n_no, (refused_no or "")[:90], n_yes))

    snap0 = st.snapshot()["objects"]
    rep["замеры"]["начальное состояние"] = snap0
    check("все слоты по умолчанию выключены — сеанс без объектов идёт "
          "как прежде",
          len(snap0) == 6 and not any(s["on"] for s in snap0),
          "слотов %d, включённых %d. Это условие совместимости: 28 "
          "сохранённых прогонов не знают ни про какие объекты"
          % (len(snap0), sum(1 for s in snap0 if s["on"])))

    # --- 2. правка доходит до сцены ---
    print("\n2. Правка доходит до сцены", flush=True)
    obs, ref = still(sim)
    pose = np.asarray(obs["fly"])[0]
    fwd = np.asarray(obs["fly_orientation"])
    tgt = (float(pose[0] + 6.0 * fwd[0]), float(pose[1] + 6.0 * fwd[1]))
    want = [dict(s) for s in snap0]
    want[0] = {"x": tgt[0], "y": tgt[1], "z": 1.5, "radius": 1.0,
               "grey": 0.05, "on": True}
    now = st.apply({"objects": want}, cmd_id="obj-1", t_ms=123.0)
    obs, after = still(sim, steps=3)
    d = (ref - after).max(axis=-1)
    seen = {"потемневших омматидиев": int((d > 0.05).sum()),
            "наибольшее затемнение": round(float(d.max()), 5)}
    rep["замеры"]["правка дошла"] = {
        "подтверждение": now["objects"][0], "сетчатка": seen,
        "цель, мм": [round(v, 3) for v in tgt]}
    print("   подтверждено %s; потемнело %d омматидиев, наибольшее "
          "затемнение %.3f"
          % (now["objects"][0], seen["потемневших омматидиев"],
             seen["наибольшее затемнение"]), flush=True)
    check("правка дошла до сцены: объект виден сетчатке, а не только "
          "подтверждён словами",
          now["objects"][0]["on"] is True
          and abs(now["objects"][0]["x"] - tgt[0]) < 1e-3
          and seen["потемневших омматидиев"] > 10
          and seen["наибольшее затемнение"] > 0.1,
          "подтверждение вернуло on=True и x=%.3f при заказанном %.3f; "
          "в кадре потемнело %d омматидиев с наибольшим затемнением "
          "%.3f. Подтверждение без перемены в кадре ничего не стоило бы "
          "— здесь проверено и то, и другое"
          % (now["objects"][0]["x"], tgt[0],
             seen["потемневших омматидиев"],
             seen["наибольшее затемнение"]))

    # --- 3. журнал ---
    print("\n3. Журнал правки", flush=True)
    rec = [r for r in st.log if "objects" in (r.get("стало") or {})]
    hist = st.pool.slots[0].история
    rep["замеры"]["журнал"] = {
        "записей о объектах": len(rec),
        "последняя запись": {k: rec[-1][k] for k in
                            ("что", "модельное, мс", "команда")}
        if rec else None,
        "правок в истории слота": len(hist),
        "номера команд в истории": [h.get("команда") for h in hist],
    }
    ok_log = (len(rec) == 1
              and rec[-1]["команда"] == "obj-1"
              and rec[-1]["модельное, мс"] == 123.0
              and rec[-1]["было"]["objects"][0]["on"] is False
              and rec[-1]["стало"]["objects"][0]["on"] is True
              and any(h.get("команда") == "obj-1" for h in hist))
    print("   записей %d, команда %s, модельное время %s мс"
          % (len(rec), rec[-1]["команда"] if rec else None,
             rec[-1]["модельное, мс"] if rec else None), flush=True)
    check("правка записана: номер команды, модельное время, что было и "
          "что стало", ok_log,
          "в журнале живых условий одна запись о объектах: команда %r, "
          "модельное время %s мс, было on=False, стало on=True. В "
          "истории самого слота %d правок, и номер команды там тот же. "
          "Правка, которой нет в журнале, не считается сделанной"
          % (rec[-1]["команда"] if rec else None,
             rec[-1]["модельное, мс"] if rec else None, len(hist)))

    # --- 4. отказы до записи ---
    print("\n4. Отказы до записи", flush=True)
    before = st.snapshot()["objects"]
    bad = {
        "слотов меньше": [{"on": True}],
        "значение за пределом": [dict(s) for s in before],
        "лишнее поле": [dict(s) for s in before],
        "не список": {"on": True},
    }
    bad["значение за пределом"][0] = dict(before[0], x=10_000.0)
    bad["лишнее поле"][0] = dict(before[0], цвет="красный")
    refusals = {}
    for tag, payload in bad.items():
        try:
            st.apply({"objects": payload}, cmd_id="bad-" + tag, t_ms=200.0)
            refusals[tag] = None
        except Exception as e:                        # noqa: BLE001
            refusals[tag] = str(e)[:120]
    after_bad = st.snapshot()["objects"]
    unchanged = before == after_bad
    rep["замеры"]["отказы"] = {"тексты": refusals,
                               "модель не изменилась": bool(unchanged)}
    for tag, msg in refusals.items():
        print("   %-22s %s" % (tag, (msg or "НЕ ОТКАЗАЛ")[:80]), flush=True)
    check("все четыре неверные правки отклонены ДО записи, и модель "
          "осталась прежней",
          all(refusals.values()) and unchanged,
          "отклонены: %s. Состояние пула до и после попыток совпало — "
          "значит проверка идёт целиком до правки, а не по ходу неё. "
          "Отказ посреди записи оставил бы сцену в состоянии, которого "
          "никто не заказывал"
          % ", ".join(k for k, v in refusals.items() if v))

    del sim
    note("управление объектами идёт через существующую ручку живых "
         "условий, а не через свой протокол: проверка до записи, "
         "применение на границе шага, запись с номером команды и "
         "модельным временем. Зрительного отклика сети на объекты это "
         "не добавляет и не обещает: сигнал не доходит до T4, T5 и "
         "LPLC2 (`VISION.md` §5).")

    rep["замеры"]["проба заняла, с"] = round(time.perf_counter() - t0, 1)
    (OUT / "objects_live.json").write_text(
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
