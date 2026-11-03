r"""Сокращённый съём при зрении: та же траектория и чего это стоит.

`obs_cost.py` показал, что четыре поля правил совпадают побитово и что
расписание зрения сохраняется. Этого мало: правила читают поля, а
траекторию ведёт весь шаг целиком. Здесь строятся **две мухи** на одном
зерне — с сокращённым съёмом и без — и прогоняются одинаковой чередой
действий. Требование объявлено до замера: `qpos`, `qvel` и время
обязаны совпасть **побитово** на каждом сравнении, а не «в пределах
точности».

Проверка умеет провалиться: если бы сокращённый съём менял хоть что-то
из того, что читают правила, траектории разошлись бы — ровно так и
разошёлся в своё время отвергнутый `fast_physics`.

Время меряется там же и **чередующимися кругами**: сборка одна, мухи
две, и круги идут по очереди, чтобы разогрев и дыхание машины попали в
обе.

    cd D:\fly-brain\lab
    set PYTHONIOENCODING=utf-8
    ..\.venv\Scripts\python.exe probe\rule_obs_equal.py
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("FLYBRAIN_OUT",
                      str(Path(__file__).parent / "_rule_obs_out"))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

OUT = Path(__file__).parent
SUB = 100
STEPS = 8          # шагов управления
ROUNDS = 7

rep = {"проверки": {}, "замеры": {}, "находки": []}
fails = []


def check(name, ok, info=""):
    print(("  ok     " if ok else "  ПРОВАЛ ") + name
          + (("  " + str(info)) if info else ""), flush=True)
    rep["проверки"][name] = {"пройдено": bool(ok), "замер": str(info)}
    if not ok:
        fails.append(name)


def собрать(rule_obs, vision):
    from flybrain import body as B

    # Походка ИМЕННО `legacy`: правка, которую проверяет эта проба,
    # действует только там. На умолчании (`gait="fast"`) контроллер
    # свой, и он переиспользует наблюдение прошлого шага сам
    # (`gait._rules_observation`), так что сокращать там нечего — и
    # первая редакция этой пробы честно показала ×1,006, сравнив
    # быструю походку саму с собой.
    cfg = B.BodyConfig(duration_s=1.0, control_ms=20.0, seed=0,
                       device="cuda", readout="ALPN", gait="legacy",
                       vision=vision, vision_hz=100.0, rule_obs=rule_obs)
    sources = [[25.0, 0.0, 1.5], [-25.0, 0.0, 1.5]]
    sim, _ = B._build_body(cfg, sources)
    return sim


def main():                                                   # noqa: C901
    vision = True
    print("собираю две мухи, зрение включено", flush=True)
    t0 = time.perf_counter()
    быстрая = собрать(True, vision)
    эталон = собрать(False, vision)
    print("  собраны за %.1f с" % (time.perf_counter() - t0), flush=True)
    # Проверка на то, что сравниваются РАЗНЫЕ сборки. Без неё проба
    # однажды уже сравнила быструю походку саму с собой и объявила
    # побитовое совпадение — совершенно правдиво и совершенно
    # бессмысленно.
    check("сборки разные", type(быстрая).__name__ != type(эталон).__name__,
          "%s против %s" % (type(быстрая).__name__, type(эталон).__name__))

    # Череда действий одна и та же, и она не постоянная: постоянная
    # команда не заставила бы правила походки сработать ни разу.
    rng = np.random.default_rng(7)
    череда = [np.array([0.6 + 0.6 * rng.random(), 0.6 + 0.6 * rng.random()])
              for _ in range(STEPS)]

    def прогнать(sim):
        for a in череда:
            for _ in range(SUB):
                sim.step(a)
        d = sim.physics.data.ptr
        return (np.array(d.qpos, copy=True), np.array(d.qvel, copy=True),
                float(d.time),
                int(len(getattr(sim.fly, "_vision_update_mask", []))))

    q1, v1, t1, m1 = прогнать(быстрая)
    q0, v0, t0_, m0 = прогнать(эталон)
    dq = float(np.abs(q1 - q0).max())
    dv = float(np.abs(v1 - v0).max())
    rep["замеры"]["после %d шагов управления" % STEPS] = {
        "макс |разн| qpos": dq, "макс |разн| qvel": dv,
        "время": [t1, t0_],
        "обновлений сетчатки": [m1, m0],
    }
    check("траектория совпала побитово", dq == 0.0 and dv == 0.0,
          "qpos %.3e, qvel %.3e" % (dq, dv))
    check("модельное время совпало", t1 == t0_, "%.6f против %.6f" % (t1, t0_))
    check("обновлений сетчатки поровну", m1 == m0, "%d против %d" % (m1, m0))

    # --- цена, чередующимися кругами --------------------------------
    a = череда[0]

    def круг(sim):
        t = time.perf_counter()
        for _ in range(SUB):
            sim.step(a)
        return (time.perf_counter() - t) * 1000.0

    for _ in range(2):
        круг(быстрая)
        круг(эталон)
    б, э = [], []
    for _ in range(ROUNDS):
        б.append(круг(быстрая))
        э.append(круг(эталон))
    мб, мэ = float(np.median(б)), float(np.median(э))
    rep["замеры"]["шаг управления, мс"] = {
        "с сокращённым съёмом": round(мб, 2),
        "эталон": round(мэ, 2),
        "выигрыш, мс": round(мэ - мб, 2),
        "ускорение": round(мэ / max(мб, 1e-9), 3),
    }
    print("  шаг управления: %.1f мс против %.1f мс, выигрыш %.1f мс (×%.3f)"
          % (мб, мэ, мэ - мб, мэ / max(мб, 1e-9)), flush=True)
    check("сокращённый съём быстрее", мб < мэ,
          "%.1f против %.1f мс" % (мб, мэ))
    # Сравнение ПАРНОЕ, как принято в этом проекте, а не по размахам.
    # Круги чередуются, поэтому круг i у обеих сборок снят в соседние
    # мгновения и видел один и тот же фон машины; разность внутри пары
    # этот фон сокращает. Первая редакция требовала, чтобы размахи не
    # пересекались, и провалилась на машине, где один круг гуляет на
    # сорок миллисекунд при разнице медиан в семнадцать, — требование
    # было не к правке, а к тишине в комнате.
    пары = [э[i] - б[i] for i in range(len(б))]
    rep["замеры"]["разности пар, мс"] = [round(v, 1) for v in пары]
    плюс = sum(1 for v in пары if v > 0)
    check("выигрыш держится на большинстве пар", плюс >= len(пары) - 1,
          "%d пар из %d, медиана разности %+.1f мс"
          % (плюс, len(пары), float(np.median(пары))))
    rep["находки"].append(
        "на полном шаге со зрением (258,3 мс по PERF_RT §1) это %.1f %% "
        % (100.0 * (мэ - мб) / 258.3))
    print("  " + rep["находки"][-1], flush=True)

    (OUT / "rule_obs_equal.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")
    print("\nпровалов: %d" % len(fails), flush=True)
    for name in fails:
        print("  - " + name, flush=True)


if __name__ == "__main__":
    main()
    sys.exit(1 if fails else 0)
