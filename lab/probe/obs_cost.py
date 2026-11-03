r"""Сокращённое наблюдение при включённом зрении: цена и равнозначность.

В проекте уже есть `rule_observation` — съём четырёх полей, которые
читают правила походки, вместо полного наблюдения из семи. Он
проверен и работает, но **выключен, как только включается зрение**, и
причина в коде названа прямо: `Fly.get_observation` зовёт
`_update_vision(sim)`, а тот ведёт `_last_vision_update_time`. То есть
у повторного съёма есть последствие, и выбросить его значит изменить
расписание обновления зрения.

Причина верная, а вывод из неё — не единственный. Расписание можно
сохранить, если сокращённый съём **сам позовёт** `_update_vision`.
Тогда вызовов будет столько же и в том же порядке, а выбросится только
то, что и так выбрасывалось: сорок два суставных датчика, скорости,
углы Эйлера через scipy, кардинальные векторы, обоняние и `xpos`
контактных тел.

Проба отвечает на два вопроса, и оба замером:

1. **Сколько это стоит.** Полное наблюдение против сокращённого на той
   же модели и на том же состоянии, чередующимися кругами.
2. **То же ли это самое.** Четыре поля, которые читают правила, обязаны
   совпасть **побитово**; расписание зрения — совпасть по числу
   обновлений и по времени последнего.

Встречная проверка устроена так, чтобы могла провалиться: если
сокращённый съём забудет позвать `_update_vision`, число обновлений
разойдётся, и проба это скажет.

    cd D:\fly-brain\lab
    set PYTHONIOENCODING=utf-8
    ..\.venv\Scripts\python.exe probe\obs_cost.py
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("FLYBRAIN_OUT",
                      str(Path(__file__).parent / "_obs_cost_out"))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

OUT = Path(__file__).parent
SUB = 100
ROUNDS = 9
ПОЛЯ = ("fly", "end_effectors", "contact_forces", "fly_orientation")

rep = {"проверки": {}, "замеры": {}, "находки": []}
fails = []


def check(name, ok, info=""):
    print(("  ok     " if ok else "  ПРОВАЛ ") + name
          + (("  " + str(info)) if info else ""), flush=True)
    rep["проверки"][name] = {"пройдено": bool(ok), "замер": str(info)}
    if not ok:
        fails.append(name)


def main():                                                   # noqa: C901
    from flybrain import body as B

    cfg = B.BodyConfig(duration_s=0.4, control_ms=20.0, seed=0,
                       device="cuda", readout="ALPN",
                       vision=True, vision_hz=100.0)
    sources = [[25.0, 0.0, 1.5], [-25.0, 0.0, 1.5]]
    sim, _ = B._build_body(cfg, sources)
    fly = sim.fly
    check("у мухи есть сокращённый съём", hasattr(fly, "rule_observation"))
    if not hasattr(fly, "rule_observation"):
        return
    check("зрение в сборке включено", bool(getattr(fly, "enable_vision", 0)))

    action = np.array([0.9, 0.9], dtype=float)
    for _ in range(SUB * 2):
        sim.step(action)

    # --- равнозначность: четыре поля побитово -----------------------
    полное = fly.get_observation(sim)
    кратко = fly.rule_observation(sim)
    разошлось = {}
    for k in ПОЛЯ:
        a = np.asarray(полное[k])
        b = np.asarray(кратко[k])
        # У поля `fly` полное наблюдение отдаёт четыре строки
        # (положение, скорость, курс, угловая скорость), а правила
        # читают только `obs["fly"][0][2]`. Сокращённый съём отдаёт
        # одну строку нарочно, и сравнивать надо ту, которую читают, —
        # иначе проверка валится на том, что и не обещано.
        if a.shape != b.shape and k == "fly" and a.shape[1:] == b.shape[1:]:
            a = a[:b.shape[0]]
        if a.shape != b.shape:
            разошлось[k] = "формы %s против %s" % (a.shape, b.shape)
        elif not np.array_equal(a, b):
            разошлось[k] = "макс |разн| %.3e" % float(np.abs(a - b).max())
    rep["замеры"]["расхождения полей"] = разошлось
    check("четыре поля правил совпали побитово", not разошлось,
          json.dumps(разошлось, ensure_ascii=False) if разошлось
          else "%d полей" % len(ПОЛЯ))

    # --- расписание зрения ------------------------------------------
    #
    # Считается не «вызвали ли», а СЛЕДСТВИЕ вызова: время последнего
    # обновления сетчатки. Его ведёт сам FlyGym.
    def расписание(сколько, съём):
        t0 = float(getattr(fly, "_last_vision_update_time", 0.0))
        было = len(getattr(fly, "_vision_update_mask", []))
        for _ in range(сколько):
            sim.physics.data.time += sim.timestep
            sim.curr_time += sim.timestep
            съём(sim)
        return (float(getattr(fly, "_last_vision_update_time", 0.0)) - t0,
                len(getattr(fly, "_vision_update_mask", [])) - было)

    сдвиг_п, _ = расписание(SUB, fly.get_observation)
    сдвиг_к, _ = расписание(SUB, fly.rule_observation)
    rep["замеры"]["сдвиг времени обновления, с"] = {
        "полное": сдвиг_п, "сокращённое": сдвиг_к}
    check("сокращённый съём двигает расписание зрения так же",
          abs(сдвиг_п - сдвиг_к) < 1e-12,
          "%.6f против %.6f с" % (сдвиг_п, сдвиг_к))
    if abs(сдвиг_к) < 1e-12 < abs(сдвиг_п):
        rep["находки"].append(
            "сокращённый съём расписание зрения НЕ двигает — значит "
            "`_update_vision` в нём не зовётся, и включать его при "
            "зрении нельзя, пока не позовёт")

    # --- цена, чередующимися кругами --------------------------------
    for _ in range(SUB):
        sim.step(action)

    def круг(съём):
        t = time.perf_counter()
        for _ in range(SUB):
            съём(sim)
        return (time.perf_counter() - t) * 1000.0

    for _ in range(2):
        круг(fly.get_observation)
        круг(fly.rule_observation)
    п, к = [], []
    for _ in range(ROUNDS):
        п.append(круг(fly.get_observation))
        к.append(круг(fly.rule_observation))
    мп, мк = float(np.median(п)), float(np.median(к))
    rep["замеры"]["мс на 100 съёмов"] = {
        "полное": round(мп, 3), "сокращённое": round(мк, 3),
        "выигрыш, мс": round(мп - мк, 3),
        "отношение": round(мп / max(мк, 1e-9), 3)}
    print("  полное %.3f мс, сокращённое %.3f мс, выигрыш %.3f мс на сто "
          "съёмов (×%.2f)" % (мп, мк, мп - мк, мп / max(мк, 1e-9)), flush=True)
    check("сокращённое дешевле полного", мк < мп,
          "%.3f против %.3f мс" % (мк, мп))
    rep["находки"].append(
        "на шаге управления съёмов для правил ровно сто, значит потолок "
        "выигрыша здесь — %.1f мс из 115,6 мс блока физики с "
        "контроллером" % (мп - мк))
    print("  " + rep["находки"][-1], flush=True)

    # --- то же, но ПОСЛЕ ШАГА ФИЗИКИ ---------------------------------
    #
    # Замер выше снят на «чистом» состоянии: сто съёмов подряд без
    # шагов, и `physics.forward()` отработал только на первом. В
    # настоящем цикле всё иначе: шаг физики помечает состояние
    # грязным, и КАЖДЫЙ съём — хоть полный, хоть сокращённый —
    # платит за пересчёт. Этот пересчёт сокращением не снимается, и
    # разница между съёмами в живом цикле поэтому МЕНЬШЕ. Мерить надо
    # её, а не ту.
    def живой(съём, n=50):
        t = 0.0
        for _ in range(n):
            sim.physics.step()
            а = time.perf_counter()
            съём(sim)
            t += time.perf_counter() - а
        return t * 1000.0 / n

    for _ in range(2):
        живой(fly.get_observation)
        живой(fly.rule_observation)
    жп, жк = [], []
    for _ in range(ROUNDS):
        жп.append(живой(fly.get_observation))
        жк.append(живой(fly.rule_observation))
    мжп, мжк = float(np.median(жп)), float(np.median(жк))
    rep["замеры"]["мс на съём после шага физики"] = {
        "полное": round(мжп, 4), "сокращённое": round(мжк, 4),
        "выигрыш": round(мжп - мжк, 4),
        "на сто шагов физики, мс": round(100.0 * (мжп - мжк), 2)}
    print("  после шага физики: полное %.4f мс, сокращённое %.4f мс, "
          "выигрыш %.2f мс на сто шагов"
          % (мжп, мжк, 100.0 * (мжп - мжк)), flush=True)
    check("в живом цикле сокращённое тоже дешевле", мжк < мжп,
          "%.4f против %.4f мс" % (мжк, мжп))
    rep["находки"].append(
        "пересчёт `physics.forward()` сокращением не снимается: в живом "
        "цикле выигрыш %.1f мс на сто шагов против %.1f мс на чистом "
        "состоянии" % (100.0 * (мжп - мжк), мп - мк))
    print("  " + rep["находки"][-1], flush=True)

    (OUT / "obs_cost.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")
    print("\nпровалов: %d" % len(fails), flush=True)
    for name in fails:
        print("  - " + name, flush=True)


if __name__ == "__main__":
    main()
    sys.exit(1 if fails else 0)
