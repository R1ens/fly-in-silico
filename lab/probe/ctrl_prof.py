"""Сколько стоит ПИТОНОВСКИЙ КОНТРОЛЛЕР ПОХОДКИ отдельной строкой.

В разложении шага управления из `V2.md` контроллера нет: он зовётся из
того же цикла, что и физика (`body.step(action)` — это
`HybridTurningController.step`), и целиком спрятан внутри строки
«физика». Здесь он выделен.

Что мерится. `HybridTurningController.step` на каждый шаг физики делает:

* `super().get_observation()` — ПОЛНОЕ наблюдение (179 сенсоров, поза,
  силы контактов, запах) ещё до того, как физика шагнула;
* правила втягивания и спотыкания, шесть ног питоновским циклом;
* `CPGNetwork.step` — шесть осцилляторов, Эйлер;
* на каждую ногу: кубический сплайн `PreprogrammedSteps.get_joint_angles`
  и линейная интерполяция `phasic_multiplier` (scipy), то есть 6 + 6
  вызовов на шаг физики и 600 + 600 на шаг управления;
* `super().step(action)` — вот это уже физика: `mj_step` плюс ВТОРОЕ
  полное наблюдение в `Fly.post_step`.

Разложение собирается обёртками на методах, а не cProfile: cProfile
раздувает питоновскую часть и для долей времени не годится. cProfile
гоняется отдельно (`--profile`) и только для порядка строк.

    cd D:\\fly-brain\\lab
    set PYTHONIOENCODING=utf-8
    ..\\.venv\\Scripts\\python.exe probe\\ctrl_prof.py --duration 1.0
"""

import argparse
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

OUT = Path(__file__).parent

# Накопитель: имя -> [суммарное время, число вызовов].
ACC = OrderedDict()


def _acc(name):
    row = ACC.get(name)
    if row is None:
        row = ACC[name] = [0.0, 0]
    return row


def timed(name, fn):
    row = _acc(name)
    pc = time.perf_counter

    def wrapper(*a, **kw):
        t = pc()
        try:
            return fn(*a, **kw)
        finally:
            row[0] += pc() - t
            row[1] += 1

    wrapper.__wrapped__ = fn
    return wrapper


def patch_classes(coarse=False):
    """Обёртки на классах. Ставятся один раз, до сборки тела.

    ``coarse`` — только три обёртки (шаг контроллера, шаг физики и то
    наблюдение, что контроллер читает сам). Мелкое разложение стоит
    около пяти тысяч вызовов обёртки на шаг управления, то есть
    единицы миллисекунд, и главное число надо брать с грубого прогона.
    """
    from flygym_gymnasium.simulation import SingleFlySimulation
    from flygym_gymnasium.fly import Fly
    from flygym_gymnasium.examples.locomotion import (
        CPGNetwork, PreprogrammedSteps)
    from flygym_gymnasium.examples.locomotion.turning_controller import (
        HybridTurningController as HTC)

    # Быстрый контроллер — ПОДКЛАСС, и его `step` обёртку на базе не
    # зовёт. Класс собирается лениво, поэтому его надо собрать тут же,
    # иначе строка «контроллер: шаг целиком» выйдет пустой, а разложение
    # — бессмысленным. Поймано замером: первый прогон дал нули.
    from flybrain import gait as G

    FAST = G.controller_class()

    if coarse:
        pairs = [
            (HTC, "step", "контроллер: шаг целиком"),
            (FAST, "step", "контроллер: шаг целиком"),
            (SingleFlySimulation, "step", "физика: шаг целиком"),
            (SingleFlySimulation, "get_observation",
             "наблюдение ДО шага (контроллер)"),
        ]
        for cls, meth, name in pairs:
            setattr(cls, meth, timed(name, cls.__dict__[meth]))
        return
    pairs = [
        (HTC, "step", "контроллер: шаг целиком"),
        (FAST, "step", "контроллер: шаг целиком"),
        (FAST, "_rules_observation", "наблюдение для правил (кэш)"),
        (FAST, "_phasic", "фазовый множитель (все ноги)"),
        (FAST, "_joint_angles", "сплайн углов (все ноги)"),
        (HTC, "_retraction_rule_find_leg", "правило втягивания: поиск ноги"),
        (HTC, "_update_persistence_counter", "счётчик удержания"),
        (HTC, "_update_correction_amount", "обновление поправки"),
        (HTC, "_stumbling_rule_check_condition", "правило спотыкания"),
        (SingleFlySimulation, "step", "физика: шаг целиком"),
        (SingleFlySimulation, "get_observation",
         "наблюдение ДО шага (контроллер)"),
        (Fly, "get_observation", "наблюдение: все вызовы"),
        (Fly, "pre_step", "запись ctrl"),
        (CPGNetwork, "step", "CPG: Эйлер"),
        (PreprogrammedSteps, "get_joint_angles", "сплайн углов (на ногу)"),
        (PreprogrammedSteps, "get_adhesion_onoff", "прилипание (на ногу)"),
    ]
    for cls, meth, name in pairs:
        if meth not in cls.__dict__:
            continue
        setattr(cls, meth, timed(name, cls.__dict__[meth]))


def patch_instance(body, coarse=False):
    """Обёртки на экземплярах: физика MuJoCo, арена, фазовый множитель."""
    if coarse:
        return
    ph = body.physics
    ph.step = timed("MuJoCo: mj_step", ph.step)
    ph.forward = timed("MuJoCo: mj_forward", ph.forward)
    body.arena.step = timed("арена: шаг", body.arena.step)
    body.arena.get_olfaction = timed("арена: запах",
                                     body.arena.get_olfaction)
    pm = body.phasic_multiplier
    for leg in list(pm):
        pm[leg] = timed("фазовый множитель (на ногу)", pm[leg])


def run(duration, readout, device=None, profile=False, coarse=False,
        gait="fast"):
    from flybrain import body as B, data as D

    neurons = D.load_neurons()
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    weights = D.load_weights(device=device)

    patch_classes(coarse=coarse)
    cfg = B.BodyConfig(duration_s=duration, control_ms=20.0, seed=0,
                       device=device, readout=readout, gait=gait)
    t0 = time.perf_counter()
    if profile:
        import cProfile
        import pstats
        import io

        pr = cProfile.Profile()
        pr.enable()
    run_ = B.run_episode(cfg=cfg, neurons=neurons, weights=weights,
                         progress=False,
                         on_body=lambda b: patch_instance(b, coarse=coarse))
    if profile:
        pr.disable()
        buf = io.StringIO()
        pstats.Stats(pr, stream=buf).sort_stats("tottime").print_stats(30)
        prof_text = buf.getvalue()
    else:
        prof_text = None
    wall = time.perf_counter() - t0
    return run_, wall, prof_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=1.0)
    ap.add_argument("--readout", default="ALPN")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--coarse", action="store_true")
    ap.add_argument("--gait", default="legacy", choices=("legacy", "fast"))
    ap.add_argument("--tag", default="базовая")
    a = ap.parse_args()

    run_, wall, prof_text = run(a.duration, a.readout, profile=a.profile,
                                coarse=a.coarse, gait=a.gait)
    steps = len(run_.records)
    t_sim = run_.timing["sim_s"]

    def ms(name):
        row = ACC.get(name)
        return (0.0, 0) if row is None else (row[0] * 1000.0 / steps, row[1])

    def val(name):
        return ms(name)[0]

    ctrl_all = val("контроллер: шаг целиком")
    phys_all = val("физика: шаг целиком")
    obs_pre = val("наблюдение ДО шага (контроллер)")
    obs_both = val("наблюдение: все вызовы")
    obs_post = obs_both - obs_pre
    mj = val("MuJoCo: mj_step") + val("MuJoCo: mj_forward")
    # Логика контроллера: его шаг минус физика минус то наблюдение, что он
    # читает сам. Наблюдение названо отдельной строкой нарочно: это не
    # «мозг» и не MuJoCo, а сборка наблюдения FlyGym на питоне.
    ctrl_logic = ctrl_all - phys_all - obs_pre
    flygym_rest = phys_all - mj - obs_post - val("запись ctrl") \
        - val("арена: шаг")

    rep = {
        "метка": a.tag,
        "контроллер": a.gait,
        "длительность, с": a.duration,
        "шагов управления": steps,
        "по стене, с": round(wall, 2),
        "цикл эпизода, с": t_sim,
        "шаг управления, мс": round(t_sim * 1000.0 / steps, 2),
        "строки, мс на шаг управления": {},
        "итог, мс на шаг управления": {
            "тело целиком (контроллер + физика)": round(ctrl_all, 2),
            "— логика контроллера походки": round(ctrl_logic, 2),
            "— наблюдение ДО шага (лишнее?)": round(obs_pre, 2),
            "— наблюдение ПОСЛЕ шага (Fly.post_step)": round(obs_post, 2),
            "— MuJoCo (mj_step + mj_forward)": round(mj, 2),
            "— прочее FlyGym внутри шага": round(flygym_rest, 2),
            "мозг и запись (остаток шага)": round(
                t_sim * 1000.0 / steps - ctrl_all, 2),
        },
    }
    for name in ACC:
        v, c = ms(name)
        rep["строки, мс на шаг управления"][name] = {
            "мс": round(v, 3), "вызовов на шаг управления": c // max(steps, 1)}

    print()
    print("=== шаг управления, мс по стене: %.2f (%d шагов) ==="
          % (t_sim * 1000.0 / steps, steps))
    for k, v in rep["итог, мс на шаг управления"].items():
        print("  %-42s %8.2f" % (k, v))
    print()
    print("--- внутри, по строкам ---")
    for name, d in rep["строки, мс на шаг управления"].items():
        print("  %-38s %8.3f мс  × %5d"
              % (name, d["мс"], d["вызовов на шаг управления"]))

    rep["метрики"] = run_.metrics
    rep["калибровка"] = run_.calib
    if prof_text:
        (OUT / ("ctrl_prof_%s.txt" % a.tag)).write_text(prof_text,
                                                        encoding="utf-8")
        print("\ncProfile: probe/ctrl_prof_%s.txt" % a.tag)
    p = OUT / ("ctrl_prof_%s.json" % a.tag)
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    print("отчёт: %s" % p)


if __name__ == "__main__":
    main()
