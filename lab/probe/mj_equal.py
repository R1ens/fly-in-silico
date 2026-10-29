"""Побитовое равенство трёх правок обвязки MuJoCo.

Шлюз `probe/body_equal.py` проверяет равенство на эпизоде целиком, по
отпечаткам. Здесь проверяется то же самое на уровне одного шага — чтобы
при расхождении было видно, какая именно правка и какое именно поле, а
не только «отпечатки разошлись».

Три проверки, и каждая прямая:

1. **`mj_forwardSkip` против `mj_forward`.** На настоящем шаге эпизода, в
   той самой точке, где dm_control зовёт пересчёт по «грязному» флагу,
   состояние копируется (`copy.copy(MjData)`). На копии считается полный
   `mj_forward`, на исходном — `mj_forwardSkip(STAGE_VEL)`. Сравниваются
   ВСЕ массивы `mjData`, не только сенсоры.
2. **Дешёвое наблюдение против прежнего.** На одном и том же состоянии
   зовутся `_FastObsFly.get_observation` и исходный
   `Fly.get_observation`, поля сравниваются побайтово.
3. **Сокращённое наблюдение правил.** Четыре поля, которые читают
   правила походки, сравниваются с теми же полями полного наблюдения.

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\mj_equal.py
"""

import copy as _copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

OUT = Path(__file__).parent
N_STEPS = 400
fails = []
rep = {"проверки": {}}


def check(name, ok, info=""):
    print(("  ok     " if ok else "  ПРОВАЛ ") + name
          + (("  " + str(info)) if info else ""))
    rep["проверки"][name] = {"пройдено": bool(ok), "замер": str(info)}
    if not ok:
        fails.append(name)


# Поля `mjData`, которые НЕ сравниваем, и почему каждое:
#
# * `contact`, `efc_*` — переменной длины, живут в арене `mjData`;
#   сравниваются отдельно, по `ncon`/`nefc` и `efc_force`;
# * `solver*`, `timer*`, `warning*` — счётчики самого MuJoCo, не
#   состояние модели;
# * `buffer`, `arena`, `threadpool`, `signal`, `maxuse*`, `nbuffer`,
#   `narena`, `pbase`, `pstack`, `parena` — служебные;
# * `*island*` — поиск островов ограничений в этой модели ВЫКЛЮЧЕН
#   (`mjENBL_ISLAND` не стоит, `nisland == 0`), поэтому массивы островов
#   — неинициализированный черновик арены, который ни один расчёт не
#   читает. Проба это проверяет отдельно, а не берёт на веру.
SKIP_FIELDS = {"contact", "efc_", "solver", "timer", "warning", "buffer",
               "arena", "threadpool", "signal", "maxuse", "nbuffer",
               "narena", "pbase", "pstack", "parena"}


def data_arrays(d):
    out = {}
    for name in dir(d):
        if name.startswith("_") or "island" in name \
                or any(name.startswith(s) for s in SKIP_FIELDS):
            continue
        try:
            v = getattr(d, name)
        except Exception:
            continue
        if isinstance(v, np.ndarray) and v.dtype.kind in "fiub":
            out[name] = v
    return out




def main():
    import mujoco
    from flygym_gymnasium import Fly as PlainFly
    from flybrain import body as B

    action = np.array([1.0, 1.0])

    # ------------------------------------------------- 1. forwardSkip
    cfg = B.BodyConfig(duration_s=1.0, seed=0)
    sim, _ = B._build_body(cfg, B.DEFAULT_SOURCES, None, head_stab=None)
    sim.reset(seed=0)
    physics = sim.physics
    st = getattr(physics, "_fb_accfwd", None)
    check("дешёвый пересчёт установлен", st is not None, str(st))

    model = physics.model.ptr
    data = physics.data.ptr
    worst = {}
    counters = {"дешёвых": 0, "полных": 0}
    fields = {"n": 0, "острова": None, "ncon=nefc": 0}
    island_on = bool(int(model.opt.enableflags)
                     & int(mujoco.mjtEnableBit.mjENBL_ISLAND))
    check("поиск островов ограничений выключен (иначе их массивы нельзя "
          "исключать из сверки)", not island_on,
          "enableflags %d" % int(model.opt.enableflags))

    # Перехват ровно в точке пересчёта: если сейчас пойдёт дешёвая
    # дорога, снимаем копию состояния, считаем на ней ПОЛНЫЙ `mj_forward`,
    # потом отдаём управление настоящему (дешёвому) пересчёту и сравниваем
    # все массивы `mjData`. Так сравниваются не два эпизода, а два
    # пересчёта одного и того же состояния.
    real_forward = physics.forward

    def forward_checked():
        cheap = st["stepped"] and st["ctrl_only"]
        if not cheap:
            counters["полных"] += 1
            return real_forward()
        snap = _copy.copy(data)
        mujoco.mj_forward(model, snap)
        real_forward()
        counters["дешёвых"] += 1
        a, b = data_arrays(data), data_arrays(snap)
        both = sorted(set(a) & set(b))
        fields["n"] = len(both)
        for name in both:
            if a[name].shape != b[name].shape:
                continue
            if not np.array_equal(a[name], b[name]):
                d = float(np.max(np.abs(
                    a[name].astype(float) - b[name].astype(float))))
                worst[name] = max(worst.get(name, 0.0), d)
        # Переменной длины: число контактов, число строк ограничений и
        # силы ограничений — сравниваются отдельно, по нынешней длине.
        fields["острова"] = int(data.nisland)
        if int(data.ncon) != int(snap.ncon) or int(data.nefc) != int(snap.nefc):
            worst["ncon/nefc"] = 1.0
        else:
            fields["ncon=nefc"] = (int(data.ncon), int(data.nefc))
            nefc = int(data.nefc)
            if nefc and not np.array_equal(data.efc_force[:nefc],
                                           snap.efc_force[:nefc]):
                worst["efc_force"] = max(
                    worst.get("efc_force", 0.0),
                    float(np.max(np.abs(data.efc_force[:nefc]
                                        - snap.efc_force[:nefc]))))

    physics.forward = forward_checked
    for _ in range(50):
        sim.step(action)
    for _ in range(N_STEPS):
        sim.step(action)
    physics.forward = real_forward
    checked = counters["дешёвых"]

    rep["полей mjData сравнено"] = fields["n"]
    rep["шагов сравнено (forwardSkip)"] = checked
    rep["расхождения mjData"] = worst
    check("mj_forwardSkip(STAGE_VEL) = mj_forward, все массивы mjData",
          not worst,
          "%d шагов, %d полей%s" % (checked, rep["полей mjData сравнено"],
                                    "" if not worst
                                    else ", разошлись: " + json.dumps(
                                        worst, ensure_ascii=False)))
    rep["счётчик пересчётов"] = dict(counters)
    rep["островов в последней точке"] = fields["острова"]
    rep["ncon/nefc в последней точке"] = fields["ncon=nefc"]
    sim.close()

    # ------------------------------------------------- 2 и 3. наблюдения
    cfg = B.BodyConfig(duration_s=1.0, seed=0)
    sim, _ = B._build_body(cfg, B.DEFAULT_SOURCES, None, head_stab=None)
    sim.reset(seed=0)
    fly = sim.fly
    bad_obs, bad_rule = {}, {}
    n_cmp = 0
    for k in range(N_STEPS):
        sim.step(action)
        fast = fly.get_observation(sim)
        plain = PlainFly.get_observation(fly, sim)
        rule = fly.rule_observation(sim)
        n_cmp += 1
        for key in plain:
            fa = np.ascontiguousarray(fast[key])
            pl = np.ascontiguousarray(plain[key])
            if fa.shape != pl.shape or fa.dtype != pl.dtype \
                    or fa.tobytes() != pl.tobytes():
                bad_obs[key] = bad_obs.get(key, 0) + 1
        for key in ("fly", "end_effectors", "contact_forces",
                    "fly_orientation"):
            pl = np.ascontiguousarray(plain[key])
            ru = np.ascontiguousarray(rule[key])
            if key == "fly":
                # Правило читает только `obs["fly"][0][2]`.
                same = (ru[0][2].tobytes() == pl[0][2].tobytes())
            else:
                same = (ru.shape == pl.shape and ru.dtype == pl.dtype
                        and ru.tobytes() == pl.tobytes())
            if not same:
                bad_rule[key] = bad_rule.get(key, 0) + 1
    check("дешёвое наблюдение = прежнее, побайтово", not bad_obs,
          "%d шагов, поля: %s" % (n_cmp, ", ".join(sorted(plain))))
    if bad_obs:
        rep["разошлись поля наблюдения"] = bad_obs
        print("     разошлись:", bad_obs)
    check("наблюдение правил = те же поля полного, побайтово", not bad_rule,
          "%d шагов" % n_cmp)
    if bad_rule:
        rep["разошлись поля правил"] = bad_rule
        print("     разошлись:", bad_rule)
    rep["шагов сравнено (наблюдения)"] = n_cmp
    sim.close()

    # ------------------------------------------- 4. отметка правил сработала
    # Правка «наблюдение правил» однажды уже оказалась пустой (её обходил
    # `super()` в апстриме), и замер этого не показывал — разница
    # терялась в шуме машины. Поэтому считаем вызовы, а не верим.
    cfg = B.BodyConfig(duration_s=1.0, seed=0)
    sim, _ = B._build_body(cfg, B.DEFAULT_SOURCES, None, head_stab=None)
    sim.reset(seed=0)
    cls = type(sim.fly)
    cnt = {"полных": 0, "правил": 0}
    orig_full, orig_rule = cls.get_observation, cls.rule_observation

    def counted_full(self, s):
        if not self._fb_rule_stage:
            cnt["полных"] += 1
        return orig_full(self, s)

    def counted_rule(self, s):
        cnt["правил"] += 1
        return orig_rule(self, s)

    cls.get_observation, cls.rule_observation = counted_full, counted_rule
    n = 150
    for _ in range(n):
        sim.step(action)
    cls.get_observation, cls.rule_observation = orig_full, orig_rule
    rep["вызовов наблюдения на %d шагов физики" % n] = dict(cnt)
    check("на шаг физики ровно одно полное наблюдение и одно для правил",
          cnt["полных"] == n and cnt["правил"] == n,
          "полных %d, правил %d на %d шагов" % (cnt["полных"],
                                                cnt["правил"], n))
    sim.close()

    (OUT / "mj_equal.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nпроверок %d, провалов %d" % (len(rep["проверки"]), len(fails)))
    for f in fails:
        print("  - " + f)
    print("отчёт: probe/mj_equal.json")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
