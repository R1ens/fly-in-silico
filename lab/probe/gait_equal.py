"""Побитовая сверка быстрого контроллера походки по частям.

`probe/body_equal.py` сверяет эпизод целиком отпечатками — это итог, и
он нужен. Но когда отпечатки разойдутся, он не скажет, где. Здесь
сверяются САМИ ВЫРАЖЕНИЯ, на случайных входах и на границах, по частям:

1. **Кусочный полином.** Шесть `CubicSpline` апстрима против одного
   сложенного `PPoly` из `flybrain.gait`. Значения фаз берутся и внутри
   периода, и далеко за ним (фазы CPG растут без границ), и ровно в
   узлах.
2. **Фазовый множитель.** Шесть `interp1d` апстрима против выписанной
   линейной интерполяции — включая точные узлы, где `searchsorted`
   меняет сторону.
3. **Проекция сил контактов.** Один `np.dot` на 18 строк против шести по
   три, на float32, в трёх масштабах величин.
4. **Прилипание.** Цепочка сравнений апстрима против маски.
5. **Поправки.** `max(0, …)` апстрима против `np.maximum`, с нулём и
   отрицательным нулём среди входов.
6. **Углы ног целиком.** `get_joint_angles` апстрима на ногу плюс
   поправка против того же массивами, на случайных фазах и амплитудах.

Встречный замер. Проба обязана ловить расхождение, а не печатать «ok» на
любом коде: в конце тот же сравниватель гоняется по НАРОЧНО испорченной
арифметике (последний бит мантиссы у одного коэффициента), и он должен
провалиться.

    cd D:\\fly-brain\\lab
    set PYTHONIOENCODING=utf-8
    ..\\.venv\\Scripts\\python.exe probe\\gait_equal.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

OUT = Path(__file__).parent
fails = []
rep = {"проверки": {}}


def check(name, ok, info=""):
    print(("  ok     " if ok else "  ПРОВАЛ ") + name
          + (("  " + str(info)) if info else ""))
    rep["проверки"][name] = {"пройдено": bool(ok), "замер": str(info)}
    if not ok:
        fails.append(name)


def same(a, b):
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    return a.shape == b.shape and a.dtype == b.dtype \
        and a.tobytes() == b.tobytes()


def build_controller(gait):
    """Контроллер на пустой арене: нужна только его арифметика."""
    from flygym_gymnasium import Fly
    from flygym_gymnasium.preprogrammed import default_leg_sensor_placements

    from flybrain import gait as G

    fly = Fly(enable_adhesion=True, enable_olfaction=False,
              contact_sensor_placements=default_leg_sensor_placements,
              init_pose="stretch", control="position",
              spawn_pos=(0.0, 0.0, 0.2))
    return G.build(gait=gait, fly=fly, cameras=[], timestep=2e-4)


def main():
    rng = np.random.default_rng(7)
    fast = build_controller("fast")
    ps = fast.preprogrammed_steps
    legs = ps.legs
    n = len(legs)

    # --- 1. кусочный полином ---
    bad = worst = 0
    for trial in range(4000):
        if trial < 1000:
            ph = rng.random(n) * 2 * np.pi
        elif trial < 2000:
            ph = (rng.random(n) - 0.5) * 400.0        # далеко за период
        else:
            ph = rng.choice(fast._psi.x, n) + (trial % 3 - 1) * 0.0
        got = fast._psi(ph).reshape(n, n, fast._ndof)[fast._rows, fast._rows]
        for i, leg in enumerate(legs):
            want = ps._psi_funcs[leg](np.array([ph[i]])).squeeze()
            if not same(got[i], want):
                bad += 1
                worst = max(worst, float(np.max(np.abs(got[i] - want))))
    check("кусочный полином: один сложенный против шести",
          bad == 0, "расхождений %d, максимум %g" % (bad, worst))

    # --- 2. фазовый множитель ---
    bad = 0
    tries = 0
    for trial in range(4000):
        ph = rng.random(n) * 2 * np.pi
        got = fast._phasic(ph)
        for i, leg in enumerate(legs):
            want = np.float64(fast.phasic_multiplier[leg](ph[i]))
            tries += 1
            if not same(got[i], want):
                bad += 1
    # точные узлы и края
    for i, leg in enumerate(legs):
        for v in list(fast._px[i]) + [0.0, 2 * np.pi]:
            ph = np.full(n, float(v))
            got = fast._phasic(ph)
            want = np.float64(fast.phasic_multiplier[leg](float(v)))
            tries += 1
            if not same(got[i], want):
                bad += 1
    check("фазовый множитель: линейная интерполяция",
          bad == 0, "расхождений %d из %d" % (bad, tries))

    # --- 3. проекция сил контактов ---
    bad = 0
    idx = fast._stumb
    for _ in range(4000):
        cf = (rng.standard_normal((len(ps.legs) * 10, 3))
              * rng.choice([1e-4, 1.0, 1e4])).astype(np.float32)
        ori = rng.standard_normal(3).astype(np.float32)
        proj = np.dot(cf[idx], ori)
        cond_fast = (proj.reshape(-1, fast._n_stumb)
                     < fast.stumbling_force_threshold).any(axis=1)
        for i, leg in enumerate(legs):
            want = (np.dot(cf[fast.stumbling_sensors[leg], :], ori)
                    < fast.stumbling_force_threshold).any()
            if bool(cond_fast[i]) != bool(want):
                bad += 1
    check("правило спотыкания: проекция одним dot", bad == 0,
          "расхождений %d" % bad)

    # --- 4. прилипание ---
    bad = 0
    for _ in range(4000):
        ph = (rng.random(n) - 0.3) * 40.0
        mod = ph % (2 * np.pi)
        got = ~((fast._swing[:, 0] < mod) & (mod < fast._swing[:, 1]))
        for i, leg in enumerate(legs):
            want = ps.get_adhesion_onoff(leg, ph[i])
            if bool(got[i]) != bool(want):
                bad += 1
    check("прилипание: маска против цепочки сравнений", bad == 0,
          "расхождений %d" % bad)

    # --- 5. поправки ---
    bad = 0
    rates = fast.correction_rates["retraction"]
    dec = rates[1] * fast.timestep
    inc = rates[0] * fast.timestep
    pool = np.concatenate([rng.random(200) * 2.0, [0.0, -0.0, dec, inc],
                           np.full(4, dec)])
    for v in pool:
        cur = np.full(n, float(v))
        for cond in (True, False):
            got = np.where(np.full(n, cond), cur + inc,
                           np.maximum(0.0, cur - dec))
            want = np.array([(float(v) + inc) if cond
                             else float(max(0, float(v) - dec))
                             for _ in range(n)])
            if not same(got, want):
                bad += 1
    check("поправки: np.maximum против max(0, …)", bad == 0,
          "расхождений %d" % bad)

    # --- 6. углы ног целиком ---
    bad = worst = 0
    for _ in range(2000):
        ph = (rng.random(n) - 0.2) * 30.0
        mag = rng.random(n) * 1.5
        net = (rng.random(n) - 0.2) * 0.5
        got = fast._joint_angles(ph, mag) + net[:, None] * fast._corr_vec
        want = []
        for i, leg in enumerate(legs):
            a = ps.get_joint_angles(leg, ph[i], mag[i])
            a = a + net[i] * fast.correction_vectors[leg[1]]
            want.append(a)
        want = np.array(want)
        if not same(got, want):
            bad += 1
            worst = max(worst, float(np.max(np.abs(got - want))))
    check("углы ног: массивами против цикла по ногам", bad == 0,
          "расхождений %d, максимум %g" % (bad, worst))

    # --- встречный замер: сравниватель обязан ловить ---
    # Портится СВОБОДНЫЙ член на всех участках первой ноги: правка
    # старшего коэффициента пропадала бы на малых `s`, и «не поймано»
    # означало бы слабый вход, а не слабый сравниватель.
    c = fast._psi.c.copy()
    fast._psi.c[-1, :, 0] = np.nextafter(fast._psi.c[-1, :, 0], np.inf)
    caught = 0
    for _ in range(400):
        ph = rng.random(n) * 2 * np.pi
        got = fast._psi(ph).reshape(n, n, fast._ndof)[fast._rows, fast._rows]
        for i, leg in enumerate(legs):
            want = ps._psi_funcs[leg](np.array([ph[i]])).squeeze()
            if not same(got[i], want):
                caught += 1
    fast._psi.c[...] = c
    check("встречный замер: испорченный последний бит замечен",
          caught > 0, "поймано на %d случаях" % caught)

    rep["провалов"] = len(fails)
    p = OUT / "gait_equal.json"
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    print("\nотчёт: %s" % p)
    if fails:
        print("ПРОВАЛЫ: " + ", ".join(fails))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
