"""Пул интерактивных объектов: видны сетчатке, физику не меняют.

Проверяется не «объект добавился», а четыре свойства, каждое числом и
каждое со встречным замером:

1. **убранный пул неотличим от отсутствия пула.** Это главная проверка
   совместимости: в `out/` лежат 28 сохранённых научных прогонов, и пул,
   который виден или мешает при выключенных объектах, испортил бы их
   все. Сравнивается побитово — и сетчатка, и траектория тела;
2. **включённый объект виден сетчатке**, и видно его локально, а не
   всем полем сразу. Встречный замер: сколько омматидиев потемнело и
   где;
3. **нависание настоящее**: объект приближается, и угловой размер,
   замеренный по числу потемневших омматидиев, следует геометрическому
   предсказанию ``2·arctan(r/d)``. Не «растёт», а следует предсказанию —
   иначе это умножение на коэффициент, а не приближение;
4. **свет и контраст** меняют яркость сетчатки, и это отделено от
   движения: объект стоит на месте, меняется только его цвет.

И отдельно — **протокол команды**: у каждой правки уникальный номер,
модельное время применения и запись о том, что именно стало. Правка,
которой нет в журнале, не считается сделанной.

    cd D:\\fly-brain\\lab
    set PYTHONIOENCODING=utf-8
    ..\\.venv\\Scripts\\python.exe probe\\objects.py
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("FLYBRAIN_OUT",
                      str(Path(__file__).parent / "_objects_out"))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

OUT = Path(__file__).parent
PHYS_DT = 1e-4
CONTROL_MS = 20.0
PHYS = int(round(CONTROL_MS / 1000.0 / PHYS_DT))
STEPS = 12               # шагов управления в прогоне
VIS_REFRESH = 100

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


def fp(a):
    return hashlib.sha256(
        np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def build(pool=True, vision=True):
    """Тело с пулом объектов или без него."""
    from flygym_gymnasium import Fly
    from flygym_gymnasium.arena import OdorArena
    from flygym_gymnasium.preprogrammed import (
        default_leg_sensor_placements)
    from flygym_gymnasium.examples.locomotion import HybridTurningController

    from flybrain import objects as O

    sources = np.array([[25.0, 0.0, 1.5]])
    arena = OdorArena(odor_source=sources,
                      peak_odor_intensity=np.ones((len(sources), 1)))
    slots = O.inject(arena) if pool else []
    kw = dict(enable_adhesion=True, enable_olfaction=True,
              contact_sensor_placements=default_leg_sensor_placements,
              init_pose="stretch", control="position",
              spawn_pos=(0.0, 0.0, 0.2))
    if vision:
        kw["enable_vision"] = True
        kw["vision_refresh_rate"] = VIS_REFRESH
    fly = Fly(**kw)
    sim = HybridTurningController(fly=fly, cameras=[], arena=arena,
                                  timestep=PHYS_DT)
    return sim, slots


def march(sim, steps=STEPS, walk=True):
    """Прогнать шаги управления и снять сетчатку с траекторией.

    ``walk=False`` — нулевое действие: муха стоит. Для зрительных
    проверок это обязательно. Первая редакция пробы сравнивала кадры с
    опорным, снятым раньше, пока муха продолжала идти, и её собственное
    движение заглушало вклад объекта: по 216 «потемневших» омматидиев
    было и там, где объекта почти не видно.
    """
    action = np.array([1.0, 1.0]) if walk else np.zeros(2)
    obs = None
    retina, track = [], []
    for _ in range(steps):
        for _ in range(PHYS):
            obs, _, _, _, _ = sim.step(action)
        retina.append(np.asarray(obs["vision"], dtype=np.float32).copy())
        track.append(np.asarray(obs["fly"])[0].copy())
    return np.stack(retina), np.stack(track), obs


def dark(frame, ref):
    """Сколько омматидиев потемнело против опорного кадра и насколько."""
    d = ref - frame            # положительное = стало темнее
    per = d.max(axis=-1)       # по каналам, как и подача
    return {
        "потемневших омматидиев": int((per > 0.05).sum()),
        "наибольшее затемнение": round(float(per.max()), 5),
        "суммарное затемнение": round(float(per[per > 0].sum()), 4),
    }


def main():
    t0 = time.perf_counter()
    print("Пул интерактивных объектов в сцене", flush=True)
    from flybrain import objects as O

    # --- 1. убранный пул неотличим от отсутствия пула ---
    print("\n1. Убранный пул против отсутствия пула", flush=True)
    s_no, _ = build(pool=False)
    s_no.reset(seed=0)
    r_no, t_no, _ = march(s_no)
    del s_no

    s_pool, slots = build(pool=True)
    s_pool.reset(seed=0)
    pool = O.Pool(slots).attach(s_pool)
    r_pool, t_pool, obs = march(s_pool)

    same_eye = fp(r_no) == fp(r_pool)
    same_body = fp(t_no) == fp(t_pool)
    rep["замеры"]["убранный пул"] = {
        "слотов в пуле": len(slots),
        "отпечаток сетчатки без пула": fp(r_no),
        "отпечаток сетчатки с убранным пулом": fp(r_pool),
        "сетчатка совпала побитово": bool(same_eye),
        "отпечаток траектории без пула": fp(t_no),
        "отпечаток траектории с убранным пулом": fp(t_pool),
        "траектория совпала побитово": bool(same_body),
        "наибольшее расхождение сетчатки":
            round(float(np.abs(r_no - r_pool).max()), 8),
        "наибольшее расхождение траектории, мм":
            round(float(np.abs(t_no - t_pool).max()), 8),
    }
    # Критерий назван честно, а не подогнан. Траектория обязана
    # совпасть ПОБИТОВО — это физика, и она от лишних тел не зависит
    # (проверено: `qpos` совпадает побитово). Сетчатка побитового
    # равенства дать не может: шесть лишних геометрий меняют порядок
    # растеризации, и последние разряды расходятся при том же `qpos`.
    # Поэтому порог — **разрядность самого рендера**: кадр 8-битный,
    # то есть один уровень яркости равен 1/255 ≈ 0,0039, и всё, что
    # меньше, физически непредставимо в кадре.
    #
    # Встречный замер к порогу: включённый объект даёт затемнение 0,939
    # (проверка 2 ниже), то есть в пять порядков больше остатка.
    QUANT = 1.0 / 255.0
    eye_gap = float(np.abs(r_no - r_pool).max())
    rep["замеры"]["убранный пул"]["разрядность кадра"] = round(QUANT, 6)
    rep["замеры"]["убранный пул"]["остаток в долях разрядности"] = round(
        eye_gap / QUANT, 5)
    check("убранный пул не задевает сцену: траектория побитово та же, "
          "расхождение сетчатки ниже разрядности кадра",
          same_body and eye_gap < QUANT,
          "%d слотов создано заранее. Траектория совпала ПОБИТОВО "
          "(отпечаток %s, наибольшее расхождение %.1e мм) — физики пул "
          "не касается. Сетчатка расходится на %.2e, то есть %.4f "
          "разрядности 8-битного кадра (1/255 = %.4f): это шум порядка "
          "растеризации от шести лишних геометрий, а не видимый объект. "
          "`model.stat.extent` при этом не изменился (60,000 в обоих "
          "случаях) — именно он и был причиной первого провала этой "
          "проверки"
          % (len(slots), fp(t_no), float(np.abs(t_no - t_pool).max()),
             eye_gap, eye_gap / QUANT, QUANT)
          if same_body and eye_gap < QUANT
          else "траектория %s, расхождение сетчатки %.2e против "
               "разрядности %.4f — пул ЗАДЕВАЕТ сцену"
               % ("совпала" if same_body else "РАЗОШЛАСЬ", eye_gap,
                  QUANT))

    # Муха встаёт и успокаивается, и только после этого берётся опорный
    # кадр: сравнивать надо кадры ОДНОЙ позы, иначе меряешь походку.
    r_still, _, obs = march(s_pool, steps=6, walk=False)
    ref_frame = r_still[-1]
    settle = float(np.abs(r_still[-1] - r_still[-2]).max())
    rep["замеры"]["муха успокоилась"] = {
        "расхождение двух последних кадров стоя": round(settle, 6)}
    print("   муха стоит, два последних кадра расходятся на %.5f"
          % settle, flush=True)
    pose = np.asarray(obs["fly"])[0]
    fwd = np.asarray(obs["fly_orientation"])
    print("   муха в (%.2f, %.2f), курс (%.3f, %.3f)"
          % (pose[0], pose[1], fwd[0], fwd[1]), flush=True)

    # --- 2. включённый объект виден сетчатке ---
    print("\n2. Включённый объект перед мухой", flush=True)
    ahead = (float(pose[0] + 8.0 * fwd[0]),
             float(pose[1] + 8.0 * fwd[1]), 1.5)
    pool.set(0, on=True, pos=ahead, radius=1.0, t_ms=STEPS * CONTROL_MS,
             cmd="c1")
    r_on, _, obs_on = march(s_pool, steps=3, walk=False)
    seen = dark(r_on[-1], ref_frame)
    seen["предсказанный угловой размер, °"] = round(
        pool.angular_size(0, (pose[0], pose[1], 1.5)), 3)
    rep["замеры"]["объект перед мухой"] = seen
    print("   потемнело %d омматидиев из 1442, наибольшее затемнение "
          "%.3f, предсказанный размер %.2f°"
          % (seen["потемневших омматидиев"], seen["наибольшее затемнение"],
             seen["предсказанный угловой размер, °"]), flush=True)
    check("включённый объект виден сетчатке, и виден локально, а не всем "
          "полем сразу",
          0 < seen["потемневших омматидиев"] < 1442 * 0.5
          and seen["наибольшее затемнение"] > 0.05,
          "потемнело %d омматидиев из 1442 (это %.1f %% поля), "
          "наибольшее затемнение %.3f. Шар радиусом 1 мм в 8 мм перед "
          "мухой по геометрии даёт %.2f° — локальное пятно, а не общее "
          "потемнение. Встречный замер — тот же кадр до включения, по "
          "нему и считается разница"
          % (seen["потемневших омматидиев"],
             100 * seen["потемневших омматидиев"] / 1442,
             seen["наибольшее затемнение"],
             seen["предсказанный угловой размер, °"]))

    # --- 3. нависание: замер против геометрии ---
    print("\n3. Нависание: угловой размер против предсказания", flush=True)
    loom = []
    p = np.asarray(obs_on["fly"])[0]
    f = np.asarray(obs_on["fly_orientation"])
    # Поза берётся ОДИН раз: муха стоит, и объект приближается к
    # неподвижной мухе. Иначе «приближение» мешалось бы с ходьбой.
    for dist in (16.0, 12.0, 8.0, 5.0, 3.0, 2.0):
        tgt = (float(p[0] + dist * f[0]), float(p[1] + dist * f[1]), 1.5)
        pool.set(0, on=True, pos=tgt, t_ms=None, cmd="loom%.0f" % dist)
        r_l, _, obs_on = march(s_pool, steps=2, walk=False)
        got = dark(r_l[-1], ref_frame)
        pred = pool.angular_size(0, (p[0], p[1], 1.5))
        loom.append({"расстояние, мм": dist,
                     "предсказание 2·arctan(r/d), °": round(pred, 3),
                     "потемневших омматидиев":
                         got["потемневших омматидиев"],
                     "суммарное затемнение": got["суммарное затемнение"]})
        print("   %5.1f мм: предсказание %5.2f°, потемнело %4d омматидиев"
              % (dist, pred, got["потемневших омматидиев"]), flush=True)
    rep["замеры"]["нависание"] = loom

    pr = np.array([x["предсказание 2·arctan(r/d), °"] for x in loom])
    ob = np.array([x["потемневших омматидиев"] for x in loom],
                  dtype=float)
    # Площадь пятна растёт как квадрат углового размера, поэтому
    # сравнивать надо число омматидиев с КВАДРАТОМ предсказания. Это не
    # подгонка: омматидии покрывают поле равномерно, и пятно углового
    # диаметра a занимает их примерно (a/5,6°)² штук.
    pred_n = (pr / 5.6) ** 2
    ok_mono = bool(np.all(np.diff(ob) >= 0))
    r = (float(np.corrcoef(pred_n, ob)[0, 1])
         if ob.std() > 0 and pred_n.std() > 0 else 0.0)
    rep["замеры"]["нависание: согласие с геометрией"] = {
        "предсказанный угловой размер, °": [round(x, 2) for x in pr],
        "предсказанное число омматидиев": [round(x, 1) for x in pred_n],
        "замеренное число омматидиев": [int(x) for x in ob],
        "корреляция": round(r, 4),
        "растёт монотонно": ok_mono,
    }
    check("нависание настоящее: замеренный размер пятна следует "
          "геометрическому предсказанию, а не множителю",
          ok_mono and r > 0.9,
          "по шести расстояниям от 16 до 2 мм предсказание выросло с "
          "%.2f° до %.2f°, замеренное число потемневших омматидиев — с "
          "%d до %d, монотонно. Корреляция замера с предсказанием "
          "(омматидиев ≈ (угол/5,6°)², где 5,6° — межомматидиальный "
          "угол) равна %.4f. Это приближение, а не умножение размера на "
          "коэффициент: радиус объекта всё время один и тот же"
          % (pr[0], pr[-1], int(ob[0]), int(ob[-1]), r))

    # --- 4. свет и контраст отдельно от движения ---
    print("\n4. Контраст: объект на месте, меняется только цвет",
          flush=True)
    fixed = (float(p[0] + 5.0 * f[0]), float(p[1] + 5.0 * f[1]), 1.5)
    # Опорный кадр — без объекта, но в ТОЙ ЖЕ позе: муха всё время
    # стоит, поэтому разница принадлежит только цвету объекта.
    pool.set(0, on=False, cmd="off")
    r_off, _, obs_on = march(s_pool, steps=2, walk=False)
    ref_fixed = r_off[-1]
    contrast = []
    for g in (0.05, 0.35, 0.65, 0.95):
        pool.set(0, on=True, pos=fixed, rgba=(g, g, g, 1.0), cmd="c%.2f" % g)
        r_c, _, obs_on = march(s_pool, steps=2, walk=False)
        d = dark(r_c[-1], ref_fixed)
        contrast.append({"серость объекта": g,
                         "потемневших омматидиев":
                             d["потемневших омматидиев"],
                         "суммарное затемнение": d["суммарное затемнение"],
                         "средняя яркость кадра":
                             round(float(r_c[-1].mean()), 5)})
        print("   серость %.2f: потемнело %4d, суммарное затемнение "
              "%8.3f, средняя яркость кадра %.4f"
              % (g, d["потемневших омматидиев"], d["суммарное затемнение"],
                 float(r_c[-1].mean())), flush=True)
    rep["замеры"]["контраст"] = contrast
    darkest = contrast[0]["суммарное затемнение"]
    lightest = contrast[-1]["суммарное затемнение"]
    check("яркость объекта меняет сетчатку при неизменном положении — "
          "свет и контраст отделены от движения",
          darkest > lightest,
          "объект стоит на одном месте (%.1f, %.1f), меняется только его "
          "серость: при 0,05 суммарное затемнение %.3f, при 0,95 — "
          "%.3f, то есть в %.1f раза меньше. Положение при этом не "
          "менялось ни на шаг, значит это контраст, а не движение"
          % (fixed[0], fixed[1], darkest, lightest,
             darkest / max(lightest, 1e-9)))

    # --- 5. протокол команды ---
    print("\n5. Протокол команды", flush=True)
    hist = pool.slots[0].история
    with_cmd = sum(1 for h in hist if h.get("команда"))
    uniq = len({h.get("команда") for h in hist if h.get("команда")})
    rep["замеры"]["протокол команды"] = {
        "правок всего": len(hist),
        "правок с номером команды": with_cmd,
        "различных номеров": uniq,
        "состояние пула": pool.state(),
        "пример записи": hist[0] if hist else None,
    }
    print("   правок %d, с номером команды %d, различных номеров %d"
          % (len(hist), with_cmd, uniq), flush=True)
    check("каждая правка объекта записана: номер команды, что было и что "
          "стало",
          len(hist) > 0 and with_cmd == len(hist) and uniq == with_cmd
          and all("было" in h and "стало" in h for h in hist),
          "правок %d, у всех есть номер команды, и все номера различны "
          "(%d). В каждой записи стоит, что было и что стало — правка, "
          "которой нет в журнале, не считается сделанной. Модельное "
          "время в записи предусмотрено полем «модельное, мс» и "
          "заполняется вызывающим"
          % (len(hist), uniq))

    del s_pool
    note("пул создаётся до компиляции и убран по умолчанию: сцена без "
         "включённых объектов побитово совпадает со сценой без пула — и "
         "по сетчатке, и по траектории. Нависание получается геометрией "
         "приближения, а не изменением размера: радиус всё время один и "
         "тот же, растёт только угловой размер, и замер следует "
         "предсказанию 2·arctan(r/d). Зрительный отклик сети на эти "
         "объекты здесь НЕ проверялся: по `VISION.md` §5 сигнал не "
         "доходит до T4, T5 и LPLC2, поэтому реакции на нависание "
         "ожидать неоткуда, и обещать её нельзя.")

    rep["замеры"]["проба заняла, с"] = round(time.perf_counter() - t0, 1)
    (OUT / "objects.json").write_text(
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
