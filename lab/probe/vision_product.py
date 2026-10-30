"""Зрение В ПРОДУКТЕ: через `body.run_episode`, а не мимо него.

Прежние зрительные пробы (`vision_chain`, `vision_live`, `vision_gl`,
`objects`) собирали `Fly(enable_vision=True)` СВОИМИ руками. Это и было
главной находкой: замеренные числа лежали на дороге, до которой продукт
не доходил — у `BodyConfig` поля зрения не существовало вовсе.

Здесь ровно наоборот: ни одного `Fly(...)` своими руками. Всё идёт тем
самым путём, которым идут командная строка (`cli.cmd_body`) и работник
(`jobs._job_body`): `spec.RunSpec` -> `to_body_config` ->
`body.run_episode`. Если зрение не проведено в продукт, эта проба
провалится на первом же пункте.

Что проверяется, и у каждого пункта встречный замер:

1. **историческое умолчание** — эпизод без зрения не несёт о зрении ни
   одного поля; встречный замер: с зрением несёт их все;
2. **арифметика переходника** — частоты совпадают с `retina.rates`
   ПОБИТОВО при единичном контрасте; встречный замер: при контрасте 2
   расходятся;
3. **оба сложных глаза** — подано 10 582 фоторецептора обоих сторон;
   встречный замер: один глаз даёт ровно свою половину;
4. **настоящие наблюдения сцены** — кадр меняется от шага к шагу; ни
   один шаг не повторил предыдущий;
5. **измеримый отклик настоящих популяций** — L1, L2, L3, Tm1 против
   нуля при выключенном зрении и против перемешанной привязки;
6. **свет и контраст** — попарно, в одном процессе: яркость сцены и
   контраст кадра меняют поданные частоты измеримо;
7. **нависание** — угловой размер следует 2·arctan(r/d), сетчатка
   темнеет; встречный замер: неподвижный объект углового размера не
   меняет, а светлое нависание вместо тёмного меняет знак;
8. **отметки времени обновления** — измеренная частота обновления
   совпадает с заказанной на 100 Гц и на 250 Гц;
9. **происхождение привязки** — в сохранённом прогоне есть отпечаток
   набора и итог внешней проверки.

Разогрев здесь укорочен (`dn_tau_s = 0,1` вместо 0,5), и это
объявленная подмена: он стоит 15 шагов вместо 75, а ни один зрительный
замер от постоянной сглаживания моторного съёма не зависит. Числа
моторной части из этой пробы брать нельзя.

    cd D:\\fly-brain\\lab
    set PYTHONIOENCODING=utf-8
    ..\\.venv\\Scripts\\python.exe probe\\vision_product.py
"""

import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("FLYBRAIN_OUT",
                      str(Path(__file__).parent / "_vision_product_out"))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

OUT = Path(__file__).parent

# Длительность зрительных эпизодов, с модельного времени. Десять шагов
# управления: зрительные замеры снимаются на каждом шаге, и больше
# нужно только нависанию (ему 20).
T_SHORT = 0.2
T_LOOM = 0.4
TAU = 0.1              # укороченный разогрев, см. докстринг

# Популяции раннего зрения. Имена — ТОЧНЫЕ типы: свободный `type:X`
# значит «содержит X» и даёт, например, 6400 нейронов на `type:T4`,
# среди которых ни одного T4 (`RESEARCH.md` §9).
EARLY = (("L1", "L1"), ("L2", "L2"), ("L3", "L3"), ("Tm1", "Tm1"))
DEEP = (("T4", ("T4a", "T4b", "T4c", "T4d")),
        ("T5", ("T5a", "T5b", "T5c", "T5d")),
        ("LPLC2", ("LPLC2",)), ("LC4", ("LC4",)), ("DNp01", ("DNp01",)))

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


def num(v, nd=4):
    return None if v is None else round(float(v), nd)


# ------------------------------------------------------------ прогон

def episode(**body_kw):
    """Эпизод ТЕМ ЖЕ путём, что командная строка и работник.

    Собирается настоящая `spec.RunSpec`, из неё — `BodyConfig`, и только
    потом зовётся `body.run_episode`. Своего `Fly(...)` здесь нет
    нарочно: проба обязана ломаться, если зрение не проведено в
    продукт.
    """
    from flybrain import body as B, spec as S

    kw = {"dn_tau_s": TAU, "control_ms": 20.0, "physics_dt": 2e-4}
    kw.update(body_kw)
    t = kw.pop("t", T_SHORT)
    sp = S.RunSpec(kind="body", t=t, seed=0, body=S.BodySpec(**kw))
    cfg = sp.to_body_config(DEV)
    return B.run_episode(cfg=cfg, neurons=NEURONS, weights=WEIGHTS,
                         progress=False), cfg, sp


def pop_hz(run, names):
    """Частота популяции за весь эпизод, Гц на нейрон.

    Считается по `run.counts` — счётчикам спайков за весь сеанс — и
    `timing["duration_s"]`, то есть тем же способом, каким считает
    `sim.Result.rate_table`. Выборка — точным селектором `celltype=`.
    """
    from flybrain import data as D

    if isinstance(names, str):
        names = (names,)
    idx = np.unique(np.concatenate(
        [D.select(NEURONS, "celltype=%s" % n) for n in names]))
    dur = float(run.timing.get("duration_s") or 1.0)
    if not len(idx):
        return None, 0
    return float(run.counts[idx].sum()) / (len(idx) * dur), int(len(idx))


def vis_series(run, key):
    return np.array([r.get(key) for r in run.records
                     if r.get(key) is not None], dtype=float)


# ------------------------------------------------------------------ 1

def part1_default():
    print("\n1. Историческое умолчание: зрение выключено", flush=True)
    off, cfg_off, sp_off = episode()
    on, _cfg_on, _sp = episode(vision=True)

    has_off = "зрение" in off.timing
    has_on = "зрение" in on.timing
    rep["замеры"]["умолчание"] = {
        "BodyConfig.vision по умолчанию": bool(cfg_off.vision),
        "BodySpec.vision по умолчанию": bool(sp_off.body.vision),
        "блок «зрение» в прогоне без зрения": has_off,
        "блок «зрение» в прогоне со зрением": has_on,
        "поля vis_* в записи шага без зрения":
            sorted(k for k in off.records[0] if k.startswith("vis")),
        "поля vis_* в записи шага со зрением":
            sorted(k for k in on.records[0] if k.startswith("vis")),
    }
    check("зрение выключено по умолчанию и в BodyConfig, и в спеке",
          not cfg_off.vision and not sp_off.body.vision,
          "BodyConfig.vision=%s, BodySpec.vision=%s"
          % (cfg_off.vision, sp_off.body.vision))
    check("прогон без зрения не несёт о зрении ни одного поля",
          not has_off and not any(k.startswith("vis")
                                  for k in off.records[0]),
          "блока «зрение» нет, полей vis_* в записи шага нет — то есть "
          "форма `track.parquet` у прежнего эпизода не изменилась")
    check("встречный замер: прогон со зрением несёт их все",
          has_on and {"vis_t_ms", "vis_n", "vis_hz", "vis_sat",
                      "vis_lag_ms"} <= set(on.records[0]),
          "блок «зрение» есть, в записи шага %s"
          % sorted(k for k in on.records[0] if k.startswith("vis")))
    return off, on


# ------------------------------------------------------------------ 2

def part2_adapter():
    print("\n2. Арифметика переходника против retina.rates", flush=True)
    from flybrain import retina as R, vision as V

    ad = V.make("retina", max_hz=150.0, ref=0.5).attach(NEURONS, "cpu")
    g = np.random.default_rng(7)
    frame = g.random((2, 721, 2), dtype=np.float32)

    want = R.rates(ad.maps, frame, max_hz=150.0, ref=0.5, channel="max")
    flat = []
    for key in sorted(ad.maps, key=lambda t: (t[0], t[1])):
        flat.append(np.asarray(want[key], dtype=np.float32))
    want_flat = np.concatenate(flat)
    got = ad.hz(frame, contrast=1.0)

    same = np.array_equal(want_flat, got)
    got2 = ad.hz(frame, contrast=2.0)
    diff2 = float(np.abs(got2 - got).max())
    rep["замеры"]["переходник"] = {
        "фоторецепторов": int(len(got)),
        "побитово равно retina.rates при контрасте 1": bool(same),
        "наибольшее расхождение при контрасте 1":
            float(np.abs(want_flat - got).max()),
        "наибольшее расхождение при контрасте 2": diff2,
        "средняя частота при контрасте 1": num(got.mean()),
        "средняя частота при контрасте 2": num(got2.mean()),
    }
    check("частоты переходника совпадают с retina.rates ПОБИТОВО",
          same, "%d фоторецепторов, наибольшее расхождение %.3g"
                % (len(got), float(np.abs(want_flat - got).max())))
    check("встречный замер: при контрасте 2 расходятся", diff2 > 1.0,
          "наибольшее расхождение %.3f Гц, средняя %.2f против %.2f Гц"
          % (diff2, got2.mean(), got.mean()))
    return ad


def part3_eyes(ad):
    print("\n3. Оба сложных глаза", flush=True)
    from flybrain import vision as V

    left = V.make("retina", eyes="left").attach(NEURONS, "cpu")
    right = V.make("retina", eyes="right").attach(NEURONS, "cpu")
    nb, nl, nr = len(ad.indices), len(left.indices), len(right.indices)
    overlap = len(np.intersect1d(left.indices, right.indices))
    sides = {"left": 0, "right": 0}
    for key in ad.maps:
        sides[key[0]] += len(ad.maps[key].neurons)
    rep["замеры"]["глаза"] = {
        "оба глаза, фоторецепторов": nb,
        "слева": sides["left"], "справа": sides["right"],
        "только левый глаз": nl, "только правый": nr,
        "пересечение левого и правого": overlap,
        "привязок в наборе": len(ad.maps),
    }
    check("подаются ОБА глаза, и половины сходятся с целым",
          nb == nl + nr and nl == sides["left"] and nr == sides["right"]
          and overlap == 0,
          "%d = %d слева + %d справа, пересечение %d, привязок %d "
          "(3 типа × 2 стороны)" % (nb, nl, nr, overlap, len(ad.maps)))
    check("встречный замер: один глаз даёт ровно свою половину",
          0 < nl < nb and 0 < nr < nb,
          "левый %d, правый %d из %d" % (nl, nr, nb))


# ------------------------------------------------------------------ 4

def part4_frames(on):
    print("\n4. Настоящие наблюдения сцены, а не подставной вход",
          flush=True)
    hz = vis_series(on, "vis_hz")
    sat = vis_series(on, "vis_sat")
    same_pairs = int(np.sum(np.diff(hz) == 0.0)) if len(hz) > 1 else -1
    rep["замеры"]["кадр"] = {
        "шагов с подачей": int(len(hz)),
        "средняя частота фоторецепторов, Гц": num(hz.mean()),
        "разброс по шагам, Гц": num(hz.std()),
        "наименьшая, Гц": num(hz.min()), "наибольшая, Гц": num(hz.max()),
        "доля в потолке, среднее": num(sat.mean()),
        "совпавших подряд шагов": same_pairs,
        "наибольшая разница между соседними шагами, Гц":
            num(np.abs(np.diff(hz)).max()) if len(hz) > 1 else None,
    }
    check("кадр живой: ни один шаг не повторил предыдущий",
          len(hz) > 1 and same_pairs == 0,
          "шагов %d, совпавших подряд %d, разброс %.3f Гц, наибольшая "
          "разница между соседними %.3f Гц"
          % (len(hz), same_pairs, hz.std(), np.abs(np.diff(hz)).max()))
    check("встречный замер к «кадр подставной»: частоты в середине "
          "шкалы, а не на потолке и не в нуле",
          0.05 < float(sat.mean()) < 0.95 and hz.min() > 1.0,
          "в потолке %.1f %% фоторецепторов, частоты %.1f…%.1f Гц при "
          "потолке 150" % (100 * sat.mean(), hz.min(), hz.max()))


# ------------------------------------------------------------------ 5

def part5_populations(off, on):
    print("\n5. Измеримый отклик НАСТОЯЩИХ популяций", flush=True)
    shuf, _c, _s = episode(vision=True, vision_shuffle=True)
    rows = {}
    for title, sel in EARLY:
        a, n = pop_hz(off, sel)
        b, _ = pop_hz(on, sel)
        c, _ = pop_hz(shuf, sel)
        rows[title] = {"нейронов": n, "выключено": num(a, 5),
                       "зрение": num(b, 5), "перемешано": num(c, 5),
                       "зрению принадлежит": num((b or 0) - (a or 0), 5),
                       "структуре принадлежит": num((b or 0) - (c or 0), 5)}
        print("   %-7s n %5d  выкл %8.4f  зрение %8.4f  перемеш %8.4f"
              % (title, n, a or 0, b or 0, c or 0), flush=True)
    deep = {}
    for title, sel in DEEP:
        b, n = pop_hz(on, sel)
        deep[title] = {"нейронов": n, "зрение": num(b, 5)}
    pr, npr = pop_hz(on, ("R1-6", "R7", "R8"))
    pr_off, _ = pop_hz(off, ("R1-6", "R7", "R8"))
    rep["замеры"]["популяции"] = {"фоторецепторы": {
        "нейронов": npr, "выключено": num(pr_off, 5), "зрение": num(pr, 5)},
        "раннее зрение": rows, "глубже": deep}

    # Сравнения пишутся через `is not None`, а не через `or 0`: ноль в
    # питоне ложен, и `(pr_off or 1) == 0.0` отдавало False РОВНО
    # ТОГДА, когда замер идеален. Поймано на первом же запуске этой
    # пробы: три пункта провалились на правильных числах.
    check("фоторецепторы отвечают, а при выключенном зрении молчат",
          pr is not None and pr > 50.0
          and pr_off is not None and pr_off == 0.0,
          "%d нейронов: %.2f Гц со зрением против %.4f без"
          % (npr, pr if pr is not None else float("nan"),
             pr_off if pr_off is not None else float("nan")))
    lit = [t for t, v in rows.items()
           if v["зрение"] is not None and v["зрение"] > 0.5
           and v["выключено"] is not None and v["выключено"] == 0.0]
    check("ранняя зрительная активность измерима и принадлежит зрению",
          len(lit) >= 3,
          "отозвались и были нулём без зрения: %s" % ", ".join(lit))
    struct = {t: v["структуре принадлежит"] for t, v in rows.items()}
    check("встречный замер: отклик зависит от пространственной "
          "структуры при том же суммарном входе",
          any(v is not None and abs(v) > 1e-3 for v in struct.values()),
          "разница настоящей и перемешанной привязки по популяциям: %s"
          % json.dumps(struct, ensure_ascii=False))
    silent = [t for t, v in deep.items()
              if v["зрение"] is not None and v["зрение"] == 0.0]
    check("ОГРАНИЧЕНИЕ НА МЕСТЕ: глубже раннего зрения сигнал не "
          "проходит, и это видно числом, а не словами",
          len(silent) == len(DEEP),
          "ровно 0,000 Гц у всех: %s — поведение на зрение не "
          "опирается" % ", ".join(silent))
    return shuf


# ------------------------------------------------------------------ 6

def part6_light_contrast(on):
    print("\n6. Свет и контраст: попарно, в одном процессе", flush=True)
    base = float(vis_series(on, "vis_hz").mean())
    dim, _c, _s = episode(vision=True, vision_light=0.3)
    hi, _c, _s = episode(vision=True, vision_contrast=2.0)
    dim_hz = float(vis_series(dim, "vis_hz").mean())
    hi_hz = float(vis_series(hi, "vis_hz").mean())
    light = (dim.timing["зрение"]["свет сцены"] or {})
    rep["замеры"]["свет и контраст"] = {
        "свет 1,0: средняя частота, Гц": num(base),
        "свет 0,3: средняя частота, Гц": num(dim_hz),
        "свет 0,3: отношение": num(dim_hz / base if base else None),
        "свет 0,3: diffuse в модели": light.get("diffuse"),
        "свет 0,3: множитель в модели": light.get("множитель"),
        "контраст 2,0: средняя частота, Гц": num(hi_hz),
        "контраст 2,0: отношение": num(hi_hz / base if base else None),
    }
    check("свет сцены меняет кадр глаза измеримо, и правка доехала "
          "до НАСТОЯЩИХ источников MuJoCo",
          dim_hz < base * 0.95 and light.get("множитель") is not None
          and abs(float(light["множитель"]) - 0.3) < 1e-3,
          "%.2f Гц при свете 0,3 против %.2f при 1,0 (отношение %.3f); "
          "diffuse в модели %s, множитель %s"
          % (dim_hz, base, dim_hz / base, light.get("diffuse"),
             light.get("множитель")))
    check("контраст меняет поданные частоты измеримо",
          abs(hi_hz - base) > base * 0.02,
          "%.2f Гц при контрасте 2 против %.2f при 1 (отношение %.3f)"
          % (hi_hz, base, hi_hz / base))
    check("встречный замер: при свете 1 и контрасте 1 модель не "
          "трогается вовсе",
          (on.timing["зрение"]["свет сцены"] or {}).get("множитель") is None,
          "множителя в записи нет — значит `scene_light` не звался, и "
          "кадр остался прежним до бита")


# ------------------------------------------------------------------ 7

def part7_loom():
    """Нависание — ПОПАРНО против опоры без объекта.

    Абсолютный знак здесь ничего не доказывает, и это замер, а не
    рассуждение: муха идёт своей походкой, сцена вокруг неё меняется, и
    средняя яркость кадра за 400 мс растёт сама — у опоры без объекта
    на +1,4 Гц. Тёмный шар в 6,8° углового размера закрывает десятки
    омматидиев из 1442, и его вклад этим дрейфом заглушается.

    Поэтому замер строится парами на ОДНОМ зерне: «тот же эпизод с
    объектом» минус «тот же эпизод без объекта», шаг против шага. Эта
    разница и есть вклад объекта. Встречных замеров к ней два: светлый
    шар на том же отрезке (знак обязан перевернуться) и неподвижный
    (роста углового размера нет, значит и хода разницы быть не
    должно).
    """
    print("\n7. Нависание: попарно против опоры без объекта", flush=True)
    from flybrain import objects as O, spec as S

    bare, _c, _s = episode(t=T_LOOM, **S.vision_preset("bare"))
    base = vis_series(bare, "vis_hz")
    out = {"опора без объекта": {
        "частота: начало, Гц": num(base[0], 3),
        "частота: конец, Гц": num(base[-1], 3),
        "дрейф за эпизод, Гц": num(base[-1] - base[0], 3)}}
    runs = {"bare": bare}
    for name in ("loom-dark", "loom-light", "still"):
        kw = S.vision_preset(name)
        run, _c, _s = episode(t=T_LOOM, **kw)
        runs[name] = run
        hz = vis_series(run, "vis_hz")
        # Угловой размер считается по заданному треку и положению мухи
        # этого шага — то есть предсказание геометрии, а не замер по
        # сетчатке. Сравнение предсказания с замером и есть проверка.
        track = O.Track(kw["vis_track"])
        ang = []
        for r in run.records:
            want = track.at(r["t_ms"])
            if 0 not in want:
                ang.append(0.0)
                continue
            p = np.asarray(want[0]["pos"], dtype=float)
            d = math.sqrt((p[0] - r["x"]) ** 2 + (p[1] - r["y"]) ** 2
                          + p[2] ** 2)
            rad = want[0]["radius"]
            ang.append(180.0 if d <= rad
                       else math.degrees(2.0 * math.atan(rad / d)))
        ang = np.asarray(ang)
        # Вклад объекта: шаг против шага против опоры без объекта.
        n = min(len(hz), len(base))
        dd = hz[:n] - base[:n]
        out[name] = {
            "шагов": len(hz),
            "угловой размер: начало, °": num(ang[0], 3),
            "угловой размер: конец, °": num(ang[-1], 3),
            "угловой размер: рост, раз": num(ang[-1] / ang[0], 3)
            if ang[0] > 0 else None,
            "частота: начало, Гц": num(hz[0], 3),
            "частота: конец, Гц": num(hz[-1], 3),
            "вклад объекта: начало, Гц": num(dd[0], 4),
            "вклад объекта: конец, Гц": num(dd[-1], 4),
            "вклад объекта: ход за эпизод, Гц": num(dd[-1] - dd[0], 4),
            "корреляция углового размера с вкладом объекта":
                num(np.corrcoef(ang[:n], dd)[0, 1], 4)
                if ang[:n].std() > 1e-9 and dd.std() > 1e-9 else None,
        }
        print("   %-11s угловой %6.2f°→%6.2f°  вклад объекта "
              "%+7.4f→%+7.4f Гц"
              % (name, ang[0], ang[-1], dd[0], dd[-1]), flush=True)
    rep["замеры"]["нависание"] = out

    d = out["loom-dark"]
    check("тёмное нависание: угловой размер растёт геометрией, а "
          "ВКЛАД ОБЪЕКТА в поданную частоту уходит вниз",
          d["угловой размер: рост, раз"] is not None
          and d["угловой размер: рост, раз"] > 1.3
          and d["вклад объекта: ход за эпизод, Гц"] < 0,
          "угловой размер %.2f° → %.2f° (в %.2f раза); вклад объекта "
          "%+.4f → %+.4f Гц, то есть сдвинулся на %+.4f. Опора без "
          "объекта за то же время дрейфует на %+.3f Гц — потому и "
          "замер попарный"
          % (d["угловой размер: начало, °"], d["угловой размер: конец, °"],
             d["угловой размер: рост, раз"],
             d["вклад объекта: начало, Гц"], d["вклад объекта: конец, Гц"],
             d["вклад объекта: ход за эпизод, Гц"],
             out["опора без объекта"]["дрейф за эпизод, Гц"]))
    st = out["still"]
    check("встречный замер: у неподвижного объекта угловой размер не "
          "растёт, и хода вклада нет",
          st["угловой размер: рост, раз"] is not None
          and abs(st["угловой размер: рост, раз"] - 1.0) < 0.35
          and abs(st["вклад объекта: ход за эпизод, Гц"])
          < abs(d["вклад объекта: ход за эпизод, Гц"]),
          "угловой размер %.2f° → %.2f° (в %.2f раза — остаток от "
          "собственной походки мухи); ход вклада %+.4f против %+.4f у "
          "нависающего"
          % (st["угловой размер: начало, °"],
             st["угловой размер: конец, °"],
             st["угловой размер: рост, раз"],
             st["вклад объекта: ход за эпизод, Гц"],
             d["вклад объекта: ход за эпизод, Гц"]))
    lt = out["loom-light"]
    check("встречный замер: СВЕТЛОЕ нависание даёт вклад другого "
          "знака — отличие ровно в серости объекта",
          lt["вклад объекта: конец, Гц"] > d["вклад объекта: конец, Гц"],
          "вклад в конце: тёмное %+.4f Гц, светлое %+.4f Гц при одном и "
          "том же отрезке траектории и одном зерне"
          % (d["вклад объекта: конец, Гц"],
             lt["вклад объекта: конец, Гц"]))
    return runs


# ------------------------------------------------------------------ 8

def part8_clock(on):
    print("\n8. Отметки времени обновления зрения", flush=True)
    fast, _c, _s = episode(vision=True, vision_hz=250.0)
    out = {}
    for name, run in (("100 Гц", on), ("250 Гц", fast)):
        v = run.timing["зрение"]
        t = vis_series(run, "vis_t_ms")
        lag = vis_series(run, "vis_lag_ms")
        n = vis_series(run, "vis_n")
        out[name] = {
            "заказано, Гц": v["частота обновления заказана, Гц"],
            "у FlyGym, Гц": v["частота обновления у FlyGym, Гц"],
            "измерено, Гц": v["частота обновления измерена, Гц"],
            "обновлений за сеанс": v["обновлений кадра за сеанс"],
            "обновлений за шаг: наименьшее": int(n.min()),
            "обновлений за шаг: наибольшее": int(n.max()),
            "отметки не убывают": bool(np.all(np.diff(t) >= 0)),
            "отставание кадра: наибольшее, мс": num(lag.max(), 3),
        }
        print("   %-7s заказано %5.0f, измерено %7.2f, обновлений %4d, "
              "за шаг %d…%d" % (name, out[name]["заказано, Гц"],
                                out[name]["измерено, Гц"],
                                out[name]["обновлений за сеанс"],
                                out[name]["обновлений за шаг: наименьшее"],
                                out[name]["обновлений за шаг: наибольшее"]),
          flush=True)
    rep["замеры"]["расписание"] = out
    ok = all(abs(v["измерено, Гц"] - v["заказано, Гц"])
             <= 0.02 * v["заказано, Гц"] for v in out.values())
    check("измеренная частота обновления совпадает с заказанной и на "
          "100 Гц, и на 250 Гц",
          ok, "100 Гц: измерено %.2f; 250 Гц: измерено %.2f"
              % (out["100 Гц"]["измерено, Гц"],
                 out["250 Гц"]["измерено, Гц"]))
    check("отметки времени кадра не убывают, и подан всегда кадр не "
          "новее шага",
          all(v["отметки не убывают"] for v in out.values())
          and all(v["отставание кадра: наибольшее, мс"] >= 0
                  for v in out.values()),
          "наибольшее отставание: 100 Гц %.1f мс, 250 Гц %.1f мс"
          % (out["100 Гц"]["отставание кадра: наибольшее, мс"],
             out["250 Гц"]["отставание кадра: наибольшее, мс"]))
    check("встречный замер: на 250 Гц обновлений за шаг больше, чем на "
          "100 Гц",
          (out["250 Гц"]["обновлений за шаг: наименьшее"]
           > out["100 Гц"]["обновлений за шаг: наибольшее"]),
          "100 Гц: %d…%d за шаг; 250 Гц: %d…%d"
          % (out["100 Гц"]["обновлений за шаг: наименьшее"],
             out["100 Гц"]["обновлений за шаг: наибольшее"],
             out["250 Гц"]["обновлений за шаг: наименьшее"],
             out["250 Гц"]["обновлений за шаг: наибольшее"]))


# ------------------------------------------------------------------ 9

def part9_provenance(on, shuf):
    print("\n9. Происхождение привязки в сохранённом прогоне", flush=True)
    from flybrain import body as B, data as D, runs as RUNS

    d = D.OUT / "vision-product-provenance"
    B.save(on, d)
    found = RUNS.scan()
    got = next((r for r in found if r.path == d), None)
    vis = (got.vision or {}) if got is not None else {}
    pr = vis.get("привязка") or {}
    ext = (pr.get("внешняя проверка") or {}).get("итог") or ""
    fp_on = pr.get("отпечаток набора")
    fp_shuf = (shuf.timing["зрение"]["привязка"] or {}).get("отпечаток набора")
    cav = [c for c in (got.caveats if got is not None else ())
           if "привязк" in c.lower()]
    rep["замеры"]["происхождение"] = {
        "прогон найден каталогом": got is not None,
        "переходник": pr.get("переходник"),
        "отпечаток набора": fp_on,
        "отпечаток перемешанной": fp_shuf,
        "отпечатки различны": fp_on != fp_shuf,
        "привязок в наборе": len(pr.get("привязки") or {}),
        "итог внешней проверки есть": bool(ext),
        "оговорка про привязку доехала до каталога": len(cav),
    }
    check("в сохранённом прогоне видно, КАКОЙ привязкой считалось",
          bool(fp_on) and len(pr.get("привязки") or {}) == 6
          and pr.get("переходник") == "retina",
          "переходник %r, отпечаток набора %s, привязок 6 "
          "(3 типа × 2 стороны), у каждой свой отпечаток"
          % (pr.get("переходник"), fp_on))
    check("встречный замер: у перемешанной привязки отпечаток другой",
          bool(fp_shuf) and fp_on != fp_shuf,
          "настоящая %s, перемешанная %s" % (fp_on, fp_shuf))
    check("ограничение не замазано: итог внешней проверки лежит в "
          "прогоне и доезжает до каталога оговоркой",
          "ОПРОВЕРГНУТА" in ext and len(cav) >= 1,
          "в прогоне: «%s…»; оговорок про привязку в каталоге %d"
          % (ext[:90], len(cav)))


def main():
    global NEURONS, WEIGHTS, DEV

    t0 = time.perf_counter()
    print("Зрение в продукте: через body.run_episode", flush=True)
    import torch

    from flybrain import data as D

    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    NEURONS = D.load_neurons()
    WEIGHTS = D.load_weights(device=DEV)
    print("  устройство %s, нейронов %d" % (DEV, len(NEURONS)), flush=True)

    off, on = part1_default()
    ad = part2_adapter()
    part3_eyes(ad)
    part4_frames(on)
    shuf = part5_populations(off, on)
    part6_light_contrast(on)
    part7_loom()
    part8_clock(on)
    part9_provenance(on, shuf)

    note("зрение проведено в продукт целиком: спека -> BodyConfig -> "
         "эпизод, и проба не собирает `Fly(...)` своими руками ни разу. "
         "Поэтому она умеет провалиться: убери поле `vision` из "
         "`BodySpec` — и пункт 1 падает на `TypeError`, а не проходит "
         "мимо.")
    note("ограничение остаётся измеренным, а не обойдённым: ранняя "
         "зрительная активность есть, глубже раннего зрения ровно "
         "0,000 Гц, привязка инженерная и опровергнута внешним "
         "свидетелем. Всё это лежит в самом прогоне и показывается в "
         "окне — заклеивать это продукт не стал.")
    rep["итог"] = {"проверок": len(rep["проверки"]), "провалов": len(fails),
                   "за, с": round(time.perf_counter() - t0, 1)}
    (OUT / "vision_product.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nпроверок %d, провалов %d, за %.0f с"
          % (len(rep["проверки"]), len(fails),
             time.perf_counter() - t0), flush=True)
    if fails:
        print("провалились: " + ", ".join(fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
