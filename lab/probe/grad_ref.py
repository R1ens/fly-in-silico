"""Две точки отсчёта метрики `grad_cos`: грудь против антенн.

`ANTENNA.md` §4 оставил владельцу проекта открытый вопрос. Метрика
`grad_cos`, по которой считается `grad_alignment` всех прогонов, берёт
курс из `obs["fly_orientation"]` — это ось x КОРНЕВОГО тела, — а
положение от груди. Обонятельные же сенсоры висят на
`LFuniculus`/`RFuniculus`, за шейными и антеннальными суставами, и
середина антенн лежит на 1,03 мм впереди груди. То есть метрика
сравнивает направление на источник не с тем курсом, с которого муха
нюхает, и не из той точки.

Трогать это молча нельзя: изменились бы все прежние числа
`grad_alignment`. Поэтому сделано то, что требует единый план для
научно неоднозначных решений: прежняя метрика оставлена значением по
умолчанию, рядом поставлена явно помеченная альтернатива
(`BodyConfig.grad_ref = "antennae"`), и здесь они сравниваются на одном
зерне.

Устройство сравнения. Метрика никуда не возвращается — она только
записывается, — поэтому физика двух прогонов обязана совпасть
ПОБИТОВО. Это и есть встречный замер: совпадение траектории
доказывает, что всё расхождение принадлежит метрике, а не поведению.
Если траектории разойдутся, сравнивать нечего, и проба скажет об этом
прямо, а не спишет разницу на метрику.

Что проверяется:

1. физика двух прогонов совпала побитово (встречный замер);
2. курс от антенн не перевёрнут — он отличается от курса груди на
   единицы градусов, а не на 180°. Встречный замер к ошибке знака при
   повороте оси антенн на 90°, которую иначе не видно вовсе: при
   перевёрнутом знаке `grad_cos` просто поменял бы знак, и выглядело бы
   это как «другая метрика», а не как ошибка;
3. расстояние от антенн систематически МЕНЬШЕ расстояния от груди, и
   разница близка к замеренному выносу 1,03 мм;
4. насколько расходятся сами метрики — числом, в долях и в градусах
   пеленга;
5. насколько расходится итоговый `grad_alignment` эпизода — то есть
   цена вопроса для всех прежних прогонов.

    cd D:\\fly-brain\\lab
    set PYTHONIOENCODING=utf-8
    ..\\.venv\\Scripts\\python.exe probe\\grad_ref.py
"""

import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("FLYBRAIN_OUT",
                      str(Path(__file__).parent / "_grad_ref_out"))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

OUT = Path(__file__).parent
DURATION = 1.2          # с модельного времени
SEED = 0

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


def episode(ref):
    from flybrain import body as B

    cfg = B.BodyConfig(duration_s=DURATION, seed=SEED, grad_ref=ref,
                       video=False, view=False)
    t0 = time.perf_counter()
    res = B.run_episode(cfg=cfg, progress=False)
    recs = res["records"] if isinstance(res, dict) else res.records
    met = res["metrics"] if isinstance(res, dict) else res.metrics
    print("    %-9s %d шагов за %.1f с" % (ref, len(recs),
                                           time.perf_counter() - t0),
          flush=True)
    return recs, met


def col(recs, key):
    return np.array([r[key] for r in recs], dtype=float)


def main():
    t0 = time.perf_counter()
    print("Точка отсчёта метрики grad_cos: грудь против антенн", flush=True)
    print("  два эпизода, одно зерно %d, %.1f с модельного времени"
          % (SEED, DURATION), flush=True)

    a_rec, a_met = episode("thorax")
    b_rec, b_met = episode("antennae")

    n = min(len(a_rec), len(b_rec))
    rep["замеры"]["устройство"] = {
        "зерно": SEED, "модельного времени, с": DURATION,
        "шагов": n,
        "шагов в прогоне груди": len(a_rec),
        "шагов в прогоне антенн": len(b_rec),
    }

    # --- 1. физика совпала побитово ---
    same = {}
    for k in ("x", "y", "theta", "odour_l", "odour_r", "dn_l", "dn_r",
              "asym", "amp_l", "amp_r", "spikes"):
        a, b = col(a_rec[:n], k), col(b_rec[:n], k)
        same[k] = bool(np.array_equal(a, b))
    bad = sorted(k for k, v in same.items() if not v)
    rep["замеры"]["физика совпала"] = same
    check("физика двух прогонов совпала побитово — всё расхождение "
          "принадлежит метрике, а не поведению",
          not bad,
          "совпали все %d ряда: положение, курс, яркость на антеннах, "
          "съём, асимметрия, амплитуды шага и спайки. Метрика никуда не "
          "возвращается, и это доказано, а не предположено"
          % len(same) if not bad else "разошлись ряды: " + ", ".join(bad))
    if bad:
        rep["замеры"]["проба заняла, с"] = round(time.perf_counter() - t0, 1)
        (OUT / "grad_ref.json").write_text(
            json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        return 1

    # --- 2. курс от антенн не перевёрнут ---
    # Курс груди в записи есть прямо (`theta`), курс от антенн надо
    # вынуть обратно из метрики: grad_cos = cos(пеленг), и знак потерян,
    # поэтому сравниваем не углы, а положение. Разницу курсов берём
    # через разницу расстояний и косинусов там, где источник один и
    # далеко: честнее измерить её отдельно, прямым замером на
    # неподвижном теле.
    from flybrain import body as B

    cfg = B.BodyConfig(duration_s=0.1, seed=SEED, grad_ref="antennae")
    held = {}

    def grab(bd):
        raw = np.asarray(
            bd.physics.bind(bd.fly._antennae_sensors).sensordata,
            dtype=float).reshape(-1, 3)
        l, r = B._odour_channels(bd.fly)
        p_l, p_r = raw[l], raw[r]
        obs = bd.fly.get_observation(bd)
        x, y, th = B._pose(obs)
        mid = 0.5 * (p_l + p_r)
        ax, ay = float(p_l[0] - p_r[0]), float(p_l[1] - p_r[1])
        held["курс груди, рад"] = th
        held["курс антенн, рад"] = math.atan2(ay, ax) - 0.5 * math.pi
        held["вынос середины антенн, мм"] = float(
            np.hypot(mid[0] - x, mid[1] - y))
        held["разнос антенн, мм"] = float(np.hypot(ax, ay))
        held["середина антенн"] = [round(float(v), 4) for v in mid[:2]]
        held["грудь"] = [round(x, 4), round(y, 4)]

    B.run_episode(cfg=cfg, progress=False, on_body=grab)
    d_th = math.degrees(abs((held["курс антенн, рад"]
                             - held["курс груди, рад"] + math.pi)
                            % (2 * math.pi) - math.pi))
    held["расхождение курсов, °"] = round(d_th, 3)
    rep["замеры"]["на собранном теле"] = {
        k: (round(v, 4) if isinstance(v, float) else v)
        for k, v in held.items()}
    check("курс от антенн не перевёрнут: он отличается от курса груди на "
          "единицы градусов, а не на 180°",
          d_th < 15.0,
          "расхождение курсов %.2f° при разносе антенн %.4f мм и выносе "
          "середины антенн вперёд на %.3f мм. Ошибись знак поворота оси "
          "на 90°, здесь стояло бы около 180°, и подмену нельзя было бы "
          "отличить от «другой метрики»"
          % (d_th, held["разнос антенн, мм"],
             held["вынос середины антенн, мм"]))

    # --- 3. расстояние от антенн меньше ---
    da, db = col(a_rec[:n], "dist"), col(b_rec[:n], "dist")
    ok_rows = (da >= 0) & (db >= 0)
    dd = (da - db)[ok_rows]
    dist = {
        "шагов с источником": int(ok_rows.sum()),
        "расстояние от груди, мм: среднее": round(float(da[ok_rows].mean()), 3),
        "расстояние от антенн, мм: среднее": round(float(db[ok_rows].mean()), 3),
        "антенны ближе на, мм: среднее": round(float(dd.mean()), 4),
        "антенны ближе на, мм: разброс": round(float(dd.std()), 4),
        "антенны ближе на, мм: наименьшее": round(float(dd.min()), 4),
        "антенны ближе на, мм: наибольшее": round(float(dd.max()), 4),
        "шагов, где антенны ДАЛЬШЕ": int((dd < 0).sum()),
        "вынос, замеренный на собранном теле, мм":
            round(held["вынос середины антенн, мм"], 3),
    }
    rep["замеры"]["расстояние"] = dist
    # Первая редакция этой проверки ждала, что антенны систематически
    # БЛИЖЕ, и провалилась — правильно провалилась. Вынос направлен
    # вперёд ПО КУРСУ, а не к источнику: при источнике впереди антенны
    # ближе, при источнике за спиной дальше, и за эпизод, в котором муха
    # крутится, среднее обнуляется само. Систематического смещения здесь
    # и не должно быть.
    #
    # Правильная подпись выноса — не среднее, а ПРЕДЕЛ: разница двух
    # расстояний есть проекция выноса на направление к источнику,
    # поэтому она обязана укладываться в сам вынос и обязана до него
    # доходить. Два числа вместо одного, и оба выведены, а не назначены.
    off = held["вынос середины антенн, мм"]
    reach = max(abs(dd.min()), abs(dd.max()))
    dist["наибольшая разница по модулю, мм"] = round(float(reach), 4)
    dist["вынос, мм"] = round(off, 4)
    dist["наибольшая разница в долях выноса"] = round(reach / off, 4)
    check("разница расстояний есть проекция выноса антенн: укладывается "
          "в вынос и доходит до него",
          reach <= 1.1 * off and reach >= 0.8 * off
          and abs(dd.mean()) < 0.3 * off,
          "от груди в среднем %.2f мм, от антенн %.2f мм. По шагам "
          "разница идёт от %.3f до %.3f мм при выносе середины антенн "
          "вперёд %.3f мм, то есть доходит до %.0f %% выноса и за него "
          "не выходит; среднее %.3f мм — около нуля, потому что вынос "
          "направлен по курсу, а не к источнику: на %d шагах из %d "
          "источник оказался позади, и там антенны дальше. "
          "Систематического смещения тут быть и не должно"
          % (dist["расстояние от груди, мм: среднее"],
             dist["расстояние от антенн, мм: среднее"],
             dd.min(), dd.max(), off, 100 * reach / off, dd.mean(),
             dist["шагов, где антенны ДАЛЬШЕ"], len(dd)))

    # --- 4. насколько расходятся метрики ---
    ga, gb = col(a_rec[:n], "grad_cos"), col(b_rec[:n], "grad_cos")
    m = ok_rows
    diff = np.abs(ga - gb)[m]
    # В градусы пеленга: grad_cos = cos(пеленг), поэтому разница
    # косинусов переводится через сами углы, а не умножением.
    bear_a = np.degrees(np.arccos(np.clip(ga[m], -1.0, 1.0)))
    bear_b = np.degrees(np.arccos(np.clip(gb[m], -1.0, 1.0)))
    bear = np.abs(bear_a - bear_b)
    metric = {
        "grad_cos от груди: среднее": round(float(ga[m].mean()), 4),
        "grad_cos от антенн: среднее": round(float(gb[m].mean()), 4),
        "расхождение: среднее": round(float(diff.mean()), 4),
        "расхождение: наибольшее": round(float(diff.max()), 4),
        "расхождение в градусах пеленга: среднее": round(float(bear.mean()), 3),
        "расхождение в градусах пеленга: наибольшее": round(float(bear.max()), 3),
        "шагов, где знак метрики разный": int(((ga[m] > 0) != (gb[m] > 0)).sum()),
        "шагов всего": int(m.sum()),
    }
    rep["замеры"]["метрика"] = metric
    check("расхождение двух метрик названо числом, а не словом",
          np.isfinite(diff).all(),
          "в среднем %.4f по grad_cos и %.2f° по пеленгу, наибольшее "
          "%.4f и %.2f°; знак метрики разошёлся на %d шагах из %d"
          % (diff.mean(), bear.mean(), diff.max(), bear.max(),
             metric["шагов, где знак метрики разный"],
             metric["шагов всего"]))

    # --- 5. цена вопроса для прежних прогонов ---
    align = {
        "grad_alignment от груди": a_met.get("grad_alignment"),
        "grad_alignment от антенн": b_met.get("grad_alignment"),
        "dist_end_mm от груди": a_met.get("dist_end_mm"),
        "dist_end_mm от антенн": b_met.get("dist_end_mm"),
        "path_mm от груди": a_met.get("path_mm"),
        "path_mm от антенн": b_met.get("path_mm"),
    }
    ga_m = float(align["grad_alignment от груди"] or 0.0)
    gb_m = float(align["grad_alignment от антенн"] or 0.0)
    align["расхождение grad_alignment"] = round(abs(ga_m - gb_m), 4)
    rep["замеры"]["итоговые показатели эпизода"] = align
    print("\n  итог эпизода: grad_alignment %.4f от груди против %.4f от "
          "антенн (расхождение %.4f)"
          % (ga_m, gb_m, abs(ga_m - gb_m)), flush=True)

    # Разброс grad_alignment ВНУТРИ условия замерен пробой lesion.py и
    # равен 0,887 на трёх зёрнах. Сравнивать расхождение метрик надо с
    # ним, иначе «большое» и «маленькое» ничего не значат.
    spread = 0.887
    rel = abs(ga_m - gb_m) / spread
    align["разброс grad_alignment внутри условия (probe/lesion.py)"] = spread
    align["расхождение в долях разброса"] = round(rel, 3)
    check("расхождение метрик сопоставлено с разбросом внутри условия, "
          "а не объявлено большим или малым на глаз",
          True,
          "расхождение grad_alignment %.4f при разбросе внутри условия "
          "%.3f (три зерна, probe/lesion.py), то есть %.1f %% разброса. "
          "Вывод о том, какая метрика правильнее, из этого НЕ следует: "
          "здесь измерено только, насколько они расходятся"
          % (abs(ga_m - gb_m), spread, 100 * rel))

    note("обе метрики остаются в проекте, и прежняя — значением по "
         "умолчанию: `BodyConfig.grad_ref = \"thorax\"`, то есть все "
         "прежние прогоны и все прогоны из окна считаются как раньше. "
         "Альтернатива `\"antennae\"` ставится только пробой и в "
         "`spec.py`/`catalog.py` не объявлена нарочно. Что именно "
         "правильнее — вопрос не к замеру: курс груди проще и "
         "воспроизводим, курс антенн ближе к тому, чем муха нюхает, но "
         "зависит от свободных шейных суставов, которые в "
         "NeuroMechFly ничем не стабилизированы (жёсткость 10 и 1, "
         "приводов нет). Решение за владельцем проекта; замеры для него "
         "здесь.")

    rep["замеры"]["проба заняла, с"] = round(time.perf_counter() - t0, 1)
    (OUT / "grad_ref.json").write_text(
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
