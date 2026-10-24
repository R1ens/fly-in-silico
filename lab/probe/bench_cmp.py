"""Сравнение двух снимков эталона: что ускорилось и что при этом изменилось.

Уровни из брифа различаются здесь буквально:

* **уровень 1** — отпечатки совпали побайтово. Такую оптимизацию можно
  пускать в научные прогоны без оговорок.
* **уровень 2** — отпечатки разошлись, но числа близки в пределах
  названного заранее порога. Требует отдельного разговора и пометки.
* **расхождение** — разошлись величины, которые расходиться не должны.

Порог для уровня 2 задаётся явно и один на все величины, чтобы его нельзя
было подобрать под результат.

    ..\\.venv\\Scripts\\python.exe probe\\bench_cmp.py эталон быстрый
"""

import argparse
import json
import sys
from pathlib import Path

OUT = Path(__file__).parent

# Относительный порог уровня 2. Взят не из оптимизации, а из того, что
# один бит мантиссы float32 — это 6e-8, и накопление по десяти тысячам
# шагов даёт запас на три-четыре порядка больше.
TOL = 1e-4


def num_paths(d, prefix=""):
    """Все числовые поля снимка с их путями."""
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(num_paths(v, prefix + "/" + str(k)))
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        out[prefix] = float(d)
    return out


# Поля, которые обязаны меняться: это и есть измеряемая скорость.
TIMING = ("/время/", "/память/", "/мс на шаг", "realtime_ratio",
          "setup_s", "sim_s", "/всего, с", "/загрузка")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("было")
    ap.add_argument("стало")
    ap.add_argument("--ожидаемо", nargs="*", default=[], metavar="СЦЕНАРИЙ",
                    help="сценарии, где расхождение — не поломка, а "
                         "сознательная правка модели. Их расхождение "
                         "печатается целиком, но не считается провалом")
    args = ap.parse_args()
    expected = set(getattr(args, "ожидаемо") or [])

    a = json.loads((OUT / ("bench_v2_%s.json" % getattr(args, "было"))
                    ).read_text(encoding="utf-8"))
    b = json.loads((OUT / ("bench_v2_%s.json" % getattr(args, "стало"))
                    ).read_text(encoding="utf-8"))

    print("было:  %s (%s)" % (a["метка"], a["когда"]))
    print("стало: %s (%s)" % (b["метка"], b["когда"]))
    print()

    worst = 0
    rows = []
    for name, got in b["сценарии"].items():
        ref = a["сценарии"].get(name)
        if ref is None or "ошибка" in got or "ошибка" in ref:
            rows.append((name, "нет пары", "—", "—"))
            continue
        # 1. отпечатки
        fa = {k: v for k, v in num_paths({}).items()}
        prints_a, prints_b = {}, {}
        for block in ("активность", "моторные команды", "тело", "сенсоры",
                      "популяции по сторонам"):
            for k, v in (ref.get(block) or {}).items():
                if isinstance(v, str) and "отпечаток" in k:
                    prints_a[block + "/" + k] = v
            for k, v in (got.get(block) or {}).items():
                if isinstance(v, str) and "отпечаток" in k:
                    prints_b[block + "/" + k] = v
        bad_prints = [k for k in prints_a
                      if prints_b.get(k) != prints_a[k]]

        # 2. числа, кроме времени и памяти
        na, nb = num_paths(ref), num_paths(got)
        drift = []
        for k, v in na.items():
            if any(t in k for t in TIMING):
                continue
            w = nb.get(k)
            if w is None:
                continue
            if v == w:
                continue
            rel = abs(w - v) / max(abs(v), 1e-12)
            drift.append((k, v, w, rel))
        drift.sort(key=lambda x: -x[3])

        speed = ref["время"]["всего, с"] / max(got["время"]["всего, с"], 1e-9)
        if not bad_prints and not drift:
            level = "уровень 1: побайтово то же"
        elif drift and max(d[3] for d in drift) <= TOL:
            level = "уровень 2: в пределах %.0e" % TOL
            worst = max(worst, 2)
        elif name in expected:
            level = "расхождение ОЖИДАЕМО: правка модели"
            worst = max(worst, 2)
        else:
            level = "РАСХОЖДЕНИЕ"
            worst = 3
        rows.append((name, level, "×%.2f" % speed,
                     "отпечатков разошлось %d" % len(bad_prints)
                     if bad_prints else "отпечатки совпали"))
        if drift and level != "уровень 1: побайтово то же":
            for k, v, w, rel in drift[:5]:
                rows.append(("", "   " + k, "%.6g -> %.6g" % (v, w),
                             "%.2e" % rel))

    print("%-10s %-34s %-12s %s" % ("сценарий", "итог", "скорость", "чем"))
    for r in rows:
        print("%-10s %-34s %-12s %s" % r)

    print()
    if worst == 0:
        print("ВСЁ СОВПАЛО ПОБАЙТОВО — уровень 1 по всем сценариям")
    elif worst == 2:
        if expected:
            print("расхождения только там, где их ждали: %s"
                  % ", ".join(sorted(expected)))
            print("это правка модели, а не оптимизация — результат "
                  "изменился сознательно и помечен")
        else:
            print("числа сошлись в пределах порога — уровень 2, "
                  "нужна пометка")
    else:
        print("ЕСТЬ РАСХОЖДЕНИЯ — в научные прогоны так нельзя")
    return 0 if worst <= 1 else (0 if worst == 2 else 1)


if __name__ == "__main__":
    raise SystemExit(main())
