"""Проверка системы компоновки без окна и без GPU.

Бриф требует десять свойств (пункт 6). Здесь каждое проверяется числом,
а не на глаз, и на четырёх размерах окна плюс четырёх масштабах DPI —
именно тех, что названы в брифе.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flybrain.desk import layout as L
from flybrain.desk.tokens import S, Scale

SIZES = ((1280, 720), (1500, 880), (1920, 1080), (2560, 1440))
DPIS = (1.0, 1.25, 1.5, 2.0)

fails = []


def check(name, ok, info=""):
    mark = "ok " if ok else "ПРОВАЛ"
    print("  " + mark + " " + name + (("  " + info) if info else ""))
    if not ok:
        fails.append(name)


def zones(sh):
    return [(n, getattr(sh, n)) for n in
            ("topbar", "inspector", "centre", "secondary", "timeline",
             "footer") if sh.has(n)]


def main():
    print("1-2. закрепление и пределы, 10. изменение размера окна")
    for dpi in DPIS:
        sc = Scale(dpi=dpi)
        for (w, h) in SIZES:
            sc.set_size(w, h)
            st = L.DockState()
            sh = L.compute(w, h, sc, st)
            zs = zones(sh)
            # 7. ничего не перекрывается
            bad = [(a, b) for i, (a, ra) in enumerate(zs)
                   for (b, rb) in zs[i + 1:] if ra.overlaps(rb)]
            # ничего не выходит за окно
            out = [n for n, r in zs
                   if r.x < 0 or r.y < 0 or r.x1 > w or r.y1 > h]
            # вся высота покрыта без щелей
            cover = sum(r.h for n, r in zs if n in
                        ("topbar", "centre", "timeline", "footer"))
            tag = f"dpi {dpi:g} {w}x{h}"
            check(tag + " без наложений", not bad, str(bad))
            check(tag + " в границах", not out, str(out))
            check(tag + " высота сомкнута", cover == h,
                  f"{cover} из {h}")
            # высота верхней полосы не зависит от размера окна при том же dpi
            check(tag + " верх закреплён",
                  sh.topbar.h == sc.px(S.topbar),
                  f"{sh.topbar.h} = {sc.px(S.topbar)}")

    print("3. ширина боковых панелей меняется и зажата пределами")
    sc = Scale(dpi=1.0)
    sc.set_size(1600, 900)
    st = L.DockState()
    L.start_drag(st, sc, "inspector", 280, 400)
    L.drag_splitter(st, sc, "inspector", 280 + 900, 400)
    check("рейка не шире максимума", st.inspector_w == S.inspector_max,
          f"{st.inspector_w:.0f} = {S.inspector_max}")
    L.start_drag(st, sc, "inspector", 420, 400)
    L.drag_splitter(st, sc, "inspector", 420 - 900, 400)
    check("рейка не уже минимума", st.inspector_w == S.inspector_min,
          f"{st.inspector_w:.0f} = {S.inspector_min}")
    L.start_drag(st, sc, "secondary", 1300, 400)
    L.drag_splitter(st, sc, "secondary", 1300 - 60, 400)
    check("правая колонка тянется в обратную сторону",
          st.secondary_w == S.secondary + 60, f"{st.secondary_w:.0f}")
    st2 = L.DockState()
    sc2 = Scale(dpi=2.0)
    sc2.set_size(1600, 900)
    L.start_drag(st2, sc2, "inspector", 560, 400)
    L.drag_splitter(st2, sc2, "inspector", 560 + 100, 400)
    check("8. при dpi 2 ход курсора в 100 px даёт 50 логических",
          abs(st2.inspector_w - (S.inspector + 50)) < 0.01,
          f"{st2.inspector_w:.1f}")

    print("9. компоновка сохраняется между переключениями и разворотом")
    st = L.DockState(inspector_w=333.0, secondary_w=251.0, timeline_h=190.0)
    before = (st.inspector_w, st.secondary_w, st.timeline_h)
    st.maximized = True
    sh = L.compute(1600, 900, sc, st)
    only_centre = sh.has("centre") and not any(
        sh.has(n) for n in ("inspector", "secondary", "timeline"))
    check("разворот оставляет только центр", only_centre,
          str([n for n, _ in zones(sh)]))
    check("разворот не трогает размеры зон",
          (st.inspector_w, st.secondary_w, st.timeline_h) == before)
    st.maximized = False
    sh = L.compute(1600, 900, sc, st)
    check("возврат восстанавливает раскладку",
          sh.inspector.w == sc.px(333.0)
          and sh.secondary.w == sc.px(251.0)
          and sh.timeline.h == sc.px(190.0),
          f"{sh.inspector.w}/{sh.secondary.w}/{sh.timeline.h}")

    print("узкое окно: зоны отключаются по порядку, а не сжимаются")
    sc.set_size(820, 560)
    st = L.DockState()
    sh = L.compute(820, 560, sc, st)
    check("на 820 px правая колонка ушла",
          not sh.has("secondary") and "secondary" in sh.dropped,
          str(sh.dropped))
    check("центр не уже минимума", sh.centre.w >= sc.px(L.CENTRE_MIN_W),
          f"{sh.centre.w} >= {sc.px(L.CENTRE_MIN_W)}")
    sc.set_size(560, 420)
    sh = L.compute(560, 420, sc, st)
    check("на 560 px ушла и рейка", not sh.has("inspector"),
          str(sh.dropped))
    check("центру осталась высота", sh.centre.h > 0, str(sh.centre))

    print("свёрнутые зоны")
    sc.set_size(1600, 900)
    st = L.DockState()
    st.toggle("timeline")
    sh = L.compute(1600, 900, sc, st)
    check("свёрнутая нижняя полоса отдаёт высоту центру",
          not sh.has("timeline")
          and sh.centre.h == 900 - sc.px(S.topbar) - sc.px(S.footer),
          f"{sh.centre.h}")
    check("разделителя свёрнутой зоны нет",
          "timeline" not in sh.splitters)

    print("4. независимый скролл")
    a, b = L.ScrollState(), L.ScrollState()
    a.content, a.view = 1400, 500
    b.content, b.view = 600, 500
    a.by(300)
    check("прокрутка одной не двигает другую",
          a.offset == 300 and b.offset == 0)
    a.by(5000)
    check("прокрутка зажата по содержимому", a.offset == 900,
          f"{a.offset:.0f} = 1400-500")
    b.by(-100)
    check("вверх не уходит за ноль", b.offset == 0)
    c = L.ScrollState(offset=400.0, content=1000, view=900)
    c.clamp()
    check("рост области подтягивает прокрутку", c.offset == 100,
          f"{c.offset:.0f}")

    print("курсор выкладки")
    st_ = L.Stack(L.Rect(0, 0, 200, 100), gap=8)
    took = [st_.take(30) for _ in range(5)]
    check("5. курсор не отдаёт коробок за нижним краем",
          all(r.y1 <= 100 for r in took),
          str([r.y1 for r in took]))
    check("исчерпанный курсор отдаёт пустое",
          took[-1].h == 0 and not st_.fits(1), str(took[-1]))
    g = L.Rect(0, 0, 300, 50)
    cols = g.cols(3, gap=10)
    check("колонки смыкаются с зазором",
          cols[0].w == 93 and cols[-1].x1 == 300,
          str([(c.x, c.w) for c in cols]))

    print("вписывание кадра по отношению сторон")
    fit = L.Rect(0, 0, 1000, 300).fit_aspect(4 / 3)
    check("кадр не растягивается за панелью",
          fit.h == 300 and fit.w == 400 and fit.cx == 500,
          str(fit))
    fit = L.Rect(0, 0, 300, 1000).fit_aspect(4 / 3)
    check("и по другой оси тоже", fit.w == 300 and fit.h == 225, str(fit))

    print("6. укладка и перенос текста")

    class FakeAtlas:
        def width(self, s):
            return len(s) * 10

    at = FakeAtlas()
    lines = L.wrap_text(at, "зонд стороны по семейству нейронов", 120)
    check("перенос по словам в заданную ширину",
          all(at.width(x) <= 120 for x in lines) and len(lines) >= 3,
          str(lines))
    long = L.wrap_text(at, "каталогизированныйидентификатор", 100)
    check("одно длинное слово режется по символам",
          all(at.width(x) <= 100 for x in long) and len(long) >= 4,
          str(long))
    check("пустое даёт пустое", L.wrap_text(at, "", 100) == []
          and L.wrap_text(at, "текст", 0) == [])

    print("попадание в разделитель")
    sh = L.compute(1600, 900, sc, L.DockState())
    sx = sh.splitters["inspector"]
    check("курсор на полосе рейки опознан",
          L.hit_splitter(sh, sx.cx, sx.cy) == "inspector")
    check("курсор в центре не на полосе",
          L.hit_splitter(sh, sh.centre.cx, sh.centre.cy) == "")

    print()
    if fails:
        print("ПРОВАЛОВ: " + str(len(fails)))
        for f in fails:
            print("  - " + f)
        return 1
    print("все проверки пройдены")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
