"""Пункты 3 и 5: перехват событий и длинное имя популяции.

Работник не поднимается: проверяются попадание щелчка, подсказки и
усечение подписи, а не счёт. Кадр тела подставляется искусственный, чтобы
во вьюпорте была картинка — именно она в прошлом интерфейсе съедала
полосу инструментов, и проверять перекрытие надо с ней.

Щелчок подставляется как в брифе: курсор в центр детали, кадр с
`set_mouse(x, y, True)`, кадр с `set_mouse(x, y, False)`; нажатие
считается по отпусканию.

    cd D:\\fly-brain\\lab && ..\\.venv\\Scripts\\python.exe probe\\bodyfinal_events.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT = Path(__file__).parent

LONG = ("DNa02_very_long_population_name_for_layout_test_"
        "LAL_descending_left_right_hemisphere")


def main():
    from flybrain.desk import layout as L
    from flybrain.desk.app import App
    from flybrain.desk.shell import Offscreen
    from flybrain.desk.tokens import S
    from probe.app_shot import FakeSession

    sh = Offscreen((1500, 880))
    ses = FakeSession()
    ses.alive = True
    ses.ready = True
    # Кадр тела: тёплое пятно посередине, как у настоящего.
    img = np.zeros((720, 960, 3), np.uint8)
    img[:, :, 2] = 40
    img[260:460, 380:580, 0] = 220
    img[260:460, 380:580, 1] = 150
    ses.body_frame = img
    ses.body_seq = 7
    app = App(sh, ses)
    app.st.worker.device = "cuda"
    app.st.worker.n_neurons = 138639
    app.st.worker.readouts = ("ALPN", "DNa02", LONG, "LAL")
    app.st.readout = "ALPN"
    app.st.note("окно открыто")

    # --- запись подсказок: оборачиваем рисование, ничего не меняя в коде ---
    tips = []
    orig = sh.ctx._draw_tip

    def spy(r, text):
        tips.append(text)
        return orig(r, text)

    sh.ctx._draw_tip = spy

    rep = {"щелчки": [], "подсказки": [], "усечение": {}, "перекрытие": []}

    def frame(t=None):
        if t is not None:
            sh.t = t
        app.frame()

    def park():
        sh.inp.set_mouse(-4000, -4000, False)
        frame()

    def click(wid, rect=None):
        """Щёлкнуть по детали. Возвращает её прямоугольник или None."""
        park()
        frame()
        r = rect if rect is not None else sh.ctx.rect_of(wid)
        if r is None:
            return None
        sh.inp.set_mouse(r.cx, r.cy, False)
        frame()
        sh.inp.set_mouse(r.cx, r.cy, True)
        frame()
        sh.inp.set_mouse(r.cx, r.cy, False)
        frame()
        park()
        return r

    def case(name, wid, before, action, after, rect=None):
        b = before()
        r = action(wid, rect)
        a = after()
        ok = (a != b) if not isinstance(b, tuple) else (a != b)
        rec = {"деталь": wid, "что": name, "было": b, "стало": a,
               "прямоугольник": None if r is None else
               [r.x, r.y, r.w, r.h], "сработало": bool(ok)}
        rep["щелчки"].append(rec)
        print("  %-22s %-34s %s -> %s  %s"
              % (wid, name, b, a, "ок" if ok else "НЕ СРАБОТАЛО"
                 if r is not None else "НЕТ ДЕТАЛИ"))
        return ok

    cam = app.st.camera("body")
    dock = app.st.ws.dock()

    print("\n1. полоса инструментов вьюпорта")
    for i, key in enumerate(("back", "side", "top", "front")):
        cam.view = "back" if key != "back" else "front"
        case("вид " + key, "body/tb/view/%d" % i,
             lambda: cam.view, click, lambda: cam.view)
    cam.view = "back"
    cam.dist = 1.0
    case("приблизить", "body/tb/zoom_in", lambda: round(cam.dist, 4),
         click, lambda: round(cam.dist, 4))
    case("отдалить", "body/tb/zoom_out", lambda: round(cam.dist, 4),
         click, lambda: round(cam.dist, 4))
    case("сброс вида", "body/tb/reset",
         lambda: (round(cam.dist, 4), cam.free), click,
         lambda: (round(cam.dist, 4), cam.free))
    case("развернуть", "body/tb/full", lambda: dock.maximized, click,
         lambda: dock.maximized)
    if dock.maximized:
        click("body/tb/full")

    print("\n2. кнопки зон в верхней полосе")
    for z in ("inspector", "secondary", "timeline"):
        case("зона " + z, "zone/" + z,
             lambda z=z: dock.is_open(z), click,
             lambda z=z: dock.is_open(z))
        if not dock.is_open(z):
            click("zone/" + z)

    print("\n3. вкладки правой колонки")
    for i in range(4):
        app.st.ws.set_tab(0 if i else 1)
        case("вкладка %d" % i, "sec/%d" % i,
             lambda: app.st.ws.tab(), click, lambda: app.st.ws.tab())
    app.st.ws.set_tab(0)

    print("\n4. переключатель пространств")
    case("пространство", "ws/1", lambda: app.st.ws.current, click,
         lambda: app.st.ws.current)
    app.st.ws.switch("body")
    frame()

    print("\n5. рейка: кнопка, числовые поля, ползунок, отключённый тумблер")
    cam.yaw = 30.0
    cam.free = True
    case("сбросить камеру", "b/camreset", lambda: (cam.yaw, cam.free),
         click, lambda: (cam.yaw, cam.free))
    # Шаговые кнопки числового поля появляются только под курсором,
    # поэтому сперва навести, потом брать прямоугольник.
    park()
    frame()
    r = sh.ctx.rect_of("b/t")
    sh.inp.set_mouse(r.cx, r.cy, False)
    frame()
    for sub, nm in (("b/tcaret-up", "длина +"), ("b/tcaret-down", "длина -")):
        rr = sh.ctx.rect_of(sub)
        case(nm, sub, lambda: round(app.st.t_run, 3),
             click, lambda: round(app.st.t_run, 3), rect=rr)
        # вернуть курсор на поле, иначе шаговые кнопки исчезнут
        sh.inp.set_mouse(r.cx, r.cy, False)
        frame()
    park()
    # Ползунок: тянем в правый край дорожки.
    before = round(cam.dist, 3)
    rs = sh.ctx.rect_of("b/dist")
    sh.inp.set_mouse(rs.x + rs.w // 2, rs.cy, True)
    frame()
    sh.inp.set_mouse(rs.x1 - 2, rs.cy, True)
    frame()
    sh.inp.set_mouse(rs.x1 - 2, rs.cy, False)
    frame()
    park()
    rep["щелчки"].append({"деталь": "b/dist", "что": "тянуть ползунок",
                          "было": before, "стало": round(cam.dist, 3),
                          "сработало": round(cam.dist, 3) != before})
    print("  %-22s %-34s %s -> %s  %s"
          % ("b/dist", "тянуть ползунок", before, round(cam.dist, 3),
             "ок" if round(cam.dist, 3) != before else "НЕ СРАБОТАЛО"))
    cam.reset()
    # Отключённый тумблер: щелчок обязан НЕ сработать.
    park()
    frame()
    rt = sh.ctx.rect_of("b/contacts")
    got = click("b/contacts", rt)
    rep["щелчки"].append({"деталь": "b/contacts",
                          "что": "отключённый тумблер (щелчок не должен "
                                 "менять состояние)",
                          "прямоугольник": None if rt is None else
                          [rt.x, rt.y, rt.w, rt.h],
                          "сработало": False, "ожидалось": "без изменений"})
    print("  %-22s %-34s прямоугольник %s, состояние не меняется"
          % ("b/contacts", "отключённый тумблер",
             None if rt is None else (rt.w, rt.h)))

    print("\n6. раскрытый список выхода на рулёжку")
    park()
    frame()
    rc = sh.ctx.rect_of("b/out")
    click("b/out", rc)
    frame()
    opened = sh.ctx.popup
    print("  список раскрыт: %r" % opened)
    # Прямоугольники пунктов контекст не запоминает (combo не зовёт probe
    # для пунктов), поэтому считаем их так же, как их считает combo.
    sc = sh.sc
    opts = list(app.st.worker.readouts)
    row = sc.px(S.row)
    ph = row * min(9, len(opts)) + sc.px(S.xs) * 2
    below = rc.y1 + sc.px(S.xs)
    pr = L.Rect(rc.x, below if below + ph <= sc.h else
                max(0, rc.y - ph - sc.px(S.xs)), rc.w, ph)
    area = pr.inset(sc.px(S.xs))
    rep["пункты списка"] = {"rect_of('b/out/o0')":
                            sh.ctx.rect_of("b/out/o0") is not None}
    for i, opt in enumerate(opts):
        if opt == app.st.readout:
            continue
        cell = L.Rect(area.x, area.y + i * row, area.w, row)
        was = app.st.readout
        # список уже раскрыт предыдущим щелчком по полю
        if sh.ctx.popup != "b/out":
            click("b/out", rc)
            frame()
        sh.inp.set_mouse(cell.cx, cell.cy, False)
        frame()
        sh.inp.set_mouse(cell.cx, cell.cy, True)
        frame()
        sh.inp.set_mouse(cell.cx, cell.cy, False)
        frame()
        park()
        ok = app.st.readout == opt
        rep["щелчки"].append({"деталь": "b/out/o%d" % i, "что": "пункт " + opt,
                              "было": was, "стало": app.st.readout,
                              "сработало": ok})
        print("  %-22s %-34s %s -> %s  %s"
              % ("b/out/o%d" % i, "пункт " + opt[:28], was[:18],
                 app.st.readout[:28], "ок" if ok else "НЕ СРАБОТАЛО"))

    print("\n7. перекрытие: щелчок по вьюпорту под полосой инструментов")
    # Камера не должна поворачиваться от нажатия на полосу, и наоборот —
    # нажатие по пустому вьюпорту должно её поворачивать.
    park()
    frame()
    tbr = sh.ctx.rect_of("body/tb/reset")
    vw = sh.ctx.rect_of("body/view")
    before = (round(cam.yaw, 3), round(cam.pitch, 3))
    sh.inp.set_mouse(tbr.cx, tbr.cy, True)
    frame()
    sh.inp.set_mouse(tbr.cx + 40, tbr.cy + 30, True)
    frame()
    sh.inp.set_mouse(tbr.cx + 40, tbr.cy + 30, False)
    frame()
    park()
    after_bar = (round(cam.yaw, 3), round(cam.pitch, 3))
    cam.reset()
    frame()
    sh.inp.set_mouse(vw.cx, vw.cy, True)
    frame()
    sh.inp.set_mouse(vw.cx + 50, vw.cy + 20, True)
    frame()
    sh.inp.set_mouse(vw.cx + 50, vw.cy + 20, False)
    frame()
    park()
    after_view = (round(cam.yaw, 3), round(cam.pitch, 3))
    rep["перекрытие"] = [
        {"что": "протяжка, начатая на полосе инструментов",
         "yaw,pitch": after_bar, "ожидание": "вращение не должно начаться"},
        {"что": "протяжка по пустому вьюпорту", "yaw,pitch": after_view,
         "ожидание": "вращение должно начаться"}]
    print("  с полосы: yaw,pitch = %s (ждём 0,0)" % (after_bar,))
    print("  по виду:  yaw,pitch = %s (ждём не 0)" % (after_view,))
    cam.reset()

    print("\n8. колесо над вьюпортом и над раскрытым списком")
    frame()
    d0 = round(cam.dist, 4)
    sh.inp.set_mouse(vw.cx, vw.cy, False)
    sh.inp.wheel = 2.0
    frame()
    d1 = round(cam.dist, 4)
    print("  колесо над видом: %s -> %s" % (d0, d1))
    rep["колесо"] = {"над видом": [d0, d1]}
    cam.reset()
    park()

    print("\n9. подсказки: наведение на каждую деталь")
    wanted = ("tl/run", "body/tb/zoom_in", "body/tb/zoom_out",
              "body/tb/reset", "body/tb/full", "body/tb/view/0",
              "body/tb/view/1", "body/tb/view/2", "body/tb/view/3",
              "zone/inspector", "zone/secondary", "zone/timeline",
              "b/out", "b/dist", "b/t", "b/seed", "b/camreset",
              "b/contacts", "b/run/info", "b/cam/info", "b/layers/info")
    for wid in wanted:
        park()
        sh.t = 0.3
        frame()
        r = sh.ctx.rect_of(wid)
        if r is None:
            rep["подсказки"].append({"деталь": wid, "есть": False,
                                     "подсказка": None})
            print("  %-20s НЕТ ДЕТАЛИ" % wid)
            continue
        tips.clear()
        sh.inp.set_mouse(r.cx, r.cy, False)
        sh.t = 0.35
        frame()
        sh.t = 2.5           # задержка подсказки прошла
        frame()
        got = tips[-1] if tips else None
        rep["подсказки"].append({"деталь": wid, "есть": True,
                                 "подсказка": got})
        print("  %-20s %s" % (wid, (got[:62] + "…") if got and len(got) > 62
                              else (got or "ПОДСКАЗКИ НЕТ")))
    park()

    print("\n10. длинное имя популяции (пункт 5)")
    app.st.readout = LONG
    frame()
    frame()
    rc = sh.ctx.rect_of("b/out")
    at = sh.fonts.body
    avail = max(0, rc.w - sh.sc.px(S.sm) * 3 - sh.sc.px(12))
    shown = at.fit(LONG, avail)
    rep["усечение"] = {
        "полное имя": LONG,
        "длина полного, символов": len(LONG),
        "ширина полного, px": at.width(LONG),
        "ширина поля, px": rc.w,
        "доступно под текст, px": avail,
        "что влезает": shown,
        "усечено": shown != LONG,
        "st.readout после отрисовки": app.st.readout,
        "st.readout == полное имя": app.st.readout == LONG,
    }
    for k, v in rep["усечение"].items():
        print("  %-28s %s" % (k, v))
    # Подсказка над полем: содержит ли она полное имя
    tips.clear()
    sh.inp.set_mouse(rc.cx, rc.cy, False)
    sh.t = 0.35
    frame()
    sh.t = 2.5
    frame()
    tip_text = tips[-1] if tips else None
    rep["усечение"]["подсказка при наведении"] = tip_text
    rep["усечение"]["полное имя есть в подсказке"] = bool(
        tip_text and LONG in tip_text)
    print("  подсказка при наведении: %r" % tip_text)
    print("  полное имя есть в подсказке: %s"
          % rep["усечение"]["полное имя есть в подсказке"])
    # То же в раскрытом списке: усечён ли пункт
    park()
    frame()
    click("b/out", rc)
    frame()
    area_w = rc.w - sh.sc.px(S.xs) * 2
    rep["усечение"]["ширина пункта списка, px"] = area_w - sh.sc.px(S.md)
    rep["усечение"]["пункт списка усечён"] = (
        at.width(LONG) > area_w - sh.sc.px(S.md))
    print("  пункт списка усечён: %s"
          % rep["усечение"]["пункт списка усечён"])
    sh.ctx.popup = ""
    park()
    frame()
    sh.shot(OUT / "bodyfinal_long_readout.png")

    print("\n11. полоса инструментов при тесном центре")
    rep["полоса при разных размерах"] = []
    for (w, h, dpi) in ((1500, 880, 1.0), (1280, 720, 1.0),
                        (1500, 880, 1.5), (1920, 1080, 1.0),
                        (1920, 1080, 2.0)):
        sh.resize(w, h, dpi)
        park()
        frame()
        frame()
        have = [n for n in ("view/0", "view/1", "view/2", "view/3",
                            "zoom_out", "zoom_in", "reset", "full")
                if sh.ctx.rect_of("body/tb/" + n) is not None]
        run_btn = sh.ctx.rect_of("tl/run") is not None
        rec = {"размер": [w, h], "dpi": dpi, "кнопок полосы": len(have),
               "есть": have, "кнопка пуска доступна": run_btn}
        rep["полоса при разных размерах"].append(rec)
        print("  %dx%d dpi%.2f: %d/8 кнопок %s; пуск: %s"
              % (w, h, dpi, len(have), have, run_btn))
    sh.resize(1500, 880, 1.0)

    # Та же проверка, но зона исполнения спрятана вручную.
    dock.timeline_open = False
    park()
    frame()
    frame()
    rep["пуск при спрятанной полосе исполнения"] = (
        sh.ctx.rect_of("tl/run") is not None)
    print("  полоса исполнения спрятана: кнопка пуска доступна: %s"
          % rep["пуск при спрятанной полосе исполнения"])
    dock.timeline_open = True

    bad_click = [c for c in rep["щелчки"]
                 if not c.get("сработало") and "не должен" not in c.get("что", "")]
    no_tip = [t for t in rep["подсказки"] if not t["подсказка"]]
    print("\nитог: щелчков проверено %d, не сработало %d; "
          "подсказок проверено %d, нет подсказки у %d"
          % (len(rep["щелчки"]), len(bad_click), len(rep["подсказки"]),
             len(no_tip)))
    for c in bad_click:
        print("  !! не сработало: %s / %s" % (c["деталь"], c["что"]))
    for t in no_tip:
        print("  !! без подсказки: %s (деталь есть: %s)"
              % (t["деталь"], t["есть"]))
    rep["итог"] = {"щелчков": len(rep["щелчки"]), "не сработало": len(bad_click),
                   "подсказок": len(rep["подсказки"]),
                   "без подсказки": len(no_tip)}
    (OUT / "bodyfinal_events.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("отчёт: probe/bodyfinal_events.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
