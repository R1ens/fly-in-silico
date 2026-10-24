"""Тестовый экран компонентов.

Бриф требует экран, на котором видно каждую деталь и каждое её состояние.
Здесь он и собран, и работает двумя путями:

* `--shot` — без окна: поверхность кадра создаётся нужного размера,
  экран рисуется, накопитель читается в PNG. Так проверяются все
  требуемые размеры окна и все масштабы DPI, не завися от того, что окно
  в этой среде меняет размер само.
* `--live` — живое окно: то же самое, но можно наводить, нажимать и
  набирать. Esc закрывает; `LOOK_SECONDS=20` закрывает сама.

Матрица состояний нарисована честно: курсор и фокус подставляются той же
структуре ввода, которой пользуется настоящая мышь, поэтому «под
курсором» на снимке — это ровно то, что увидит владелец, а не отдельная
краска для картинки.

    cd D:\\fly-brain\\lab
    ..\\.venv\\Scripts\\python.exe probe\\gallery.py --shot
    ..\\.venv\\Scripts\\python.exe probe\\gallery.py --live
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flybrain.desk import widgets as W                          # noqa: E402
from flybrain.desk import icons                                 # noqa: E402
from flybrain.desk.canvas import Canvas                         # noqa: E402
from flybrain.desk.layout import Rect, Stack                    # noqa: E402
from flybrain.desk.textinput import QUERY, TextInput            # noqa: E402
from flybrain.desk.tokens import C, R, S, Scale, T              # noqa: E402

OUT = Path(__file__).parent
SIZES = ((1280, 720), (1500, 880), (1920, 1080))
DPIS = (1.0, 1.25, 1.5, 2.0)
STATES = ("обычное", "под курсором", "нажатое", "в фокусе", "отключено",
          "выбрано")


class Demo:
    """Состояние экрана, которое живёт между кадрами."""

    def __init__(self):
        self.query = TextInput("class:gustatory+side:left", allow=QUERY)
        self.name = TextInput("")
        self.t_run = 0.8
        self.rate = 180.0
        self.bins = 10.0
        self.gain = 0.42
        self.seg = 1
        self.tab = 0
        self.cam = 2
        self.combo = 3
        self.on_decay = True
        self.on_only = False
        self.view = "play"
        self.chips = {"AL": True, "MB": False, "LAL": True, "CX": False}


# ------------------------------------------------------------ состояния

def _force(ctx, state, r, wid):
    """Подставить ввод так, чтобы деталь оказалась в нужном состоянии."""
    inp = ctx.inp
    inp.mx, inp.my, inp.down = -4000, -4000, False
    ctx.active = ""
    if ctx.focus == wid:
        ctx.focus = ""
    if state == "под курсором":
        inp.mx, inp.my = r.cx, r.cy
        ctx._hover_since[wid] = ctx.now        # подсказку не показываем
    elif state == "нажатое":
        inp.mx, inp.my, inp.down = r.cx, r.cy, True
        ctx.active = wid
    elif state == "в фокусе":
        ctx.focus = wid


def state_matrix(ctx, area, demo):
    """Шесть состояний на каждую деталь, в столбцах."""
    sc, f = ctx.sc, ctx.f
    rows = (
        ("Кнопка", lambda r, i, st: W.button(
            ctx, "m/b%d" % i, r, "Пуск", kind="primary",
            enabled=(st != "отключено"), selected=(st == "выбрано"))),
        ("Кнопка, обычная", lambda r, i, st: W.button(
            ctx, "m/d%d" % i, r, "Отчёт", icon="chart",
            enabled=(st != "отключено"), selected=(st == "выбрано"))),
        ("Кнопка, тихая", lambda r, i, st: W.button(
            ctx, "m/g%d" % i, r, "Сброс", kind="ghost",
            enabled=(st != "отключено"), selected=(st == "выбрано"))),
        ("Кнопка, отказ", lambda r, i, st: W.button(
            ctx, "m/x%d" % i, r, "Стоп", kind="danger",
            enabled=(st != "отключено"), selected=(st == "выбрано"))),
        ("Иконка", lambda r, i, st: W.icon_button(
            ctx, "m/i%d" % i, Rect(r.x, r.y, r.h, r.h), "camera",
            enabled=(st != "отключено"), selected=(st == "выбрано"))),
        ("Переключатель", lambda r, i, st: W.toggle(
            ctx, "m/t%d" % i, r, st == "выбрано", "след",
            enabled=(st != "отключено"))),
        ("Поле ввода", lambda r, i, st: W.text_field(
            ctx, "m/f%d" % i, r, demo.name, "имя прогона",
            enabled=(st != "отключено"))),
        ("Числовое поле", lambda r, i, st: W.numeric(
            ctx, "m/n%d" % i, r, 0.8, 0.1, 60.0, 0.1, "с",
            enabled=(st != "отключено"))),
        ("Список", lambda r, i, st: W.combo(
            ctx, "m/c%d" % i, r, ["AL", "MB", "LAL"], 1,
            enabled=(st != "отключено"))),
        ("Ползунок", lambda r, i, st: W.slider(
            ctx, "m/s%d" % i, r, 0.42, 0.0, 1.0, "усиление",
            enabled=(st != "отключено"))),
        ("Чип", lambda r, i, st: W.chip(
            ctx, "m/h%d" % i, r, "side:left", closable=True,
            enabled=(st != "отключено"), selected=(st == "выбрано"))),
    )

    at = ctx.f.micro
    lab_w = max(ctx.f.small.width(n) for n, _ in rows) + sc.px(S.md)
    head, body = area.cut_top(at.gh + sc.px(S.xs), sc.px(S.sm))
    grid = Rect(body.x + lab_w, body.y, body.w - lab_w, body.h)
    cols = grid.cols(len(STATES), gap=sc.px(S.sm))
    for cell, name in zip(cols, STATES):
        ctx.dl.text(at, at.fit(name, cell.w), cell.x,
                    head.y + head.h - at.gh, C.faint)

    rh = sc.px(S.control)
    st_ = Stack(body, gap=sc.px(S.sm))
    used = head.h + sc.px(S.sm)
    saved = (ctx.inp.mx, ctx.inp.my, ctx.inp.down, ctx.focus, ctx.active)
    for ri, (name, fn) in enumerate(rows):
        if not st_.fits(rh):
            break
        row = st_.take(rh)
        used += rh + sc.px(S.sm)
        ctx.dl.text(ctx.f.small, name, row.x,
                    row.y + (row.h - ctx.f.small.gh) // 2, C.dim,
                    max_px=lab_w - sc.px(S.md))
        for ci, state in enumerate(STATES):
            cell = Rect(cols[ci].x, row.y, cols[ci].w, row.h)
            wid = "m/%d/%d" % (ri, ci)
            _force(ctx, state, cell, wid)
            fn(cell, ri * 10 + ci, state)
    (ctx.inp.mx, ctx.inp.my, ctx.inp.down, ctx.focus,
     ctx.active) = saved
    return used


# ------------------------------------------------------------ живой блок

def live_block(ctx, area, demo):
    """То же самое, но настоящими руками: здесь всё работает."""
    sc, f = ctx.sc, ctx.f
    st = Stack(area, gap=sc.px(S.md))

    W.section(ctx, st, "Сборка прогона")
    row = st.take(sc.px(S.control))
    demo.seg = W.segmented(ctx, "g/seg", row,
                           ["Стимуляция", "Арена", "Тело", "Зонд"],
                           demo.seg)
    row = st.take(sc.px(S.control))
    W.text_field(ctx, "g/q", row, demo.query, "селектор", icon="search",
                 tip="class:gustatory+side:left")
    row = st.take(sc.px(S.control))
    half = row.cols(2, gap=sc.px(S.sm))
    demo.t_run = W.numeric(ctx, "g/t", half[0], demo.t_run, 0.1, 120.0,
                           0.1, "с", label="длина")
    demo.bins = W.numeric(ctx, "g/bin", half[1], demo.bins, 1.0, 100.0,
                          1.0, "мс", label="бин")
    row = st.take(sc.px(S.control))
    demo.combo = W.combo(ctx, "g/cb", row,
                         ["AL — усик", "MB — грибовидное тело",
                          "LAL — нижнее боковое", "CX — центральный",
                          "AMMC", "ATL", "SEZ", "LOP", "ME", "LO",
                          "AVLP", "PVLP"], demo.combo)
    row = st.take(sc.px(38))
    demo.rate = W.slider(ctx, "g/r", row, demo.rate, 0.0, 400.0,
                         "частота стимула", "Гц", fmt="{:.0f}", step=5)
    row = st.take(sc.px(38))
    demo.gain = W.slider(ctx, "g/g", row, demo.gain, 0.0, 1.0,
                         "усиление карты", fmt="{:.2f}")

    row = st.take(sc.px(S.row))
    demo.on_decay = W.toggle(ctx, "g/dk", row, demo.on_decay,
                             "гасить след после прогона")
    row = st.take(sc.px(S.row))
    demo.on_only = W.toggle(ctx, "g/oa", row, demo.on_only,
                            "показывать только активные")

    W.rule(ctx, st)
    row = st.take(sc.px(S.row))
    x = row.x
    for name, on in list(demo.chips.items()):
        w = f.small.width(name) + sc.px(S.xl) + sc.px(S.md)
        cell = Rect(x, row.y, min(w, max(0, row.x1 - x)), row.h)
        if cell.w < sc.px(S.xl):
            break
        hit, closed = W.chip(ctx, "g/ch" + name, cell, name,
                             selected=on, closable=True)
        if hit:
            demo.chips[name] = not on
        if closed:
            demo.chips.pop(name, None)
        x = cell.x1 + sc.px(S.sm)

    row = st.take(sc.px(S.control_lg))
    cells = row.cols(3, gap=sc.px(S.sm))
    W.button(ctx, "g/run", cells[0], "Пуск", kind="primary", icon="play")
    W.button(ctx, "g/stop", cells[1], "Стоп", kind="danger", icon="stop")
    W.button(ctx, "g/rep", cells[2], "Отчёт", icon="chart")
    return st.used


# ------------------------------------------------------------ правая часть

def meters_block(ctx, area, demo):
    sc, f = ctx.sc, ctx.f
    st = Stack(area, gap=sc.px(S.sm))

    demo.tab = W.tabs(ctx, "r/tabs", st.take(sc.px(S.row + 6)),
                      ["Показания", "Области", "Журнал"], demo.tab,
                      counts=(None, 36, 12))
    st.skip(sc.px(S.sm))

    if demo.tab == 0:
        row = st.take(sc.px(S.row))
        W.status(ctx, row, "run", pulse=True)
        st.skip(sc.px(S.xs))
        row = st.take(ctx.f.metric_big.gh + ctx.f.small.gh + sc.px(S.xs))
        third = row.cols(3, gap=sc.px(S.sm))
        W.metric(ctx, third[0], "4,17", "модельное, с", big=True)
        W.metric(ctx, third[1], "11,8", "по стене, с", big=True)
        W.metric(ctx, third[2], "0,35", "доля реального", big=True,
                 color=C.warn)
        W.rule(ctx, st)
        for k, v, u in (("нейронов активно", "9 142", "шт"),
                        ("спайков в бине", "255", "шт"),
                        ("шагов движка", "1 167", "1/с"),
                        ("кадров тела", "30", "1/с"),
                        ("память работника", "1 284", "МБ")):
            if not st.fits(sc.px(S.row)):
                break
            W.kv(ctx, st.take(sc.px(S.row)), k, v, u)
        W.rule(ctx, st)
        if st.fits(sc.px(34)):
            W.progress(ctx, st.take(sc.px(34)), 0.62, "прогон")
        if st.fits(sc.px(34)):
            W.progress(ctx, st.take(sc.px(34)), 0.0,
                       "подготовка: загрузка коннектома",
                       indeterminate=True)
    elif demo.tab == 1:
        area2 = st.take_all()
        content, token = W.scroll(ctx, "r/reg", area2)
        inner = Stack(content, gap=sc.px(S.xs))
        names = ("AL", "AMMC", "ATL", "AVLP", "CAN", "CX", "FLA", "GNG",
                 "GOR", "IB", "ICL", "LAL", "LH", "LO", "LOP", "ME",
                 "MB", "NO", "OCG", "PB", "PLP", "PRW", "PVLP", "SAD",
                 "SCL", "SEZ", "SIP", "SLP", "SMP", "SPS", "VES", "WED")
        for i, n in enumerate(names):
            row = inner.take(ctx.sc.px(S.row))
            v = (len(n) * 37 + i * 11) % 400
            frac = v / 400.0
            ctx.dl.rect(row.x, row.y, row.x + int(row.w * frac), row.y1,
                        C.plot, 0.22, 2.0)
            ctx.dl.text(ctx.f.body, n, row.x + sc.px(S.sm),
                        row.y + (row.h - ctx.f.body.gh) // 2, C.ink)
            ctx.dl.text_right(ctx.f.metric, str(v), row.x1 - sc.px(S.sm),
                              row.y + (row.h - ctx.f.metric.gh) // 2,
                              C.dim)
        W.scroll_done(ctx, "r/reg", area2, inner.used, token)
    else:
        area2 = st.take_all()
        content, token = W.scroll(ctx, "r/log", area2)
        inner = Stack(content, gap=sc.px(2))
        lines = [
            "работник: коннектом 138 639 нейронов, 15 091 983 связи",
            "работник: веса на устройстве cuda, 302 МБ",
            "прогон exp-0007: class:gustatory+side:left, 0,8 с",
            "предупреждение: зрение не подключено к оптическим долям",
            "бин 40 из 80, 9 142 активных нейрона",
            "остановка по кнопке: усечение до 4,17 с",
            "готово: out/exp-0007, отчёт собран",
        ] * 3
        for i, s in enumerate(lines):
            h = W.label(ctx, Rect(content.x, inner.area.y, content.w,
                                  ctx.f.small.gh * 3), s, role="small",
                        color=C.dim if i % 7 != 3 else C.warn,
                        wrapped=True)
            inner.take(h, sc.px(S.xs))
        W.scroll_done(ctx, "r/log", area2, inner.used, token)


# ------------------------------------------------------------ весь экран

def draw_screen(ctx, demo, w, h, note=""):
    sc, f = ctx.sc, ctx.f
    dl = ctx.dl
    win = Rect(0, 0, w, h)

    # верхняя полоса
    top, rest = win.cut_top(sc.px(S.topbar))
    dl.rect(top.x, top.y, top.x1, top.y1, C.panel, 1.0, 0.0)
    dl.rect(top.x, top.y1 - sc.hair(), top.x1, top.y1, C.border_soft,
            1.0, 0.0)
    pad = sc.px(S.lg)
    at = f.workspace
    dl.text(at, "Компоненты", top.x + pad, top.y + (top.h - at.gh) // 2,
            C.ink)
    x = top.x + pad + at.width("Компоненты") + sc.px(S.xl)
    nav = Rect(x, top.y + (top.h - sc.px(S.control)) // 2,
               min(sc.px(400), max(0, top.x1 - x - sc.px(260))),
               sc.px(S.control))
    if nav.w > sc.px(200):
        W.segmented(ctx, "top/ws", nav,
                    ["Эксперимент", "Мозг", "Тело", "Анализ"], 2)
    right = Rect(top.x1 - sc.px(240), top.y, sc.px(228), top.h)
    W.status(ctx, right, "run", "Идёт: exp-0007", pulse=True)

    foot, rest = rest.cut_bottom(sc.px(S.footer))
    dl.rect(foot.x, foot.y, foot.x1, foot.y1, C.panel, 1.0, 0.0)
    am = f.micro
    info = note or ("%d\u00d7%d  масштаб %.2f" % (w, h, sc.k))
    dl.text(am, info, foot.x + sc.px(S.md),
            foot.y + (foot.h - am.gh) // 2, C.faint)
    dl.text_right(am, "иконок %d  кеглей %d" % (len(icons.NAMES),
                                                len(W.Fonts.ROLES)),
                  foot.x1 - sc.px(S.md), foot.y + (foot.h - am.gh) // 2,
                  C.faint)

    body = rest.inset(sc.px(S.md))
    left, right2 = body.cut_left(int(body.w * 0.62), sc.px(S.md))

    # левая колонка: матрица состояний сверху, живой блок снизу
    mh = int(left.h * 0.68)
    mat, low = left.cut_top(mh, sc.px(S.md))
    inner = W.panel(ctx, mat, "Состояния деталей",
                    right="прокручивается")
    content, token = W.scroll(ctx, "mat", inner)
    used = state_matrix(ctx, Rect(content.x, content.y, content.w,
                                  1 << 16), demo)
    W.scroll_done(ctx, "mat", inner, used, token)

    vp, tools = low.cut_top(int(low.h * 0.52), sc.px(S.md))
    dl.rect(vp.x, vp.y, vp.x1, vp.y1, C.sunken, 1.0, sc.px(R.panel))
    # сетка — чтобы было видно, что полоса инструментов плавает над видом
    step = sc.px(28)
    for gx in range(vp.x + step, vp.x1, step):
        dl.rect(gx, vp.y, gx + sc.hair(), vp.y1, C.grid, 1.0, 0.0)
    for gy in range(vp.y + step, vp.y1, step):
        dl.rect(vp.x, gy, vp.x1, gy + sc.hair(), C.grid, 1.0, 0.0)
    W.label(ctx, Rect(vp.x + sc.px(S.md), vp.y + sc.px(S.md), vp.w,
                      f.small.gh), "вьюпорт: полоса инструментов поверх",
            role="small", color=C.faint)
    hit, _picks = W.viewport_toolbar(
        ctx, "vp/tb",
        Rect(vp.x + sc.px(S.md), vp.y1 - sc.px(S.control) - sc.px(S.md) * 2,
             vp.w - sc.px(S.md) * 2, sc.px(S.control) + sc.px(S.xs) * 2),
        [("seg", "view", ["сзади", "сбоку", "сверху"], 0,
          ["камера позади", "профиль", "сверху"]), ("-",),
         ("play", "play", "пуск"), ("pause", "pause", "пауза"),
         ("stop", "stop", "стоп"), ("-",),
         ("camera", "camera", "снимок"),
         ("reset", "reset", "сбросить камеру"),
         ("layers", "layers", "слои отладки"), ("-",),
         ("full", "fullscreen", "развернуть", True)])
    if hit:
        demo.view = hit

    kinds = W.panel(ctx, tools, "Разделители и состояния задачи")
    kst = Stack(kinds, gap=sc.px(S.sm))
    row = kst.take(sc.px(S.row))
    cells = row.cols(6, gap=sc.px(S.sm))
    for cell, key in zip(cells, ("idle", "prep", "run", "cancel", "done",
                                 "fail")):
        W.status(ctx, cell, key, pulse=(key in ("run", "cancel")))
    if kst.fits(sc.px(S.lg)):
        row = kst.take(sc.px(S.lg))
        c2 = row.cols(2, gap=sc.px(S.xl))
        W.splitter_handle(ctx, "sp/h1", c2[0], axis="h")
        W.splitter_handle(ctx, "sp/h2", c2[1], axis="h", dragging=True)

    # правая колонка
    inner = W.panel(ctx, right2, "")
    up, down = inner.cut_top(int(inner.h * 0.46), sc.px(S.md))
    content, token = W.scroll(ctx, "live", up)
    used = live_block(ctx, Rect(content.x, content.y, content.w,
                                1 << 16), demo)
    W.scroll_done(ctx, "live", up, used, token)
    dl.rect(down.x, down.y - sc.px(S.sm), down.x1,
            down.y - sc.px(S.sm) + sc.hair(), C.border_soft, 1.0, 0.0)
    meters_block(ctx, down, demo)


# ------------------------------------------------------------ запуски

def shots():
    """Снимки без окна: все требуемые размеры и масштабы."""
    import numpy as np
    import taichi as ti
    from PIL import Image

    from flybrain.ui import gpu

    ti.init(arch=ti.vulkan, offline_cache=True)
    surf = gpu.Surface()
    made = []
    t_all = time.perf_counter()
    for (w, h) in SIZES:
        for dpi in DPIS:
            sc = Scale(dpi=dpi, w=w, h=h)
            fonts = W.Fonts(sc)
            inp = W.Input()
            inp.mx = inp.my = -4000
            main, over = Canvas(), Canvas()
            # Фаза мигания каретки выбрана так, чтобы на снимке она была
            # видна: иначе «в фокусе» неотличимо от обычного.
            ctx = W.Ctx(main, over, fonts, sc, inp, now=0.3)
            ctx.begin(0.3)
            surf.ensure(w, h)
            surf.clear(C.bg)
            demo = Demo()
            t0 = time.perf_counter()
            draw_screen(ctx, demo, w, h,
                        note="%d\u00d7%d  DPI %d %%  масштаб %.2f"
                             % (w, h, round(dpi * 100), sc.k))
            ctx.end()
            launches = main.launches() + over.launches()
            main.flush(surf)
            over.flush(surf)
            build = (time.perf_counter() - t0) * 1000.0
            a = np.clip(surf.acc.to_numpy(), 0.0, 1.0)
            img = (a.transpose(1, 0, 2)[::-1] * 255).astype(np.uint8)
            name = "gal_%dx%d_dpi%d.png" % (w, h, round(dpi * 100))
            Image.fromarray(img).save(OUT / name)
            made.append(name)
            print("  %-26s кадр %5.1f мс  запусков ядра %2d  кегли %s"
                  % (name, build, launches,
                     "/".join(str(sc.size(r)) for r in
                              ("body", "heading", "metric"))))
    # Отдельный снимок с раскрытым списком: закрытый список на экране
    # есть, а раскрытый иначе не покажешь.
    sc = Scale(dpi=1.0, w=1500, h=880)
    fonts = W.Fonts(sc)
    inp = W.Input()
    inp.mx = inp.my = -4000
    main, over = Canvas(), Canvas()
    ctx = W.Ctx(main, over, fonts, sc, inp, now=0.3)
    ctx.begin(0.3)
    ctx.popup = "g/cb"
    surf.ensure(1500, 880)
    surf.clear(C.bg)
    draw_screen(ctx, Demo(), 1500, 880, note="раскрытый список")
    ctx.end()
    main.flush(surf)
    over.flush(surf)
    a = np.clip(surf.acc.to_numpy(), 0.0, 1.0)
    Image.fromarray((a.transpose(1, 0, 2)[::-1] * 255).astype(np.uint8)
                    ).save(OUT / "gal_popup.png")
    made.append("gal_popup.png")
    print("  gal_popup.png              раскрытый список")
    print("снимков: %d за %.1f с" % (len(made),
                                     time.perf_counter() - t_all))
    return made


def live():
    from flybrain.desk.shell import Shell

    sh = Shell("FlyBrain — компоненты", (1600, 900))
    demo = Demo()
    limit = float(os.environ.get("LOOK_SECONDS", "0"))
    t0 = time.perf_counter()
    shot = False
    print("клавиатура:", "нативная" if sh.keys.native else sh.keys.failure)
    print("dpi:", sh.dpi_info)
    while sh.running:
        if limit and time.perf_counter() - t0 > limit:
            break
        ctx = sh.begin()
        for ev in list(ctx.inp.events):
            if getattr(ev, "name", "") == "escape":
                if ctx.focus:
                    ctx.focus = ""
                elif ctx.popup:
                    ctx.popup = ""
                else:
                    sh.stop()
        draw_screen(ctx, demo, sh.w, sh.h,
                    note="%d\u00d7%d  %.0f кадр/с  %d запусков  %.1f мс"
                         % (sh.w, sh.h, sh.fps, sh.launches, sh.frame_ms))
        sh.end()
        if not shot and time.perf_counter() - t0 > 1.5:
            print("снимок:", sh.shot(OUT / "gal_live.png"))
            print(sh.describe())
            shot = True
    sh.close()


if __name__ == "__main__":
    if "--live" in sys.argv:
        live()
    else:
        shots()
