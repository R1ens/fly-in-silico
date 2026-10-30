"""Рабочее пространство «Тело».

Центр — живой вид MuJoCo теми кадрами, что снимает работник. Кадр
приходит в общую память 960×720 и кладётся во вьюпорт **с сохранением
отношения сторон** и билинейной выборкой: при растягивании окна картинка
не искажается и не рассыпается на лестницы.

Про камеру честно: кадр рисует работник, а он делает это на шаге
управления. Значит вид поворачивается не в тот же миг, когда владелец
повёл мышью, а на ближайшем шаге — и только пока прогон идёт. В простое
на экране остаётся последний снятый кадр, и панель говорит об этом
прямо.

Про зрение честно вдвойне, и честность эта переехала. Прежде входа в
оптические доли не было вовсе, и вкладка «Зрение» говорила ровно это.
Теперь вход есть (`BodyConfig.vision`, по умолчанию выключен), и обрыв
стоит ДАЛЬШЕ — там, где он измерен: ранние зрительные нейроны (L1, L2,
L3, Tm1) отвечают, а детекторы движения и нависания (T4, T5, LPLC2,
LC4, DNp01) дают ровно 0,000 Гц. Вкладка показывает это схемой с
перечёркнутой связью и говорит отдельно, чем является сама привязка
омматидиев: она инженерная и опровергнута внешним свидетелем по
взаимному совмещению двух популяций.

Длинные пояснения живут в подсказках на значке «i», а не абзацами в
рейке: панель управления не должна быть стеной текста.
"""

from . import layout as L
from . import state as ST
from . import widgets as W
from .state import BODY_VIEWS, VISION
from .tokens import C, R, S

__all__ = ["SPACE"]

TABS = ("Показания", "Путь", "Зрение", "Журнал")
FRAME_AR = 960.0 / 720.0

CAM_HELP = ("Поворот — потянуть мышью по виду, приближение — колесом. "
            "Кадр снимает работник на шаге управления, поэтому вид "
            "откликается во время прогона, а в простое остаётся "
            "последний снятый кадр.")
LAYER_HELP = ("Выключено не для вида. Маркеры касания MuJoCo нарисованы в "
              "масштабе сцены, а муха длиной пять миллиметров — они "
              "закрывают собой кадр целиком (probe/contacts.py). Слой "
              "включим, когда подберём масштаб маркеров. Других слоёв "
              "работник пока не отдаёт.")


# ---------------------------------------------------------------- центр

def centre(app, ctx, r):
    dl, sc, f = ctx.dl, app.sh.sc, app.sh.fonts
    st = app.st
    cam = st.camera("body")
    dock = st.ws.dock()

    dl.rect(r.x, r.y, r.x1, r.y1, C.sunken, 1.0, sc.px(R.panel))
    view = r.inset(sc.px(S.sm))
    frame = app.ses.body_frame if app.ses is not None else None
    if st.run.phase == "prep":
        # В подготовке кадров нового эпизода ещё нет по определению.
        frame = None
    box = view.fit_aspect(FRAME_AR)

    if frame is not None:
        dl.image(frame, box.x, box.y, box.w, box.h)
    else:
        dl.rect(box.x, box.y, box.x1, box.y1, C.bg, 1.0, sc.px(R.control))
        mid = L.Rect(box.x + sc.px(S.xl), box.cy - sc.px(34),
                     box.w - sc.px(S.xl) * 2, sc.px(70))
        W.label(ctx, mid, "Кадров тела ещё не было", role="heading",
                color=C.dim, align="centre")
        W.label(ctx, L.Rect(mid.x + mid.w // 6,
                            mid.y + f.heading.gh + sc.px(S.sm),
                            mid.w - mid.w // 3, sc.px(70)),
                "Кадр снимает работник на шаге управления. Нажмите "
                "«Пуск» внизу: первый кадр появится после сборки тела, "
                "это около тридцати секунд.",
                role="small", color=C.faint, wrapped=True)

    # Полоса инструментов: вид — подписями, действия — значками с
    # подсказками, группы разделены.
    pad = sc.px(S.md)
    keys = [v[0] for v in BODY_VIEWS]
    names = [v[1] for v in BODY_VIEWS]
    idx = keys.index(cam.view) if cam.view in keys else 0
    tb = L.Rect(view.x + pad, view.y + pad, view.w - pad * 2,
                sc.px(S.control) + sc.px(S.xs) * 2)
    hit, picks = W.viewport_toolbar(ctx, "body/tb", tb, [
        ("seg", "view", names, idx,
         ["камера позади мухи", "профиль, головой вправо",
          "сверху, головой вверх", "лицом к мухе"]),
        ("-",),
        ("zoom_out", "minus", "отдалить"),
        ("zoom_in", "plus", "приблизить"),
        ("reset", "reset", "вернуть вид по умолчанию"),
        ("-",),
        ("full", "fullscreen",
         "свернуть вид (F11)" if dock.maximized else "развернуть вид (F11)",
         dock.maximized),
    ])
    if "view" in picks:
        cam.view = keys[picks["view"]]
        cam.yaw = cam.pitch = 0.0
        cam.free = False
    if hit == "reset":
        cam.reset()
    elif hit == "zoom_in":
        cam.zoom(0.8)
    elif hit == "zoom_out":
        cam.zoom(1.25)
    elif hit == "full":
        dock.maximized = not dock.maximized

    # Вращение мышью и приближение колесом.
    hover, held, _, _ = ctx.probe("body/view", view, True)
    # Точка отсчёта ставится в первом кадре нажатия и снимается, как
    # только кнопку отпустили. Иначе первый кадр протяжки съедал
    # разницу с тем местом, где курсор был раньше — хоть из другой зоны
    # окна, — и вид прыгал на сотни градусов.
    last = getattr(app, "_drag_last", None)
    if held:
        if last is not None:
            dx, dy = ctx.inp.mx - last[0], ctx.inp.my - last[1]
            if dx or dy:
                cam.nudge(dx * 0.4, -dy * 0.3)
        app._drag_last = (ctx.inp.mx, ctx.inp.my)
    else:
        app._drag_last = None
    if hover and ctx.inp.wheel:
        cam.zoom(0.9 ** ctx.inp.wheel)

    # Подпись вида поверх кадра: что видно и насколько это свежо.
    title = dict((k, t) for k, t, _c, _s in BODY_VIEWS).get(cam.view,
                                                            cam.view)
    bits = ["вид: " + title]
    if cam.free:
        bits.append("повёрнут вручную")
    if abs(cam.dist - 1.0) > 0.01:
        bits.append(("приближение %.2f×" % cam.dist).replace(".", ","))
    if not st.run.busy:
        bits.append("кадр обновится на следующем прогоне")
    note = "  ·  ".join(bits)
    back = ctx.layer(ctx.overlay)
    at = f.small
    cw = at.width(note) + sc.px(S.md) * 2
    cap = L.Rect(view.x + pad, view.y1 - pad - at.gh - sc.px(S.sm),
                 min(cw, view.w - pad * 2), at.gh + sc.px(S.sm))
    ctx.dl.rect(cap.x, cap.y, cap.x1, cap.y1, C.bg, 0.78, sc.px(R.control))
    W.label(ctx, L.Rect(cap.x + sc.px(S.md), cap.y + sc.px(S.xs) // 2 + 1,
                        cap.w - sc.px(S.md) * 2, at.gh), note,
            role="small", color=C.dim)
    ctx.layer(back)


# --------------------------------------------------------------- рейка

def inspector(app, ctx, r):
    sc, f = app.sh.sc, app.sh.fonts
    st = app.st
    content, token = W.scroll(ctx, "body/insp", r)
    s = L.Stack(L.Rect(content.x, content.y, content.w, 1 << 16),
                gap=sc.px(S.sm))

    W.section(ctx, s, "Прогон", first=True, wid="b/run",
              info="Эпизод в теле: мозг считает команды, MuJoCo считает "
                   "ноги. Длина — модельное время, не по стене: одна "
                   "секунда модельного идёт около минуты.")
    st.t_run = W.numeric(ctx, "b/t", s.take(sc.px(S.control)), st.t_run,
                         0.2, 120.0, 0.5, "с", label="длина",
                         tip="модельное время эпизода")
    st.seed = int(W.numeric(ctx, "b/seed", s.take(sc.px(S.control)),
                            st.seed, 0, 10000, 1, "", label="зерно",
                            fmt="{:.0f}",
                            tip="одно и то же зерно даёт тот же эпизод"))
    W.label(ctx, s.take(f.small.gh + sc.px(S.xs)), "выход на рулёжку",
            role="small", color=C.dim)
    opts = list(st.worker.readouts) or ["(работник не готов)"]
    cur = opts.index(st.readout) if st.readout in opts else 0
    i = W.combo(ctx, "b/out", s.take(sc.px(S.control)), opts, cur,
                enabled=bool(st.worker.readouts),
                tip="популяция, разницу в которой читают как поворот")
    if st.worker.readouts and opts[i] != st.readout:
        st.readout = opts[i]

    W.section(ctx, s, "Камера", wid="b/cam", info=CAM_HELP)
    cam = st.camera("body")
    cam.dist = W.slider(ctx, "b/dist", s.take(sc.px(40)), cam.dist, 0.3,
                        4.0, "отдаление", "×", fmt="{:.2f}",
                        tip="множитель к охвату, который считается от "
                            "габарита мухи")
    row = s.take(sc.px(S.control))
    half = row.cols(2, gap=sc.px(S.sm))
    if W.button(ctx, "b/camreset", half[0], "Сбросить", icon="reset",
                tip="вид по умолчанию, без ручного поворота"):
        cam.reset()
    title = dict((k, t) for k, t, _c, _sp in BODY_VIEWS).get(cam.view, "")
    W.label(ctx, L.Rect(half[1].x,
                        half[1].y + (half[1].h - f.small.gh) // 2,
                        half[1].w, f.small.gh),
            title + (" · вручную" if cam.free else ""), role="small",
            color=C.faint, align="right")

    W.section(ctx, s, "Слои отладки", wid="b/layers", info=LAYER_HELP)
    W.toggle(ctx, "b/contacts", s.take(sc.px(S.row)), False,
             "точки касания лап", enabled=False,
             tip="маркеры MuJoCo при размере мухи закрывают кадр; "
                 "подробнее — значок рядом с заголовком")

    W.scroll_done(ctx, "body/insp", r, s.used, token)


# ------------------------------------------------------- правая колонка

def secondary(app, ctx, r, tab):
    if tab == 0:
        _meters(app, ctx, r)
    elif tab == 1:
        _track(app, ctx, r)
    elif tab == 2:
        _vision(app, ctx, r)
    else:
        _log(app, ctx, r)


METER_GROUPS = (
    ("Команды на ноги", (("команда влево", "dn_l", 3, ""),
                         ("команда вправо", "dn_r", 3, ""),
                         ("перекос", "asym", 4, ""))),
    ("Запах", (("слева", "odour_l", 5, ""),
               ("справа", "odour_r", 5, ""),
               ("на источник", "grad_cos", 3, ""),
               ("до источника", "dist", 1, "мм"))),
    ("Счёт", (("спайков за шаг", "spikes", 0, "шт"),)),
)


def _meters(app, ctx, r):
    sc, f = app.sh.sc, app.sh.fonts
    ses = app.ses
    s = L.Stack(r, gap=sc.px(S.xs))
    extra = dict(getattr(ses, "extra", {}) or {})
    if not extra:
        W.label(ctx, s.take(f.small.gh * 3),
                "Показаний пока нет: они приходят с шагом управления "
                "идущего прогона.", role="small", color=C.faint,
                wrapped=True)
        return
    first = True
    for title, rows in METER_GROUPS:
        if not s.fits(sc.px(S.row) * 2):
            break
        W.section(ctx, s, title, first=first)
        first = False
        for name, key, nd, unit in rows:
            if not s.fits(sc.px(S.row)):
                break
            v = extra.get(key)
            txt = "—" if v is None else (
                ("{:,.0f}".format(v).replace(",", " ")) if nd == 0
                else ("{:.%df}" % nd).format(float(v)).replace(".", ","))
            # `dist = -1` значит «ни одного включённого источника», а не
            # ноль миллиметров (body.py), так что числа здесь быть не
            # должно. Так же сделано в песочнице, в «Где муха».
            off = key == "dist" and v is not None and float(v) < 0
            W.kv(ctx, s.take(sc.px(S.row)), name,
                 "нет включённых" if off else txt, "" if off else unit)
    if s.fits(sc.px(S.row) * 2):
        W.section(ctx, s, "Поток")
        # «Активных нейронов» здесь нет нарочно: счётчик сеанса считает
        # их по сглаженному следу с постоянной 50 мс, подобранной под
        # стимуляцию, а в теле кадр приходит раз в секунду с лишним. К
        # следующему кадру от следа не остаётся ничего, и вышел бы
        # честный на вид ноль при тринадцати тысячах спайков за шаг.
        W.kv(ctx, s.take(sc.px(S.row)), "кадров активности",
             str(int(getattr(ses, "frames_seen", 0) or 0)), "шт")
        if s.fits(sc.px(S.row)):
            W.kv(ctx, s.take(sc.px(S.row)), "кадров тела",
                 str(int(getattr(ses, "body_seq", 0) or 0)), "шт")


def _track(app, ctx, r):
    """Путь мухи по арене. Рисуется из тех же записей шага."""
    dl, sc, f = ctx.dl, app.sh.sc, app.sh.fonts
    track = app.st.track
    content, token = W.scroll(ctx, "body/track", r)
    s = L.Stack(L.Rect(content.x, content.y, content.w, 1 << 16),
                gap=sc.px(S.sm))
    W.section(ctx, s, "Путь по арене", first=True,
              info="Координаты берутся из записей шага эпизода: в рядах, "
                   "которые шлёт задача, их нет, и окно копит их само.")
    plot = s.take(content.w)
    box = plot.fit_aspect(1.0)
    dl.rect(box.x, box.y, box.x1, box.y1, C.sunken, 1.0, sc.px(R.control))
    if len(track) < 2:
        W.label(ctx, L.Rect(box.x, box.cy, box.w, f.small.gh),
                "пути ещё нет", role="small", color=C.faint,
                align="centre")
        W.scroll_done(ctx, "body/track", r, s.used, token)
        return
    px = [p[1] for p in track]
    py = [p[2] for p in track]
    lo = min(min(px), min(py)) - 2.0
    hi = max(max(px), max(py)) + 2.0
    span = max(1e-3, hi - lo)
    inner = box.inset(sc.px(S.sm))
    pts = [(inner.x + (x - lo) / span * inner.w,
            inner.y1 - (y - lo) / span * inner.h) for x, y in zip(px, py)]
    grid_mm = next((g for g in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
                    if span / g <= 8.0), 100.0)
    step = inner.w / max(1.0, span / grid_mm)
    gx = inner.x
    while gx < inner.x1:
        dl.rect(gx, inner.y, gx + sc.hair(), inner.y1, C.grid, 1.0, 0.0)
        gx += step
    gy = inner.y1
    while gy > inner.y:
        dl.rect(inner.x, gy, inner.x1, gy + sc.hair(), C.grid, 1.0, 0.0)
        gy -= step
    dl.polyline(pts, C.plot, max(1.0, sc.px(1.5)))
    hx, hy = pts[-1]
    dl.rect(hx - sc.px(4), hy - sc.px(4), hx + sc.px(4), hy + sc.px(4),
            C.accent, 1.0, sc.px(4))
    # Подпись масштаба — под рамкой, а не внутри: у нижнего края она
    # ложилась на саму рамку.
    W.label(ctx, s.take(f.micro.gh, sc.px(S.sm)),
            "сторона %.0f мм · клетка %g мм" % (span, grid_mm),
            role="micro", color=C.faint)
    if s.fits(sc.px(S.row) * 2):
        W.section(ctx, s, "Пройдено")
        W.kv(ctx, s.take(sc.px(S.row)), "шагов", str(len(track)), "шт")
        if s.fits(sc.px(S.row)):
            d = sum(((px[i] - px[i - 1]) ** 2 + (py[i] - py[i - 1]) ** 2)
                    ** 0.5 for i in range(1, len(track)))
            W.kv(ctx, s.take(sc.px(S.row)), "длина пути",
                 ("%.1f" % d).replace(".", ","), "мм")
    W.scroll_done(ctx, "body/track", r, s.used, token)


# Схема зрения: что откуда берётся и где обрыв. Рисуется явно, потому
# что иначе разрыв приходится описывать словами, а словам верят хуже,
# чем перечёркнутой связи.
#
# Звеньев теперь четыре, а не три, и обрыв переехал. Прежняя схема
# ставила крест между омматидиями и оптическими долями — это было верно,
# пока входа не существовало. Теперь вход есть, и крест стоит там, где
# обрыв ИЗМЕРЕН: между ранними зрительными нейронами (L1, L2, L3, Tm1 —
# отвечают) и детекторами движения и нависания (T4, T5, LPLC2, LC4,
# DNp01 — ровно 0,000 Гц).
#
# Второй элемент кортежа — ключ строки в `state.VISION`, третий — есть
# ли связь ПОСЛЕ этого звена.
VISION_CHAIN = (("Камера глаза", "camera", True),
                ("Омматидии, 721×2", "retina", True),
                ("Фоторецепторы, 10 582", "wired", True),
                ("L1, L2, L3, Tm1", "early", False),
                ("T4, T5, LPLC2, DNp01", "deep", False))


def _vision_diagram(ctx, r, on=None):
    """Схема цепочки. ``on`` — включено ли зрение в этом прогоне.

    При выключенном зрении перечёркнуты ВСЕ связи после омматидиев: не
    потому, что пути нет, а потому, что этот прогон его не считал. Это
    разные вещи, и подпись у креста поэтому тоже разная.
    """
    dl, sc, f = ctx.dl, ctx.sc, ctx.f
    at = f.small
    bh = at.gh + sc.px(S.sm) + sc.px(2)
    gap = sc.px(18)
    n = len(VISION_CHAIN)
    total = bh * n + gap * (n - 1)
    if r.h < total or r.w < sc.px(120):
        return 0
    y = r.y
    for i, (title, _key, link_ok) in enumerate(VISION_CHAIN):
        # Само звено «есть», если до него связь дошла. Первые два есть
        # всегда: кадр и омматидии считает FlyGym независимо от того,
        # подаём мы их в сеть или нет.
        if i <= 1:
            ok = True
        elif on is False:
            ok = False
        else:
            ok = all(w[2] for w in VISION_CHAIN[:i])
        col = C.ok if ok else C.danger
        dl.rect(r.x, y, r.x1, y + bh, C.raised if ok else C.panel, 1.0,
                sc.px(R.control))
        dl.outline(r.x, y, r.x1, y + bh, col, sc.hair(),
                   1.0 if ok else 0.9, sc.px(R.control))
        dl.rect(r.x, y + sc.px(S.xs), r.x + sc.px(3), y + bh - sc.px(S.xs),
                col, 1.0, 1.0)
        dl.text(at, title, r.x + sc.px(S.md), y + (bh - at.gh) // 2,
                C.ink if ok else C.dim, max_px=r.w - sc.px(S.lg))
        if i < n - 1:
            ax, y0, y1 = r.cx, y + bh, y + bh + gap
            # Связь после этого звена: либо есть, либо перечёркнута, и
            # у креста своя подпись — «входа нет» и «зрение выключено»
            # это разные вещи.
            live = link_ok and (on is not False or i == 0)
            if live:
                dl.rect(ax - sc.hair(), y0 + sc.px(3), ax + sc.hair(),
                        y1 - sc.px(3), C.ok, 1.0, 0.0)
                dl.seg(ax - sc.px(4), y1 - sc.px(7), ax, y1 - sc.px(3),
                       C.ok, 1.6)
                dl.seg(ax + sc.px(4), y1 - sc.px(7), ax, y1 - sc.px(3),
                       C.ok, 1.6)
            else:
                # Пунктир с перечёркиванием: связи нет.
                yy = y0 + sc.px(3)
                while yy < y1 - sc.px(3):
                    dl.rect(ax - sc.hair(), yy, ax + sc.hair(),
                            min(yy + sc.px(3), y1 - sc.px(3)),
                            C.faint, 1.0, 0.0)
                    yy += sc.px(6)
                m = (y0 + y1) // 2
                d = sc.px(5)
                dl.seg(ax - d, m - d, ax + d, m + d, C.danger, 1.8)
                dl.seg(ax + d, m - d, ax - d, m + d, C.danger, 1.8)
                why = ("зрение выключено" if on is False
                       else ("не проходит" if i >= 2 else "входа нет"))
                W.label(ctx, L.Rect(ax + sc.px(S.md), m - f.micro.gh // 2,
                                    r.x1 - ax - sc.px(S.md), f.micro.gh),
                        why, role="micro", color=C.danger)
        y += bh + gap
    return total


def _vision_state(app):
    """Включено ли зрение в том прогоне, который сейчас на экране.

    Спрашивается у самого прогона, а не у рейки настройки: окно обязано
    говорить про ТОТ эпизод, чьи кадры показывает. `None` — неизвестно
    (простой, либо прогон без такого признака), и тогда схема рисуется
    без утверждений о выключенности.
    """
    ses = app.ses
    vis = (ses.extra or {}).get("vis_hz") if ses is not None else None
    if vis is not None:
        return True, None
    sb = getattr(app.st, "sb", None)
    if ses is not None and getattr(ses, "busy", False) and sb is not None:
        return bool(sb.vision), None
    return None, None


def _vision(app, ctx, r):
    """Шесть разных вещей, которые нельзя смешивать."""
    sc, f = app.sh.sc, app.sh.fonts
    content, token = W.scroll(ctx, "body/vision", r)
    s = L.Stack(L.Rect(content.x, content.y, content.w, 1 << 16),
                gap=sc.px(S.sm))
    on, prov = _vision_state(app)
    rows = ST.vision_rows(on=on, provenance=prov)

    W.section(ctx, s, "Путь сигнала", first=True,
              info="Схема показывает обрыв буквально и ровно там, где он "
                   "ИЗМЕРЕН: кадр и омматидии настоящие, вход в "
                   "фоторецепторы есть, ранние зрительные нейроны "
                   "отвечают, а детекторы движения и нависания дают "
                   "ровно 0,000 Гц. Перечёркнутая связь — это замер, а "
                   "не оформление.")
    used = _vision_diagram(ctx, s.area, on=on)
    if used:
        s.take(used, sc.px(S.md))

    W.section(ctx, s, "Что есть")
    for key in ("camera", "retina", "wired", "early"):
        _vision_row(app, ctx, s, key, rows)

    W.section(ctx, s, "Чего нет и чем является то, что есть")
    for key in ("deep", "binding"):
        _vision_row(app, ctx, s, key, rows)

    W.section(ctx, s, "Ограничение")
    h = W.label(ctx, s.area,
                "Картинки глаза в этом окне нет: работник отдаёт кадр "
                "камеры сцены, а не кадр омматидиев, и рисовать второй "
                "из первого значило бы показывать не то, что "
                "считается. Числа поданных частот и отметки времени "
                "обновления кадра — в «Анализе», видом «Таймлайн».",
                role="small", color=C.warn, wrapped=True)
    s.take(h)
    W.scroll_done(ctx, "body/vision", r, s.used, token)


def _vision_row(app, ctx, s, key, rows=None):
    sc = app.sh.sc
    title, state, text = (rows or VISION)[key]
    if not s.fits(sc.px(S.row)):
        return
    row = s.take(sc.px(S.row))
    # «есть» и «измерена» — это да; «выключена», «НЕ установлена» и
    # «ИНЖЕНЕРНАЯ, опровергнута» — нет. Выводится по списку, а не по
    # вхождению подстроки: подстрока однажды совпала бы не с тем.
    ok = state in ("есть", "измерена")
    W.status(ctx, row, "done" if ok else "fail", title)
    W.label(ctx, row, state, role="small", align="right",
            color=C.ok if ok else C.danger)
    h = W.label(ctx, L.Rect(row.x, s.area.y, row.w, s.area.h), text,
                role="small", color=C.faint, wrapped=True)
    s.take(h, sc.px(S.sm))


def _log(app, ctx, r):
    sc, f = app.sh.sc, app.sh.fonts
    st = app.st
    content, token = W.scroll(ctx, "body/log", r)
    s = L.Stack(L.Rect(content.x, content.y, content.w, 1 << 16),
                gap=sc.px(2))
    if not st.log:
        W.label(ctx, s.take(f.small.gh), "журнал пуст", role="small",
                color=C.faint)
    for _ts, level, text in st.log[-200:]:
        col = {"warn": C.warn, "error": C.danger}.get(level, C.dim)
        h = W.label(ctx, L.Rect(content.x, s.area.y, content.w,
                                s.area.h), text, role="small",
                    color=col, wrapped=True)
        s.take(h, sc.px(S.xs))
    W.scroll_done(ctx, "body/log", r, s.used, token)


SPACE = {"centre": centre, "inspector": inspector, "secondary": secondary,
         "tabs": TABS}
