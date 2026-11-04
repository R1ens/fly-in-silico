"""Рабочее пространство «Мозг»: исследование коннектома FlyWire v783.

Центр — настоящее облако из 138 639 нейронов, которое рисует
`ui/brainview.py` своим ядром. Второй реализации коннектома здесь нет:
геометрия приходит от работника один раз при знакомстве, позиции
квантованы в uint16, цвета нейропилей берутся из общей палитры.

Что это не просто облако точек:

* нейропиль выбирается наведением и щелчком, у выбранного показываются
  его настоящие числа;
* селектор-DSL тот же, что у командной строки, и находит он те же
  индексы — поиск уходит в работника и окно на нём не стоит;
* вкладка «Связи» спрашивает входящие и исходящие связи выбранного
  набора и различает четыре разные величины: нейроны, рёбра, синапсы и
  направление;
* режим активности красит по настоящим кадрам LIF, с временно́й меткой и
  подписанной шкалой.

Чего здесь сознательно нет: линий между нейронами. Пятнадцать миллионов
рёбер не несут смысла на экране, а рисовать правдоподобные, но выдуманные
траектории аксонов — хуже, чем не рисовать ничего. Связи показываются
таблицей и прямо названы логическими, а не морфологическими.
"""

import time

import numpy as np

from . import layout as L
from . import widgets as W
from .state import BRAIN_MODES, SCIENCE, SEL_VIEWS
from .tokens import C, R, S

__all__ = ["SPACE"]

TABS = ("Область", "Активность", "Связи", "Сведения")

SEARCH_HELP = ("Тот же селектор, что у командной строки: `region:AL`, "
               "`type:LC10`, `class:gustatory+side:left`. Плюс — "
               "пересечение, запятая — объединение. Enter ищет.")
LINK_HELP = ("Связи логические, а не морфологические: показано, кто с кем "
             "соединён и сколькими синапсами, а не как идёт аксон. "
             "Траектории по этим данным не восстанавливаются, и рисовать "
             "их было бы выдумкой.")

EXAMPLES = ("region:AL", "region:MB", "type:LC10", "class:gustatory",
            "super:descending", "nt:gaba+side:left")


# ---------------------------------------------------------------- центр

def centre(app, ctx, r):
    dl, sc, f = ctx.dl, app.sh.sc, app.sh.fonts
    st = app.st
    b = st.brain
    dock = st.ws.dock()
    bv = app.brain

    view = r.inset(sc.px(S.sm))
    if bv is None:
        # Подложка нужна только пустому месту: облако рисуется прямо в
        # накопитель кадра, и любая фигура, положенная в список команд,
        # легла бы поверх него — списки выливаются после сплатов.
        dl.rect(r.x, r.y, r.x1, r.y1, C.sunken, 1.0, sc.px(R.panel))
        _waiting(app, ctx, view)
        return

    # Карта рисуется прямо в поверхность кадра: это сплаты, а не картинка.
    #
    # Активность надо ОТДАТЬ карте, и это забывалось: `set_activity` звал
    # только прежний `ui/app.py`, а здесь никто. Карта брала нули,
    # заведённые в конструкторе, и режим «Активность» рисовал
    # неподвижный силуэт — при живом счёте, при работающем счётчике
    # «сработало» и при шкале в герцах под ним. Молчало ровно то, ради
    # чего режим и сделан.
    disp = getattr(app.ses, "disp", None)
    if disp is not None:
        bv.set_activity(disp)
    bv.show_labels = b.labels
    bv.only_active = b.only_active
    bv.sel_view = b.sel_view
    bv.mode = "activity" if b.mode == "activity" else "regions"
    # В анатомии цвет нейропиля не зависит от активности: показывают
    # строение. В активности долю цвета отдаём самой активности.
    bv.region_floor = 0.25 if b.mode == "activity" else 1.0
    bv.dim = b.dim
    bv.rad0 = 1.25 * b.point
    bv.rad1 = 2.6 * b.point
    bv.focus = b.region if (b.region >= 0 and b.sel_idx is None) else None

    hover, held, _, _ = ctx.probe("brain/view", view, True)
    vp = (view.x, view.y, view.w, view.h)
    t0 = time.perf_counter()
    bv.interact(vp, (ctx.inp.mx, ctx.inp.my), held,
                ctx.inp.wheel if hover else 0.0)
    if hover and not held:
        # Наведение считается только при движении курсора и только по
        # видимому облаку: это около миллисекунды, но каждый кадр оно ни
        # к чему.
        last = getattr(app, "_brain_hover_at", None)
        now = (ctx.inp.mx, ctx.inp.my)
        if last != now:
            app._brain_hover_at = now
            b.hover = -1 if bv.region_at(now) is None else bv.region_at(now)
            b.hover_neuron = bv.hover if bv.hover is not None else -1
    app._brain_pick_ms = (time.perf_counter() - t0) * 1000.0

    # Подписи кладутся в тот же фасад, что и остальной текст: иначе они
    # попадут в список, который рисуют без атласа, и пропадут.
    t1 = time.perf_counter()
    bv.draw(app.sh.surf, ctx.dl, f.small, vp, gain=1.0)
    app._brain_draw_ms = (time.perf_counter() - t1) * 1000.0

    # Щелчок по области. Отличить его от вращения можно только по ходу
    # курсора: нажатие всегда «удерживается», и прежняя проверка «не
    # тащили» глушила вообще все щелчки. Запоминаем место нажатия и
    # считаем щелчком отпускание рядом с ним.
    if ctx.inp.pressed and hover:
        app._brain_press = (ctx.inp.mx, ctx.inp.my)
    if ctx.inp.released:
        press = getattr(app, "_brain_press", None)
        app._brain_press = None
        if press is not None and hover:
            moved = abs(ctx.inp.mx - press[0]) + abs(ctx.inp.my - press[1])
            if moved <= 4:
                i = bv.region_at((ctx.inp.mx, ctx.inp.my))
                if i is not None:
                    _pick_region(app, int(i))
                elif b.sel_idx is None:
                    b.region = -1
                    bv.set_focus(None)

    _toolbar(app, ctx, view)
    _legend(app, ctx, view)


def _waiting(app, ctx, view):
    """Коннектом ещё не пришёл. Говорим, чего ждём и сколько это."""
    sc, f = app.sh.sc, app.sh.fonts
    w = app.st.worker
    mid = L.Rect(view.x + view.w // 4, view.cy - sc.px(44),
                 view.w // 2, sc.px(90))
    if not w.alive:
        head, text = ("Работника нет",
                      "Без процесса со счётом коннектом не загрузить. "
                      "Проверьте журнал внизу.")
    elif w.last_error:
        head, text = ("Коннектом не загрузился", w.last_error)
    else:
        head = "Коннектом загружается"
        text = ("138 639 нейронов и матрица на 15 091 983 связи. "
                "Это около тридцати секунд при первом запуске: "
                + (w.stage or "идёт"))
    W.label(ctx, mid, head, role="heading", color=C.dim, align="centre")
    W.label(ctx, L.Rect(mid.x, mid.y + f.heading.gh + sc.px(S.sm), mid.w,
                        sc.px(70)), text, role="small", color=C.faint,
            wrapped=True)
    if not w.ready and w.alive:
        bar = L.Rect(mid.x, mid.y + sc.px(76), mid.w, sc.px(6))
        W.progress(ctx, bar, max(0.0, w.progress),
                   indeterminate=(w.progress <= 0.0))


def _toolbar(app, ctx, view):
    sc = app.sh.sc
    st = app.st
    b = st.brain
    bv = app.brain
    dock = st.ws.dock()
    pad = sc.px(S.md)
    modes = [t for _k, t in BRAIN_MODES]
    keys = [k for k, _t in BRAIN_MODES]
    tb = L.Rect(view.x + pad, view.y + pad, view.w - pad * 2,
                sc.px(S.control) + sc.px(S.xs) * 2)
    hit, picks = W.viewport_toolbar(ctx, "brain/tb", tb, [
        ("seg", "mode", modes, keys.index(b.mode) if b.mode in keys else 0,
         ["цвет по нейропилям", "цвет по кадрам динамики LIF"]),
        ("-",),
        ("all", "brain", "показать весь мозг"),
        ("focus", "expand", "навести камеру на выделенное"),
        ("reset", "reset", "сбросить поворот и приближение"),
        ("-",),
        ("labels", "list", "подписи нейропилей", b.labels),
        ("active", "eye", "только сработавшие нейроны", b.only_active),
        ("-",),
        ("full", "fullscreen",
         "свернуть вид (F11)" if dock.maximized else "развернуть вид (F11)",
         dock.maximized),
    ])
    if "mode" in picks:
        # Режим цвета не трогает выделение: это разные состояния.
        b.mode = keys[picks["mode"]]
    if hit == "all":
        bv.fit_to(None)
        bv.zoom = 1.0
    elif hit == "focus":
        if b.sel_idx is not None and len(b.sel_idx):
            bv.fit_to(b.sel_idx)
        elif b.region >= 0:
            bv.fit_to(np.nonzero(bv.region_code == b.region)[0])
        bv.zoom = 1.0
    elif hit == "reset":
        bv.view_i = 3
        _, bv.yaw, bv.pitch = __import__(
            "flybrain.ui.brainview", fromlist=["VIEWS"]).VIEWS[3]
        bv.zoom = 1.0
        bv.fit_to(None)
    elif hit == "labels":
        b.labels = not b.labels
    elif hit == "active":
        b.only_active = not b.only_active
    elif hit == "full":
        dock.maximized = not dock.maximized


def _legend(app, ctx, view):
    """Подпись снизу: что сейчас показано и чем покрашено."""
    sc, f = app.sh.sc, app.sh.fonts
    st, b, bv = app.st, app.st.brain, app.brain
    ses = app.ses
    at = f.small
    pad = sc.px(S.md)
    back = ctx.layer(ctx.overlay)
    dl = ctx.dl

    if b.mode == "activity":
        seen = int(getattr(ses, "frames_seen", 0) or 0)
        if not seen:
            note = "активность: кадров ещё не было"
        else:
            age = float(getattr(ses, "stale_s", 0.0) or 0.0)
            note = ("активность: сглаженный след, τ 50 мс · "
                    "последний кадр %s с назад · модельное %.2f с"
                    % (("%.1f" % age).replace(".", ","),
                       float(getattr(ses, "t_ms", 0.0)) / 1000.0))
    else:
        note = "цвет: нейропиль, палитра одна с отчётом"
    if b.sel_idx is not None:
        note += "  ·  выделено %s нейронов" % _num(b.sel_n)
    elif b.region >= 0:
        note += "  ·  область " + bv.region_names[b.region]

    cw = at.width(note) + pad * 2
    cap = L.Rect(view.x + pad, view.y1 - pad - at.gh - sc.px(S.sm),
                 min(cw, view.w - pad * 2), at.gh + sc.px(S.sm))
    dl.rect(cap.x, cap.y, cap.x1, cap.y1, C.bg, 0.78, sc.px(R.control))
    W.label(ctx, L.Rect(cap.x + pad, cap.y + sc.px(S.xs) // 2 + 1,
                        cap.w - pad * 2, at.gh), note, role="small",
            color=C.dim)

    # Шкала активности: полоса магмы с настоящими границами в герцах.
    if b.mode == "activity" and int(getattr(ses, "frames_seen", 0) or 0):
        from ..ui import theme
        sw = min(sc.px(220), view.w // 3)
        sh_ = sc.px(8)
        bar = L.Rect(view.x1 - pad - sw, cap.y + (cap.h - sh_) // 2, sw,
                     sh_)
        steps = 40
        for i in range(steps):
            # Полоса подписана герцами, поэтому вдоль неё идёт ЧАСТОТА,
            # а цвет берётся тем же корнем, что и у точек в ядре
            # (`gpu.k_splat`). Иначе полоса обещала бы один цвет, а
            # карта показывала бы другой на той же частоте.
            t = (i / (steps - 1.0)) ** 0.5
            col = theme.magma_at(t)
            x0 = bar.x + int(bar.w * i / steps)
            x1 = bar.x + int(bar.w * (i + 1) / steps)
            dl.rect(x0, bar.y, x1 + 1, bar.y1, col, 1.0, 0.0)
        scale = float(getattr(ses, "scale", 0.0) or 0.0)
        W.label(ctx, L.Rect(bar.x - sc.px(40), bar.y - sc.px(1),
                            sc.px(36), f.micro.gh), "0", role="micro",
                color=C.faint, align="right")
        W.label(ctx, L.Rect(bar.x1 + sc.px(S.xs), bar.y - sc.px(1),
                            sc.px(90), f.micro.gh),
                "%.0f Гц" % scale, role="micro", color=C.faint)
    ctx.layer(back)


# --------------------------------------------------------------- рейка

def inspector(app, ctx, r):
    sc, f = app.sh.sc, app.sh.fonts
    st, b, bv = app.st, app.st.brain, app.brain
    content, token = W.scroll(ctx, "brain/insp", r)
    s = L.Stack(L.Rect(content.x, content.y, content.w, 1 << 16),
                gap=sc.px(S.sm))

    # --- поиск ---
    W.section(ctx, s, "Поиск", first=True, wid="br/find", info=SEARCH_HELP)
    ti = ctx.editor("brain/q", b.query)
    row = s.take(sc.px(S.control))
    W.text_field(ctx, "brain/q", row, ti, "селектор", icon="search",
                 enabled=st.worker.ready,
                 on_enter=lambda text: _search(app, text))
    b.query = ti.text
    if b.query_note:
        h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                    b.query_note, role="small",
                    color=C.danger if b.query_bad else C.dim, wrapped=True)
        s.take(h, sc.px(S.xs))
    row = s.take(sc.px(S.control))
    half = row.cols(2, gap=sc.px(S.sm))
    if W.button(ctx, "brain/go", half[0], "Найти", kind="primary",
                icon="search", enabled=st.worker.ready):
        _search(app, ti.text)
    if W.button(ctx, "brain/clr", half[1], "Сбросить", icon="close",
                enabled=(b.sel_idx is not None or b.region >= 0)):
        b.clear_selection()
        if bv is not None:
            bv.clear_selection()
            bv.set_focus(None)
            bv.fit_to(None)

    W.label(ctx, s.take(f.micro.gh + sc.px(S.xs)), "примеры",
            role="micro", color=C.faint)
    x = content.x
    row = s.take(sc.px(S.row))
    for ex in EXAMPLES:
        w = f.small.width(ex) + sc.px(S.md) * 2 + sc.px(S.sm)
        if x + w > content.x1:
            if not s.fits(sc.px(S.row)):
                break
            row = s.take(sc.px(S.row))
            x = content.x
        hit, _closed = W.chip(ctx, "brain/ex/" + ex,
                              L.Rect(x, row.y, w, row.h), ex)
        if hit:
            ti.set_text(ex)
            _search(app, ex)
        x += w + sc.px(S.xs)

    # --- отображение ---
    # Переключатель «Анатомия / Активность» живёт только в полосе над
    # вьюпортом: там, где смотрят. Дубля в рейке нет нарочно.
    W.section(ctx, s, "Отображение", wid="br/show",
              info="Режим цвета переключается в полосе над картой. "
                   "Здесь — подробности показа.")
    b.labels = W.toggle(ctx, "brain/lab", s.take(sc.px(S.row)), b.labels,
                        "подписи нейропилей")
    b.only_active = W.toggle(ctx, "brain/oa", s.take(sc.px(S.row)),
                             b.only_active, "только сработавшие",
                             enabled=(b.mode == "activity"),
                             tip="имеет смысл только в режиме активности")
    b.point = W.slider(ctx, "brain/pt", s.take(sc.px(40)), b.point, 0.5,
                       2.5, "размер точки", "×", fmt="{:.2f}")
    b.dim = W.slider(ctx, "brain/dim", s.take(sc.px(40)), b.dim, 0.0, 0.5,
                     "приглушение остального", fmt="{:.2f}",
                     tip="во сколько гаснет то, что не выделено")

    # --- выделение ---
    W.section(ctx, s, "Выделение", wid="br/sel",
              info="Выделение не зависит от режима цвета: переключение "
                   "анатомии и активности его не сбрасывает, и наоборот.")
    vnames = [t for _k, t, _h in SEL_VIEWS]
    vkeys = [k for k, _t, _h in SEL_VIEWS]
    vi = W.segmented(ctx, "brain/selview", s.take(sc.px(S.control)),
                     vnames, vkeys.index(b.sel_view)
                     if b.sel_view in vkeys else 0,
                     enabled=(b.sel_idx is not None),
                     tips=[h for _k, _t, h in SEL_VIEWS])
    if vkeys[vi] != b.sel_view:
        b.sel_view = vkeys[vi]
    if bv is None:
        W.label(ctx, s.take(f.small.gh), "карта ещё не готова",
                role="small", color=C.faint)
    elif b.sel_idx is not None:
        W.kv(ctx, s.take(sc.px(S.row)), "набор", b.sel_label or "—")
        W.kv(ctx, s.take(sc.px(S.row)), "нейронов", _num(b.sel_n), "шт")
        for k, v in list(b.sel_facts.get("regions", {}).items())[:4]:
            if not s.fits(sc.px(S.row)):
                break
            W.kv(ctx, s.take(sc.px(S.row)), "  " + k, _num(v), "шт")
    elif b.region >= 0:
        W.kv(ctx, s.take(sc.px(S.row)), "нейропиль",
             bv.region_names[b.region])
        W.kv(ctx, s.take(sc.px(S.row)), "нейронов",
             _num(int(bv.counts[b.region])), "шт")
    else:
        h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                    "Ничего не выбрано. Щёлкните по области на карте или "
                    "найдите популяцию селектором.", role="small",
                    color=C.faint, wrapped=True)
        s.take(h, sc.px(S.xs))

    W.scroll_done(ctx, "brain/insp", r, s.used, token)


# ------------------------------------------------------- правая колонка

def secondary(app, ctx, r, tab):
    if tab == 0:
        _tab_region(app, ctx, r)
    elif tab == 1:
        _tab_activity(app, ctx, r)
    elif tab == 2:
        _tab_links(app, ctx, r)
    else:
        _tab_about(app, ctx, r)


def _tab_region(app, ctx, r):
    sc, f = app.sh.sc, app.sh.fonts
    st, b, bv = app.st, app.st.brain, app.brain
    content, token = W.scroll(ctx, "brain/t0", r)
    s = L.Stack(L.Rect(content.x, content.y, content.w, 1 << 16),
                gap=sc.px(S.sm))
    if bv is None:
        W.label(ctx, s.take(f.small.gh * 2), "Коннектом ещё не загружен.",
                role="small", color=C.faint, wrapped=True)
        W.scroll_done(ctx, "brain/t0", r, s.used, token)
        return

    i = b.region if b.region >= 0 else b.hover
    if i < 0:
        W.section(ctx, s, "Нейропиль", first=True)
        h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                    "Наведите курсор на карту или щёлкните по области. "
                    "Подписи крупных областей нанесены прямо на облако.",
                    role="small", color=C.faint, wrapped=True)
        s.take(h)
        W.scroll_done(ctx, "brain/t0", r, s.used, token)
        return

    name = bv.region_names[i]
    live = (b.region < 0)
    W.section(ctx, s, name + (" · под курсором" if live else ""),
              first=True)
    W.kv(ctx, s.take(sc.px(S.row)), "код области", str(i))
    W.kv(ctx, s.take(sc.px(S.row)), "нейронов", _num(int(bv.counts[i])),
         "шт")
    total = int(np.sum(bv.counts))
    W.kv(ctx, s.take(sc.px(S.row)), "доля коннектома",
         ("%.2f" % (100.0 * bv.counts[i] / max(total, 1))).replace(".", ","),
         "%")
    sup = app.region_super.get(name, "")
    if sup:
        W.kv(ctx, s.take(sc.px(S.row)), "семейство", sup)
    lev = app.region_level(i)
    if lev is not None:
        W.kv(ctx, s.take(sc.px(S.row)), "средняя активность",
             ("%.1f" % lev).replace(".", ","), "Гц")

    row = s.take(sc.px(S.control))
    half = row.cols(2, gap=sc.px(S.sm))
    if W.button(ctx, "brain/showreg", half[0], "Показать область",
                icon="expand",
                tip="выделить нейроны области и навести камеру"):
        _pick_region(app, i, select=True)
    if W.button(ctx, "brain/linkreg", half[1], "Связи", icon="chart",
                tip="спросить входящие и исходящие связи"):
        _pick_region(app, i, select=True)
        st.ws.set_tab(2)
        _ask_links(app)

    # Разбор «откуда известна область» относится к выделенному набору, а
    # заголовок вкладки может показывать область под курсором. Показываем
    # блок только когда это одно и то же, иначе числа врут про не ту
    # область.
    src = {}
    if b.region == i and b.sel_src == "region":
        src = (b.sel_facts or {}).get("region_src") or {}
    if src:
        W.section(ctx, s, "Откуда известна область")
        tot = max(1, sum(src.values()))
        for k, v in sorted(src.items(), key=lambda kv: -kv[1])[:4]:
            if not s.fits(sc.px(S.row)):
                break
            nice = {"knn_soma": "голосование соседних сом",
                    "annotation": "аннотация",
                    "аннотация": "аннотация"}.get(k, k)
            W.kv(ctx, s.take(sc.px(S.row)), nice,
                 ("%.0f" % (100.0 * v / tot)), "%")

    W.section(ctx, s, "Что здесь точно")
    h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                "Точка на карте — положение сомы нейрона, а не объём "
                "нейропиля: нейрон, отнесённый к этой области, может "
                "сидеть телом далеко от неё. Число нейронов — из таблицы "
                "коннектома. Средняя активность считается по кадрам "
                "идущего прогона; без прогона её нет. Объёма нейропиля и "
                "списка трактов в этих данных нет, и выдумывать их нечем.",
                role="small", color=C.faint, wrapped=True)
    s.take(h)
    W.scroll_done(ctx, "brain/t0", r, s.used, token)


def _tab_activity(app, ctx, r):
    """Три разные вещи порознь: весь мозг, выбранный набор, последний кадр.

    Смешивать их нельзя. И отдельно: ноль — это измеренный ноль, а
    «нет данных» — это нет данных; в интерфейсе они выглядят по-разному.
    """
    sc, f = app.sh.sc, app.sh.fonts
    st, b, bv = app.st, app.st.brain, app.brain
    ses = app.ses
    content, token = W.scroll(ctx, "brain/t1", r)
    s = L.Stack(L.Rect(content.x, content.y, content.w, 1 << 16),
                gap=sc.px(S.sm))
    seen = int(getattr(ses, "frames_seen", 0) or 0)
    act = getattr(ses, "act", None)

    if not seen:
        W.section(ctx, s, "Данных нет", first=True,
                  info="Активность появляется только во время прогона. "
                       "Это не нулевая активность, а отсутствие "
                       "измерений вообще.")
        h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                    "Кадров динамики ещё не приходило. Запустите прогон "
                    "внизу или в пространстве «Эксперимент»: как только "
                    "пойдут бины, здесь появятся измерения.",
                    role="small", color=C.faint, wrapped=True)
        s.take(h)
        W.scroll_done(ctx, "brain/t1", r, s.used, token)
        return

    # --- весь мозг ---
    W.section(ctx, s, "Весь мозг", first=True,
              info="Сглаженный след по всем 138 639 нейронам: спайки за "
                   "бин, переведённые в герцы и сглаженные экспонентой "
                   "с постоянной 50 мс. Сглаживание — фильтр "
                   "измеренного, а не досочинённый сигнал.")
    W.kv(ctx, s.take(sc.px(S.row)), "пик следа",
         ("%.0f" % float(getattr(ses, "peak_hz", 0.0))), "Гц")
    W.kv(ctx, s.take(sc.px(S.row)), "нейронов со следом",
         _num(int(getattr(ses, "n_active", 0))), "шт")
    W.kv(ctx, s.take(sc.px(S.row)), "верх шкалы",
         ("%.0f" % float(getattr(ses, "scale", 0.0))), "Гц")

    # --- выбранный набор ---
    W.section(ctx, s, "Выбранный набор",
              info="То же, но только по выделенным нейронам. Молчащий "
                   "набор и отсутствие выделения — разные состояния.")
    if b.sel_idx is None:
        h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                    "Ничего не выделено. Найдите популяцию селектором "
                    "или щёлкните по области на карте.", role="small",
                    color=C.faint, wrapped=True)
        s.take(h, sc.px(S.sm))
    elif act is None:
        W.kv(ctx, s.take(sc.px(S.row)), "состояние", "нет массива следа")
    else:
        a = np.asarray(act)[np.asarray(b.sel_idx, dtype=np.int64)]
        live = int((a > 0.5).sum())
        W.kv(ctx, s.take(sc.px(S.row)), "нейронов в наборе",
             _num(int(len(a))), "шт")
        W.kv(ctx, s.take(sc.px(S.row)), "со следом", _num(live), "шт")
        if live:
            W.kv(ctx, s.take(sc.px(S.row)), "средняя по набору",
                 ("%.2f" % float(a.mean())).replace(".", ","), "Гц")
            W.kv(ctx, s.take(sc.px(S.row)), "средняя по сработавшим",
                 ("%.1f" % float(a[a > 0.5].mean())).replace(".", ","),
                 "Гц")
            W.kv(ctx, s.take(sc.px(S.row)), "наибольшая",
                 ("%.0f" % float(a.max())), "Гц")
        else:
            # Ноль измеренный, а не отсутствующий: так и написано.
            W.kv(ctx, s.take(sc.px(S.row)), "средняя по набору", "0", "Гц")
            h = W.label(ctx, L.Rect(content.x, s.area.y, content.w,
                                    s.area.h),
                        "Набор молчал: за %s принятых кадров ни один из "
                        "его нейронов не дал спайка. Это измеренный "
                        "ноль, а не нехватка данных."
                        % _num(seen), role="small", color=C.warn,
                        wrapped=True)
            s.take(h, sc.px(S.sm))

    # --- последний кадр ---
    W.section(ctx, s, "Последний кадр",
              info="Сам кадр, а не след: столько нейронов пришло в "
                   "последнем бине и столько в нём было спайков.")
    W.kv(ctx, s.take(sc.px(S.row)), "кадров принято", _num(seen), "шт")
    W.kv(ctx, s.take(sc.px(S.row)), "модельное время",
         ("%.3f" % (float(getattr(ses, "t_ms", 0.0)) / 1000.0)
          ).replace(".", ","), "с")
    W.kv(ctx, s.take(sc.px(S.row)), "получен",
         ("%.1f" % float(getattr(ses, "stale_s", 0.0))).replace(".", ","),
         "с назад")
    W.kv(ctx, s.take(sc.px(S.row)), "нейронов в кадре",
         _num(int(getattr(ses, "last_n", 0))), "шт")
    W.kv(ctx, s.take(sc.px(S.row)), "спайков в кадре",
         _num(int(getattr(ses, "last_spikes", 0))), "шт")

    # --- по областям ---
    W.section(ctx, s, "Верхние области",
              info="Среднее по всем нейронам области, включая молчащие: "
                   "в области с тысячами нейронов десяток горящих даёт "
                   "доли герца. Это не ошибка, а то, что значит среднее.")
    lev = app.region_levels()
    if lev is None:
        W.label(ctx, s.take(f.small.gh), "нет данных", role="small",
                color=C.faint)
    else:
        order = np.argsort(-lev)[:10]
        hi = max(float(lev[order[0]]), 1e-6)
        shown = 0
        for i in order:
            if not s.fits(sc.px(S.row)) or lev[i] <= 0.0:
                break
            row = s.take(sc.px(S.row))
            ctx.dl.rect(row.x, row.y, row.x + int(row.w * lev[i] / hi),
                        row.y1, C.plot, 0.22, 2.0)
            ctx.dl.text(f.body, bv.region_names[i], row.x + sc.px(S.sm),
                        row.y + (row.h - f.body.gh) // 2, C.ink)
            ctx.dl.text_right(f.metric,
                              ("%.2f" % lev[i]).replace(".", ","),
                              row.x1 - sc.px(S.sm),
                              row.y + (row.h - f.metric.gh) // 2, C.dim)
            shown += 1
        if not shown:
            W.label(ctx, s.take(f.small.gh),
                    "во всех областях след нулевой", role="small",
                    color=C.warn)
    W.scroll_done(ctx, "brain/t1", r, s.used, token)


# Столбцы таблицы связей: ключ, подпись, выравнивание вправо, подсказка.
# Три числа рядом легко принять за одно и то же, поэтому у каждого своё
# объяснение, и щелчок по шапке сортирует именно по нему.
LINK_COLS = (
    ("partner", "партнёр", 0,
     "группа партнёров: нейропиль, тип или класс — что выбрано ниже"),
    ("neurons", "нейронов", 1,
     "сколько РАЗНЫХ нейронов-партнёров участвует в связи. Для "
     "входящих и исходящих это разные множества, и совпадать они не "
     "обязаны"),
    ("edges", "рёбер", 1,
     "сколько пар «пресинаптический — постсинаптический». Рёбер не "
     "меньше, чем нейронов: один партнёр может быть связан со многими "
     "нейронами набора"),
    ("synapses", "синапсов", 1,
     "суммарное число синапсов по всем этим рёбрам. Это счёт из "
     "реконструкции, а не сила связи в милливольтах"),
)


def _tab_links(app, ctx, r):
    sc, f = app.sh.sc, app.sh.fonts
    st, b = app.st, app.st.brain
    content, token = W.scroll(ctx, "brain/t2", r)
    s = L.Stack(L.Rect(content.x, content.y, content.w, 1 << 16),
                gap=sc.px(S.sm))

    W.section(ctx, s, "Связи выбранного", first=True, wid="br/links",
              info=LINK_HELP)
    if b.sel_idx is None and b.region < 0:
        h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                    "Сначала выберите нейропиль или найдите популяцию: "
                    "связи считаются для выбранного набора, а не для "
                    "всего коннектома.", role="small", color=C.faint,
                    wrapped=True)
        s.take(h)
        W.scroll_done(ctx, "brain/t2", r, s.used, token)
        return

    dirs = ["входящие", "исходящие"]
    i = W.segmented(ctx, "brain/dir", s.take(sc.px(S.control)), dirs,
                    0 if b.links_dir == "in" else 1,
                    tips=["кто подаёт на выбранных",
                          "куда подают выбранные"])
    nd = "in" if i == 0 else "out"
    bys = ["область", "тип", "класс"]
    bk = ["region", "type", "class"]
    j = W.segmented(ctx, "brain/by", s.take(sc.px(S.control)), bys,
                    bk.index(b.links_by) if b.links_by in bk else 0,
                    tips=["группировать партнёров по нейропилю",
                          "по типу клетки", "по классу клетки"])
    if nd != b.links_dir or bk[j] != b.links_by:
        b.links_dir, b.links_by = nd, bk[j]
        _ask_links(app)

    if b.links_state in ("idle", "none"):
        row = s.take(sc.px(S.control))
        if W.button(ctx, "brain/asklinks", row, "Посчитать связи",
                    kind="primary", icon="chart",
                    enabled=st.worker.ready):
            _ask_links(app)
        h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                    "Считает работник: матрица на 15 миллионов связей "
                    "живёт в его процессе, и окно на этом не стоит.",
                    role="small", color=C.faint, wrapped=True)
        s.take(h)
    elif b.links_state == "wait":
        W.progress(ctx, s.take(sc.px(34)), 0.0, "считаю связи",
                   indeterminate=True)
        h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                    "Первый запрос строит указатель по столбцам матрицы: "
                    "это несколько секунд, дальше ответы быстрые.",
                    role="small", color=C.faint, wrapped=True)
        s.take(h)
    elif b.links_state == "error":
        h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                    b.links_note or "связи посчитать не удалось",
                    role="small", color=C.danger, wrapped=True)
        s.take(h)
    else:
        W.label(ctx, s.take(f.micro.gh + sc.px(S.xs)),
                "для набора «%s», %s мс · %s"
                % (b.links_for, ("%.0f" % b.links_ms),
                   "кто подаёт на набор" if b.links_dir == "in"
                   else "куда подаёт набор"),
                role="micro", color=C.faint)
        head = s.take(f.small.gh + sc.px(S.sm))
        cols = _link_cols(app, head)
        for (key, title, right, tip) in LINK_COLS:
            cell = cols[[c[0] for c in LINK_COLS].index(key)]
            hid = "brain/lkh/" + key
            hov, _h, clicked, _f = ctx.probe(hid, cell, True)
            on = (b.links_sort == key)
            mark = (" ▼" if on else "")
            W.label(ctx, L.Rect(cell.x, cell.y
                                + (cell.h - f.small.gh) // 2, cell.w,
                                f.small.gh), title + mark, role="small",
                    color=C.ink if on else (C.dim if hov else C.faint),
                    align="right" if right else "left")
            ctx.want_tip(hid, cell, tip + ". Щелчок — сортировать по "
                                          "этому столбцу.", delay=0.3)
            if clicked and key != "partner":
                b.links_sort = key
                _sort_links(b)
        line = s.take(sc.hair(), sc.px(S.xs))
        ctx.dl.rect(line.x, line.y, line.x1, line.y + sc.hair(),
                    C.border_soft, 1.0, 0.0)
        for k, rec in enumerate(b.links_rows):
            if not s.fits(sc.px(S.row)):
                break
            line = s.take(sc.px(S.row))
            sub = "brain/lk/%d" % k
            hov, held, clicked, _fc = ctx.probe(sub, line, True)
            if hov or held:
                # Строка подсвечивается целиком: иначе непонятно, к
                # какому партнёру относится число под курсором.
                ctx.dl.rect(line.x - sc.px(S.xs), line.y,
                            line.x1 + sc.px(S.xs), line.y1,
                            C.pressed if held else C.hover, 1.0,
                            sc.px(R.control))
                ctx.dl.rect(line.x - sc.px(S.xs), line.y,
                            line.x - sc.px(S.xs) + sc.px(2), line.y1,
                            C.accent, 1.0, 1.0)
            cells = _link_cols(app, line)
            for (key, _t, right, _tip), cell in zip(LINK_COLS, cells):
                v = rec.get(key, "—")
                txt = v if isinstance(v, str) else _num(v)
                if right:
                    W.label(ctx, L.Rect(cell.x, cell.y
                                        + (cell.h - f.small.gh) // 2,
                                        cell.w, f.small.gh), txt,
                            role="small", color=C.ink, align="right")
                else:
                    W.label(ctx, L.Rect(cell.x, cell.y
                                        + (cell.h - f.small.gh) // 2,
                                        cell.w, f.small.gh), txt,
                            role="small", color=C.ink)
            ctx.want_tip(sub, line,
                         "%s: %s нейронов, %s рёбер, %s синапсов "
                         "(возбуждающих %s, тормозных %s). Щелчок — "
                         "перейти к этой группе на карте."
                         % (rec.get("partner", "—"),
                            _num(rec.get("neurons", 0)),
                            _num(rec.get("edges", 0)),
                            _num(rec.get("synapses", 0)),
                            _num(rec.get("exc", 0)),
                            _num(rec.get("inh", 0))))
            if clicked:
                _goto_partner(app, rec)
        if b.links_note:
            W.label(ctx, s.take(f.micro.gh + sc.px(S.sm)), b.links_note,
                    role="micro", color=C.faint)
    W.scroll_done(ctx, "brain/t2", r, s.used, token)


def _sort_links(b):
    """Пересортировать уже полученные строки. Работник не тревожится."""
    key = b.links_sort if b.links_sort in ("neurons", "edges",
                                           "synapses") else "synapses"
    b.links_rows.sort(key=lambda r: -float(r.get(key, 0) or 0))


def _link_cols(app, row):
    sc = app.sh.sc
    num = sc.px(62)
    first = max(sc.px(40), row.w - num * 3 - sc.px(S.sm))
    x = row.x
    out = [L.Rect(x, row.y, first, row.h)]
    x += first
    for _ in range(3):
        out.append(L.Rect(x, row.y, num, row.h))
        x += num
    return out


def _tab_about(app, ctx, r):
    sc, f = app.sh.sc, app.sh.fonts
    content, token = W.scroll(ctx, "brain/t3", r)
    s = L.Stack(L.Rect(content.x, content.y, content.w, 1 << 16),
                gap=sc.px(S.sm))
    w = app.st.worker
    W.section(ctx, s, "Данные", first=True)
    W.kv(ctx, s.take(sc.px(S.row)), "нейронов", _num(w.n_neurons), "шт")
    W.kv(ctx, s.take(sc.px(S.row)), "связей", _num(15091983), "шт")
    W.kv(ctx, s.take(sc.px(S.row)), "нейропилей", _num(len(w.regions)),
         "шт")
    W.kv(ctx, s.take(sc.px(S.row)), "устройство", w.device or "—")

    W.section(ctx, s, "Оговорки")
    for title, value, text in SCIENCE:
        if not s.fits(sc.px(S.row)):
            break
        row = s.take(sc.px(S.row))
        W.label(ctx, L.Rect(row.x, row.y + (row.h - f.body.gh) // 2,
                            row.w, f.body.gh), title, role="body",
                color=C.ink)
        W.label(ctx, row, value, role="small", align="right",
                color=C.warn if value in ("нет", "предсказан",
                                          "не подключён",
                                          "не запись живой мухи")
                else C.dim)
        h = W.label(ctx, L.Rect(row.x, s.area.y, row.w, s.area.h), text,
                    role="small", color=C.faint, wrapped=True)
        s.take(h, sc.px(S.sm))
    W.scroll_done(ctx, "brain/t3", r, s.used, token)


# ------------------------------------------------------------- действия

def _num(v):
    try:
        return "{:,}".format(int(v)).replace(",", " ")
    except (TypeError, ValueError):
        return str(v)


def _search(app, text):
    """Поиск уходит в работника: окно на нём не стоит."""
    st, b = app.st, app.st.brain
    text = (text or "").strip()
    b.query = text
    ti = app.sh.ctx._edits.get("brain/q")
    if ti is not None and ti.text != text:
        ti.set_text(text)
    if not text:
        b.query_note = "пустой селектор"
        b.query_bad = True
        return
    if app.ses is None or not st.worker.ready:
        b.query_note = "работник не готов"
        b.query_bad = True
        return
    b.query_note = "ищу…"
    b.query_bad = False
    req = app.ses.ask("select_idx", text)
    b.pending[req] = "select"


def _pick_region(app, i, select=False):
    """Выбрать нейропиль. `select` ещё и выделяет его нейроны."""
    b, bv = app.st.brain, app.brain
    b.region = int(i)
    b.links_state = "idle"
    b.links_rows = []
    if bv is None:
        return
    name = bv.region_names[i]
    if select:
        idx = np.nonzero(bv.region_code == i)[0]
        b.sel_idx = idx
        b.sel_n = int(len(idx))
        b.sel_label = "region:" + name
        b.sel_src = "region"
        b.sel_facts = {"regions": {name: int(len(idx))}}
        bv.set_selection(idx, b.sel_label)
        bv.fit_to(idx)
        bv.zoom = 1.0
        # Сам выбор мгновенный — он считается по уже загруженным кодам
        # областей. А за подробностями (откуда взялась область у этих
        # нейронов, какие там типы) идём к работнику: камеру и выделение
        # ответ не трогает.
        if app.ses is not None and app.st.worker.ready:
            req = app.ses.ask("select_idx", "region:" + name)
            b.pending[req] = "facts"
    else:
        bv.set_focus(i)


def _ask_links(app):
    st, b = app.st, app.st.brain
    if app.ses is None or not st.worker.ready:
        b.links_state = "error"
        b.links_note = "работник не готов"
        return
    idx = b.sel_idx
    label = b.sel_label
    if idx is None and b.region >= 0 and app.brain is not None:
        idx = np.nonzero(app.brain.region_code == b.region)[0]
        label = "region:" + app.brain.region_names[b.region]
    if idx is None or not len(idx):
        b.links_state = "none"
        return
    b.links_state = "wait"
    b.links_for = label
    req = app.ses.ask("links", {"idx": np.asarray(idx, dtype=np.int32),
                                "dir": b.links_dir, "by": b.links_by,
                                "limit": 40})
    b.pending[req] = "links"


def _goto_partner(app, rec):
    """Перейти к партнёру: выделить его и навести камеру."""
    b = app.st.brain
    name = str(rec.get("partner", ""))
    if not name or name == "—":
        return
    key = {"region": "region:", "type": "type=", "class": "class:"}.get(
        b.links_by, "type:")
    _search(app, key + name)


def take_answers(app):
    """Разобрать ответы работника. Зовётся раз в кадр из оболочки."""
    b = app.st.brain
    if app.ses is None or not b.pending:
        return
    for req in list(b.pending):
        ans = app.ses.answer(req)
        if ans is None:
            continue
        kind = b.pending.pop(req)
        if not isinstance(ans, dict):
            continue
        if kind == "select":
            _took_select(app, ans)
        elif kind == "facts":
            if not ans.get("error"):
                app.st.brain.sel_facts = {
                    "regions": ans.get("regions", {}),
                    "types": ans.get("types", {}),
                    "sides": ans.get("sides", {}),
                    "region_src": ans.get("region_src", {})}
        elif kind == "links":
            _took_links(app, ans)


def _took_select(app, ans):
    b, bv = app.st.brain, app.brain
    if ans.get("error"):
        b.query_note = "селектор не разобран: " + str(ans["error"])[:120]
        b.query_bad = True
        return
    n = int(ans.get("n", 0))
    if not n:
        b.query_note = "ничего не нашлось"
        b.query_bad = True
        b.sel_idx = None
        b.sel_n = 0
        if bv is not None:
            bv.clear_selection()
        return
    idx = np.asarray(ans.get("idx"), dtype=np.int64)
    b.sel_idx = idx
    b.sel_n = n
    b.sel_label = b.query
    b.sel_src = "search"
    b.sel_facts = {"regions": ans.get("regions", {}),
                   "types": ans.get("types", {}),
                   "sides": ans.get("sides", {}),
                   "region_src": ans.get("region_src", {})}
    b.query_bad = False
    b.query_note = "нашлось %s нейронов" % _num(n)
    b.region = -1
    b.links_state = "idle"
    b.links_rows = []
    if bv is not None:
        bv.set_selection(idx, b.sel_label)
        bv.set_focus(None)
        bv.fit_to(idx)
        bv.zoom = 1.0


def _took_links(app, ans):
    b = app.st.brain
    if ans.get("error"):
        b.links_state = "error"
        b.links_note = str(ans["error"])[:200]
        return
    rows = ans.get("rows") or []
    if isinstance(rows, dict):
        rows = rows.get("rows", [])
    b.links_rows = list(rows)[:40]
    _sort_links(b)
    b.links_ms = float(ans.get("ms", 0.0))
    b.links_state = "ready" if b.links_rows else "none"
    tot = ans.get("total") or {}
    bits = []
    if tot:
        bits.append("всего %s партнёров, %s рёбер, %s синапсов"
                    % (_num(tot.get("partners", 0)),
                       _num(tot.get("edges", 0)),
                       _num(tot.get("synapses", 0))))
        hid = int(tot.get("hidden_groups", 0) or 0)
        if hid:
            bits.append("не показано групп: %s (%s синапсов)"
                        % (_num(hid), _num(tot.get("hidden_synapses", 0))))
        if tot.get("device"):
            bits.append("счёт на " + str(tot["device"]))
    b.links_note = " · ".join(bits)


SPACE = {"centre": centre, "inspector": inspector, "secondary": secondary,
         "tabs": TABS, "answers": take_answers}
