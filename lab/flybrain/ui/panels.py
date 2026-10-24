"""Панели окна: рейка сборки, показания, полоса тела, подвал, карта.

Каждая панель — функция, которая получает контекст деталей, сеанс и свою
коробку, и ничего не знает про окно. Так панель можно нарисовать в проверке
без окна и посмотреть на неё картинкой.
"""

import numpy as np

from . import kit, theme
from .query import KEYS

# Ключи, предлагаемые сборщиком. Порядок — по тому, как часто нужны.
PICK_KEYS = ("type", "class", "region", "side")
KEY_LABELS = {k: label for k, label, _ in KEYS}

__all__ = ["rail", "meters", "strip", "foot", "brain_toolbar", "hover_tip"]

# Виды задач в двух строках. Одной строкой шесть русских подписей не
# помещаются даже в широкой рейке, а резать их до огрызков хуже, чем
# перенести.
MODE_ROW1 = (("stim", "Стимуляция", "Стим"), ("arena", "Арена", "Арена"),
             ("body", "Тело", "Тело"))
MODE_ROW2 = (("lateral", "Зонд", "Зонд"), ("demos", "Демо", "Демо"),
             ("agent", "Агент", "Агент"), ("runs", "Прогоны", "Спис."))
MODE_KEYS = tuple(k for k, _, _ in MODE_ROW1 + MODE_ROW2)


def _section(ui, col, title):
    """Заголовок раздела: полужирный текст и волосяная линия под ним.

    Рейка без разделов читалась как стопка одинаковых серых коробок —
    кнопка, список и ползунок выглядели одинаково важными.
    """
    col.skip(ui.px(4))
    r = col.take(ui.strong.gh + ui.px(2))
    ui._txt(ui.strong, title, r.x0, r.y0, theme.DIM, max_px=r.w)
    ui.over.hline(r.x0, r.x1, r.y1 + ui.px(2), theme.HAIRLINE)
    col.skip(ui.px(6))
    return r


def _note(ui, col, text, color=None, small=True):
    """Строка пояснения, обрезанная по ширине колонки.

    `ui.label` ширины не знает и спокойно рисует за край панели — в узкой
    рейке длинное пояснение залезало прямо на карту мозга.
    """
    atlas = ui.small if small else ui.font
    r = col.take(ui.line(small))
    ui._txt(atlas, text, r.x0, r.y0, color or theme.DIM, max_px=r.w)
    return r


def _mode_labels(ui, row, width):
    """Подписи по ширине: в узкой рейке полные не помещаются и лезут за край."""
    full = [n for _, n, _ in row]
    need = max(ui.small.width(n) for n in full) + ui.px(16)
    return full if width >= need * len(row) else [s for _, _, s in row]


# ---------------------------------------------------------------- рейка

def rail(ui, sess, st, b):
    """Левая рейка: что запускаем и с какими числами."""
    ui.back.rect(b.x0, b.y0, b.x1, b.y1, theme.PANEL)
    ui.back.vline(b.x1 - 1, b.y0, b.y1, theme.HAIRLINE)

    # Место под пуск, стоп и строку состояния отрезается заранее. Если
    # считать их после переменного списка, в тесном окне список съедает всю
    # высоту и кнопки наезжают на ползунки.
    area = b.inset(ui.pad, ui.pad)
    foot_h = ui.row * 2 + ui.gap + ui.line(True) + ui.px(8)
    bottom, body = area.cut_bottom(min(foot_h, max(0, area.h // 2)), ui.gap)
    col = kit.Column(body, ui.gap)

    t = col.take(ui.title.gh + ui.px(2))
    ui.label("Мозг мухи", t.x0, t.y0, theme.INK, face="title")
    sub = (str(sess.n) + " нейронов · " + (sess.device or "—")
           if sess.ready else "поднимаю работника…")
    ui.label(sub, col.area.x0, col.take(ui.line(True)).y0,
             theme.DIM, small=True)
    col.skip(ui.px(6))
    ui.over.hline(col.area.x0, col.area.x1, col.take(1).y0, theme.HAIRLINE)
    col.skip(ui.px(6))

    keys1 = [k for k, _, _ in MODE_ROW1]
    keys2 = [k for k, _, _ in MODE_ROW2]
    cur1 = keys1.index(st.mode) if st.mode in keys1 else -1
    cur2 = keys2.index(st.mode) if st.mode in keys2 else -1
    w = col.area.w
    a = ui.segmented("mode1", col.take(ui.row), _mode_labels(ui, MODE_ROW1, w),
                     cur1)
    col.skip(ui.px(2))
    b2 = ui.segmented("mode2", col.take(ui.row), _mode_labels(ui, MODE_ROW2, w),
                      cur2)
    if 0 <= a < len(keys1) and a != cur1:
        st.mode = keys1[a]
    elif 0 <= b2 < len(keys2) and b2 != cur2:
        st.mode = keys2[b2]
    col.skip(ui.px(8))

    builder = {"stim": _rail_stim, "arena": _rail_arena, "body": _rail_body,
               "lateral": _rail_lateral, "demos": _rail_demos,
               "agent": _rail_agent, "runs": _rail_runs}.get(st.mode)
    if builder is not None:
        builder(ui, sess, st, col)

    bcol = kit.Column(bottom, ui.px(4))
    busy = sess.busy
    txt = _state_text(sess)
    ui.label(txt, bottom.x0, bcol.take(ui.line(True)).y0,
             theme.DANGER if sess.state == "error" else
             (theme.ACCENT if busy else theme.DIM), small=True)
    if bcol.left >= ui.row * 2 + ui.gap:
        go, stop = bcol.rest.rows(2, ui.gap)
    else:
        go, stop = bcol.rest.cols(2, ui.gap)
    if ui.button("go", go, _go_label(st), kind="primary",
                 enabled=sess.ready and not busy):
        st.want_run = True
    if ui.button("stop", stop, "Стоп", kind="danger", enabled=busy):
        sess.cancel()
    return st


def _go_label(st):
    """На кнопке видно, во что ввязываешься.

    Серия из шести эпизодов в теле — это полчаса счёта, и узнавать об этом
    после нажатия поздно.
    """
    from .. import jobs

    try:
        from . import app as _app

        secs = jobs.estimate(_app._spec(st))
    except Exception:                                          # noqa: BLE001
        secs = 0.0
    if secs < 1.0:
        return "Пуск"
    if secs < 90.0:
        return f"Пуск · ~{secs:.0f} с"
    return f"Пуск · ~{secs / 60.0:.0f} мин"


def _state_text(sess):
    if not sess.ready:
        return (sess.boot_phase or "работник поднимается") + "…"
    if sess.state == "error":
        return "ошибка: " + (sess.error or "")[:40]
    if sess.busy:
        p = sess.phase or sess.state
        return p + " · " + format(sess.t_ms / 1000.0, ".2f") + " с"
    if sess.state == "done":
        return "готово · " + str(sess.frames_seen) + " кадров"
    return "готов"


def _rail_stim(ui, sess, st, col):
    _section(ui, col, "кого зажечь")

    # Собранный запрос показывается строкой DSL: ровно то, что примет
    # командная строка, чтобы окно и `fly run` нельзя было развести.
    qbox = col.take(ui.row)
    ui.dl.rect(qbox.x0, qbox.y0, qbox.x1, qbox.y1, theme.BG)
    ui.dl.frame(qbox.x0, qbox.y0, qbox.x1, qbox.y1, theme.HAIRLINE)
    text = st.qb.text() or "ничего не выбрано"
    ui._txt(ui.num_small, text, qbox.x0 + ui.px(8),
            ui._mid(qbox, ui.num_small),
            theme.ACCENT if st.qb.text() else theme.FAINT,
            max_px=qbox.w - ui.px(16))
    col.skip(ui.px(4))

    terms = [t.label() for g in st.qb.groups for t in g]
    if terms:
        chip_h = ui.row * max(1, (len(terms) + 1) // 2)
        chosen = ui.chips("terms", col.take(chip_h), terms, set(terms))
        if len(chosen) != len(terms):
            gone = [t for t in terms if t not in chosen]
            for gi, g in enumerate(st.qb.groups):
                for ti_, t in enumerate(list(g)):
                    if t.label() in gone:
                        st.qb.remove(gi, ti_)
                        break
        col.skip(ui.px(4))

    row = col.take(ui.row)
    half = row.cols(2, ui.gap)
    if ui.button("q_or", half[0], "или", enabled=not st.qb.is_empty()):
        st.qb.new_group()
    if ui.button("q_clear", half[1], "очистить", enabled=not st.qb.is_empty()):
        st.qb.clear()
    col.skip(ui.px(8))

    # Четыре ключа, которыми собирается почти всё: тип, класс, нейропиль,
    # сторона. Медиатор и надкласс добавятся, когда понадобятся — лишняя
    # строка выбора стоит дороже, чем редкий ключ.
    labels = [KEY_LABELS[k] for k in PICK_KEYS]
    need = max(ui.small.width(s) for s in labels) + ui.px(14)
    if col.area.w >= need * len(labels):
        st.key_i = ui.segmented("qkey", col.take(ui.row), labels, st.key_i)
    else:
        # В узкой рейке четыре подписи в строку сливаются в кашу.
        a = ui.segmented("qkeyA", col.take(ui.row), labels[:2],
                         st.key_i if st.key_i < 2 else -1)
        bsel = ui.segmented("qkeyB", col.take(ui.row), labels[2:],
                            st.key_i - 2 if st.key_i >= 2 else -1)
        if 0 <= a < 2 and a != st.key_i:
            st.key_i = a
        elif 0 <= bsel < 2 and bsel + 2 != st.key_i:
            st.key_i = bsel + 2
    key = PICK_KEYS[st.key_i]
    col.skip(ui.px(6))

    facets = sess.facets()
    vals = facets.get(key.rstrip("="), [])
    labels = [v + "   " + str(n) for v, n in vals]
    # Сперва отмеряется место ползункам, список берёт остаток. Порядок
    # важен: ползунок без строки подписи молча теряет её, а список без
    # нескольких строк просто не нужен — пусть уходит он.
    full = ui.row + ui.small.gh
    # Зазор между взятиями тоже место: без него третьему ползунку
    # не хватало на строку подписи, и он оставался голой дорожкой.
    room = col.left - ui.gap * 3 - ui.px(6)
    sh_ = max(ui.row, min(full, room // 3))
    lb_h = min(max(room - sh_ * 3, 0), ui.row * 11)
    if lb_h >= ui.row * 2:
        pick = ui.list_box("qvals", col.take(lb_h), labels, st.val_i)
        if pick != st.val_i and 0 <= pick < len(vals):
            st.val_i = pick
            st.qb.add(key, vals[pick][0])
        col.skip(ui.px(6))

    st.rate = ui.slider("rate", col.take(sh_), "частота, Гц",
                        st.rate, 10.0, 400.0, 10.0)
    st.t_stim = ui.slider("tst", col.take(sh_), "длительность, с",
                          st.t_stim, 0.2, 5.0, 0.1)
    st.bin_ms = ui.slider("bin", col.take(sh_), "бин, мс",
                          st.bin_ms, 5.0, 50.0, 5.0)


def _rail_body(ui, sess, st, col):
    _section(ui, col, "чем рулить")
    outs = list(sess.hello.readouts) if sess.ready else []
    if outs:
        if st.readout not in outs:
            st.readout = "ALPN" if "ALPN" in outs else outs[0]
        full = ui.row + ui.small.gh
        room = col.left - ui.row - ui.gap * 4 - ui.px(6)
        sh_ = max(ui.row, min(full, room // 2))
        lb_h = min(max(room - sh_ * 2, 0), ui.row * 10)
        if lb_h >= ui.row * 2:
            i = ui.list_box("outs", col.take(lb_h), outs,
                            outs.index(st.readout))
            st.readout = outs[max(0, min(i, len(outs) - 1))]
    else:
        sh_ = ui.row + ui.small.gh
    col.skip(ui.px(6))

    st.t_body = ui.slider("tb", col.take(sh_), "длительность, с",
                          st.t_body, 0.5, 20.0, 0.5)
    st.dn_tau = ui.slider("tau", col.take(sh_), "постоянная съёма, с",
                          st.dn_tau, 0.05, 2.0, 0.05)
    st.swap = ui.toggle("swap", col.take(ui.row),
                        "переставить антенны (контроль)", st.swap)
    _series_block(ui, st, col)


def _series_block(ui, st, col):
    """Серия вместо одного прогона — это поле спеки, а не отдельный режим.

    Поэтому и в окне это переключатель рядом с параметрами, а не седьмая
    кнопка вида задачи.
    """
    # Место кончилось — деталь не рисуется вовсе. Сжатый до нитки
    # переключатель хуже отсутствующего: он выглядит как мусор на строке
    # состояния.
    if col.left < ui.row:
        return
    st.series = ui.toggle("series", col.take(ui.row),
                          "серия с контролем", st.series)
    if st.series:
        full = ui.row + ui.small.gh
        sh_ = max(ui.row, min(full, (col.left - ui.gap * 2) // 2))
        st.series_n = int(ui.slider("sn", col.take(sh_), "эпизодов",
                                    st.series_n, 2, 30, 1))
        st.series_t = ui.slider("stt", col.take(sh_), "эпизод, с",
                                st.series_t, 1.0, 20.0, 0.5)


def _readout_list(ui, sess, col, current, key, height_rows):
    """Список съёмов. Пустая строка наверху значит «как в настройках по умолчанию»."""
    outs = list(sess.hello.readouts) if sess.ready else []
    if not outs:
        return current
    items = ["нисходящие (по умолчанию)"] + outs
    cur = (outs.index(current) + 1) if current in outs else 0
    lb_h = min(max(col.left - height_rows, 0), ui.row * 9)
    if lb_h < ui.row * 2:
        return current
    i = ui.list_box(key, col.take(lb_h), items, cur)
    col.skip(ui.px(6))
    return "" if i <= 0 else outs[min(i - 1, len(outs) - 1)]


def _rail_arena(ui, sess, st, col):
    _section(ui, col, "чем рулить")
    full = ui.row + ui.small.gh
    tail = full * 2 + ui.row * 2 + ui.px(14)
    st.arena_readout = _readout_list(ui, sess, col, st.arena_readout,
                                     "arena_outs", tail)
    sh_ = max(ui.row, min(full, (col.left - ui.row * 2 - ui.px(14)) // 2))
    st.t_arena = ui.slider("ta", col.take(sh_), "длительность, с",
                           st.t_arena, 2.0, 60.0, 1.0)
    st.dn_tau = ui.slider("tau_a", col.take(sh_), "постоянная съёма, с",
                          st.dn_tau, 0.05, 2.0, 0.05)
    st.comment = ui.toggle("cmt", col.take(ui.row),
                           "комментарий локальной модели", st.comment)
    _series_block(ui, st, col)


def _rail_lateral(ui, sess, st, col):
    _section(ui, col, "зонд стороны")
    _note(ui, col, "открытый контур: антенны качаются сами,")
    _note(ui, col, "а мы смотрим, кто за ними следует")
    col.skip(ui.px(10))
    full = ui.row + ui.small.gh
    sh_ = max(ui.row, min(full, (col.left - ui.px(10)) // 2))
    st.t_lateral = ui.slider("tl", col.take(sh_), "длительность, с",
                             st.t_lateral, 2.0, 30.0, 1.0)
    st.base_hz = ui.slider("bhz", col.take(sh_), "вход, Гц",
                           st.base_hz, 20.0, 200.0, 5.0)
    col.skip(ui.px(8))
    n = len(sess.hello.populations) if sess.ready else 0
    _note(ui, col, f"перебирается {n} популяций разом", theme.FAINT)


def _rail_demos(ui, sess, st, col):
    _section(ui, col, "готовые опыты")
    from .. import spec as S

    names = [d.title for d in S.DEMOS]
    lb_h = min(max(col.left - ui.line(True) * 3 - ui.px(10), 0), ui.row * 8)
    if lb_h >= ui.row * 2:
        st.demo_i = ui.list_box("demos", col.take(lb_h), names,
                                st.demo_i % len(names))
    col.skip(ui.px(6))
    d = S.DEMOS[st.demo_i % len(S.DEMOS)]
    for line in (d.about, d.stim.activate[0].query,
                 f"{d.stim.activate[0].rate_hz:g} Гц · {d.t:g} с"
                 + (f" · без {d.stim.silence}" if d.stim.silence else "")):
        _note(ui, col, line)


def _rail_agent(ui, sess, st, col):
    from .. import spec as S

    _section(ui, col, "агент-экспериментатор")
    lb_h = min(max(col.left - ui.row - ui.small.gh - ui.line(True) * 2
                   - ui.px(12), 0), ui.row * 7)
    if lb_h >= ui.row * 2:
        st.goal_i = ui.list_box("goals", col.take(lb_h), S.GOALS,
                                st.goal_i % len(S.GOALS))
    col.skip(ui.px(6))
    full = ui.row + ui.small.gh
    if col.left >= full:
        st.agent_steps = int(ui.slider("asteps", col.take(full), "шагов",
                                       st.agent_steps, 2, 12, 1))
    model = (sess.hello.models[0] if getattr(sess.hello, "models", None)
             else "gpt-oss")
    _note(ui, col, "модель: " + str(model), theme.FAINT)
    _note(ui, col, "отчёты опытов лягут в out/", theme.FAINT)


def _rail_runs(ui, sess, st, col):
    _section(ui, col, "прогоны")
    _note(ui, col, "список и разбор — в середине окна")


# ---------------------------------------------------------------- показания

def meters(ui, sess, st, b, names, code, counts, cols):
    ui.back.rect(b.x0, b.y0, b.x1, b.y1, theme.PANEL)
    ui.back.vline(b.x0, b.y0, b.y1, theme.HAIRLINE)
    col = kit.Column(b.inset(ui.pad, ui.pad), ui.px(2))

    for lbl, val in (("время", format(sess.t_ms / 1000.0, ".2f") + " с"),
                     ("активных", str(sess.n_active)),
                     ("пик", format(sess.peak_hz, ".0f") + " Гц"),
                     ("шкала", format(sess.scale, ".0f") + " Гц"),
                     ("кадров", str(sess.frames_seen))):
        r = col.take(ui.line())
        ui.value(lbl, val, r.x0, r.y0, r.w)
    col.skip(ui.px(8))
    ui.over.hline(col.area.x0, col.area.x1, col.take(1).y0, theme.HAIRLINE)
    col.skip(ui.px(8))

    if st.mode == "body" and sess.series:
        ui.label("рулёжка", col.area.x0, col.take(ui.line(True)).y0,
                 theme.FAINT, small=True)
        dl_, dr_ = sess.series_array("dn_l", 1), sess.series_array("dn_r", 1)
        top = max(float(dl_[0]) if len(dl_) else 0.0,
                  float(dr_[0]) if len(dr_) else 0.0, 1.0)
        for nm, key in (("слева", "dn_l"), ("справа", "dn_r")):
            v = sess.series_array(key, 1)
            cur = float(v[0]) if len(v) else 0.0
            ui.meter(col.take(ui.row), cur / top, theme.ACCENT, nm,
                     format(cur, ".0f") + " Гц")
        for nm, key in (("амплитуда L", "amp_l"), ("амплитуда R", "amp_r")):
            s = sess.series_array(key, 160)
            if len(s):
                r = col.take(ui.row)
                ui.label(nm, r.x0, ui._mid(r, ui.small), theme.DIM, small=True)
                ui.spark(kit.Box(r.x0 + ui.px(86), r.y0, r.x1, r.y1), s,
                         theme.INK)
        col.skip(ui.px(8))
        ui.over.hline(col.area.x0, col.area.x1, col.take(1).y0, theme.HAIRLINE)
        col.skip(ui.px(8))

    if _result_block(ui, sess, st, col):
        return

    _section(ui, col, "области")
    lv = sess.region_levels(code, len(names), counts)
    order = np.argsort(-lv)
    top = float(lv[order[0]]) if len(order) else 1.0
    space = col.left - ui.line() * 6
    rows = max(3, int(space // (ui.row)))
    for i in order[:rows]:
        if lv[i] <= 0 and st.mode:
            break
        r = col.take(ui.row)
        name = names[int(i)]
        ui.meter(r, float(lv[i]) / max(top, 1e-9), cols[name], name,
                 format(float(lv[i]), ".1f"))
        # Щелчок по полосе области включает фокус на ней в карте.
        if ui.input.clicked and r.contains(ui.input.x, ui.input.y):
            st.focus_click = int(i)
    col.skip(ui.px(6))

    if sess.log:
        ui.over.hline(col.area.x0, col.area.x1, col.take(1).y0, theme.HAIRLINE)
        col.skip(ui.px(4))
        ui.log_view("log", col.rest, sess.log[-200:])


def _result_block(ui, sess, st, col):
    """Итог прогона там, где у стимуляции живут полосы областей.

    У зонда и у серии живой активности по областям нет смысла: смотреть надо
    на ранжирование популяций и на разницу между условиями. Возвращает True,
    если место занято итогом и полосы областей рисовать не нужно.
    """
    s = sess.summary or {}
    kind = s.get("kind", "")

    if kind == "lateral":
        _section(ui, col, "кто кодирует сторону")
        rows = s.get("top") or []
        for r in rows[: max(3, int(col.left // max(ui.row, 1)) - 1)]:
            if col.left < ui.row:
                break
            box = col.take(ui.row)
            strength = min(abs(float(r.get("r", 0.0))), 1.0)
            ui.meter(box, strength, theme.ACCENT, str(r.get("population", "")),
                     f"{float(r.get('r', 0.0)):+.2f}")
        return True

    if kind.endswith("-bench"):
        _section(ui, col, "серия с контролем")
        bench = s.get("bench") or {}
        conds = list(bench.items())
        for label, r in conds:
            if col.left < ui.line() * 2:
                break
            mean = r.get("grad_alignment_mean")
            sem = r.get("grad_alignment_sem")
            box = col.take(ui.line())
            ui.value(label, (f"{mean:+.3f}" if mean is not None else "—"),
                     box.x0, box.y0, box.w)
            if sem is not None:
                b2 = col.take(ui.line(True))
                ui._txt_right(ui.small, f"± {sem:.3f}  n={r.get('n_runs', '—')}",
                              b2.x1, b2.y0, theme.FAINT)
        if len(conds) == 2:
            from .. import index as I

            runs = [c[1].get("runs") for c in conds]
            if all(runs):
                w = I._welch(runs[0], runs[1])
                if w and col.left >= ui.line() * 3:
                    col.skip(ui.px(6))
                    for lbl, val in (
                            ("разница", f"{w['diff']:+.3f} ± {w['se']:.3f}"),
                            ("Welch t", f"{w['t']:.2f}"),
                            ("значимость",
                             ("p = %.3f" % w["p"]) if w["p"] is not None
                             else "—")):
                        if col.left < ui.line():
                            break
                        box = col.take(ui.line())
                        ui.value(lbl, val, box.x0, box.y0, box.w)
        return True

    return False


# ---------------------------------------------------------------- полоса тела

def strip(ui, sess, st, b):
    """Нижняя полоса: кадр из MuJoCo и путь мухи."""
    if b.w < 40 or b.h < 40:
        return
    left, right = b.cols(2, ui.gap)
    frame = sess.body_frame if st.mode == "body" else None
    if frame is not None:
        ui.image_box(left, np.asarray(frame), "вид сверху")
    else:
        ui.dl.rect(left.x0, left.y0, left.x1, left.y1, theme.BG)
        ui.dl.frame(left.x0, left.y0, left.x1, left.y1, theme.HAIRLINE)
        note = ("арена рисуется следом за путём" if st.mode == "arena"
                else "вид тела появится на ходу")
        ui._txt_centre(ui.small, note, left.x0, left.x1,
                       ui._mid(left, ui.small), theme.FAINT)

    # Путь копится по шагам тела, а не по кадрам окна: окно рисует 60 раз в
    # секунду, а шаг управления приходит два-три раза, и без этой проверки в
    # путь попадали бы двадцать копий одной точки.
    if sess.extra and sess.extra.get("t_ms") != st.track_t:
        st.track_t = sess.extra.get("t_ms")
        st.track.append((sess.extra.get("x", 0.0), sess.extra.get("y", 0.0)))
        if len(st.track) > 4000:
            del st.track[:2000]
    if len(st.track) > 2:
        ui.track(right, np.asarray(st.track, dtype=np.float32), theme.ACCENT)
    else:
        ui.dl.rect(right.x0, right.y0, right.x1, right.y1, theme.BG)
        ui.dl.frame(right.x0, right.y0, right.x1, right.y1, theme.HAIRLINE)
        ui._txt_centre(ui.small, "путь появится на ходу", right.x0, right.x1,
                       ui._mid(right, ui.small), theme.FAINT)


# ---------------------------------------------------------------- подвал

def foot(ui, sess, sh, b, lay):
    ui.back.hline(b.x0, b.x1, b.y0, theme.HAIRLINE)
    y = ui._mid(b, ui.small)
    left = (sess.device or "—") + " · " + str(sess.n) + " нейронов"
    if sess.frames_dropped:
        left += " · кадров потеряно " + str(sess.frames_dropped)
    ui._txt(ui.small, left, b.x0 + ui.pad, y, theme.DIM)
    right = (str(sh.w) + "×" + str(sh.h) + " · " + format(sh.fps, ".0f")
             + " кадр/с")
    ui._txt_right(ui.small, right, b.x1 - ui.pad, y, theme.FAINT)
    if lay.notes:
        ui._txt_centre(ui.small, lay.notes[0], b.x0, b.x1, y, theme.FAINT)


# ---------------------------------------------------------------- карта

def brain_toolbar(ui, bv, b):
    """Полоска над картой.

    В совсем узкой карте не рисуется вовсе: подпись переключателя там всё
    равно не прочесть, а место отнимает у самой карты.

    Подстраивается под ширину: в узком окне остаётся только переключатель
    режима, а кнопки уходят. Прибивать их к пикселям нельзя — это окно
    побывало и 1500 пикселей шириной, и 257.
    """
    if b.w < ui.px(240):
        return b
    r = kit.Box(b.x0, b.y0, b.x1, b.y0 + ui.row)
    seg_w = min(ui.px(210), max(ui.px(120), r.w // 2))
    seg, tail = r.cut_left(seg_w, ui.gap)
    mode = ui.segmented("bmode", seg, ("активность", "области"),
                        0 if bv.mode == "activity" else 1)
    bv.set_mode("activity" if mode == 0 else "regions")

    want = [("bview", bv.view_name if bv.view_i >= 0 else "свой", None),
            ("blab", "подписи", "show_labels"),
            ("bact", "активные", "only_active")]
    if bv.focus is not None:
        want.append(("bfoc", "снять фокус", "focus"))
    btn_w = ui.px(110)
    fits = max(0, tail.w // (btn_w + ui.gap))
    for key, text, attr in want[:fits]:
        piece, tail = tail.cut_right(btn_w, ui.gap)
        if attr is None:
            if ui.button(key, piece, text):
                bv.next_view()
        elif attr == "focus":
            if ui.button(key, piece, text):
                bv.set_focus(None)
        else:
            setattr(bv, attr, ui.toggle(key, piece, text, getattr(bv, attr)))
    return kit.Box(b.x0, r.y1 + ui.gap, b.x1, b.y1)


def hover_tip(ui, sess, bv, inp, vp=None):
    """Подсказка про нейрон под курсором.

    Держится внутри карты, а не внутри окна: иначе у правого края она
    наезжает на колонку показаний.
    """
    if bv.hover is None:
        return
    txt = sess.describe(bv.hover)
    if not txt:
        return
    reg = bv.region_names[int(bv.region_code[bv.hover])]
    hz = float(sess.act[bv.hover]) if sess.act is not None else 0.0
    txt = txt + " · " + reg + " · " + format(hz, ".0f") + " Гц"
    w = ui.small.width(txt) + ui.px(16)
    h = ui.line(True) + ui.px(6)
    lo_x, lo_y, hi_x, hi_y = (0, 0, ui.w, ui.h) if vp is None else (
        vp[0], vp[1], vp[0] + vp[2], vp[1] + vp[3])
    x = min(max(int(inp.x) + ui.px(16), lo_x + 2), hi_x - w - 2)
    y = min(max(int(inp.y) - h - ui.px(6), lo_y + 2), hi_y - h - 2)
    ui.top.rect(x, y, x + w, y + h, theme.PANEL, 0.97)
    ui.top.frame(x, y, x + w, y + h, theme.HAIRLINE)
    ui._txt(ui.small, txt, x + ui.px(8), y + ui.px(3), theme.INK)
