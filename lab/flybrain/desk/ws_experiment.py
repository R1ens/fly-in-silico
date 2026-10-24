"""Рабочее пространство «Эксперимент»: сборка и запуск научных сценариев.

Второго движка опытов здесь нет. Вид опыта, его поля, умолчания, пределы
и то, что он даёт на выходе, берутся из `flybrain/catalog.py`, а сам
справочник собран по `spec.py`, `jobs.py` и `cli.py` и сверяется с ними
проверкой. Запуск уходит той же `RunSpec` и тем же `jobs`, что и у
командной строки: окно ничего не упрощает и ничего не добавляет.

Три вещи, которые это пространство обязано делать честно:

* показывать **до** запуска, что именно получится — какие файлы, какие
  ряды, будут ли кадры активности и кадры картинки;
* проверять параметры и предупреждать о цене (память под растр, время
  счёта, множитель серии), не запрещая запуск там, где он законен;
* в серии различать **запланированные, выполненные, отменённые и
  ошибочные** эпизоды, а не показывать одну полоску «прогресс».

Статистическую значимость это пространство не объявляет: серия даёт
среднее, sd и sem по эпизодам, и этим ограничивается.
"""

import time

from .. import catalog as K
from . import layout as L
from . import widgets as W
from .tokens import C, R, S

__all__ = ["SPACE"]

TABS = ("Что получится", "Серия", "Результат", "Журнал")

KIND_ORDER = ("stim", "arena", "body", "lateral", "verify")


# ------------------------------------------------------------ состояние

def _state(app):
    """Состояние пространства живёт на приложении: переживает вкладки."""
    st = getattr(app, "_exp", None)
    if st is None:
        st = app._exp = {"kind": "stim", "values": {}, "terms": [
            {"query": "class:gustatory+side:left", "rate_hz": 200.0,
             "t_on_ms": 0.0, "t_off_ms": None, "name": ""}],
            "series": {}, "last_spec": None}
        _reset_values(st, "stim")
    return st


def _reset_values(st, kind):
    """Значения по умолчанию для вида — те, что рекомендует справочник."""
    st["kind"] = kind
    st["values"] = {}
    for f in K.fields_of(kind):
        if f.type == "list":
            continue
        st["values"][_fk(f)] = f.shown_default


def _fk(f):
    return (f.block + "." if f.block else "") + f.key


def _auto_out(st, kind):
    """Имя папки по отметке времени — одно на сборку, а не на кадр.

    Раньше оно собиралось прямо в `build_spec`, то есть заново каждый
    кадр: подпись «результат ляжет в out/…» бежала секундами, и обещанное
    до нажатия имя никогда не совпадало с тем, что уходило работнику.
    Имя живёт до конца этого прогона и выбирается свободным: две задачи в
    одну секунду не должны лечь в одну папку.
    """
    name = st.get("auto_out") or ""
    if name and name.endswith("-" + kind):
        return name
    base = time.strftime("%Y%m%d-%H%M%S-") + kind
    name = base
    try:
        from ..data import OUT

        for i in range(2, 100):
            if not (OUT / name).exists():
                break
            name = "%s-%d" % (base, i)
    except Exception:                                         # noqa: BLE001
        pass                      # нет доступа к диску — хватит отметки
    st["auto_out"] = name
    return name


def build_spec(app):
    """Собрать словарь спеки из состояния. Ровно то, что уйдёт работнику."""
    st = _state(app)
    kind = st["kind"]
    out = {"kind": kind, "open_report": False}
    for f in K.fields_of(kind):
        if f.type == "list":
            continue
        v = st["values"].get(_fk(f), f.shown_default)
        if v is None or (isinstance(v, str) and not v.strip()):
            continue            # пусто значит «умолчание модуля», не ноль
        if f.block:
            out.setdefault(f.block, {})[f.key] = v
        else:
            out[f.key] = v
    # Папка результата. Работник сохраняет прогон только когда имя
    # задано, иначе опыт считается и пропадает — а «доступ к результатам
    # завершённого опыта» без папки невозможен. Пустое поле значит не
    # «не сохранять», а «назови сам», как у командной строки.
    if not out.get("out"):
        out["out"] = _auto_out(st, kind)
    if kind == "stim":
        terms = [dict(t) for t in st["terms"] if str(t.get("query", "")).strip()]
        for t in terms:
            if t.get("t_off_ms") in ("", None):
                t.pop("t_off_ms", None)
            t.pop("name", None) if not t.get("name") else None
        out.setdefault("stim", {})["activate"] = terms
    return out


def apply_spec(app, spec):
    """Положить готовую спеку в окно сборки. Ничего не запускает.

    Так работает «Повторить опыт» из «Анализа»: настройки прошлого
    прогона переносятся сюда, человек их видит, при желании правит и
    пускает сам. Имя папки нарочно сбрасывается — повтор создаёт новый
    результат, исходный не трогается.
    """
    spec = dict(spec or {})
    kind = str(spec.get("kind") or "")
    if kind not in KIND_ORDER:
        return False, ("вид %r это окно не запускает: в «Эксперименте» "
                       "пять видов, агент живёт отдельно" % kind)
    st = _state(app)
    _reset_values(st, kind)
    for fld in K.fields_of(kind):
        if fld.type == "list":
            continue
        src = (spec.get(fld.block) or {}) if fld.block else spec
        if isinstance(src, dict) and fld.key in src:
            st["values"][_fk(fld)] = src[fld.key]
    st["values"]["out"] = None
    st["auto_out"] = ""
    if kind == "stim":
        terms = [t for t in ((spec.get("stim") or {}).get("activate") or [])
                 if isinstance(t, dict)]
        st["terms"] = [{"query": str(t.get("query") or ""),
                        "rate_hz": float(t.get("rate_hz") or 200.0),
                        "t_on_ms": float(t.get("t_on_ms") or 0.0),
                        "t_off_ms": (None if t.get("t_off_ms") is None
                                     else float(t["t_off_ms"])),
                        "name": str(t.get("name") or "")}
                       for t in terms] or [
            {"query": "", "rate_hz": 200.0, "t_on_ms": 0.0,
             "t_off_ms": None, "name": ""}]
    # Текстовые поля помнят свой текст в редакторах окна: без сброса в них
    # остались бы прежние строки, и показанное разошлось бы со спекой.
    ctx = getattr(app.sh, "ctx", None)
    if ctx is not None:
        for i in range(8):
            ctx.drop_editor("exp/term%d" % i)
        for fld in K.fields_of(kind):
            ctx.drop_editor("exp/f/" + _fk(fld))
    return True, ""


# ---------------------------------------------------------------- центр

def centre(app, ctx, r):
    sc, f = app.sh.sc, app.sh.fonts
    st = _state(app)
    kind = st["kind"]
    spec = build_spec(app)
    problems = K.validate(spec)
    warnings = K.limits(spec)

    inner = W.panel(ctx, r, "")
    content, token = W.scroll(ctx, "exp/mid", inner)
    s = L.Stack(L.Rect(content.x, content.y, content.w, 1 << 16),
                gap=sc.px(S.sm))

    rec = K.KINDS[kind]
    W.section(ctx, s, rec.title, first=True, wid="exp/kind",
              info=rec.about)
    h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                rec.about, role="small", color=C.dim, wrapped=True)
    s.take(h, sc.px(S.sm))
    for note in (rec.notes or ())[:2]:
        h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                    "— " + note, role="small", color=C.faint, wrapped=True)
        s.take(h, sc.px(S.xs))

    if kind == "stim":
        _terms_editor(app, ctx, s, content, float(spec.get("t") or 1.0))

    # --- проверка ---
    W.section(ctx, s, "Проверка", wid="exp/check",
              info="Слева красным — то, что мешает запуску. Жёлтым — "
                   "цена: память, время, множитель серии. "
                   "Предупреждение запуску не мешает.")
    if problems:
        for p in problems:
            h = W.label(ctx, L.Rect(content.x, s.area.y, content.w,
                                    s.area.h), "✕  " + p, role="small",
                        color=C.danger, wrapped=True)
            s.take(h, sc.px(S.xs))
    else:
        W.label(ctx, s.take(f.small.gh + sc.px(S.xs)),
                "параметры в порядке", role="small", color=C.ok)
    for wmsg in warnings:
        h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                    "!  " + wmsg, role="small", color=C.warn, wrapped=True)
        s.take(h, sc.px(S.xs))

    # --- запуск ---
    W.section(ctx, s, "Запуск")
    W.label(ctx, s.take(f.small.gh + sc.px(S.xs)),
            "результат ляжет в out/" + str(spec.get("out", "")),
            role="small", color=C.faint)
    run = app.st.run
    row = s.take(sc.px(S.control_lg))
    cells = row.cols(3, gap=sc.px(S.sm))
    ready = app.st.worker.ready and not problems and not run.busy
    if run.busy:
        if W.button(ctx, "exp/stop", cells[0], "Стоп", kind="danger",
                    icon="stop", enabled=(run.phase != "cancel"),
                    tip="остановить между шагами"):
            app.cancel_run()
    else:
        tip = ("запустить опыт" if ready else
               (problems[0] if problems else "работник не готов"))
        if W.button(ctx, "exp/run", cells[0], "Запустить",
                    kind="primary", icon="play", enabled=ready, tip=tip):
            st["last_spec"] = spec
            app.start_run(spec, title=rec.title)
            st["auto_out"] = ""   # следующему прогону — своя папка
    # Перейти туда, где это видно
    where = {"stim": ("brain", "Смотреть в «Мозге»"),
             "arena": ("brain", "Смотреть в «Мозге»"),
             "verify": ("brain", "Смотреть в «Мозге»"),
             "lateral": ("brain", "Смотреть в «Мозге»"),
             "body": ("body", "Смотреть в «Теле»")}.get(kind)
    if where and W.button(ctx, "exp/goto", cells[1], where[1],
                          icon="expand",
                          tip="перейти туда, где виден ход этого опыта"):
        app.st.ws.switch(where[0])
    if W.button(ctx, "exp/defaults", cells[2], "Умолчания", icon="reset",
                enabled=not run.busy,
                tip="вернуть значения, рекомендованные для этого вида"):
        _reset_values(st, kind)

    if run.busy:
        s.skip(sc.px(S.sm))
        W.progress(ctx, s.take(sc.px(34)),
                   max(0.0, run.progress) if run.progress > 0 else 0.0,
                   "идёт: " + (run.stage or rec.title),
                   indeterminate=(run.progress <= 0))

    W.scroll_done(ctx, "exp/mid", inner, s.used, token)


def _terms_editor(app, ctx, s, content, spec_t=1.0):
    """Список стимулов: кого зажечь, как сильно и когда."""
    sc, f = app.sh.sc, app.sh.fonts
    st = _state(app)
    W.section(ctx, s, "Стимулы", wid="exp/terms",
              info="Каждая строка — пуассоновский вход в свою популяцию. "
                   "Пустое «гаснет» значит «горит до конца прогона», а "
                   "не ноль.")
    drop = None
    for i, t in enumerate(st["terms"]):
        if not s.fits(sc.px(S.control) * 2 + sc.px(S.sm)):
            break
        row = s.take(sc.px(S.control))
        kill, field = row.cut_right(sc.px(34), sc.px(S.sm))
        ti = ctx.editor("exp/term%d" % i, str(t.get("query", "")))
        W.text_field(ctx, "exp/term%d" % i, field, ti, "селектор",
                     icon="search")
        t["query"] = ti.text
        if W.icon_button(ctx, "exp/termdel%d" % i, kill, "close",
                         enabled=len(st["terms"]) > 1,
                         tip="убрать этот стимул"):
            drop = i
        row = s.take(sc.px(S.control))
        three = row.cols(3, gap=sc.px(S.sm))
        t["rate_hz"] = W.numeric(ctx, "exp/rate%d" % i, three[0],
                                 float(t.get("rate_hz") or 200.0),
                                 0.0, K.MAX_POISSON_HZ, 10.0, "Гц",
                                 label="частота", fmt="{:.0f}")
        t["t_on_ms"] = W.numeric(ctx, "exp/ton%d" % i, three[1],
                                 float(t.get("t_on_ms") or 0.0),
                                 0.0, 600000.0, 50.0, "мс",
                                 label="начало", fmt="{:.0f}",
                                 tip="Начало, мс — с какой миллисекунды "
                                     "прогона стимул включается.")
        off = t.get("t_off_ms")
        val = W.numeric(ctx, "exp/toff%d" % i, three[2],
                        float(off) if off is not None else 0.0,
                        0.0, 600000.0, 50.0, "мс", label="конец",
                        fmt="{:.0f}",
                        tip="Конец, мс — с какой миллисекунды стимул "
                            "гаснет. Ноль здесь не «погасить сразу»: "
                            "`off = steps if t_off_ms is None` в sim.run, "
                            "то есть стимул горит до конца прогона.")
        t["t_off_ms"] = None if val <= 0 else val
        # Смысл нулевого конца пишется словами под строкой, а не только в
        # подсказке: «конец 0» читается как «выключен», и это ровно
        # наоборот.
        t_full = float(spec_t) * 1000.0
        on = float(t.get("t_on_ms") or 0.0)
        if val <= 0:
            say = ("горит с %g мс и до конца прогона (%g мс) — "
                   "ноль в «конце» значит именно это" % (on, t_full))
            col = C.warn
        elif val <= on:
            say = ("конец %g мс не позже начала %g мс: окно пустое, "
                   "стимул не зажжётся" % (val, on))
            col = C.danger
        else:
            say = ("горит с %g до %g мс — окно %g мс из %g мс прогона"
                   % (on, val, val - on, t_full))
            col = C.faint
        h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                    say, role="small", color=col, wrapped=True)
        s.take(h, sc.px(S.xs))
    if drop is not None and len(st["terms"]) > 1:
        st["terms"].pop(drop)
        for i in range(len(st["terms"]) + 1):
            ctx.drop_editor("exp/term%d" % i)
    row = s.take(sc.px(S.control))
    if W.button(ctx, "exp/termadd", row.cols(2, gap=sc.px(S.sm))[0],
                "Добавить стимул", icon="plus",
                enabled=len(st["terms"]) < 8):
        st["terms"].append({"query": "", "rate_hz": 200.0,
                            "t_on_ms": 0.0, "t_off_ms": None, "name": ""})


# --------------------------------------------------------------- рейка

def inspector(app, ctx, r):
    sc, f = app.sh.sc, app.sh.fonts
    st = _state(app)
    content, token = W.scroll(ctx, "exp/insp", r)
    s = L.Stack(L.Rect(content.x, content.y, content.w, 1 << 16),
                gap=sc.px(S.sm))

    W.section(ctx, s, "Вид опыта", first=True, wid="exp/pick",
              info="Пять настоящих видов прогона, те же, что у "
                   "командной строки. Агента здесь нет: он порождает "
                   "подпрогоны и живёт отдельно.")
    titles = [K.KINDS[k].title for k in KIND_ORDER]
    cur = KIND_ORDER.index(st["kind"]) if st["kind"] in KIND_ORDER else 0
    i = W.combo(ctx, "exp/kindsel", s.take(sc.px(S.control)), titles, cur,
                enabled=not app.st.run.busy)
    if KIND_ORDER[i] != st["kind"]:
        _reset_values(st, KIND_ORDER[i])

    # Поля вида: общие сверху, блок вида ниже — так их и группирует
    # справочник. Редкие (Ollama, заголовок отчёта, первый сид серии)
    # помечены в справочнике `advanced` и уходят под раскрытие: они
    # нужны в одном прогоне из двадцати, а место в рейке занимали всегда.
    all_f = [x for x in K.fields_of(st["kind"]) if x.type != "list"]
    общие = [x for x in all_f if x.block is None and not x.advanced]
    свои = [x for x in all_f if x.block and not x.advanced]
    редкие = [x for x in all_f if x.advanced]
    for title, group in (("Общее", общие), ("Параметры опыта", свои)):
        if not group:
            continue
        W.section(ctx, s, title)
        for fld in group:
            if not s.fits(sc.px(S.control)):
                break
            _field(app, ctx, s, content, fld)
    if редкие:
        open_ = bool(st.get("show_rare"))
        st["show_rare"] = W.section(
            ctx, s, "Редкие настройки", wid="exp/rare", fold=open_,
            right="%d" % len(редкие),
            info="Разбор и комментарий локальной моделью, модель Ollama, "
                 "заголовок отчёта, первый сид серии, предохранитель "
                 "длины. Поля никуда не делись: они те же поля "
                 "`RunSpec`, просто свёрнуты.")
        if st["show_rare"]:
            for fld in редкие:
                if not s.fits(sc.px(S.control)):
                    break
                _field(app, ctx, s, content, fld)

    W.scroll_done(ctx, "exp/insp", r, s.used, token)


def _nice(v):
    """Подпись варианта: число без хвоста, строка как есть."""
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    return str(v)


def _field(app, ctx, s, content, fld):
    """Один параметр. Вид органа управления — по типу из справочника."""
    sc, f = app.sh.sc, app.sh.fonts
    st = _state(app)
    key = _fk(fld)
    val = st["values"].get(key, fld.shown_default)
    wid = "exp/f/" + key
    tip = fld.hint or fld.label
    if fld.none_means:
        tip += " · пусто значит: " + fld.none_means
    busy = app.st.run.busy

    if fld.type == "bool":
        st["values"][key] = W.toggle(ctx, wid, s.take(sc.px(S.row)),
                                     bool(val), fld.label, enabled=not busy,
                                     tip=tip)
        return
    if fld.type in ("float", "int"):
        lo = fld.min if fld.min is not None else 0.0
        hi = fld.max if fld.max is not None else 1e9
        step = 1.0 if fld.type == "int" else max(
            0.01, round((hi - lo) / 100.0, 3) if hi < 1e6 else 1.0)
        # Подпись в строку с полем помещается не всегда: «Длительность
        # эпизода серии» в рейке 280 px превращалась в «Длительность э…»,
        # а таких полей у арены два разных. Длинную кладём над полем.
        inline = f.body.width(fld.label) <= content.w * 0.42
        if not inline:
            W.label(ctx, s.take(f.small.gh + sc.px(S.xs)), fld.label,
                    role="small", color=C.dim)
        v = W.numeric(ctx, wid, s.take(sc.px(S.control)),
                      float(val if val is not None else lo), lo, hi, step,
                      fld.unit or "", label=fld.label if inline else "",
                      fmt="{:.0f}" if fld.type == "int" else "{:g}",
                      enabled=not busy, tip=tip)
        st["values"][key] = int(v) if fld.type == "int" else float(v)
        return
    if fld.type == "choice":
        raw = list(fld.choices) if fld.choices else []
        if not raw and "probe_candidates" in (fld.choices_from or ""):
            raw = list(app.st.worker.readouts)
        labels = [_nice(o) for o in raw]
        if fld.none_means:
            raw = [None] + raw
            labels = ["— " + fld.none_means] + labels
        cur = 0
        for idx, o in enumerate(raw):
            if o == val or (o is None and val is None):
                cur = idx
                break
        W.label(ctx, s.take(f.small.gh + sc.px(S.xs)), fld.label,
                role="small", color=C.dim)
        j = W.combo(ctx, wid, s.take(sc.px(S.control)),
                    labels or ["(нет вариантов)"], cur,
                    enabled=bool(raw) and not busy, tip=tip)
        if raw:
            st["values"][key] = raw[min(j, len(raw) - 1)]
        return
    # строка и селектор
    W.label(ctx, s.take(f.small.gh + sc.px(S.xs)), fld.label,
            role="small", color=C.dim)
    ti = ctx.editor(wid, "" if val is None else str(val))
    W.text_field(ctx, wid, s.take(sc.px(S.control)), ti,
                 fld.none_means or "",
                 icon="search" if fld.type == "query" else None,
                 enabled=not busy)
    st["values"][key] = ti.text or None


# ------------------------------------------------------- правая колонка

def secondary(app, ctx, r, tab):
    if tab == 0:
        _tab_measures(app, ctx, r)
    elif tab == 1:
        _tab_series(app, ctx, r)
    elif tab == 2:
        _tab_result(app, ctx, r)
    else:
        _tab_log(app, ctx, r)


def _tab_measures(app, ctx, r):
    sc, f = app.sh.sc, app.sh.fonts
    st = _state(app)
    kind = st["kind"]
    m = K.measures_of(kind)
    content, token = W.scroll(ctx, "exp/t0", r)
    s = L.Stack(L.Rect(content.x, content.y, content.w, 1 << 16),
                gap=sc.px(S.sm))

    W.section(ctx, s, "Файлы в out/", first=True,
              info="Что появится в папке прогона. Список взят из кода "
                   "сохранения, а не придуман.")
    for name, when, what in m.files:
        if not s.fits(sc.px(S.row)):
            break
        row = s.take(f.body.gh + sc.px(S.xs))
        W.label(ctx, row, name, role="body", color=C.ink)
        h = W.label(ctx, L.Rect(row.x, s.area.y, row.w, s.area.h),
                    when + " · " + what, role="small", color=C.faint,
                    wrapped=True)
        s.take(h, sc.px(S.sm))

    W.section(ctx, s, "Что придёт в окно по ходу")
    rows = [("кадры активности", "да" if m.activity_frames else "нет"),
            ("кадры картинки", "да" if m.picture_frames else "нет")]
    ser = K.series_of(kind)
    rows.append(("ряды величин", ", ".join(ser) if ser else "нет"))
    for name, val in rows:
        if not s.fits(sc.px(S.row)):
            break
        row = s.take(sc.px(S.row))
        W.label(ctx, L.Rect(row.x, row.y + (row.h - f.body.gh) // 2,
                            row.w, f.body.gh), name, role="body",
                color=C.dim)
        W.label(ctx, row, val, role="small", align="right",
                color=C.ok if val not in ("нет",) else C.faint)

    if m.summary:
        W.section(ctx, s, "В сводке прогона")
        for item in m.summary[:10]:
            if not s.fits(f.small.gh):
                break
            W.label(ctx, s.take(f.small.gh + sc.px(2)), "· " + str(item),
                    role="small", color=C.faint)
    W.scroll_done(ctx, "exp/t0", r, s.used, token)


def KINDS_TITLE(kind):
    rec = K.KINDS.get(kind)
    return rec.title if rec is not None else str(kind)


def _planned(spec):
    """Сколько эпизодов запланировано. Серия считает норму и контроль."""
    rep = int(spec.get("repeat", 0) or 0)
    return rep * 2 if rep else 1


def _tab_series(app, ctx, r):
    """Счёт эпизодов серии. Исходы не пересекаются — это проверяется.

    Запись о серии ведёт `App`: он знает номер задачи и ход, который шлёт
    работник, и закрывает серию ровно один раз — в момент, когда прогон
    кончился. Вкладка только показывает, ничего не досчитывая: пока счёт
    жил здесь, при уходе в другое пространство эпизоды переставали
    считаться, а чужой одиночный прогон наполнял чужую запись.
    """
    sc, f = app.sh.sc, app.sh.fonts
    st = _state(app)
    run = app.st.run
    ser = app.series
    content, token = W.scroll(ctx, "exp/t1", r)
    s = L.Stack(L.Rect(content.x, content.y, content.w, 1 << 16),
                gap=sc.px(S.sm))

    planned_now = _planned(build_spec(app))
    W.section(ctx, s, "Эпизоды", first=True,
              info="Серия гоняет по два эпизода на повтор: норму и "
                   "контроль (перестановку антенн). Выполненные приходят "
                   "по ходу от работника — это доля эпизодов, а не "
                   "времени. Запланировано = выполнено + отменено + с "
                   "ошибкой: эти исходы не пересекаются.")
    if ser is None:
        if planned_now > 1:
            h = W.label(ctx, L.Rect(content.x, s.area.y, content.w,
                                    s.area.h),
                        "Серия настроена: %d эпизодов (%d повтора × два "
                        "условия). Счёт появится здесь после пуска."
                        % (planned_now, planned_now // 2),
                        role="small", color=C.dim, wrapped=True)
        else:
            h = W.label(ctx, L.Rect(content.x, s.area.y, content.w,
                                    s.area.h),
                        "Серия не настроена. Поставьте «эпизодов в серии» "
                        "больше нуля в параметрах опыта — тогда прогон "
                        "пойдёт эпизодами, и здесь появится их счёт.",
                        role="small", color=C.faint, wrapped=True)
        s.take(h)
        W.scroll_done(ctx, "exp/t1", r, s.used, token)
        return

    planned = int(ser.get("planned", 0) or 0)
    done = int(ser.get("done", 0) or 0)
    cancelled = int(ser.get("cancelled", 0) or 0)
    failed = int(ser.get("failed", 0) or 0)
    closed = bool(ser.get("closed"))
    left = max(0, planned - done - cancelled - failed)
    mine = (run.job_id == ser.get("job")) and run.busy

    W.status(ctx, s.take(sc.px(S.row)),
             ("run" if mine else {"done": "done", "fail": "fail"}.get(
                 ser.get("phase", ""), "idle")),
             ("идёт: %s" % (ser.get("title") or ser.get("kind"))) if mine
             else ("серия %s: %s" % (ser.get("job", "")[:8],
                                     _outcome_word(ser))))
    rows = [("запланировано", planned, C.ink),
            ("выполнено", done, C.ok if done else C.dim),
            ("отменено", cancelled, C.warn if cancelled else C.dim),
            ("с ошибкой", failed, C.danger if failed else C.dim)]
    if not closed:
        rows.append(("ещё не начато", left, C.dim))
    for name, val, col in rows:
        if not s.fits(sc.px(S.row)):
            break
        row = s.take(sc.px(S.row))
        W.label(ctx, L.Rect(row.x, row.y + (row.h - f.body.gh) // 2,
                            row.w, f.body.gh), name, role="body",
                color=C.dim)
        W.label(ctx, row, str(val), role="small", align="right", color=col)
    if s.fits(sc.px(34)):
        W.progress(ctx, s.take(sc.px(34)), done / max(planned, 1),
                   "выполнено %d из %d" % (done, planned))
    # Сумма сходится по построению, но показать её дешевле, чем доверять:
    # именно здесь раньше стояло «выполнено 4» и «отменено 4» сразу.
    total = done + cancelled + failed + (0 if closed else left)
    if s.fits(f.small.gh):
        W.label(ctx, s.take(f.small.gh + sc.px(S.xs)),
                "%d + %d + %d%s = %d" % (
                    done, cancelled, failed,
                    "" if closed else (" + %d" % left), total),
                role="small",
                color=C.faint if total == planned else C.danger)
    if ser.get("episode_t") and s.fits(f.small.gh):
        W.label(ctx, s.take(f.small.gh + sc.px(S.xs)),
                "эпизод %g с" % float(ser["episode_t"]),
                role="small", color=C.faint)
    if closed and ser.get("out_dir") and s.fits(sc.px(S.control)):
        if W.button(ctx, "exp/serout", s.take(sc.px(S.control)),
                    "Открыть папку серии", icon="table",
                    tip=str(ser["out_dir"])):
            _open_path(app, ser["out_dir"])
    h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                "Серия даёт среднее, стандартное отклонение и sem по "
                "эпизодам в bench.json. Значимость по ним окно не "
                "объявляет: два-три эпизода её не дают, и считать её "
                "за вас было бы подлогом.", role="small", color=C.warn,
                wrapped=True)
    s.take(h)
    W.scroll_done(ctx, "exp/t1", r, s.used, token)


def _outcome_word(ser):
    if not ser.get("closed"):
        return "идёт"
    if int(ser.get("failed", 0) or 0):
        return "прервана ошибкой"
    if int(ser.get("cancelled", 0) or 0):
        return "остановлена"
    return "пройдена целиком"


def _tab_result(app, ctx, r):
    sc, f = app.sh.sc, app.sh.fonts
    last = app.st.last
    content, token = W.scroll(ctx, "exp/t2", r)
    s = L.Stack(L.Rect(content.x, content.y, content.w, 1 << 16),
                gap=sc.px(S.sm))
    if last is None:
        W.section(ctx, s, "Результата ещё нет", first=True)
        h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                    "Здесь появится последний законченный опыт: вид, "
                    "номер задачи, время и папка с файлами.",
                    role="small", color=C.faint, wrapped=True)
        s.take(h)
        W.scroll_done(ctx, "exp/t2", r, s.used, token)
        return

    outcome = last.outcome
    ok = outcome == "done"
    word = {"done": ": пройден целиком", "cancelled": ": остановлен",
            "fail": ": отказ"}.get(outcome, ": " + outcome)
    W.section(ctx, s, "Последний опыт", first=True)
    W.status(ctx, s.take(sc.px(S.row)),
             "done" if ok else ("cancel" if outcome == "cancelled"
                                else "fail"),
             (last.title or last.kind) + word)
    if outcome == "cancelled":
        h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                    "Опыт остановлен не дойдя до конца. Файлы сохранены, "
                    "но длительность в них подрезана до посчитанного "
                    "(jobs._truncate), и частоты относятся к этому "
                    "укороченному времени, а не к заказанному.",
                    role="small", color=C.warn, wrapped=True)
        s.take(h, sc.px(S.xs))
    W.kv(ctx, s.take(sc.px(S.row)), "номер задачи", last.job_id or "—")
    W.kv(ctx, s.take(sc.px(S.row)), "модельное время",
         ("%.2f" % last.t_sim).replace(".", ","), "с")
    W.kv(ctx, s.take(sc.px(S.row)), "по стене",
         ("%.1f" % last.t_wall).replace(".", ","), "с")
    if last.out_dir:
        W.kv(ctx, s.take(sc.px(S.row)), "папка",
             last.out_dir.replace("\\", "/").split("/")[-1])
        row = s.take(sc.px(S.control))
        half = row.cols(2, gap=sc.px(S.sm))
        if W.button(ctx, "exp/open", half[0], "Открыть папку",
                    icon="table", tip=last.out_dir):
            _open_path(app, last.out_dir)
        if W.button(ctx, "exp/report", half[1], "Отчёт", icon="chart",
                    tip="report.html, если он есть"):
            _open_path(app, last.out_dir, "report.html")
    elif not ok:
        h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                    last.message or "подробности в журнале", role="small",
                    color=C.danger, wrapped=True)
        s.take(h)
    else:
        h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                    "Файлов нет: работник сохраняет прогон только когда "
                    "задано имя папки, а при отказе сохранять нечего — "
                    "объекта прогона не получилось.",
                    role="small", color=C.faint, wrapped=True)
        s.take(h)
    if last.out_dir and s.fits(f.small.gh):
        W.label(ctx, s.take(f.small.gh + sc.px(S.xs)),
                "рядом лежит run.json: вид, отменён ли, когда кончился",
                role="small", color=C.faint)
    W.scroll_done(ctx, "exp/t2", r, s.used, token)


def _open_path(app, out_dir, name=""):
    """Открыть папку или файл средствами системы. Ничего не считает."""
    import os
    from pathlib import Path

    p = Path(out_dir)
    if name:
        p = p / name
    if not p.exists():
        app.st.note("нет такого пути: " + str(p), "warn")
        return
    try:
        os.startfile(str(p))                                  # noqa: S606
        app.st.note("открыто: " + str(p))
    except Exception as exc:                                  # noqa: BLE001
        app.st.note("не открылось: " + str(exc)[:120], "warn")


def _tab_log(app, ctx, r):
    sc, f = app.sh.sc, app.sh.fonts
    st = app.st
    content, token = W.scroll(ctx, "exp/t3", r)
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
    W.scroll_done(ctx, "exp/t3", r, s.used, token)


SPACE = {"centre": centre, "inspector": inspector, "secondary": secondary,
         "tabs": TABS}
