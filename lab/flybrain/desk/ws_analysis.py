"""Рабочее пространство «Анализ»: сохранённые прогоны и что в них лежит.

Второго хранилища здесь нет и форматы не меняются. Каталог строит
`flybrain/runs.py` по тем же файлам, что пишут `sim.save`, `loop.save`,
`body.save` и `worker._save`; графики перечисляет и читает
`flybrain/plots.py`; статистику по сериям считает `flybrain/stats.py`;
сопоставимость величин решает `flybrain/compare.py`. Окно только
показывает и спрашивает.

Три правила, на которых держится это пространство:

* **Существование папки успехом не является.** Прогон может быть
  завершён, остановлен на полпути, упасть, оказаться неполным — или его
  исход может быть неизвестен, и так выглядят все прогоны командной
  строки. Пять состояний, и они не смешиваются.
* **График рисуется только там, где в файлах есть ось.** Временные ряды
  берутся из `track.parquet` (`t_ms`) и `bins.npz` (номер бина × `bin_ms`).
  Из агрегатов временных рядов не делается.
* **Тяжёлое читается по выбору и в стороне.** При открытии пространства
  не открывается ни один `parquet` и ни один `npz`: обзор каталога стоит
  три сотых секунды. Таблицы и графики грузятся отдельной нитью, и
  загрузку можно отменить.
"""

import threading
import time

import numpy as np

from .. import compare as CP
from .. import plots as PL
from .. import replay as RP
from .. import runs as RN
from .. import stats as ST
from .. import timeline as TLM
from . import charts
from . import layout as L
from . import widgets as W
from .tokens import C, R, S

__all__ = ["SPACE"]

TABS = ("Параметры", "Файлы", "Оговорки", "Действия")
VIEWS = ("Обзор", "Графики", "Таблицы", "Сравнение", "Статистика",
         "Таймлайн")

KIND_FILTER = (("", "все виды"), ("stim", "Стимуляция"), ("arena", "Арена"),
               ("body", "В теле"), ("lateral", "Перебор популяций"),
               ("verify", "Сверка"), ("bench", "Серия"), ("agent", "Агент"),
               ("unknown", "Неопознанные"))

STATE_FILTER = (("", "любое состояние"), ("done", "завершён"),
                ("cancelled", "остановлен"), ("failed", "отказ"),
                ("partial", "неполный"), ("broken", "повреждён"),
                ("unknown", "исход неизвестен"))

STATE_COLOR = {"done": C.ok, "cancelled": C.warn, "failed": C.danger,
               "partial": C.warn, "broken": C.danger, "unknown": C.faint}

STATE_DOT = {"done": "done", "cancelled": "cancel", "failed": "fail",
             "partial": "prep", "broken": "fail", "unknown": "idle"}

# Оговорки, которые относятся ко всему пространству, а не к одному прогону.
SCIENCE = (
    "Повреждённый файл не прячется и не меняет вид прогона: запись "
    "остаётся в каталоге с именем файла и самой ошибкой.",
    "Прогон на диске — это запись, а не повторный счёт. Числа здесь те же, "
    "что получились тогда; пересчитать их этим окном нельзя.",
    "Порог записи 0,5 Гц: нейроны, молчавшие весь прогон, в файлы не "
    "попали вовсе. «Нет в файле» и «молчал» здесь одно и то же, и среднее "
    "по области считается по попавшим, а не по всей области.",
    "Исход прогона командной строки в файлах не записан. Окно говорит "
    "«неизвестно» и не выдаёт это за успех.",
    "Отменённый прогон сохраняется с подрезанной длительностью "
    "(`jobs._truncate`): частоты в нём относятся к посчитанному времени, а "
    "не к заказанному.",
    "Шаги внутри эпизода зависимы: поворот держится около сотни "
    "миллисекунд, градиент — секунды. За независимые наблюдения они не "
    "берутся ни в одном расчёте этого пространства.",
    "Значимость это окно не объявляет. Оно показывает среднее, разброс, "
    "sem, интервал и p там, где их честно посчитать, и называет вещи "
    "своими именами.",
    "Объединять серии вручную можно, но проверить непересечение эпизодов "
    "нечем: первый сид серии в файлы не пишется.",
)


# ------------------------------------------------------------- состояние

class _Loader:
    """Чтение тяжёлого файла в стороне от кадра. Отменяется.

    Отменить чтение `parquet` посреди него нельзя — отменяется ожидание:
    окно перестаёт ждать ответ и больше его не примет. Нить при этом
    дочитывает и умирает сама, не трогая состояние.
    """

    def __init__(self):
        self.key = ""
        self.started = 0.0
        self.result = None
        self.error = ""
        self._thread = None
        self._done = threading.Event()

    @property
    def busy(self):
        return self._thread is not None and not self._done.is_set()

    @property
    def waited(self):
        return (time.perf_counter() - self.started) if self.started else 0.0

    def start(self, key, fn):
        self.cancel()
        self.key = key
        self.started = time.perf_counter()
        self.result = None
        self.error = ""
        self._done = threading.Event()
        done = self._done

        def work():
            try:
                out = fn()
            except Exception as exc:                          # noqa: BLE001
                if not done.is_set():
                    self.error = str(exc)[:200]
                done.set()
                return
            if not done.is_set():
                self.result = out
            done.set()

        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()

    def cancel(self):
        if self._thread is not None and not self._done.is_set():
            self._done.set()           # ответ больше не нужен
            self.result = None
            self.error = "отменено"
        self._thread = None

    def take(self):
        """Забрать готовое. Вернуть (есть ли ответ, ответ, ошибка)."""
        if self._thread is None or not self._done.is_set():
            return False, None, ""
        self._thread = None
        return True, self.result, self.error


def _state(app):
    st = getattr(app, "_ana", None)
    if st is None:
        st = app._ana = {
            "runs": None, "scanned": 0.0, "scan_ms": 0.0,
            "kind": 0, "state": 0, "view": 0,
            "cursor": "", "sel": [], "plot": "", "table": "",
            "detail": None, "detail_for": "",
            "data": None, "data_key": "",
            "loader": _Loader(), "msg": "", "msg_kind": "dim",
            "pool": False,
        }
    return st


def _ensure(app):
    """Каталог читается один раз и по кнопке. Тяжёлого не открывает."""
    st = _state(app)
    if st["runs"] is None:
        t0 = time.perf_counter()
        try:
            st["runs"] = RN.scan()
        except Exception as exc:                              # noqa: BLE001
            st["runs"] = []
            st["msg"] = "каталог не прочитался: " + str(exc)[:120]
            st["msg_kind"] = "error"
        st["scan_ms"] = (time.perf_counter() - t0) * 1000.0
        st["scanned"] = time.time()
        if st["runs"] and not st["cursor"]:
            st["cursor"] = st["runs"][0].name
        _warm_tables()
    return st


_WARMED = [False]


def _warm_tables():
    """Подтянуть pandas сразу, а не на первой же открытой таблице.

    Окно само по себе pandas не грузит — таблицы читает работник. Первый
    `read_parquet` тянет за собой pandas и pyarrow, а `scipy.stats` нужен
    для квантиля Стьюдента. Оба импорта стоят около секунды каждый и
    случались прямо в кадре: на первой открытой таблице и на первой
    серии окно замирало. Греем один раз, сразу после обзора каталога,
    когда человек ещё читает список.
    """
    if _WARMED[0]:
        return
    _WARMED[0] = True

    def work():
        try:
            import pandas  # noqa: F401
        except Exception:                                     # noqa: BLE001
            pass
        ST.warm()      # scipy: первый импорт стоит секунду с лишним

    threading.Thread(target=work, daemon=True).start()


def _filtered(st):
    runs = st["runs"] or []
    kind = KIND_FILTER[st["kind"]][0]
    state = STATE_FILTER[st["state"]][0]
    q = (st.get("query") or "").strip().lower()
    out = []
    for r in runs:
        if kind and r.kind != kind:
            continue
        if state and r.state != state:
            continue
        if q:
            hay = (r.name + " " + r.title + " " + r.kind_title + " "
                   + str((r.spec or {}).get("stim", ""))).lower()
            if q not in hay:
                continue
        out.append(r)
    return out


def _current(app):
    st = _state(app)
    for r in (st["runs"] or []):
        if r.name == st["cursor"]:
            return r
    return None


def _selected(app):
    st = _state(app)
    by = {r.name: r for r in (st["runs"] or [])}
    return [by[n] for n in st["sel"] if n in by]


def _detail(app, run):
    """Маленькие файлы выбранного прогона. Кэш на один прогон."""
    st = _state(app)
    if run is None:
        return {}
    if st["detail_for"] != run.name:
        st["detail"] = RN.detail(run)
        st["detail_for"] = run.name
    return st["detail"] or {}


# ------------------------------------------------------------- рейка

def inspector(app, ctx, r):
    sc, f = app.sh.sc, app.sh.fonts
    st = _ensure(app)
    content, token = W.scroll(ctx, "ana/insp", r)
    s = L.Stack(L.Rect(content.x, content.y, content.w, 1 << 16),
                gap=sc.px(S.sm))

    runs = st["runs"] or []
    W.section(ctx, s, "Каталог прогонов", first=True, wid="ana/cat",
              right="%d" % len(runs),
              info="Папки в out/ как они есть. Обзор читает только "
                   "маленькие JSON-ы: ни parquet, ни npz при открытии "
                   "пространства не трогаются. Прогон, который не удалось "
                   "опознать, из списка не выбрасывается — «не узнали» "
                   "честнее молчания.")
    row = s.take(sc.px(S.control))
    btn, field = row.cut_right(sc.px(34), sc.px(S.sm))
    ti = ctx.editor("ana/q", st.get("query", ""))
    W.text_field(ctx, "ana/q", field, ti, "поиск по имени и запросу",
                 icon="search")
    st["query"] = ti.text
    if W.icon_button(ctx, "ana/rescan", btn, "reset",
                     tip="перечитать out/ заново"):
        st["runs"] = None
        st["detail_for"] = ""
        _ensure(app)
    st["kind"] = W.combo(ctx, "ana/kind", s.take(sc.px(S.control)),
                         [t for _k, t in KIND_FILTER], st["kind"],
                         tip="вид прогона определяется по файлам в папке, "
                             "а не по её имени")
    st["state"] = W.combo(ctx, "ana/state", s.take(sc.px(S.control)),
                          [t for _k, t in STATE_FILTER], st["state"],
                          tip="завершён, остановлен, отказ, неполный или "
                              "исход неизвестен")

    shown = _filtered(st)
    W.section(ctx, s, "Прогоны", right="%d из %d" % (len(shown), len(runs)))
    if not shown:
        h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                    "Под фильтр ничего не попало."
                    if runs else
                    "В out/ нет ни одной папки с прогоном.",
                    role="small", color=C.faint, wrapped=True)
        s.take(h)
    rowh = f.body.gh + f.micro.gh + sc.px(S.sm)
    for run in shown:
        if not s.fits(rowh):
            break
        box = s.take(rowh, sc.px(2))
        _run_row(app, ctx, box, run)
    if st["sel"]:
        W.section(ctx, s, "В сравнении", right="%d" % len(st["sel"]))
        if s.fits(sc.px(S.control)):
            half = s.take(sc.px(S.control)).cols(2, gap=sc.px(S.sm))
            if W.button(ctx, "ana/cmp", half[0], "Сравнить", icon="table",
                        enabled=len(st["sel"]) >= 2,
                        tip="свести выбранные прогоны в одну таблицу"):
                st["view"] = 3
            if W.button(ctx, "ana/clr", half[1], "Снять выбор",
                        icon="close"):
                st["sel"] = []
    W.scroll_done(ctx, "ana/insp", r, s.used, token)


def _run_row(app, ctx, box, run):
    """Строка каталога: состояние, имя, вид и время. Щелчок выбирает."""
    dl, sc, f = ctx.dl, app.sh.sc, app.sh.fonts
    st = _state(app)
    cur = st["cursor"] == run.name
    picked = run.name in st["sel"]
    hover, _held, clicked, _fc = ctx.probe("ana/row/" + run.name, box, True)
    if cur or hover:
        dl.rect(box.x, box.y, box.x1, box.y1,
                C.selected if cur else C.hover, 1.0, sc.px(R.control))
    if cur:
        dl.rect(box.x, box.y + sc.px(2), box.x + sc.px(3), box.y1 - sc.px(2),
                C.accent, 1.0, 1.0)
    if clicked:
        st["cursor"] = run.name
        st["plot"] = ""
        st["data_key"] = ""
        st["data"] = None
        st["loader"].cancel()
    pad = sc.px(S.sm)
    d = sc.px(7)
    dl.rect(box.x + pad, box.y + sc.px(6), box.x + pad + d,
            box.y + sc.px(6) + d, STATE_COLOR.get(run.state, C.faint),
            1.0, d / 2)
    # Правый край — чип «в сравнение»: множественный выбор делается им, а
    # не щелчком с клавишей, иначе о нём невозможно догадаться.
    cw = sc.px(26)
    cr = L.Rect(box.x1 - cw, box.y + sc.px(3), cw, box.h - sc.px(6))
    # Кнопкой, а не чипом: на чип текст не влезал и обрезался в многоточие,
    # и колонка выбора читалась как «тут что-то не поместилось».
    chip_hit = W.icon_button(ctx, "ana/pick/" + run.name, cr,
                             "check" if picked else "plus", selected=picked)
    if chip_hit:
        if picked:
            st["sel"].remove(run.name)
        elif len(st["sel"]) < 6:
            st["sel"].append(run.name)
        else:
            st["msg"] = "больше шести прогонов разом не сравнить"
            st["msg_kind"] = "warn"
    ctx.want_tip("ana/pick/" + run.name, cr,
                 "убрать из сравнения" if picked else "добавить в сравнение")
    x = box.x + pad + d + sc.px(S.sm)
    right = cr.x - sc.px(S.sm)
    dl.text(f.body, run.name, x, box.y + sc.px(3), C.ink if cur else C.dim,
            max_px=max(0, right - x))
    tail = run.kind_title + " · " + run.when + " · " + run.state_word
    dl.text(f.micro, tail, x, box.y + sc.px(3) + f.body.gh, C.faint,
            max_px=max(0, right - x))
    ctx.want_tip("ana/row/" + run.name, box,
                 "%s\n%s\n%s" % (run.title, run.state_why,
                                 "файлов: " + ", ".join(run.files[:8])))


# ------------------------------------------------------------- центр

def centre(app, ctx, r):
    sc, f = app.sh.sc, app.sh.fonts
    st = _ensure(app)
    run = _current(app)
    inner = W.panel(ctx, r, "")
    bar, body = inner.cut_top(sc.px(S.control) + sc.px(S.sm), sc.px(S.xs))
    _hit, picks = W.viewport_toolbar(
        ctx, "ana/tb", bar,
        [("seg", "view", list(VIEWS), st["view"],
          ["что это за прогон и что в нём намерено",
           "графики по настоящим данным из файлов",
           "таблицы: нейроны, шаги, эпизоды",
           "несколько прогонов рядом",
           "серия: среднее, разброс, sem, критерий",
           "сеанс по модельному времени: что записано и чего нет"])])
    if "view" in picks:
        st["view"] = picks["view"]
        st["loader"].cancel()

    if run is None and st["view"] != 3:
        W.label(ctx, body, "Выберите прогон в каталоге слева.",
                role="small", color=C.faint, wrapped=True)
        return
    (_overview, _plots, _tables, _compare, _stats,
     _timeline)[st["view"]](app, ctx, body, run)


def _overview(app, ctx, r, run):
    sc, f = app.sh.sc, app.sh.fonts
    d = _detail(app, run)
    content, token = W.scroll(ctx, "ana/ov", r)
    s = L.Stack(L.Rect(content.x, content.y, content.w, 1 << 16),
                gap=sc.px(S.sm))

    W.section(ctx, s, run.title, first=True, wid="ana/ovh",
              right=run.kind_title)
    W.status(ctx, s.take(sc.px(S.row)), STATE_DOT.get(run.state, "idle"),
             "Состояние результата: " + run.state_word)
    h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                run.state_why, role="small",
                color=STATE_COLOR.get(run.state, C.faint), wrapped=True)
    s.take(h, sc.px(S.sm))
    _broken(app, ctx, s, content, run)

    if run.has_timeline:
        W.section(ctx, s, "Сеанс песочницы" if run.session
                  else "Таймлайн модельного времени", wid="ana/ovt",
                  right="вкладка «Таймлайн»",
                  info="Рядом с прогоном лежит единый таймлайн по "
                       "модельному времени: сенсорные каналы, выбранная "
                       "нейронная активность, нисходящие, моторные "
                       "команды, траектория и события среды. Смотреть "
                       "его — в виде «Таймлайн»; там же видно, на какие "
                       "моменты в записи данных нет.")
        tlm = run.timeline or {}
        for k in ("окно записи, с", "шагов записано", "шагов в сеансе",
                  "запись с, мс", "запись до, мс", "событий"):
            if tlm.get(k) is None or not s.fits(sc.px(S.row)):
                continue
            W.kv(ctx, s.take(sc.px(S.row)), k, _num(tlm[k]))
        if run.has_state and s.fits(sc.px(S.row)):
            W.kv(ctx, s.take(sc.px(S.row)), "снимок состояния",
                 "есть: сеанс можно продолжить")

    W.section(ctx, s, "Когда и сколько")
    W.kv(ctx, s.take(sc.px(S.row)), "папка", run.name)
    W.kv(ctx, s.take(sc.px(S.row)), "последняя запись", run.when)
    if run.t_requested is not None:
        W.kv(ctx, s.take(sc.px(S.row)), "заказано",
             ("%g" % run.t_requested).replace(".", ","), "с")
    if run.t_actual is not None:
        W.kv(ctx, s.take(sc.px(S.row)), "в файле",
             ("%g" % run.t_actual).replace(".", ","), "с")
    if run.n_episodes:
        W.kv(ctx, s.take(sc.px(S.row)), "эпизодов в серии",
             str(run.n_episodes))
        W.kv(ctx, s.take(sc.px(S.row)), "условий",
             ", ".join(run.series_conditions))

    stims = (d.get("meta.json") or {}).get("stimuli") or []
    if stims:
        W.section(ctx, s, "Стимулы", wid="ana/ovs",
                  info="Имя стимула — это сам запрос `data.select`. "
                       "Пустой «до» значит, что стимул горел до конца "
                       "прогона.")
        for t in stims[:6]:
            if not s.fits(f.body.gh + f.micro.gh):
                break
            W.label(ctx, s.take(f.body.gh), str(t.get("name") or "—"),
                    role="body", color=C.ink)
            off = t.get("t_off_ms")
            W.label(ctx, s.take(f.micro.gh + sc.px(S.xs)),
                    "%d нейронов · %g Гц · с %g мс %s"
                    % (int(t.get("n") or 0), float(t.get("rate_hz") or 0.0),
                       float(t.get("t_on_ms") or 0.0),
                       ("до %g мс" % float(off)) if off is not None
                       else "до конца прогона"),
                    role="micro", color=C.faint)
        if run.kind in ("arena", "body") and s.fits(f.micro.gh * 2):
            h = W.label(ctx, L.Rect(content.x, s.area.y, content.w,
                                    s.area.h),
                        "! У эпизода это сенсорные каналы, а не разовый "
                        "стимул: их частота пересчитывается на каждом шаге "
                        "из положения мухи, и в meta.json записано "
                        "значение на момент сохранения. Как менялся вход, "
                        "видно на графике «Запах на антеннах».",
                        role="micro", color=C.warn, wrapped=True)
            s.take(h, sc.px(S.xs))
        n_sil = (d.get("meta.json") or {}).get("n_silenced") or 0
        if n_sil and s.fits(f.micro.gh):
            W.label(ctx, s.take(f.micro.gh + sc.px(S.xs)),
                    "заглушено нейронов: %d" % int(n_sil),
                    role="micro", color=C.warn)

    pop = _population_of(run, d)
    if pop:
        W.section(ctx, s, "Популяция съёма", wid="ana/ovp",
                  info="Кого слушала модель движения. Там, где опорная "
                       "частота и усиление измерялись на разогреве, в "
                       "файле лежат ИЗМЕРЕННЫЕ значения, а не заказанные.")
        for k, v in pop:
            if not s.fits(sc.px(S.row)):
                break
            W.kv(ctx, s.take(sc.px(S.row)), k, str(v))

    ver = d.get("verify.json")
    if isinstance(ver, dict):
        W.section(ctx, s, "Сверка с апстримом", wid="ana/ovv",
                  right={"ok": "сравнение сделано",
                         "cancelled": "отменена",
                         "upstream_failed": "апстрим упал",
                         "no_upstream_file": "нет файла апстрима"}.get(
                             ver.get("outcome"), str(ver.get("outcome"))),
                  info="Паспорт сверки из verify.json: с чем сравнивали, "
                       "на каких настройках и что получилось. Отменённая "
                       "сверка помечена отменённой, а не пройденной.")
        ds = ver.get("dataset") or {}
        rows = [("набор данных", ds.get("release")),
                ("нейронов в модели", ds.get("n_neurons")),
                ("эксперимент", ver.get("experiment_name")),
                ("нейронов под стимулом",
                 (ver.get("stimulus") or {}).get("n_matched")),
                ("частота, Гц", (ver.get("stimulus") or {}).get("rate_hz"))]
        if ver.get("compared"):
            rows += [("Jaccard", ver.get("jaccard")),
                     ("корреляция частот", ver.get("corr")),
                     ("активных у нас", ver.get("n_active_mine")),
                     ("активных у апстрима", ver.get("n_active_upstream")),
                     ("файл апстрима", ver.get("upstream_file"))]
        for k, v in rows:
            if v is None or not s.fits(sc.px(S.row)):
                continue
            W.kv(ctx, s.take(sc.px(S.row)), k, _num(v))
        if ver.get("why"):
            h = W.label(ctx, L.Rect(content.x, s.area.y, content.w,
                                    s.area.h),
                        "! " + str(ver["why"]), role="small", color=C.warn,
                        wrapped=True)
            s.take(h, sc.px(S.xs))
        for lim in (ver.get("limits") or []):
            if not s.fits(f.micro.gh):
                break
            h = W.label(ctx, L.Rect(content.x, s.area.y, content.w,
                                    s.area.h), "— " + str(lim),
                        role="micro", color=C.faint, wrapped=True)
            s.take(h, sc.px(2))

    met = CP._metrics_of(run, d)
    if met:
        W.section(ctx, s, "Что намерено", wid="ana/ovm",
                  info="Величины из главного файла вида, как они там "
                       "записаны. Это измерения прогона, а не оценки и не "
                       "интерпретация.")
        for k, v in list(met.items())[:16]:
            if not s.fits(sc.px(S.row)):
                break
            label, unit, _why = CP.MEASURES.get(k, (k, "", ""))
            W.kv(ctx, s.take(sc.px(S.row)), label,
                 _num(v), unit)

    avail = PL.available(run)
    W.section(ctx, s, "Доступные графики", right="%d" % len(avail),
              wid="ana/ovg",
              info="Строится только то, под что в папке есть файлы. "
                   "Временной ряд — лишь там, где в файле есть ось "
                   "времени.")
    if not avail:
        h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                    "По этому прогону рисовать нечего: ни таблицы "
                    "активности, ни записи по шагам в папке нет.",
                    role="small", color=C.faint, wrapped=True)
        s.take(h)
    for p in avail:
        if not s.fits(f.small.gh):
            break
        W.label(ctx, s.take(f.small.gh + sc.px(2)),
                "· %s  [%s]" % (p.title, p.unit or "безразмерная"),
                role="small", color=C.dim)
    W.scroll_done(ctx, "ana/ov", r, s.used, token)


def _broken(app, ctx, s, content, run):
    """Повреждённые файлы — поимённо и с самой ошибкой.

    Папка остаётся в каталоге: часть файлов цела, и прятать запись из-за
    одного битого JSON значит потерять прогон. Но и «завершён» про неё не
    говорится.
    """
    if not run.errors:
        return
    sc, f = app.sh.sc, app.sh.fonts
    W.section(ctx, s, "Повреждённые файлы", right="%d" % len(run.errors),
              wid="ana/brk",
              info="Эти файлы лежат в папке, но не читаются. Остальные "
                   "файлы прогона не тронуты, и то, что из них строится, "
                   "показывается как обычно.")
    for name, why in run.errors.items():
        if not s.fits(f.small.gh):
            break
        W.label(ctx, s.take(f.small.gh + sc.px(2)), name, role="small",
                color=C.danger)
        h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                    why, role="micro", color=C.warn, wrapped=True)
        s.take(h, sc.px(S.xs))


def _vision_block(app, ctx, s, content, run):
    """Происхождение зрения прогона: чем считалось и что это значит.

    Отсутствие блока — это «зрение было выключено»: так посчитаны все
    прежние прогоны, и ключа `timing["зрение"]` у них нет вовсе.
    Рисовать для них «зрение: нет» значило бы заводить строку, которой
    в данных не было.

    Что здесь ОБЯЗАНО быть видно, и это требование, а не украшение:
    какой переходник считал, какой привязкой, с каким отпечатком, и
    что привязка инженерная и провалила внешнюю проверку. Последнее
    приходит не отсюда, а из оговорок прогона — они выше и жёлтым; тут
    стоит короткая строка с тем же итогом, чтобы не пришлось сличать
    два места.
    """
    vis = run.vision
    if not isinstance(vis, dict):
        return
    sc, f = app.sh.sc, app.sh.fonts
    pr = vis.get("привязка") or {}

    def num(v):
        # Прочерк, а не «None»: отсутствующее число должно выглядеть
        # отсутствующим, иначе читающий примет его за значение.
        return "—" if v is None else _num(v)

    W.section(ctx, s, "Зрение", wid="ana/vis",
              right=str(pr.get("отпечаток набора") or ""),
              info="Справа — отпечаток набора привязок "
                   "(`retina.fingerprint_all`): по нему прогон "
                   "опознаётся однозначно, и два прогона с разным "
                   "отпечатком считались разной привязкой. Частота "
                   "обновления показана и заказанной, и ИЗМЕРЕННОЙ по "
                   "отметкам FlyGym: они обязаны совпадать, и если не "
                   "совпали — верить надо измеренной.")
    rows = [
        ("переходник", "%s v%s" % (pr.get("переходник", "?"),
                                   pr.get("версия переходника", "?")), ""),
        # Версия самой привязки, а не кода переходника: прежняя
        # привязка сохраняется как историческая, исправленная придёт
        # отдельным именем, и по прогону обязано быть видно, какая
        # считала.
        ("версия привязки", str(pr.get("версия привязки") or "—"), ""),
        ("глаза", str(pr.get("глаза", "?")), ""),
        ("фоторецепторов", str(pr.get("фоторецепторов всего", "?")), "шт"),
        ("обновление: заказано",
         num(vis.get("частота обновления заказана, Гц")), "Гц"),
        ("обновление: измерено",
         num(vis.get("частота обновления измерена, Гц")), "Гц"),
        ("обновлений за сеанс",
         num(vis.get("обновлений кадра за сеанс")), "шт"),
        ("привязка", "%s / %s" % (pr.get("способ") or
                                  (pr.get("правило") or ["", "", ""])[2],
                                  pr.get("назначение") or ""), ""),
        ("перемешана (встречный замер)",
         "да" if pr.get("перемешано (встречный замер)") else "нет", ""),
        ("контраст кадра", num(vis.get("контраст")), "×"),
    ]
    light = vis.get("свет сцены") or {}
    if light.get("множитель") is not None:
        rows.append(("свет сцены", num(light.get("множитель")), "×"))
    for title, val, unit in rows:
        if not s.fits(sc.px(S.row)):
            return
        W.kv(ctx, s.take(sc.px(S.row)), title, val, unit)
    # Итог внешней проверки — одной строкой и красным. Это не оговорка
    # про картинку, а то, чем привязка является.
    ext = (pr.get("внешняя проверка") or {}).get("итог")
    if ext and s.fits(f.micro.gh * 2):
        h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                    "внешняя проверка: " + str(ext), role="micro",
                    color=C.danger, wrapped=True)
        s.take(h, sc.px(S.xs))
    what = pr.get("что каналы обеспечивают")
    if what and s.fits(f.micro.gh * 2):
        h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                    str(what), role="micro", color=C.warn, wrapped=True)
        s.take(h, sc.px(S.xs))
    track = vis.get("траектория") or []
    if track and s.fits(f.micro.gh * 2):
        W.label(ctx, s.take(f.small.gh + sc.px(2)),
                "Траектория объектов", role="small", color=C.dim)
        for w in track:
            if not s.fits(f.micro.gh):
                break
            W.label(ctx, s.take(f.micro.gh + sc.px(2)),
                    "слот %s: %s → %s мм, r %s, серость %s, %s…%s мс"
                    % (w.get("слот"), w.get("из, мм"), w.get("в, мм"),
                       w.get("радиус, мм"), w.get("серость"),
                       w.get("с, мс"), w.get("по, мс")),
                    role="micro", color=C.faint)


def _population_of(run, d):
    out = []
    ep = d.get("episode.json") or d.get("embodied.json") or {}
    body = ep.get("body") or (ep.get("config") or {})
    if body.get("motor_left"):
        out.append(("съём слева", body["motor_left"]))
    if body.get("dn_ref_hz") is not None:
        out.append(("опорная частота, Гц", _num(body["dn_ref_hz"])))
    if body.get("turn_gain_rad_s") is not None:
        out.append(("усиление поворота, рад/с", _num(body["turn_gain_rad_s"])))
    if body.get("dn_tau_s") is not None:
        out.append(("сглаживание, с", _num(body["dn_tau_s"])))
    if body.get("swap_antennae") is not None:
        out.append(("антенны переставлены",
                    "да" if body["swap_antennae"] else "нет"))
    cal = (d.get("embodied.json") or {}).get("calibration") or {}
    for k, v in cal.items():
        out.append(("разогрев: " + k, _num(v)))
    b = d.get("bench.json") or {}
    if isinstance(b, dict):
        for rec in b.values():
            if isinstance(rec, dict) and rec.get("readout"):
                out.append(("съём серии", rec["readout"]))
                break
    return out


def _num(v):
    if isinstance(v, bool):
        return "да" if v else "нет"
    if isinstance(v, float):
        s = ("%.4f" % v).rstrip("0").rstrip(".")
        return s.replace(".", ",")
    return str(v)


# ------------------------------------------------------------- графики

def _plots(app, ctx, r, run):
    sc, f = app.sh.sc, app.sh.fonts
    st = _state(app)
    avail = PL.available(run)
    content, token = W.scroll(ctx, "ana/pl", r)
    s = L.Stack(L.Rect(content.x, content.y, content.w, 1 << 16),
                gap=sc.px(S.sm))
    if not avail:
        W.section(ctx, s, "Графиков нет", first=True)
        h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                    "В папке нет ни `bins.npz`, ни `rates.parquet`, ни "
                    "`track.parquet` — рисовать не из чего. Это не "
                    "поломка прогона: перебор популяций и серия сохраняют "
                    "только свои JSON-ы.",
                    role="small", color=C.faint, wrapped=True)
        s.take(h)
        W.scroll_done(ctx, "ana/pl", r, s.used, token)
        return
    if not st["plot"] or st["plot"] not in {p.key for p in avail}:
        st["plot"] = avail[0].key

    W.section(ctx, s, "Что рисуем", first=True, right="%d" % len(avail))
    # Чипы раскладываются по строкам заранее: место под них надо отрезать
    # ДО того, как они нарисованы, иначе график наезжает на последнюю
    # строку выбора, когда подписи длиннее обычного.
    cell = sc.px(S.row)
    widths = [min(content.w, f.small.width(p.title) + sc.px(S.xl))
              for p in avail]
    lines, cur, acc = [], [], 0
    for p, w in zip(avail, widths):
        if cur and acc + w > content.w:
            lines.append(cur)
            cur, acc = [], 0
        cur.append((p, w))
        acc += w + sc.px(S.sm)
    if cur:
        lines.append(cur)
    area = s.take(len(lines) * (cell + sc.px(S.xs)), sc.px(S.sm))
    cy = area.y
    for line in lines:
        cx = content.x
        for p, w in line:
            spot = L.Rect(cx, cy, w, cell)
            hit, _c = W.chip(ctx, "ana/p/" + p.key, spot, p.title,
                             selected=(st["plot"] == p.key))
            ctx.want_tip("ana/p/" + p.key, spot,
                         p.what + "\nисточник: " + p.source)
            if hit and st["plot"] != p.key:
                st["plot"] = p.key
                st["data"] = None
                st["data_key"] = ""
                st["loader"].cancel()
            cx += w + sc.px(S.sm)
        cy += cell + sc.px(S.xs)

    key = run.name + "/" + st["plot"]
    data = _want(app, key, lambda: PL.load(run, st["plot"]))
    if data is None:
        _loading(app, ctx, s, content, "график")
    elif data.get("cancelled"):
        h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                    "Чтение отменено. Данные не загружены — показывать "
                    "нечего, и рисовать по памяти окно не станет.",
                    role="small", color=C.warn, wrapped=True)
        s.take(h, sc.px(S.sm))
        if s.fits(sc.px(S.control)):
            if W.button(ctx, "ana/again",
                        s.take(sc.px(S.control)).cols(2,
                                                      gap=sc.px(S.sm))[0],
                        "Прочитать снова", icon="reset"):
                st["data"] = None
                st["data_key"] = ""
    elif data.get("error") or "plot" not in data:
        h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                    "Этот график не прочитался: "
                    + str(data.get("error") or "файл есть, а данных для "
                                               "этого графика в нём нет"),
                    role="small", color=C.danger, wrapped=True)
        s.take(h)
    else:
        # Высота графика считается от настоящей высоты панели, а не от
        # высоты курсора прокрутки: стопка бесконечная, и по ней график
        # на окне 1280×720 получался выше самой панели.
        room = max(sc.px(200), min(r.h - sc.px(150), sc.px(620)))
        box = L.Rect(content.x, s.area.y, content.w, room)
        used = charts.draw(ctx, box, data)
        s.take(used)
    W.scroll_done(ctx, "ana/pl", r, s.used, token)


def _want(app, key, fn):
    """Данные по ключу: из кэша, иначе запустить чтение в стороне."""
    st = _state(app)
    ld = st["loader"]
    if st["data_key"] == key and st["data"] is not None:
        return st["data"]
    got, res, err = ld.take()
    if got and ld.key == key:
        if err or res is None:
            st["msg"] = "не прочиталось: " + (err or "данных нет")
            st["msg_kind"] = "error" if err else "warn"
            st["data_key"] = key
            st["data"] = {"error": err or "в файле нет данных для этого "
                                          "графика"}
            return st["data"]
        st["data"], st["data_key"] = res, key
        return res
    if not ld.busy or ld.key != key:
        ld.start(key, fn)
    return None


def _loading(app, ctx, s, content, what):
    sc, f = app.sh.sc, app.sh.fonts
    st = _state(app)
    ld = st["loader"]
    W.progress(ctx, s.take(sc.px(34)), 0.0,
               "читаю %s: %.1f с" % (what, ld.waited), indeterminate=True)
    if s.fits(sc.px(S.control)):
        half = s.take(sc.px(S.control)).cols(2, gap=sc.px(S.sm))
        if W.button(ctx, "ana/stop", half[0], "Отменить чтение",
                    icon="stop", kind="danger"):
            # Отмену надо запомнить: без отметки следующий же кадр снова
            # попросил бы те же данные, и кнопка ничего бы не меняла.
            key = ld.key
            ld.cancel()
            st["data_key"] = key
            st["data"] = {"cancelled": True}
    h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                "Файл читается отдельной нитью: окно не встаёт. Отмена "
                "перестаёт ждать ответ — само чтение досчитается и "
                "пропадёт, прервать `parquet` посреди разбора нечем.",
                role="small", color=C.faint, wrapped=True)
    s.take(h)


# ------------------------------------------------------------- таблицы

def _tables(app, ctx, r, run):
    sc, f = app.sh.sc, app.sh.fonts
    st = _state(app)
    d = _detail(app, run)
    content, token = W.scroll(ctx, "ana/tb", r)
    s = L.Stack(L.Rect(content.x, content.y, content.w, 1 << 16),
                gap=sc.px(S.sm))

    rows = d.get("lateral.json") or d.get("probes.json")
    if isinstance(rows, list) and rows:
        name = "lateral.json" if d.get("lateral.json") else "probes.json"
        W.section(ctx, s, "Популяции", first=True, right=name,
                  info="Строки файла как есть, отсортированы по модулю "
                       "корреляции.")
        cols = [("population", "популяция", 0.34), ("r", "r", 0.14),
                ("lag_ms", "лаг, мс", 0.14), ("mean_hz", "средняя, Гц", 0.18),
                ("asym_sd_hz", "сигма, Гц", 0.2)]
        _grid(ctx, s, content, cols,
              sorted(rows, key=lambda x: -abs(float(x.get("r") or 0.0)))[:40])
        W.scroll_done(ctx, "ana/tb", r, s.used, token)
        return

    bench = d.get("bench.json")
    if isinstance(bench, dict) and bench:
        W.section(ctx, s, "Условия серии", first=True, right="bench.json",
                  info="`n_runs` — сколько эпизодов ВЫПОЛНЕНО, а не "
                       "сколько заказано: заказ в файл не пишется.")
        info = ST.series_stats(bench)
        cols = [("name", "условие", 0.3), ("usable", "эпизодов", 0.14),
                ("mean", "среднее", 0.16), ("sd", "sd", 0.14),
                ("sem", "sem", 0.14), ("moved_mm", "к цели, мм", 0.16)]
        flat = []
        for c in info["conditions"]:
            flat.append({"name": c["name"], "usable": c["usable"],
                         "mean": c["stats"]["mean"],
                         "sd": c["stats"]["sd"], "sem": c["stats"]["sem"],
                         "moved_mm": c["moved_mm"]})
        _grid(ctx, s, content, cols, flat)
        W.section(ctx, s, "Эпизоды поимённо", wid="ana/tbe",
                  info="Сырые значения из `runs`. У прогонов, сделанных "
                       "после правки, рядом лежат семена: по ним эпизод "
                       "повторяется и проверяется парность. У прежних "
                       "серий семян нет, и там это просто порядок.")
        for c in info["conditions"]:
            if not s.fits(f.small.gh * 2):
                break
            W.label(ctx, s.take(f.small.gh + sc.px(2)),
                    c["name"] + ("  ·  сиды %s" % c["seed0"]
                                 if c.get("seed0") is not None else
                                 "  ·  семян в файле нет"),
                    role="small", color=C.ink)
            seeds = c.get("seeds") or []
            text = "  ".join(
                (("[%s] %s" % (seeds[i], _num(v))) if i < len(seeds)
                 else _num(v)) for i, v in enumerate(c["raw"]))
            h = W.label(ctx, L.Rect(content.x, s.area.y, content.w,
                                    s.area.h), text,
                        role="micro", color=C.faint, wrapped=True)
            s.take(h, sc.px(S.xs))
        W.scroll_done(ctx, "ana/tb", r, s.used, token)
        return

    if run.has("rates.parquet"):
        W.section(ctx, s, "Нейроны по частоте", first=True,
                  right="rates.parquet",
                  info="Первые сорок строк файла. В файл попали только "
                       "нейроны от 0,5 Гц: молчавших в нём нет.")
        key = run.name + "/rates"
        df = _want(app, key, lambda: RN.rates_table(
            run, columns=["label", "cell_class", "region", "side",
                          "rate_hz", "n_spikes"]))
        if df is None:
            _loading(app, ctx, s, content, "таблицу нейронов")
        elif isinstance(df, dict):
            W.label(ctx, s.take(f.small.gh),
                    "чтение отменено" if df.get("cancelled")
                    else ("не прочиталось: " + str(df.get("error"))),
                    role="small",
                    color=C.warn if df.get("cancelled") else C.danger)
        else:
            cols = [("label", "метка", 0.3), ("cell_class", "класс", 0.2),
                    ("region", "область", 0.16), ("side", "сторона", 0.12),
                    ("rate_hz", "Гц", 0.12), ("n_spikes", "спайков", 0.12)]
            _grid(ctx, s, content, cols,
                  df.head(40).to_dict("records"))
        W.scroll_done(ctx, "ana/tb", r, s.used, token)
        return

    W.section(ctx, s, "Таблиц нет", first=True)
    h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                "В папке нет ни `rates.parquet`, ни `lateral.json`, ни "
                "`bench.json` — показывать таблицей нечего.",
                role="small", color=C.faint, wrapped=True)
    s.take(h)
    W.scroll_done(ctx, "ana/tb", r, s.used, token)


def _grid(ctx, s, content, cols, rows):
    """Простая таблица: шапка, волосяная линия, строки через одну."""
    dl, sc, f = ctx.dl, ctx.sc, ctx.f
    if not rows:
        W.label(ctx, s.take(f.small.gh), "строк нет", role="small",
                color=C.faint)
        return
    head = s.take(f.micro.gh + sc.px(S.xs))
    x = head.x
    for _k, title, frac in cols:
        w = int(content.w * frac)
        dl.text(f.micro, title, x, head.y, C.faint, max_px=w - sc.px(S.xs))
        x += w
    line = s.take(sc.hair(), sc.px(S.xs))
    dl.rect(line.x, line.y, line.x1, line.y + sc.hair(), C.border_soft,
            1.0, 0.0)
    for i, row in enumerate(rows):
        if not s.fits(f.small.gh + sc.px(4)):
            break
        box = s.take(f.small.gh + sc.px(4), 0)
        if i % 2:
            dl.rect(box.x, box.y, box.x1, box.y1, C.raised, 0.35, 0.0)
        x = box.x
        for k, _t, frac in cols:
            w = int(content.w * frac)
            v = row.get(k)
            dl.text(f.small, _num(v) if v is not None else "—", x,
                    box.y + sc.px(2), C.dim, max_px=w - sc.px(S.xs))
            x += w


# ------------------------------------------------------------- таймлайн

# Группы каналов показываются в этом порядке. Порядок объявлен здесь, а
# не выведен из словаря: он читается сверху вниз как путь сигнала — от
# входа к движению.
TL_ORDER = ("сенсор", "нейронная активность", "нисходящие",
            "моторные команды", "траектория")


def _timeline(app, ctx, r, run):                              # noqa: C901
    """Сеанс по модельному времени: что записано и чего нет.

    Главное здесь — не график, а ответ на вопрос «что было в такой-то
    миллисекунде». Ползунок ходит по модельному времени непрерывно, а
    запись идёт шагами управления, поэтому между шагами окно обязано
    сказать «нет данных» и назвать причину. Числа в такие моменты не
    выдумываются и не интерполируются: на шаге 20 мс интерполяция
    гладкой траектории отличается от правды на уровне шума, то есть
    выглядела бы правдоподобно и была бы выдумкой.
    """
    sc, f = app.sh.sc, app.sh.fonts
    st = _state(app)
    content, token = W.scroll(ctx, "ana/tl", r)
    s = L.Stack(L.Rect(content.x, content.y, content.w, 1 << 16),
                gap=sc.px(S.sm))
    if not run.has_timeline:
        W.section(ctx, s, "Таймлайна нет", first=True)
        h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                    "В папке нет `timeline.npz`. Это не поломка: единый "
                    "таймлайн пишут эпизоды арены и сеансы в теле, у "
                    "которых он включён (`ArenaSpec.timeline`, "
                    "`BodySpec.timeline`), а прогоны, сделанные до него, "
                    "его и не могли оставить. Ряды по шагам у таких "
                    "прогонов есть в `track.parquet` — вид «Графики».",
                    role="small", color=C.faint, wrapped=True)
        s.take(h)
        W.scroll_done(ctx, "ana/tl", r, s.used, token)
        return

    key = run.name + "/timeline"
    view = _want(app, key, lambda: RP.View.open(run.path))
    if view is None:
        _loading(app, ctx, s, content, "таймлайн")
        W.scroll_done(ctx, "ana/tl", r, s.used, token)
        return
    if isinstance(view, dict):
        W.section(ctx, s, "Таймлайн не прочитался", first=True)
        h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                    "чтение отменено" if view.get("cancelled")
                    else str(view.get("error") or "неизвестно почему"),
                    role="small", color=C.danger, wrapped=True)
        s.take(h)
        W.scroll_done(ctx, "ana/tl", r, s.used, token)
        return

    tl = view.tl
    lo, hi = tl.window
    if lo is None:
        W.section(ctx, s, "Запись пуста", first=True)
        W.scroll_done(ctx, "ana/tl", r, s.used, token)
        return

    W.section(ctx, s, "Окно записи", first=True, wid="ana/tlw",
              right="%g — %g мс" % (lo, hi),
              info="Кольцо держит последние `keep_s` секунд модельного "
                   "времени. Что старше — вытеснено, и на такой момент "
                   "окно отвечает «нет данных: раньше начала кольца», а "
                   "не нулём.")
    W.kv(ctx, s.take(sc.px(S.row)), "шаг управления",
         _num(tl.control_ms), "мс")
    W.kv(ctx, s.take(sc.px(S.row)), "шагов записано",
         str(int(getattr(tl, "n_rows", 0))))
    W.kv(ctx, s.take(sc.px(S.row)), "шагов в сеансе",
         str(int(getattr(tl, "steps_total", 0))))
    bud = tl.budget()
    W.kv(ctx, s.take(sc.px(S.row)), "предел памяти записи",
         _num(bud["всего, МБ"]), "МБ")
    W.kv(ctx, s.take(sc.px(S.row)), "плотный растр был бы",
         _num(bud["плотный растр был бы, МБ"]), "МБ")

    # --- момент ---
    cur = st.get("tl_t")
    if cur is None or not (lo - tl.control_ms <= cur <= hi + tl.control_ms):
        cur = lo
    W.section(ctx, s, "Что было в этот момент", wid="ana/tlt",
              info="Ползунок ходит по модельному времени непрерывно, а "
                   "запись идёт шагами управления. Между шагами данных "
                   "нет, и окно говорит именно это: интерполировать "
                   "соседей значило бы выдумать состояние, которого "
                   "модель не считала.")
    row = s.take(sc.px(S.control_lg))
    cur = W.slider(ctx, "ana/tl/t", row, float(cur),
                   float(lo - tl.control_ms), float(hi + tl.control_ms),
                   label="модельное время", unit="мс", fmt="{:.1f}",
                   tip="ведите между делениями — окно скажет «нет данных»")
    st["tl_t"] = cur
    three = s.take(sc.px(S.control)).cols(3, gap=sc.px(S.sm))
    if W.button(ctx, "ana/tl/snap", three[0], "На шаг", icon="check",
                tip="подвинуть к ближайшему записанному шагу"):
        times = tl.times()
        if len(times):
            st["tl_t"] = float(times[int(np.argmin(np.abs(times - cur)))])
    if W.button(ctx, "ana/tl/mid", three[1], "Между", icon="close",
                tip="встать ровно посередине между двумя записанными "
                    "шагами: ответом обязано быть «нет данных»"):
        times = tl.times()
        if len(times) > 1:
            j = int(np.clip(np.searchsorted(times, cur), 1, len(times) - 1))
            st["tl_t"] = float((times[j - 1] + times[j]) / 2.0)
    if W.button(ctx, "ana/tl/old", three[2], "До кольца", icon="reset",
                tip="момент старше окна записи"):
        st["tl_t"] = float(lo - tl.control_ms)

    answer = tl.row(cur)
    if isinstance(answer, TLM.Hole):
        h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                    "НЕТ ДАННЫХ. " + answer.text, role="small",
                    color=C.warn, wrapped=True)
        s.take(h, sc.px(S.sm))
        h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                    "Чисел в этот момент не будет. Соседние шаги записаны, "
                    "но среднее между ними — это не состояние модели, а "
                    "догадка о нём.", role="micro", color=C.faint,
                    wrapped=True)
        s.take(h, sc.px(S.sm))
    else:
        for group, _why in TLM.GROUPS:
            if group not in TL_ORDER:
                continue
            chans = [c for c in tl.chans if c.group == group]
            if not chans or not s.fits(sc.px(S.row) * 2):
                continue
            W.section(ctx, s, group)
            for c in chans:
                if not s.fits(sc.px(S.row)):
                    break
                v, hole = tl.value(cur, c.key)
                W.kv(ctx, s.take(sc.px(S.row)), c.title,
                     _num(v) if hole is None else "нет данных", c.unit)
        for name in tl.probe_names:
            if not s.fits(sc.px(S.row)):
                break
            v, hole = tl.population(cur, name)
            W.kv(ctx, s.take(sc.px(S.row)), "зонд " + name,
                 _num(v) if hole is None else "нет данных", "Гц")

    # --- объявленное, но не подключённое ---
    absent = [c for c in TLM.channels_of(tl.kind) if c.absent]
    if absent:
        W.section(ctx, s, "Объявлено, но не подключено",
                  right="%d" % len(absent), wid="ana/tla",
                  info="Канал объявлен заранее, чтобы спросить его можно "
                       "было уже сейчас. Ответ при этом — «канал не "
                       "подключён» с причиной, а не ноль: ноль значил бы "
                       "«измерено и ничего нет».")
        for c in absent:
            if not s.fits(f.micro.gh * 2):
                break
            W.label(ctx, s.take(f.small.gh + sc.px(2)), c.title,
                    role="small", color=C.dim)
            h = W.label(ctx, L.Rect(content.x, s.area.y, content.w,
                                    s.area.h), c.absent, role="micro",
                        color=C.warn, wrapped=True)
            s.take(h, sc.px(S.xs))

    # --- нейрон ---
    W.section(ctx, s, "Активность одного нейрона", wid="ana/tln",
              info="В растр попали только нейроны зондов. На нейрон вне "
                   "зондов ответ — «нет данных», а не ноль: «не "
                   "записывался» и «молчал» — разные вещи, и путать их "
                   "нельзя.")
    nrow = s.take(sc.px(S.control))
    ti = ctx.editor("ana/tl/n", st.get("tl_n", "0"))
    W.text_field(ctx, "ana/tl/n", nrow, ti, "номер нейрона", icon="search")
    st["tl_n"] = ti.text
    try:
        neuron = int(str(ti.text).strip() or "-1")
    except ValueError:
        neuron = -1
    v, hole = tl.neuron(cur, neuron)
    h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                ("спайков за шаг: %d" % v) if hole is None
                else ("НЕТ ДАННЫХ. " + hole.text),
                role="small", color=C.ink if hole is None else C.warn,
                wrapped=True)
    s.take(h, sc.px(S.sm))
    if s.fits(f.micro.gh * 2):
        h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                    "Растр целочислен: это счёт спайков за шаг. При "
                    "рефрактерном периоде 2,2 мс и шаге %g мс нейрон не "
                    "может выдать больше %d спайков, и дробное значение "
                    "здесь было бы признаком выдумки, а не округления."
                    % (tl.control_ms, tl.raster_ceiling()),
                    role="micro", color=C.faint, wrapped=True)
        s.take(h, sc.px(S.sm))

    # --- события ---
    ev = tl.events
    W.section(ctx, s, "События сеанса", right="%d" % len(ev), wid="ana/tle",
              info="Правки условий, вмешательства и изменения среды "
                   "стоят не рядами, а событиями: они случаются в один "
                   "момент модельного времени, и растягивать их на ряд "
                   "значило бы придумывать им длительность.")
    if not ev:
        W.label(ctx, s.take(f.small.gh), "событий в записи нет",
                role="small", color=C.faint)
    for e in ev[-40:]:
        if not s.fits(f.small.gh + f.micro.gh):
            break
        t = e.get("модельное, мс")
        W.label(ctx, s.take(f.small.gh + sc.px(2)),
                "%s мс · %s · %s" % ("—" if t is None else _num(t),
                                     e.get("группа"), e.get("что")),
                role="small", color=C.ink)
        detail = str(e.get("подробно") or "")
        if detail:
            h = W.label(ctx, L.Rect(content.x, s.area.y, content.w,
                                    s.area.h), detail, role="micro",
                        color=C.warn, wrapped=True)
            s.take(h, sc.px(S.xs))

    # --- стоимость просмотра ---
    W.section(ctx, s, "Чего стоил этот просмотр", wid="ana/tlc",
              info="Подпись стоимости режима. Просмотр записанного не "
                   "зовёт движок LIF ни разу — в этом режиме движка нет "
                   "вовсе: ни весов, ни тела. Пересчёт того же сеанса "
                   "стоил бы 200 вызовов на шаг управления.")
    cost = view.cost()
    for k in ("вызовов движка LIF", "ответов", "из них «нет данных»",
              "мс на ответ"):
        if not s.fits(sc.px(S.row)):
            break
        W.kv(ctx, s.take(sc.px(S.row)), k, _num(cost[k]))
    W.scroll_done(ctx, "ana/tl", r, s.used, token)


# ------------------------------------------------------------- сравнение

def _compare(app, ctx, r, run):
    sc, f = app.sh.sc, app.sh.fonts
    st = _state(app)
    sel = _selected(app)
    content, token = W.scroll(ctx, "ana/cm", r)
    s = L.Stack(L.Rect(content.x, content.y, content.w, 1 << 16),
                gap=sc.px(S.sm))
    if len(sel) < 2:
        W.section(ctx, s, "Сравнение", first=True)
        h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                    "Выбрано прогонов: %d. Отметьте в каталоге слева "
                    "хотя бы два — кнопкой «+» справа в строке. Сравнение "
                    "показывает настройки и измеренные величины рядом и "
                    "прямо говорит, где величины несопоставимы."
                    % len(sel), role="small", color=C.faint, wrapped=True)
        s.take(h)
        W.scroll_done(ctx, "ana/cm", r, s.used, token)
        return

    t = CP.table(sel)
    W.section(ctx, s, "Сравнение", first=True, right="%d прогона" % len(sel))
    for i, run_ in enumerate(sel):
        if not s.fits(f.small.gh):
            break
        W.label(ctx, s.take(f.small.gh + sc.px(2)),
                "%d) %s — %s, %s" % (i + 1, run_.name, run_.kind_title,
                                     run_.state_word),
                role="small", color=C.dim)
    for note in t["notes"]:
        h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                    "! " + note, role="small", color=C.warn, wrapped=True)
        s.take(h, sc.px(S.xs))

    W.section(ctx, s, "Измеренные величины", wid="ana/cmm",
              info="Красным — то, что сравнивать нельзя: величина есть не "
                   "у всех, либо под одним именем лежат разные вещи. Такая "
                   "строка не убирается: её отсутствие было бы обманом "
                   "не меньшим, чем ложное сравнение.")
    _cmp_rows(ctx, s, content, t["measures"], len(sel), numbers=True)

    W.section(ctx, s, "Настройки", wid="ana/cmp2",
              info="Заказанное, а не намеренное. Там, где опорная частота "
                   "и усиление мерялись на разогреве, они помечены.")
    _cmp_rows(ctx, s, content, t["params"], len(sel), numbers=False)
    W.scroll_done(ctx, "ana/cm", r, s.used, token)


def _cmp_rows(ctx, s, content, rows, n, numbers):
    dl, sc, f = ctx.dl, ctx.sc, ctx.f
    name_w = int(content.w * 0.3)
    cell = max(sc.px(40), (content.w - name_w) // max(n, 1))
    for i, (label, unit, vals, ok, why) in enumerate(rows):
        need = f.small.gh + sc.px(4) + (f.micro.gh if why else 0)
        if not s.fits(need):
            break
        box = s.take(need, 0)
        if i % 2:
            dl.rect(box.x, box.y, box.x1, box.y1, C.raised, 0.3, 0.0)
        dl.text(f.small, label + (", " + unit if unit else ""), box.x,
                box.y + sc.px(2), C.ink if ok else C.danger,
                max_px=name_w - sc.px(S.sm))
        x = box.x + name_w
        base = None
        for j, v in enumerate(vals):
            txt = _num(v) if v is not None else "—"
            col = C.dim if ok else C.faint
            if numbers and ok and j == 0:
                base = v
            elif numbers and ok and base is not None and v is not None:
                pc, _why = CP.percent(base, v)
                if pc is not None:
                    # Процент пишется тем же цветом, что и число: зелёный
                    # читался бы как «стало лучше», а чем лучше — окно не
                    # знает и знать не может.
                    txt += "  (%+.0f %% к первому)" % pc
            dl.text(f.small, txt, x, box.y + sc.px(2), col,
                    max_px=cell - sc.px(S.xs))
            x += cell
        if why:
            dl.text(f.micro, why, box.x, box.y + f.small.gh + sc.px(2),
                    C.warn if not ok else C.faint,
                    max_px=max(0, box.w - sc.px(S.sm)))


# ------------------------------------------------------------- статистика

def _stats(app, ctx, r, run):
    sc, f = app.sh.sc, app.sh.fonts
    st = _state(app)
    content, token = W.scroll(ctx, "ana/st", r)
    s = L.Stack(L.Rect(content.x, content.y, content.w, 1 << 16),
                gap=sc.px(S.sm))

    benches = []
    for x in (_selected(app) or ([run] if run is not None else [])):
        b = (RN.detail(x) or {}).get("bench.json")
        if isinstance(b, dict) and b:
            benches.append((x.name, b))
    if not benches:
        W.section(ctx, s, "Статистики нет", first=True)
        h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                    "Статистика считается по сериям: это единственное "
                    "место, где на диске лежат поэпизодные значения "
                    "(`bench.json`, ключ `runs`). У одиночного прогона "
                    "наблюдение одно, и разброса у него нет — считать "
                    "по шагам внутри эпизода нельзя, они зависимы.",
                    role="small", color=C.faint, wrapped=True)
        s.take(h)
        W.scroll_done(ctx, "ana/st", r, s.used, token)
        return

    pooled = False
    if len(benches) > 1:
        W.section(ctx, s, "Выбрано серий", first=True,
                  right="%d" % len(benches))
        st["pool"] = W.toggle(ctx, "ana/pool", s.take(sc.px(S.row)),
                              bool(st.get("pool")),
                              "сложить эпизоды в одну выборку",
                              tip="ручное объединение: проверить, что "
                                  "эпизоды не повторяются, нечем — первый "
                                  "сид серии в файлы не пишется")
        pooled = bool(st["pool"])

    notes = []
    if pooled:
        bench, notes = ST.pool(benches)
        title = "Объединено: " + ", ".join(n for n, _b in benches)
    else:
        name, bench = benches[0]
        title = name
    info = ST.series_stats(bench)

    W.section(ctx, s, title, first=not (len(benches) > 1), wid="ana/sth",
              info="Одно наблюдение — один эпизод. Критерий непарный: "
                   "соответствия эпизодов между условиями в файлах нет.")
    for note in notes:
        h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                    "! " + note, role="small", color=C.warn, wrapped=True)
        s.take(h, sc.px(S.xs))

    for c in info["conditions"]:
        if not s.fits(sc.px(S.row) * 2):
            break
        W.section(ctx, s, c["name"],
                  right="%d эпизодов" % c["usable"])
        stt = c["stats"]
        W.kv(ctx, s.take(sc.px(S.row)), "выполнено эпизодов", str(c["done"]))
        W.kv(ctx, s.take(sc.px(S.row)), "пригодных измерений",
             str(c["usable"]))
        W.kv(ctx, s.take(sc.px(S.row)), "среднее", _num(stt["mean"]))
        W.kv(ctx, s.take(sc.px(S.row)), "стандартное отклонение",
             _num(stt["sd"]) if stt["sd"] is not None else "—")
        W.kv(ctx, s.take(sc.px(S.row)), "стандартная ошибка",
             _num(stt["sem"]) if stt["sem"] is not None else "—")
        if stt.get("ci"):
            W.kv(ctx, s.take(sc.px(S.row)), "интервал 95 %",
                 "[%s; %s]" % (_num(stt["ci"][0]), _num(stt["ci"][1])))
            if not stt.get("exact_t") and s.fits(f.micro.gh):
                W.label(ctx, s.take(f.micro.gh + sc.px(2)),
                        "интервал по нормальному приближению: scipy нет, "
                        "квантиль Стьюдента взять неоткуда",
                        role="micro", color=C.warn)
        elif stt.get("why") and s.fits(f.micro.gh):
            W.label(ctx, s.take(f.micro.gh + sc.px(2)), stt["why"],
                    role="micro", color=C.warn)

    W.section(ctx, s, "Парность эпизодов", wid="ana/stp",
              info="`bench` берёт для обоих условий одни и те же семена "
                   "(seed0 + i), то есть эпизоды под одним номером — это "
                   "одна расстановка арены и те же случайные числа. Но "
                   "семена попадают в файл только с сегодняшних прогонов: "
                   "у старых серий парность непроверяема.")
    pw = {"proven": C.ok, "broken": C.warn}.get(info["pairing"], C.warn)
    W.kv(ctx, s.take(sc.px(S.row)), "парность",
         {"proven": "доказана семенами", "broken": "семена не совпадают",
          "unknown": "непроверяема"}.get(info["pairing"],
                                         info["pairing"]))
    h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                info["pairing_why"], role="small", color=pw, wrapped=True)
    s.take(h, sc.px(S.xs))

    W.section(ctx, s, "Сравнение условий", wid="ana/stc",
              info="Критерий выбирается по данным: где пары доказаны "
                   "семенами — по разностям пар, где нет — Уэлч, "
                   "непарный. Второй критерий показывается рядом, чтобы "
                   "было видно, что именно выбрано и почему.")
    if not info["comparable"]:
        h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                    info["why"], role="small", color=C.warn, wrapped=True)
        s.take(h)
    else:
        t = info["test"]
        W.kv(ctx, s.take(sc.px(S.row)), "критерий", t["name"])
        W.kv(ctx, s.take(sc.px(S.row)), "разница средних",
             "%s ± %s" % (_num(t["diff"]), _num(t["se"])))
        W.kv(ctx, s.take(sc.px(S.row)), "интервал 95 % разницы",
             "[%s; %s]" % (_num(t["ci"][0]), _num(t["ci"][1])))
        W.kv(ctx, s.take(sc.px(S.row)), "t", _num(t["t"]))
        W.kv(ctx, s.take(sc.px(S.row)), "степеней свободы", _num(t["df"]))
        W.kv(ctx, s.take(sc.px(S.row)), "p",
             _num(t["p"]) if t["p"] is not None else "не посчитан")
        if t.get("effect") is not None:
            W.kv(ctx, s.take(sc.px(S.row)), "размер эффекта (d)",
                 _num(t["effect"]))
        h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                    ST.verdict(t, pooled=pooled), role="small", color=C.ink,
                    wrapped=True)
        s.take(h, sc.px(S.xs))
        other = info.get("unpaired")
        if other is not None and other is not t:
            W.section(ctx, s, "Для сравнения: непарный критерий",
                      wid="ana/stu",
                      info="Непарный критерий выбрасывает связь эпизодов "
                           "по семени — ту самую, ради которой серия и "
                           "гоняется одинаковыми семенами. Поэтому он "
                           "здесь не основной, а справочный.")
            W.kv(ctx, s.take(sc.px(S.row)), other["name"],
                 "p = %s" % (_num(other["p"]) if other["p"] is not None
                             else "—"))
            W.kv(ctx, s.take(sc.px(S.row)), "разница средних",
                 "%s ± %s" % (_num(other["diff"]), _num(other["se"])))

    kin = _kindred(app, benches[0][0]) if not pooled else []
    if kin:
        W.section(ctx, s, "Рядом есть такие же серии", right="%d" % len(kin),
                  wid="ana/stk",
                  info="Те же съём, сглаживание и длительность эпизода. "
                       "Сложить их можно переключателем выше, но это "
                       "ручное объединение: проверить, что эпизоды не "
                       "повторяются, нечем.")
        for name, n in kin:
            if not s.fits(sc.px(S.row)):
                break
            W.kv(ctx, s.take(sc.px(S.row)), name, "%d эпизодов" % n)
        h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                    "Отметьте их в каталоге кнопкой «+» и включите "
                    "сложение — окно пересчитает среднее, sem и критерий "
                    "по объединённой выборке и скажет, что объединение "
                    "ручное.", role="small", color=C.dim, wrapped=True)
        s.take(h, sc.px(S.xs))

    if pooled:
        W.section(ctx, s, "Каждая серия по отдельности", wid="ana/stsep",
                  info="Объединение ничего не отменяет: числа отдельных "
                       "серий остаются здесь же. Если они расходятся, "
                       "объединённое среднее это скрывает.")
        for name, bb in benches:
            one = ST.series_stats(bb)
            if not s.fits(sc.px(S.row) * 2):
                break
            W.label(ctx, s.take(f.small.gh + sc.px(2)), name, role="small",
                    color=C.ink)
            for c in one["conditions"]:
                if not s.fits(sc.px(S.row)):
                    break
                st_ = c["stats"]
                W.kv(ctx, s.take(sc.px(S.row)), "  " + c["name"],
                     "%s ± %s, n %d"
                     % (_num(st_["mean"]),
                        _num(st_["sem"]) if st_["sem"] is not None else "—",
                        c["usable"]))
            if one["comparable"] and s.fits(sc.px(S.row)):
                W.kv(ctx, s.take(sc.px(S.row)), "  критерий",
                     "%s, p = %s" % (one["test"]["name"].split(",")[0],
                                     _num(one["test"]["p"])
                                     if one["test"]["p"] is not None
                                     else "—"))

    W.section(ctx, s, "Чего эти числа не значат")
    for line in ST.CAVEATS:
        if not s.fits(f.micro.gh):
            break
        h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                    "— " + line, role="micro", color=C.faint, wrapped=True)
        s.take(h, sc.px(2))
    W.scroll_done(ctx, "ana/st", r, s.used, token)


# ------------------------------------------------------- правая колонка

def secondary(app, ctx, r, tab):
    _ensure(app)
    run = _current(app)
    if run is None:
        W.label(ctx, r, "Прогон не выбран.", role="small", color=C.faint,
                wrapped=True)
        return
    (_tab_params, _tab_files, _tab_science, _tab_actions)[tab](app, ctx, r,
                                                               run)


def _kindred(app, name):
    """Другие серии с теми же настройками: их есть смысл складывать.

    Совпадать должны съём, сглаживание и длительность эпизода — всё, что
    в `bench.json` вообще сохраняется. Первый сид серии там не лежит, и
    поручиться за непересечение эпизодов нельзя; поэтому это подсказка,
    а не предложение сложить автоматически.
    """
    st = _state(app)
    by = {x.name: x for x in (st["runs"] or [])}
    me = by.get(name)
    if me is None or me.kind != "bench":
        return []
    mine = _series_key(me)
    if mine is None:
        return []
    out = []
    for x in (st["runs"] or []):
        if x.name == name or x.kind != "bench":
            continue
        if _series_key(x) == mine:
            out.append((x.name, x.n_episodes))
    return out


def _series_key(run):
    b = (RN.detail(run) or {}).get("bench.json")
    if not isinstance(b, dict) or not b:
        return None
    for rec in b.values():
        if isinstance(rec, dict):
            return (rec.get("readout"), rec.get("dn_tau_s"),
                    rec.get("duration_s"), tuple(sorted(b.keys())))
    return None


def _tab_params(app, ctx, r, run):
    sc, f = app.sh.sc, app.sh.fonts
    d = _detail(app, run)
    content, token = W.scroll(ctx, "ana/s0", r)
    s = L.Stack(L.Rect(content.x, content.y, content.w, 1 << 16),
                gap=sc.px(S.sm))
    W.section(ctx, s, "Спека прогона", first=True, wid="ana/sp",
              right=run.name,
              info="`spec.json` — это то, что уходило работнику. По нему "
                   "прогон повторяется. У прогонов командной строки его "
                   "нет: она спеку не сохраняет.")
    if run.spec is None:
        h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                    "`spec.json` в папке нет. Настройки можно прочитать "
                    "ниже — они лежат в главном файле вида, — но собрать "
                    "из них тот же прогон одной кнопкой нельзя.",
                    role="small", color=C.warn, wrapped=True)
        s.take(h, sc.px(S.sm))
    else:
        for k, v in _flat(run.spec):
            if not s.fits(sc.px(S.row)):
                break
            W.kv(ctx, s.take(sc.px(S.row)), k, _num(v))

    ep = d.get("episode.json") or d.get("embodied.json") or {}
    cfg = ep.get("config") or (d.get("meta.json") or {}).get("config") or {}
    if cfg:
        W.section(ctx, s, "Настройки в файле")
        for k, v in cfg.items():
            if not s.fits(sc.px(S.row)):
                break
            W.kv(ctx, s.take(sc.px(S.row)), k, _num(v))
    bench = d.get("bench.json")
    if isinstance(bench, dict) and bench:
        W.section(ctx, s, "Настройки серии", wid="ana/bp",
                  info="Серия сохраняет настройки по условию, а не целиком: "
                       "заказанное число эпизодов и первый сид в файл не "
                       "попадают.")
        for cond, rec in bench.items():
            if not isinstance(rec, dict) or not s.fits(sc.px(S.row)):
                continue
            W.label(ctx, s.take(f.small.gh + sc.px(2)), cond, role="small",
                    color=C.ink)
            for k in ("readout", "dn_tau_s", "duration_s", "n_runs"):
                if rec.get(k) is None or not s.fits(sc.px(S.row)):
                    continue
                W.kv(ctx, s.take(sc.px(S.row)), k, _num(rec[k]))
    params = (d.get("meta.json") or {}).get("params") or {}
    if params:
        W.section(ctx, s, "Параметры модели", wid="ana/mp",
                  info="Константы LIF: те же для всех прогонов, но лежат "
                       "в каждом — чтобы прогон читался без кода.")
        for k, v in params.items():
            if not s.fits(sc.px(S.row)):
                break
            W.kv(ctx, s.take(sc.px(S.row)), k, _num(v))
    W.scroll_done(ctx, "ana/s0", r, s.used, token)


def _flat(d, prefix=""):
    out = []
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            out.extend(_flat(v, prefix + k + "."))
        elif isinstance(v, list):
            if v and isinstance(v[0], dict):
                for i, item in enumerate(v):
                    out.extend(_flat(item, "%s%s[%d]." % (prefix, k, i)))
            else:
                out.append((prefix + k, ", ".join(str(x) for x in v[:6])))
        else:
            out.append((prefix + k, v))
    return out


def _tab_files(app, ctx, r, run):
    sc, f = app.sh.sc, app.sh.fonts
    content, token = W.scroll(ctx, "ana/s1", r)
    s = L.Stack(L.Rect(content.x, content.y, content.w, 1 << 16),
                gap=sc.px(S.sm))
    W.section(ctx, s, "Файлы прогона", first=True,
              right="%d" % len(run.files), wid="ana/fl",
              info="Папка как она есть. Отсутствие файла не делает прогон "
                   "повреждённым: у перебора популяций нет ни растра, ни "
                   "таблицы нейронов, и это нормально.")
    total = 0
    for name in run.files:
        try:
            size = (run.path / name).stat().st_size
        except OSError:
            size = 0
        total += size
        if not s.fits(sc.px(S.row)):
            continue
        row = s.take(sc.px(S.row))
        hover, _h, clicked, _fc = ctx.probe("ana/f/" + name, row, True)
        if hover:
            ctx.dl.rect(row.x, row.y, row.x1, row.y1, C.hover, 1.0,
                        sc.px(R.control))
        if clicked:
            _open(app, run.path / name)
        W.label(ctx, L.Rect(row.x + sc.px(S.xs),
                            row.y + (row.h - f.small.gh) // 2,
                            row.w - sc.px(S.xs), f.small.gh), name,
                role="small",
                color=C.danger if name in run.errors else C.ink)
        W.label(ctx, row, _size(size), role="micro", align="right",
                color=C.faint)
        ctx.want_tip("ana/f/" + name, row, "открыть " + str(run.path / name))
    W.section(ctx, s, "Всего")
    W.kv(ctx, s.take(sc.px(S.row)), "размер папки", _size(total))
    W.kv(ctx, s.take(sc.px(S.row)), "последняя запись", run.when)
    W.kv(ctx, s.take(sc.px(S.row)), "откуда",
         {"desk": "запуск из окна", "cli": "командная строка"}.get(
             run.source, "неизвестно"))
    W.scroll_done(ctx, "ana/s1", r, s.used, token)


def _size(n):
    for unit, k in (("ГБ", 1e9), ("МБ", 1e6), ("КБ", 1e3)):
        if n >= k:
            return ("%.1f %s" % (n / k, unit)).replace(".", ",")
    return "%d Б" % n


def _tab_science(app, ctx, r, run):
    sc, f = app.sh.sc, app.sh.fonts
    content, token = W.scroll(ctx, "ana/s2", r)
    s = L.Stack(L.Rect(content.x, content.y, content.w, 1 << 16),
                gap=sc.px(S.sm))
    W.section(ctx, s, "Про этот прогон", first=True)
    h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                run.state_why, role="small",
                color=STATE_COLOR.get(run.state, C.faint), wrapped=True)
    s.take(h, sc.px(S.sm))
    if run.caveats:
        # Оговорки самого прогона: чем он отличается от эталонного.
        # Стоят первыми и жёлтым — раньше оговорок про отдельные
        # графики, потому что они про весь результат, а не про картинку.
        W.label(ctx, s.take(f.small.gh + sc.px(2)),
                "Чем этот прогон отличается от эталонного", role="small",
                color=C.dim)
        for line in run.caveats:
            if not s.fits(f.micro.gh * 2):
                break
            h = W.label(ctx, L.Rect(content.x, s.area.y, content.w,
                                    s.area.h), "— " + line, role="micro",
                        color=C.warn, wrapped=True)
            s.take(h, sc.px(S.xs))
    _vision_block(app, ctx, s, content, run)
    _broken(app, ctx, s, content, run)
    for p in PL.available(run):
        if not p.caveat or not s.fits(f.micro.gh * 2):
            continue
        W.label(ctx, s.take(f.small.gh + sc.px(2)), p.title, role="small",
                color=C.dim)
        h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                    p.caveat, role="micro", color=C.warn, wrapped=True)
        s.take(h, sc.px(S.xs))
    if run.kind == "bench":
        W.section(ctx, s, "Про серии")
        for line in ST.CAVEATS:
            if not s.fits(f.micro.gh):
                break
            h = W.label(ctx, L.Rect(content.x, s.area.y, content.w,
                                    s.area.h), "— " + line, role="micro",
                        color=C.faint, wrapped=True)
            s.take(h, sc.px(2))
    W.section(ctx, s, "Про это пространство")
    for line in SCIENCE:
        if not s.fits(f.micro.gh):
            break
        h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                    "— " + line, role="micro", color=C.faint, wrapped=True)
        s.take(h, sc.px(2))
    W.scroll_done(ctx, "ana/s2", r, s.used, token)


def _tab_actions(app, ctx, r, run):
    sc, f = app.sh.sc, app.sh.fonts
    st = _state(app)
    content, token = W.scroll(ctx, "ana/s3", r)
    s = L.Stack(L.Rect(content.x, content.y, content.w, 1 << 16),
                gap=sc.px(S.sm))
    W.section(ctx, s, "Открыть", first=True)
    row = s.take(sc.px(S.control)).cols(2, gap=sc.px(S.sm))
    if W.button(ctx, "ana/open", row[0], "Папку", icon="table",
                tip=str(run.path)):
        _open(app, run.path)
    if W.button(ctx, "ana/report", row[1], "Отчёт", icon="chart",
                enabled=run.has_report,
                tip="report.html; окно отчёты не пишет, их делает "
                    "командная строка"):
        _open(app, run.path / "report.html")

    W.section(ctx, s, "Повторить опыт", wid="ana/rep",
              info="Настройки переносятся в «Эксперимент». Сам опыт не "
                   "запускается: пуск остаётся за человеком. Исходный "
                   "результат не трогается — повтор создаёт новую папку.")
    ok, why = RN.repeatable(run)
    if W.button(ctx, "ana/repeat", s.take(sc.px(S.control_lg)),
                "Перенести в «Эксперимент»", kind="primary", icon="play",
                enabled=ok and not app.st.run.busy,
                tip=why or "открыть этот опыт в сборке"):
        from . import ws_experiment as X

        done, msg = X.apply_spec(app, run.spec)
        if done:
            app.st.ws.switch("experiment")
            app.st.note("опыт перенесён из %s: проверьте и запустите"
                        % run.name)
        else:
            app.st.note("повторить нельзя: " + msg, "warn")
    if not ok:
        h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                    why, role="small", color=C.warn, wrapped=True)
        s.take(h, sc.px(S.xs))
    elif app.st.run.busy:
        W.label(ctx, s.take(f.micro.gh + sc.px(2)),
                "сейчас идёт другой прогон", role="micro", color=C.faint)

    W.section(ctx, s, "Выгрузить", wid="ana/exp",
              info="CSV пишется в out/_export и ничего в папке прогона не "
                   "меняет.")
    # `data` может быть таблицей pandas (её кладёт вкладка «Таблицы»), а у
    # DataFrame нет истинности: `bool(df)` бросает ValueError и роняет
    # кадр. Поэтому сравнение с None, а не приведение к логическому.
    data = st.get("data")
    has_plot = (isinstance(data, dict) and "plot" in data
                and st["data_key"].startswith(run.name))
    if W.button(ctx, "ana/csv", s.take(sc.px(S.control)),
                "Показанный график в CSV", icon="table",
                enabled=has_plot,
                tip="то, что сейчас нарисовано, строками"
                    if has_plot else
                    "выгружается нарисованный график: откройте «Графики»"):
        _export(app, run)
    if st.get("msg") and s.fits(f.small.gh):
        h = W.label(ctx, L.Rect(content.x, s.area.y, content.w, s.area.h),
                    st["msg"], role="small",
                    color={"error": C.danger, "warn": C.warn}.get(
                        st.get("msg_kind"), C.dim), wrapped=True)
        s.take(h)
    W.scroll_done(ctx, "ana/s3", r, s.used, token)


def _open(app, path):
    import os

    p = str(path)
    try:
        if not os.path.exists(p):
            app.st.note("нет такого пути: " + p, "warn")
            return
        os.startfile(p)                                       # noqa: S606
        app.st.note("открыто: " + p)
    except Exception as exc:                                  # noqa: BLE001
        app.st.note("не открылось: " + str(exc)[:120], "warn")


def _export(app, run):
    """Нарисованные данные — в CSV. Папку прогона не трогаем."""
    import csv

    from ..data import OUT

    st = _state(app)
    data = st.get("data")
    if not isinstance(data, dict) or "plot" not in data:
        return
    kind = data.get("kind")
    if not kind:
        return
    out = OUT / "_export"
    out.mkdir(parents=True, exist_ok=True)
    name = "%s-%s.csv" % (run.name, st["data_key"].split("/")[-1])
    path = out / name
    try:
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, delimiter=";")
            w.writerow(["# источник", data["plot"].source])
            w.writerow(["# величина", data["plot"].what,
                        data["plot"].unit or "безразмерная"])
            if kind == "line":
                w.writerow([data.get("xlabel", "x")]
                           + [n for n, _v in data["series"]])
                for i, x in enumerate(data["x"]):
                    w.writerow([x] + [v[i] for _n, v in data["series"]])
            elif kind == "track":
                w.writerow(["x, мм", "y, мм"])
                for a, b in zip(data["x"], data["y"]):
                    w.writerow([a, b])
            elif kind == "bar":
                w.writerow(["название", "значение", "нейронов"])
                for i, lab in enumerate(data["labels"]):
                    w.writerow([lab, data["values"][i],
                                (data.get("counts") or [None] * 99)[i]])
            elif kind == "hist":
                w.writerow(["левая граница", "правая граница", "нейронов"])
                e = data["edges"]
                for i, c in enumerate(data["counts"]):
                    w.writerow([e[i], e[i + 1], c])
            elif kind == "strip":
                w.writerow(["условие", "эпизод", "значение"])
                for cond, vals, _m, _s in data["groups"]:
                    for i, v in enumerate(vals):
                        w.writerow([cond, i + 1, v])
    except Exception as exc:                                  # noqa: BLE001
        st["msg"] = "не записалось: " + str(exc)[:120]
        st["msg_kind"] = "error"
        return
    st["msg"] = "записано: " + str(path)
    st["msg_kind"] = "dim"
    app.st.note("выгружено: " + str(path))


SPACE = {"centre": centre, "inspector": inspector, "secondary": secondary,
         "tabs": TABS}
