"""Обзор сохранённых прогонов: что уже посчитано и что из этого вышло.

Папка `out/` — это и есть лабораторный журнал: двадцать с лишним прогонов,
каждый со своим отчётом. Пока единственный способ туда заглянуть — открыть
`index.html` в браузере, то есть выйти из окна и потерять сеанс. Здесь то же
содержимое разбирается прямо в окне: список слева, разбор выбранного справа, и
из разбора можно сразу перезапустить прогон теми же числами.

Обход папки стоит около шестидесяти миллисекунд: `index.scan` открывает
`rates.parquet` каждого прогона, чтобы вытащить верхние области. На кадре это
недопустимо, поэтому обход идёт не чаще раза в `RESCAN_S`, а всё тяжёлое
(`bins.npz`, `track.parquet`, журнал агента) разбирается ровно один раз — при
смене выбранной записи — и живёт разобранным в `RunsState`.

Панель ничего не знает про окно: ей дают контекст деталей, состояние и
коробку. Что делать по кнопкам — забота вызывающего кода, он передаёт
`on_open`, `on_rerun` и `on_rerun_spec`.
"""

from __future__ import annotations

import time

import numpy as np

from .. import index
from . import kit, theme

__all__ = ["RunsState", "panel", "KINDS", "KIND_LABEL"]

# Фильтр по виду. Порядок — от общего к частному, как в разделах указателя.
KINDS = (
    ("все", None),
    ("опыт", "run"),
    ("арена", "episode"),
    ("тело", "embodied"),
    ("серия", "bench"),
    ("зонд", "lateral"),
    ("агент", "agent"),
)

KIND_LABEL = {
    "run": "опыт", "episode": "арена", "embodied": "тело",
    "bench": "серия", "lateral": "зонд", "agent": "агент",
}

# Обход `out/` не чаще раза в пару секунд: прогон, который только что
# закончился, появится в списке с задержкой, незаметной глазу, а кадр не
# будет платить за двадцать чтений parquet.
RESCAN_S = 2.0

# Сколько строк показывать в таблицах зонда и серии: дальше либо не влезает,
# либо уже не информативно.
MAX_LATERAL = 24
MAX_STIMULI = 4
MAX_VERDICT_LINES = 4
MAX_GOAL_LINES = 3

# Ширина разбора в единицах `theme`, до масштаба. Строка, у которой подпись
# слева, а число у правого края окна в 1945 точек, не читается: глаз теряет,
# к чему относится число. Поэтому разбор не растягивается без предела, а
# стоит колонкой и оставляет поле справа.
TABLE_W = 900

# Левая колонка со списком. Шире рейки сборки: имена прогонов длиннее любого
# имени нейропиля, а урезанное имя в списке — это ещё один щелчок.
LIST_W = 400

_REGION_COLS = None


def _region_cols():
    """Цвет на нейропиль — тот же, что в карте мозга.

    Палитра считается от **всех** нейропилей, а не от шести верхних в
    записи: внутри супер-области светлота расходится по числу областей, и
    палитра, собранная по шести именам, дала бы прогону свои цвета, не
    совпадающие с картой.
    """
    global _REGION_COLS
    if _REGION_COLS is None:
        from .. import regions as R

        _REGION_COLS = theme.region_colors(
            {name: meta[1] for name, meta in R.NEUROPILS.items()})
    return _REGION_COLS


# ------------------------------------------------------------ числа и время

def _num(v, nd=2, sign=False):
    """Число для показаний: запятая, фиксированный знак после точки.

    Не `kit.fmt_num`: там число знаков выбирается по величине, а в колонке
    таблицы оно должно быть одинаковым, иначе цифры не стоят друг под другом.
    """
    if v is None:
        return "—"
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(x):
        return "—"
    # Знак минуса — обычный дефис: в атласе нет ни U+2212, ни тонкого
    # пробела, и напечатались бы вопросительные знаки.
    fmt = ("%+." if sign else "%.") + str(int(nd)) + "f"
    return (fmt % x).replace(".", ",")


def _grp(v):
    """Целое с разделителем разрядов: 189 325 читается, 189325 — нет."""
    try:
        return format(int(v), ",").replace(",", " ")
    except (TypeError, ValueError):
        return "—" if v is None else str(v)


def _plural(n, one, few, many):
    n = abs(int(n)) % 100
    if 11 <= n <= 14:
        return many
    d = n % 10
    if d == 1:
        return one
    if 2 <= d <= 4:
        return few
    return many


def _ago(mtime, now=None):
    """Когда это было, по-человечески.

    Отметка времени в списке не отвечает на вопрос, который к списку задают:
    «а это свежее или прошлой недели». «2 часа назад» отвечает.
    """
    try:
        dt = float(now if now is not None else time.time()) - float(mtime)
    except (TypeError, ValueError):
        return "—"
    if dt < 0:
        dt = 0.0
    if dt < 90:
        return "только что"
    if dt < 3600:
        n = int(dt // 60)
        return (str(n) + " "
                + _plural(n, "минуту", "минуты", "минут") + " назад")
    if dt < 86400:
        n = int(dt // 3600)
        return str(n) + " " + _plural(n, "час", "часа", "часов") + " назад"
    if dt < 86400 * 30:
        n = int(dt // 86400)
        return str(n) + " " + _plural(n, "день", "дня", "дней") + " назад"
    n = max(1, int(dt // (86400 * 30)))
    return str(n) + " " + _plural(n, "месяц", "месяца", "месяцев") + " назад"


# В атласе ровно те знаки, которые нужны показаниям: тонкого и неразрывного
# пробела в нём нет, а журнал агента и отчёты моделей ими полны — без замены
# на месте «410 нейронов» печатается «410?нейронов».
_SPACES = {0x00a0: " ", 0x2007: " ", 0x2009: " ", 0x202f: " ", 0x2002: " ",
           0x2003: " ", 0x2060: "", 0xfeff: "", 0x0009: "    "}


def _wrap(atlas, text, max_px):
    """Разложить текст по ширине коробки.

    Журнал агента и его вывод — обычная проза в сотню знаков строкой, а
    `log_view` строки не переносит, только обрезает. Поэтому перенос делается
    здесь: по словам, а слишком длинное слово (в журнале это куски JSON)
    режется по месту.
    """
    if max_px <= 0:
        return []
    ch_w = max(1, atlas.width("0"))
    budget = max(1, int(max_px // ch_w))
    out = []
    for para in str(text).translate(_SPACES).replace("\r", "").split("\n"):
        if not para.strip():
            out.append("")
            continue
        line = ""
        for word in para.split(" "):
            while len(word) > budget:
                if line:
                    out.append(line)
                    line = ""
                out.append(word[:budget])
                word = word[budget:]
            probe = word if not line else line + " " + word
            if atlas.width(probe) <= max_px or not line:
                line = probe
            else:
                out.append(line)
                line = word
        out.append(line)
    return out


# ---------------------------------------------------------------- состояние

def _read_track(run_dir):
    """Путь мухи из `track.parquet` — только при смене выбранной записи."""
    path = run_dir / "track.parquet"
    if not path.exists():
        return None, None
    try:
        import pandas as pd

        df = pd.read_parquet(path)
    except Exception:  # noqa: BLE001
        return None, None
    if "x" not in df.columns or "y" not in df.columns:
        return None, None
    xy = np.ascontiguousarray(
        df[["x", "y"]].to_numpy(dtype=np.float32))
    dist = None
    if "food_dist" in df.columns:
        dist = np.asarray(df["food_dist"].to_numpy(dtype=np.float64))
    return xy, dist


def _read_bins(run_dir):
    """Ход активности по бинам из `bins.npz`: сумма спайков в каждом бине."""
    path = run_dir / "bins.npz"
    if not path.exists():
        return None, ""
    try:
        with np.load(path) as z:
            series = z["bins"].sum(axis=1).astype(np.float64)
            bin_ms = float(z["bin_ms"]) if "bin_ms" in z.files else 0.0
    except Exception:  # noqa: BLE001
        return None, ""
    label = "спайков в бине"
    if bin_ms:
        label += ", бин " + _num(bin_ms, 0) + " мс"
    return series, label


class RunsState:
    """Что обзор помнит между кадрами.

    Прокрутка списка и журнала живёт в `Ui` по ключу детали — так устроен весь
    немедленный режим, и второй источник правды здесь только мешал бы. Поэтому
    состояние держит ключи, а не пиксели прокрутки.
    """

    LIST_KEY = "runs/list"
    LOG_KEY = "runs/journal"

    def __init__(self, out_dir=None):
        self.out_dir = out_dir
        self.entries = []
        self.filter_i = 0
        self.sel_name = None
        self.error = None
        # Когда последний раз обходили `out/` (монотонные секунды) и сколько
        # это стоило: обход заметен, пусть его цена будет видна.
        self.scanned_at = 0.0
        self.scan_ms = 0.0
        # Разобранные тяжёлые файлы выбранной записи.
        self.loaded = None
        self.track = None
        self.series = None
        self.series_label = ""
        self.journal = []
        self.journal_w = -1

    # -- обход out/ ----------------------------------------------------

    def refresh(self, force=False):
        now = time.monotonic()
        if not force and self.scanned_at and now - self.scanned_at < RESCAN_S:
            return
        self.scanned_at = now
        t0 = time.perf_counter()
        try:
            self.entries = index.scan(self.out_dir)
            self.error = None
        except Exception as exc:  # noqa: BLE001
            self.entries = []
            self.error = str(exc)
        self.scan_ms = (time.perf_counter() - t0) * 1000.0

    def shown(self):
        want = KINDS[self.filter_i % len(KINDS)][1]
        if want is None:
            return list(self.entries)
        return [e for e in self.entries if e.get("kind") == want]

    def current(self, shown):
        """Выбранная запись ищется по имени, а не по номеру.

        Номер смысла не имеет: обход переупорядочивает список по времени
        правки, и прогон, который только что закончился, сдвигает все
        остальные.
        """
        for e in shown:
            if e["name"] == self.sel_name:
                return e
        return shown[0] if shown else None

    def select(self, entry):
        self.sel_name = None if entry is None else entry["name"]

    # -- тяжёлое: только при смене выбранной ---------------------------

    def ensure_loaded(self, entry):
        name = None if entry is None else entry["name"]
        if name == self.loaded:
            return
        self.loaded = name
        self.track = None
        self.series = None
        self.series_label = ""
        self.journal = []
        self.journal_w = -1
        if entry is None:
            return
        kind = entry.get("kind")
        if kind in ("episode", "embodied"):
            self.track, dist = _read_track(entry["dir"])
            if dist is not None and dist.size:
                self.series = dist
                self.series_label = "расстояние до цели, мм"
        elif kind == "run":
            self.series, self.series_label = _read_bins(entry["dir"])

    def journal_lines(self, atlas, text, width):
        """Журнал, разложенный по ширине. Перекладывается при смене размера."""
        if self.journal_w != int(width) or not self.journal:
            self.journal = _wrap(atlas, text, int(width))
            self.journal_w = int(width)
        return self.journal


# ---------------------------------------------------------------- мелкие части

def _block(ui, r, w):
    """Отрезать от строки блок не шире заданного.

    Таблица, растянутая на всю ширину окна 1945 точек, разводит имя и число
    так, что глаз теряет строку. Блок стоит слева и дальше не растёт.
    """
    return kit.Box(r.x0, r.y0, min(r.x1, r.x0 + ui.px(w)), r.y1)


def _vals(ui, col, pairs, ncols=None):
    """Показания сеткой: подпись мелким кеглем, число основным.

    Колонок столько, сколько влезает **с самой длинной подписью**: иначе
    «совпадение с градиентом» в узкой ячейке урезается до «совпадение с…»,
    и показание перестаёт отвечать на вопрос, какое именно это совпадение.
    """
    if not pairs:
        return
    need = max(ui.small.width(str(a)) + ui.font.width(str(b)) + ui.px(26)
               for a, b in pairs)
    n = ncols or max(1, min(len(pairs), 6, int(col.area.w // max(1, need))))
    h = ui.font.gh + ui.px(5)
    for i in range(0, len(pairs), n):
        chunk = pairs[i:i + n]
        if col.left < h:
            return
        r = col.take(h, ui.px(2))
        for cell, (lab, val) in zip(r.cols(n, ui.px(14)), chunk):
            ui.value(lab, val, cell.x0, cell.y0, cell.w)


def _pair(ui, r, label, value, color=theme.INK):
    """Подпись слева, длинное значение справа.

    Отличается от `ui.value` тем, что урезается **значение**, а не подпись:
    имя съёма или строка стимула длиннее любой подписи, и терять надо их
    хвост, а не название показания.
    """
    lw = ui.small.width(str(label)) + ui.px(10)
    y = r.y0 + max(0, (ui.font.gh - ui.small.gh) // 2)
    ui._txt(ui.small, str(label), r.x0, y, theme.DIM, max_px=max(0, r.w - lw))
    ui._txt_right(ui.font, ui.font.fit(str(value), max(0, r.w - lw)),
                  r.x1, r.y0, color)


def _head(ui, col, text=None):
    """Разделитель и подпись раздела — одной волосяной линией, без плашек."""
    if col.left < ui.line(True) + ui.px(10):
        return False
    col.skip(ui.px(5))
    ui.over.hline(col.area.x0, col.area.x1, col.take(1).y0, theme.HAIRLINE)
    col.skip(ui.px(5))
    if text:
        r = col.take(ui.line(True), ui.px(4))
        ui._txt(ui.small, text, r.x0, r.y0, theme.FAINT, max_px=r.w)
    return True


def _lines(ui, col, lines, limit, color=theme.INK):
    """Несколько строк прозы. Обрыв помечается многоточием, а не молчанием."""
    shown = lines[:limit]
    for i, line in enumerate(shown):
        if col.left < ui.line(True):
            return
        if i == len(shown) - 1 and len(lines) > limit:
            line = line.rstrip() + " …"
        r = col.take(ui.line(True), 0)
        ui._txt(ui.small, line, r.x0, r.y0, color, max_px=r.w)


def _warn(ui, col, text):
    if col.left < ui.line(True):
        return
    r = col.take(ui.line(True), ui.px(2))
    ui._txt(ui.small, text, r.x0, r.y0, theme.DANGER, max_px=r.w)


def _regions(ui, entry, col):
    """Полосы по верхним областям: цвет области тот же, что в карте."""
    regs = entry.get("regions") or []
    if not regs or col.left < ui.row + ui.line(True) * 2:
        return
    if not _head(ui, col, "области по средней частоте"):
        return
    cols = _region_cols()
    top = max([float(r.get("mean_hz") or 0.0) for r in regs] + [1e-9])
    for r in regs:
        if col.left < ui.row:
            break
        name = str(r.get("region", "?"))
        hz = float(r.get("mean_hz") or 0.0)
        ui.meter(col.take(ui.row, ui.px(3)), hz / top,
                 cols.get(name, theme.FAINT), name,
                 _num(hz, 1) + " Гц · " + _grp(r.get("n")))


def _table_cols(ui, area_w, spec, name_min):
    """Ширины колонок таблицы и сколько их влезло.

    Колонки отмеряются по самой широкой из двух строк — подписи шапки и
    образца значения, — и отбрасываются справа, пока имени условия не станет
    хватать места. Прибивать их к пикселям нельзя: это окно бывает и 1945, и
    803 точек шириной.
    """
    out = list(spec)
    while out:
        widths = [max(ui.small.width(h), ui.font.width(sample)) + ui.px(12)
                  for h, sample, _ in out]
        if area_w - sum(widths) >= name_min:
            return out, widths
        out = out[:-1]
    return [], []


def _table_row(ui, r, name, cells, widths, ink=theme.INK, small=False):
    atlas = ui.small if small else ui.font
    x = r.x1
    for w, text in zip(reversed(widths), reversed(cells)):
        ui._txt_right(atlas, str(text), x - ui.px(2),
                      r.y0 + max(0, (r.h - atlas.gh) // 2),
                      ink if not small else theme.FAINT)
        x -= w
    name_atlas = ui.small if small else ui.font
    ui._txt(name_atlas, str(name), r.x0,
            r.y0 + max(0, (r.h - name_atlas.gh) // 2),
            theme.FAINT if small else ink,
            max_px=max(0, x - r.x0 - ui.px(6)))


# ---------------------------------------------------------------- список

MIN_NAME = 14


def _row_text(entry, chars, now, show_kind=True):
    """Строка списка: имя слева, вид и давность справа.

    Гарнитура моноширинная, поэтому колонки выравниваются добивкой
    пробелами — отдельной таблицы для трёх полей не нужно.

    Когда места мало, первым уходит вид: под выбранным фильтром он и так у
    всех строк один, а «2 часа назад» не повторяет ничего.
    """
    kind = KIND_LABEL.get(entry.get("kind"), str(entry.get("kind")))
    ago = _ago(entry["mtime"], now)
    short = ago.replace(" назад", "")
    want = [ago, short]
    if show_kind:
        want = [kind + " · " + ago, kind + " · " + short] + want
    # Берётся первый хвост, после которого имени остаётся на что смотреть;
    # если не остаётся ни после одного — самый короткий, имя важнее.
    tail = want[-1]
    for cand in want:
        if chars - len(cand) - 2 >= MIN_NAME:
            tail = cand
            break
    room = max(4, chars - len(tail) - 2)
    name = entry["name"]
    if len(name) > room:
        name = name[:room - 1] + "…"
    return name.ljust(max(0, chars - len(tail))) + tail


def _list(ui, state, shown, cur, area, now):
    col = kit.Column(area, ui.gap)

    r = col.take(ui.line())
    ui.label("прогоны", r.x0, r.y0)
    tail = str(len(state.entries)) + " в out/"
    ui._txt_right(ui.small, tail, r.x1,
                  r.y0 + max(0, (ui.font.gh - ui.small.gh) // 2), theme.FAINT)

    if state.error:
        r = col.take(ui.line(True))
        ui._txt(ui.small, "обход не удался: " + state.error, r.x0, r.y0,
                theme.DANGER, max_px=r.w)

    labels = [name for name, _ in KINDS]
    need = max(ui.font.width(s) for s in labels) + ui.px(14)
    if area.w >= need * len(labels):
        state.filter_i = ui.segmented("runs/kind", col.take(ui.row), labels,
                                      state.filter_i)
    else:
        # Семь подписей в одну строку сливаются в кашу: в узкой колонке
        # фильтр разводится на две строки, как выбор ключа в рейке сборки.
        cut = 4
        a = ui.segmented("runs/kindA", col.take(ui.row), labels[:cut],
                         state.filter_i if state.filter_i < cut else -1)
        bsel = ui.segmented("runs/kindB", col.take(ui.row), labels[cut:],
                            state.filter_i - cut if state.filter_i >= cut
                            else -1)
        if 0 <= a < cut and a != state.filter_i:
            state.filter_i = a
        elif 0 <= bsel and bsel + cut != state.filter_i:
            state.filter_i = bsel + cut

    col.skip(ui.px(4))
    rest = col.rest
    if rest.h < ui.row:
        return
    # Ширина строки считается от коробки списка за вычетом рамки, рейки
    # прокрутки и отступов самой детали.
    chars = max(10, int((rest.w - ui.px(22)) // max(1, ui.font.width("0"))))
    items = [_row_text(e, chars, now, state.filter_i == 0) for e in shown]
    at = shown.index(cur) if (cur is not None and cur in shown) else -1
    pick = ui.list_box(state.LIST_KEY, rest, items, at)
    if pick != at and 0 <= pick < len(shown):
        state.select(shown[pick])


# ---------------------------------------------------------------- разборы

def _detail_run(ui, entry, state, col):
    m = entry.get("meta") or {}
    cfg = m.get("config") or {}
    tim = m.get("timing") or {}

    stimuli = m.get("stimuli") or []
    if stimuli and _head(ui, col, "зажигали"):
        for s in stimuli[:MAX_STIMULI]:
            if col.left < ui.line():
                break
            r = col.take(ui.line(), ui.px(2))
            tail = (_grp(s.get("n")) + " @ " + _num(s.get("rate_hz"), 0)
                    + " Гц")
            _pair(ui, r, str(s.get("name", "?")), tail)
        sil = m.get("n_silenced") or 0
        if sil and col.left >= ui.line(True):
            r = col.take(ui.line(True), ui.px(2))
            ui._txt(ui.small, "заглушено нейронов: " + _grp(sil), r.x0, r.y0,
                    theme.DANGER, max_px=r.w)

    # Окно стимула бывает открытым: прогон без `t_off_ms` светит до конца.
    first = stimuli[0] if stimuli else {}
    on, off = first.get("t_on_ms"), first.get("t_off_ms")
    if on is None:
        window = "—"
    elif off is None:
        window = "с " + _num(on, 0) + " мс до конца"
    else:
        window = _num(on, 0) + "–" + _num(off, 0) + " мс"
    if _head(ui, col, "показания"):
        _vals(ui, col, [
            ("длительность", _num(cfg.get("t_run_s"), 2) + " с"),
            ("бин", _num(cfg.get("bin_ms"), 0) + " мс"),
            ("окно стимула", window),
            ("активных", _grp(m.get("n_active"))),
            ("спайков", _grp(m.get("total_spikes"))),
            ("счёт", _num(tim.get("sim_s"), 1) + " с"),
        ])

    if state.series is not None and col.left > ui.px(92):
        if _head(ui, col, state.series_label):
            ui.spark(col.take(min(ui.px(62), max(ui.px(34), col.left // 3))),
                     state.series, theme.magma_at(0.7))
    _regions(ui, entry, col)


def _detail_episode(ui, entry, state, col):
    ep = entry.get("episode") or {}
    met = ep.get("metrics") or {}
    body = ep.get("body") or {}
    cfg = ep.get("config") or {}

    motor = str(body.get("motor_left") or "").replace("+side:left", "")
    r = col.take(ui.line(), ui.px(2))
    _pair(ui, r, "выход на рулёжку", motor or "все нисходящие")
    if body.get("swap_antennae"):
        _warn(ui, col, "антенны переставлены — это контроль")

    if _head(ui, col, "показания"):
        _vals(ui, col, [
            ("постоянная съёма", _num(body.get("dn_tau_s"), 2) + " с"),
            ("совпадение с градиентом",
             _num(met.get("grad_alignment"), 3, sign=True)),
            ("длительность", _num(cfg.get("duration_s"), 1) + " с"),
            ("до еды",
             _num(met.get("food_dist_first_half_mm"), 1) + " → "
             + _num(met.get("food_dist_second_half_mm"), 1) + " мм"),
            ("визитов", _grp(met.get("food_visits"))),
            ("скорость", _num(met.get("mean_speed_mm_s"), 1) + " мм/с"),
            ("поворотов к запаху", _num(met.get("turn_to_odour"), 3)),
            ("реплик модели", _grp(len(ep.get("comments") or []))),
        ])

    if state.series is not None and col.left > ui.px(150):
        if _head(ui, col, state.series_label):
            ui.spark(col.take(min(ui.px(52), max(ui.px(30), col.left // 4))),
                     state.series, theme.magma_at(0.7))
    _detail_track(ui, state, col)


def _detail_embodied(ui, entry, state, col):
    run = entry.get("embodied") or {}
    met = run.get("metrics") or {}
    cfg = run.get("config") or {}
    cal = run.get("calibration") or {}

    r = col.take(ui.line(), ui.px(2))
    _pair(ui, r, "съём", str(cfg.get("readout") or "—"))
    if cfg.get("swap_antennae"):
        _warn(ui, col, "антенны переставлены — это контроль")
    if not cfg.get("cross", True):
        _warn(ui, col, "без перекрёста")

    if _head(ui, col, "показания"):
        _vals(ui, col, [
            ("постоянная съёма", _num(cfg.get("dn_tau_s"), 2) + " с"),
            ("совпадение с градиентом",
             _num(met.get("grad_alignment"), 3, sign=True)),
            ("длительность", _num(cfg.get("duration_s"), 1) + " с"),
            ("до источника",
             _num(met.get("dist_start_mm"), 1) + " → "
             + _num(met.get("dist_end_mm"), 1) + " мм"),
            ("прошла", _num(met.get("path_mm"), 1) + " мм"),
            ("скорость", _num(met.get("speed_mm_s"), 1) + " мм/с"),
            ("перекос на разогреве", _num(cal.get("asym_bias"), 3, sign=True)),
        ])
    _detail_track(ui, state, col)
    _regions(ui, entry, col)


def _detail_track(ui, state, col):
    """Путь: в арене это главный результат, поэтому ему остаток высоты."""
    if state.track is None or len(state.track) < 2:
        return
    if col.left < ui.px(96) + ui.line(True) * 2:
        return
    if not _head(ui, col, "траектория, янтарём — где закончила"):
        return
    # Арена квадратная, и путь в ней тоже: коробка выше своей ширины даёт
    # поля по бокам, а не более подробный путь.
    h = max(ui.px(96), min(col.left, col.area.w))
    r = col.take(h)
    # Путь не янтарный: янтарь в этом окне значит «активное и текущее», а на
    # пути текущее — только голова, её ставит сама деталь.
    ui.track(kit.Box(r.x0, r.y0, min(r.x1, r.x0 + h), r.y1), state.track,
             theme.magma_at(0.62))


def _detail_bench(ui, entry, state, col):
    data = entry.get("bench") or {}
    conds = list(data.items())
    if not conds:
        return
    first = conds[0][1]

    r = col.take(ui.line(), ui.px(2))
    _pair(ui, r, "выход", str(first.get("readout") or "все нисходящие"))
    _vals(ui, col, [
        ("эпизод", _num(first.get("duration_s"), 1) + " с"),
        ("постоянная съёма", _num(first.get("dn_tau_s"), 2) + " с"),
        ("условий", _grp(len(conds))),
    ])

    # Серия в теле пишет свои имена полей: `dist_change_mm` вместо
    # `food_dist_change_mm` и `food_steps_total` вместо `food_visits_total`.
    embodied = any("dist_change_mm" in r for _, r in conds)

    def mean_cell(r):
        mean = _num(r.get("grad_alignment_mean"), 3, sign=True)
        sem = r.get("grad_alignment_sem")
        return mean if sem is None else mean + " ± " + _num(sem, 3)

    spec = [
        ("совпадение", "+0,000 ± 0,000", mean_cell),
        ("прогонов", "000", lambda r: _grp(r.get("n_runs"))),
        ("до цели", "-000,0 мм",
         lambda r: _num(r.get("food_dist_change_mm",
                              r.get("dist_change_mm")), 1, sign=True) + " мм"),
        ("шагов у еды" if embodied else "визитов", "0000",
         lambda r: _grp(r.get("food_visits_total",
                              r.get("food_steps_total")))),
    ]
    if not _head(ui, col, "по условиям"):
        return
    table_w = _block(ui, col.area, TABLE_W).w
    used, widths = _table_cols(ui, table_w, spec,
                               ui.font.width("перестановка антенн"))
    if not used:
        used, widths = _table_cols(ui, table_w, spec[:2], ui.px(60))
    if used and col.left >= ui.line(True) + ui.row:
        r = _block(ui, col.take(ui.line(True), ui.px(3)), TABLE_W)
        _table_row(ui, r, "условие", [h for h, _, _ in used], widths,
                   small=True)
        for label, row in conds:
            if col.left < ui.row:
                break
            cell = _block(ui, col.take(ui.row, 0), TABLE_W)
            _table_row(ui, cell, label, [fn(row) for _, _, fn in used],
                       widths)
            ui.over.hline(cell.x0, cell.x1, cell.y1 - 1, theme.HAIRLINE)
        col.skip(ui.px(4))

    if len(conds) != 2:
        return
    a, b = conds[0][1].get("runs"), conds[1][1].get("runs")
    if not a or not b:
        return
    w = index._welch(list(a), list(b))
    if not w or not _head(ui, col, "вывод по Уэлчу"):
        return
    _vals(ui, col, [
        ("разница", _num(w["diff"], 3, sign=True) + " ± " + _num(w["se"], 3)),
        ("t", _num(w["t"], 2)),
        ("p", "—" if w["p"] is None else _num(w["p"], 3)),
    ])
    if col.left < ui.line(True):
        return
    r = col.take(ui.line(True), ui.px(2))
    if w["p"] is None:
        text, ink = "scipy не стоит, значимость не посчитана", theme.FAINT
    elif w["p"] < 0.05:
        text, ink = ("разница есть: p < 0,05 при "
                     + _grp(len(a)) + " и " + _grp(len(b)) + " прогонах",
                     theme.OK)
    else:
        text, ink = ("разницы не видно: контроль и норма неотличимы",
                     theme.DIM)
    ui._txt(ui.small, text, r.x0, r.y0, ink, max_px=r.w)


def _detail_lateral(ui, entry, state, col, on_rerun_spec):
    rows = sorted(entry.get("lateral") or [],
                  key=lambda r: -abs(float(r.get("r") or 0.0)))
    if not rows:
        return
    r = col.take(ui.line(True), ui.px(2))
    ui._txt(ui.small, "муха неподвижна, асимметрия входа качается синусом",
            r.x0, r.y0, theme.DIM, max_px=r.w)
    if not _head(ui, col, "верхние популяции по модулю корреляции"):
        return

    take_text = "на рулёжку"
    btn_w = ui.font.width(take_text) + ui.px(20)
    if btn_w > col.area.w // 3:
        take_text = "взять"
        btn_w = ui.font.width(take_text) + ui.px(20)
    spec = [
        ("r", "+0,000", lambda r: _num(r.get("r"), 3, sign=True)),
        ("лаг", "000 мс", lambda r: _num(r.get("lag_ms"), 0) + " мс"),
        ("средн.", "000,0 Гц", lambda r: _num(r.get("mean_hz"), 1) + " Гц"),
        ("слева/справа", "0000/0000",
         lambda r: _grp(r.get("n_left")) + "/" + _grp(r.get("n_right"))),
    ]
    body_w = _block(ui, col.area, TABLE_W).w - btn_w - ui.gap
    used, widths = _table_cols(ui, body_w, spec, ui.font.width("region:AVLP"))
    if not used:
        used, widths = _table_cols(ui, body_w, spec[:1], ui.px(50))

    if col.left >= ui.line(True) + ui.row:
        head = col.take(ui.line(True), ui.px(3))
        _table_row(ui, kit.Box(head.x0, head.y0, head.x0 + body_w, head.y1),
                   "популяция", [h for h, _, _ in used], widths, small=True)
    fits = max(0, min(MAX_LATERAL, int(col.left // (ui.row + ui.px(2)))))
    for row in rows[:fits]:
        line = _block(ui, col.take(ui.row, ui.px(2)), TABLE_W)
        name = str(row.get("population", "?"))
        btn, left = line.cut_right(btn_w, ui.gap)
        _table_row(ui, left, name, [fn(row) for _, _, fn in used], widths)
        if ui.button("runs/take/" + name, btn, take_text,
                     enabled=on_rerun_spec is not None) and on_rerun_spec:
            # Выход кладётся В БЛОК `body`, а не в корень спеки. В корне
            # `spec.from_dict` его не применял, и эпизод считался с
            # `ALPN` — то есть кнопка «взять эту популяцию» молча
            # подменяла нажатую популяцию настройкой по умолчанию.
            on_rerun_spec({"kind": "body", "body": {"readout": name}})


def _detail_agent(ui, entry, state, col):
    text = entry.get("journal") or ""
    model = ""
    for line in text.split("\n"):
        if line.startswith("модель"):
            model = line.strip()
            break
    verdict = ""
    if "### Вывод" in text:
        verdict = text.split("### Вывод", 1)[1].strip()

    # Цель и так стоит в заголовке, но там она урезана по ширине: здесь она
    # разложена по строкам целиком, без «Агент:» в начале.
    goal = str(entry.get("title") or entry["name"])
    if goal.startswith("Агент:"):
        goal = goal.split(":", 1)[1].strip()
    if _head(ui, col, "цель"):
        _lines(ui, col, _wrap(ui.small, goal, _block(ui, col.area, TABLE_W).w),
               MAX_GOAL_LINES, theme.INK)
    if model:
        r = col.take(ui.line(True), ui.px(2))
        ui._txt(ui.small, model, r.x0, r.y0, theme.DIM, max_px=r.w)
    _vals(ui, col, [
        ("шагов", _grp(text.count("\n## Шаг"))),
        ("опытов", _grp(len(entry.get("experiments") or []))),
        ("строк в журнале", _grp(text.count("\n") + 1)),
    ])

    if verdict and _head(ui, col, "вывод"):
        _lines(ui, col,
               _wrap(ui.small, verdict, _block(ui, col.area, TABLE_W).w),
               MAX_VERDICT_LINES, theme.INK)

    if col.left < ui.line(True) * 3:
        return
    if not _head(ui, col, "журнал"):
        return
    rest = col.rest
    lines = state.journal_lines(ui.small, text, max(1, rest.w - ui.px(14)))
    ui.log_view(state.LOG_KEY, rest, lines)


_DETAIL = {
    "run": _detail_run,
    "episode": _detail_episode,
    "embodied": _detail_embodied,
    "bench": _detail_bench,
    "agent": _detail_agent,
}


# ------------------------------------------------------------ разбор справа

def _buttons(ui, entry, area, on_open, on_rerun):
    """Действия на выбранном.

    «В Rerun» рисуется отключённой, а не пропускается: кнопка, которой
    однажды будет место, лучше объявлена заранее — иначе разбор перестроится,
    когда она появится.
    """
    three = (area.w - ui.gap * 2) // 3 >= ui.px(118)
    if three:
        a, b, c = area.cols(3, ui.gap)
    else:
        top, bottom = area.cut_top((area.h - ui.px(4)) // 2, ui.px(4))
        a, b = top.cols(2, ui.gap)
        c = bottom
    has_report = bool(entry and entry.get("report"))
    if ui.button("runs/open", a, "Открыть отчёт",
                 enabled=has_report and on_open is not None):
        on_open(entry["dir"] / "report.html")
    if ui.button("runs/again", b, "Прогнать снова",
                 enabled=entry is not None and on_rerun is not None):
        on_rerun(entry)
    ui.button("runs/rerun", c, "В Rerun", enabled=False)


def _detail(ui, state, entry, area, on_open, on_rerun, on_rerun_spec, now):
    if entry is None:
        ui._txt_centre(ui.small,
                       "в out/ нет прогонов этого вида", area.x0, area.x1,
                       ui._mid(area, ui.small), theme.FAINT)
        return

    # Место под кнопки отрезается заранее: если считать их после переменного
    # разбора, в тесном окне траектория или журнал съедают всю высоту и
    # кнопки наезжают на них.
    three = (area.w - ui.gap * 2) // 3 >= ui.px(118)
    foot_h = (ui.row if three else ui.row * 2 + ui.px(4)) + ui.line(True)
    foot, body = area.cut_bottom(min(foot_h, max(0, area.h // 2)), ui.gap)
    foot = _block(ui, foot, TABLE_W)

    # Заголовок — во всю ширину: название прогона не колонка цифр, урезать
    # его незачем. Колонкой стоит только разбор.
    hdr, body = body.cut_top(ui.line() + ui.px(2) + ui.line(True), ui.gap)
    hc = kit.Column(hdr, ui.px(2))
    r = hc.take(ui.line())
    ui._txt(ui.font, str(entry.get("title") or entry["name"]), r.x0, r.y0,
            theme.INK, max_px=r.w)
    r = hc.take(ui.line(True))
    sub = (entry["name"] + " · "
           + KIND_LABEL.get(entry.get("kind"), str(entry.get("kind")))
           + " · " + _ago(entry["mtime"], now))
    if not entry.get("report"):
        sub += " · без отчёта"
    ui._txt(ui.small, sub, r.x0, r.y0, theme.DIM, max_px=r.w)
    col = kit.Column(_block(ui, body, TABLE_W), ui.gap)

    if entry.get("kind") == "lateral":
        _detail_lateral(ui, entry, state, col, on_rerun_spec)
    else:
        fn = _DETAIL.get(entry.get("kind"))
        if fn is not None:
            fn(ui, entry, state, col)

    fcol = kit.Column(foot, ui.px(4))
    _buttons(ui, entry, fcol.take(foot.h - ui.line(True) - ui.px(4)),
             on_open, on_rerun)
    r = fcol.rest
    ui._txt(ui.small, "«В Rerun» появится на этапе 3", r.x0, r.y0,
            theme.FAINT, max_px=r.w)


# ---------------------------------------------------------------- панель

def panel(ui, state, b, on_open=None, on_rerun=None, on_rerun_spec=None):
    """Обзор прогонов: список слева, разбор выбранного справа.

    `on_open(path)` — открыть отчёт, `on_rerun(entry)` — повторить прогон,
    `on_rerun_spec(spec)` — завести прогон по спеке (зонд предлагает взять
    популяцию на рулёжку). Любой из них может быть `None`: тогда кнопка
    рисуется отключённой, а не пропадает.
    """
    state.refresh()
    b = kit.as_box(b)
    if b.w < ui.px(170) or b.h < ui.px(120):
        return state

    now = time.time()
    # В портретном окне колонки рядом не стоят: 257 точек на список и разбор
    # не делятся, поэтому список уезжает наверх.
    stacked = b.w < ui.px(560)
    if stacked:
        lst, det = b.cut_top(max(ui.px(150), int(b.h * 0.42)), 0)
        ui.back.rect(lst.x0, lst.y0, lst.x1, lst.y1, theme.PANEL)
        ui.back.hline(lst.x0, lst.x1, lst.y1 - 1, theme.HAIRLINE)
    else:
        lst, det = b.cut_left(
            min(ui.px(LIST_W), max(ui.px(180), int(b.w * 0.42))), 0)
        ui.back.rect(lst.x0, lst.y0, lst.x1, lst.y1, theme.PANEL)
        ui.back.vline(lst.x1 - 1, lst.y0, lst.y1, theme.HAIRLINE)

    shown = state.shown()
    cur = state.current(shown)
    _list(ui, state, shown, cur, lst.inset(ui.pad), now)

    # Выбранная могла поменяться только что, в списке: разбор тяжёлого идёт
    # после списка и ровно один раз на смену записи.
    cur = state.current(shown)
    state.ensure_loaded(cur)
    _detail(ui, state, cur, det.inset(ui.pad), on_open, on_rerun,
            on_rerun_spec, now)
    return state
