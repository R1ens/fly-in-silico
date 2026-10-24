"""Виджеты немедленного режима поверх буфера команд.

Каждая деталь определена здесь один раз и только здесь: все кнопки ведут себя
одинаково, все списки одинаково. Прежние попытки интерфейса были отвергнуты
именно за разнобой, поэтому отделка ячейки собрана в одну функцию `_cell`, а
не повторена в каждом виджете.

Два правила, которые задают всю конструкцию:

1. **Ни одного рисования.** Запуск ядра Taichi стоит около 195 микросекунд, и
   кадр с сотней деталей на отдельных запусках не уложится ни в какую частоту.
   Поэтому модуль только дописывает команды в `DrawList`; на экран их выводит
   `Surface.draw` одним запуском на тип команды. Если какой-то детали не
   хватает примитива — она обходится имеющимися (кривая искрового графика —
   столбцом вертикальных отрезков), а новое ядро не заводится.
2. **Вся метрика через масштаб.** Окно само меняет размер и бывает портретным
   (замеры: 1500x880, 803x984, 755x984), поэтому отступы, высоты строк и кегль
   считаются от площади окна, а не прибиты к пикселям. Числа берутся из
   `theme` и проходят через `Ui.px`; ни одного цвета и ни одного отступа мимо
   `theme` в этом файле нет.

Состояние между кадрами (что под курсором, что захвачено мышью, где какая
прокрутка, что в фокусе) живёт в `Ui` по строковому ключу детали — обычный
приём немедленного режима: виджет не объект, а вызов, а ключ отличает один
вызов от другого между кадрами.

Порядок кадра::

    ui.begin(w, h, inp)      # масштаб, атласы, обнуление буферов
    ...                      # вызовы деталей
    ui.flush(surf)           # снять захват и выложить кадр на поверхность

**Слои — не украшение, а необходимость.** Ядро `k_tiles` кроит все
прямоугольники одного вызова `Surface.draw` в один параллельный запуск, и
порядок записи между ними не определён. Замер: двадцать команд, где каждая
вторая кладётся поверх предыдущей, оставляют верхний цвет лишь на половине
площади. Значит, внутри одного запуска **два прямоугольника не имеют права
пересекаться**. Отсюда вся выкладка модуля:

* `back`  — подложки, на которые ложатся детали;
* `dl`    — тела деталей: заливка внутри рамки и сама рамка, между деталями
  пересечений нет;
* `over`  — данные на теле: полоса меры, кривая искры, путь, разделители;
* `top`   — указатели поверх данных: ползунок, янтарная метка текущего;
* текст   — в самом конце, двумя буферами.

Каждый слой — свой вызов `Surface.draw`, то есть свой запуск ядра и барьер
между ними; пустой слой не стоит ничего. Пересечения внутри слоя не
допускаются, кроме точек пути одного цвета, где порядок записи ни на что не
влияет.

Кеглей два, а `Surface.draw` принимает ровно один атлас на вызов: номер глифа
общий, а ширины и ремень — нет, и мелкий текст, нарисованный атласом
основного кегля, расползается. Поэтому глифы копятся в двух буферах.
"""

from __future__ import annotations

import numpy as np

from . import font as _font
from . import gpu, theme

__all__ = ["Box", "Column", "Input", "Ui", "box", "as_box", "scale_for"]

# Опорное окно, от которого отмеряется масштаб интерфейса. Выбрано не самым
# большим из замеров: на нём интерфейс выглядит плотным, крупные окна получают
# ту же плотность с небольшим запасом, портретные — сжимаются.
REF_W, REF_H = 1280.0, 800.0
SCALE_LO, SCALE_HI = 0.78, 1.30

# Прокрутка колесом: один щелчок колеса сдвигает список на три строки.
WHEEL_ROWS = 3


# ---------------------------------------------------------------- мелочи

def _mix(a, b, t):
    """Смешать цвета. Нужна, чтобы ступени фона были сплошными цветами.

    Прозрачность дала бы разный результат на панели и на фоне окна, а ступень
    должна выглядеть одинаково в любом месте окна.
    """
    return (a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t)


def fmt_num(v, step=None):
    """Число для показаний: запятая как разделитель, без хвостовых нулей."""
    if isinstance(v, str):
        return v
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    av = abs(float(v))
    if step is not None and float(step) >= 1.0 and float(v) == int(v):
        return str(int(v))
    nd = 0 if av >= 100 else (1 if av >= 10 else (2 if av >= 1 else 3))
    s = format(float(v), "." + str(nd) + "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s.replace(".", ",") if s else "0"


def _apply_fmt(fmt, v, step=None):
    if fmt is None:
        return fmt_num(v, step)
    if callable(fmt):
        return str(fmt(v))
    return str(fmt).format(v)


def scale_for(w, h):
    """Масштаб интерфейса от площади окна, с зажимом по краям.

    По площади, а не по ширине: портретное окно 803x984 по ширине выглядело бы
    крошечным, хотя места в нём столько же, сколько в 1280x720.
    """
    s = (max(1.0, float(w)) * max(1.0, float(h)) / (REF_W * REF_H)) ** 0.5
    return max(SCALE_LO, min(SCALE_HI, s))


# ---------------------------------------------------------------- коробки

class Box:
    """Прямоугольник в экранных координатах: x вправо, y вниз.

    Границы те же, что у `DrawList`: x0, y0 включительно, x1, y1 нет.
    """

    __slots__ = ("x0", "y0", "x1", "y1")

    def __init__(self, x0, y0, x1, y1):
        self.x0, self.y0 = int(x0), int(y0)
        self.x1, self.y1 = int(x1), int(y1)

    def __repr__(self):
        return ("Box(" + str(self.x0) + ", " + str(self.y0) + ", "
                + str(self.x1) + ", " + str(self.y1) + ")")

    def __iter__(self):
        return iter((self.x0, self.y0, self.x1, self.y1))

    @property
    def w(self):
        return self.x1 - self.x0

    @property
    def h(self):
        return self.y1 - self.y0

    @property
    def cx(self):
        return (self.x0 + self.x1) // 2

    @property
    def cy(self):
        return (self.y0 + self.y1) // 2

    def empty(self):
        return self.w <= 0 or self.h <= 0

    def contains(self, x, y):
        return self.x0 <= x < self.x1 and self.y0 <= y < self.y1

    def inset(self, dx, dy=None):
        dy = dx if dy is None else dy
        return Box(self.x0 + dx, self.y0 + dy, self.x1 - dx, self.y1 - dy)

    def cut_top(self, h, gap=0):
        """Отрезать сверху: вернуть (кусок, остаток)."""
        h = int(h)
        return Box(self.x0, self.y0, self.x1, self.y0 + h), \
            Box(self.x0, self.y0 + h + gap, self.x1, self.y1)

    def cut_bottom(self, h, gap=0):
        h = int(h)
        return Box(self.x0, self.y1 - h, self.x1, self.y1), \
            Box(self.x0, self.y0, self.x1, self.y1 - h - gap)

    def cut_left(self, w, gap=0):
        w = int(w)
        return Box(self.x0, self.y0, self.x0 + w, self.y1), \
            Box(self.x0 + w + gap, self.y0, self.x1, self.y1)

    def cut_right(self, w, gap=0):
        w = int(w)
        return Box(self.x1 - w, self.y0, self.x1, self.y1), \
            Box(self.x0, self.y0, self.x1 - w - gap, self.y1)

    def cols(self, n, gap=0):
        """Разбить по вертикали на n равных колонок с зазором."""
        n = max(1, int(n))
        span = (self.w - gap * (n - 1)) / n
        out = []
        for i in range(n):
            a = self.x0 + round(i * (span + gap))
            out.append(Box(a, self.y0, a + round(span), self.y1))
        out[-1].x1 = self.x1
        return out

    def rows(self, n, gap=0):
        n = max(1, int(n))
        span = (self.h - gap * (n - 1)) / n
        out = []
        for i in range(n):
            a = self.y0 + round(i * (span + gap))
            out.append(Box(self.x0, a, self.x1, a + round(span)))
        out[-1].y1 = self.y1
        return out


def box(x, y, w, h):
    """Коробка от угла и размера — привычнее для выкладки."""
    return Box(x, y, x + w, y + h)


def as_box(r):
    """Принять Box или четвёрку (x0, y0, x1, y1)."""
    return r if isinstance(r, Box) else Box(*r)


class Column:
    """Курсор вертикальной выкладки: отрезает полосы сверху вниз.

    Не деталь, а подручное средство: деталей ровно тринадцать, а выкладка —
    забота вызывающего кода.
    """

    __slots__ = ("area", "gap")

    def __init__(self, area, gap=0):
        self.area = as_box(area)
        self.gap = int(gap)

    def _clamp(self, h):
        """Не отдавать больше, чем осталось.

        Без этого `take` молча раздаёт коробки за нижним краем области, и
        деталь рисуется поверх соседней: именно так подпись состояния
        оказалась на дорожке ползунка. Пусть лучше деталь будет сжата и это
        будет видно, чем наложится и это придётся искать.
        """
        return max(0, min(int(h), self.area.h))

    def take(self, h, gap=None):
        g = self.gap if gap is None else int(gap)
        piece, self.area = self.area.cut_top(self._clamp(h), g)
        return piece

    def skip(self, h):
        _, self.area = self.area.cut_top(h, 0)

    @property
    def rest(self):
        return self.area

    @property
    def left(self):
        return self.area.h


# ---------------------------------------------------------------- ввод

class Input:
    """Состояние ввода за один кадр.

    Положение курсора — в пикселях окна, y вниз от верхнего края, как во всём
    интерфейсе. Taichi отдаёт его нормированным и снизу вверх, перевод делает
    `pull`.

    Про колесо: в GGUI его нет. Проверено на Taichi 1.7.4 — `EventType` знает
    только `Any`, `Press` и `Release`, а у события есть единственное поле
    `key`; ни «Wheel», ни дельты. Поэтому дельту колеса окно сообщает само
    методом `scroll` (из своего обратного вызова glfw, из imgui или с клавиш),
    а детали читают уже накопленное значение.

    Про клавиши: буквы приходят обычными односимвольными строками («i», «u»,
    «y», «g»), константы есть только для SHIFT, ALT, CTRL, ESCAPE, RETURN,
    TAB, BACKSPACE, SPACE, CAPSLOCK, стрелок и кнопок мыши. Про знаки
    «: + = ,» неизвестно, поэтому `pressed` просто ищет строку в списке и
    молча отвечает «нет», если такой клавиши окно не прислало.
    """

    __slots__ = ("x", "y", "down", "clicked", "released", "wheel", "keys",
                 "shift", "ctrl", "alt", "_held", "_wheel")

    def __init__(self):
        self.x = self.y = -1.0
        self.down = False
        self.clicked = False
        self.released = False
        self.wheel = 0.0
        self.keys = ()
        self.shift = self.ctrl = self.alt = False
        self._held = False          # состояние кнопки на прошлом кадре
        self._wheel = 0.0           # накопитель до начала кадра

    # -- заполнение ----------------------------------------------------

    def scroll(self, dy):
        """Сообщить дельту колеса. Положительное — содержимое вверх."""
        self._wheel += float(dy)

    def feed(self, x, y, down, keys=(), shift=False, ctrl=False, alt=False,
             wheel=0.0):
        """Кадр ввода целиком. Щелчок выводится из перехода кнопки."""
        self.x, self.y = float(x), float(y)
        down = bool(down)
        self.clicked = down and not self._held
        self.released = (not down) and self._held
        self.down = down
        self._held = down
        self.keys = tuple(str(k) for k in keys)
        self.shift, self.ctrl, self.alt = bool(shift), bool(ctrl), bool(alt)
        self.wheel = self._wheel + float(wheel)
        self._wheel = 0.0

    def pull(self, window, w, h, ti_ui):
        """Взять кадр ввода из окна Taichi.

        `ti_ui` передаётся снаружи (`taichi.ui`), чтобы модуль деталей не
        тянул окно за собой и оставался проверяемым без окна.
        """
        cx, cy = window.get_cursor_pos()
        keys = [str(e.key) for e in window.get_events(ti_ui.PRESS)]
        self.feed(cx * w, (1.0 - cy) * h,
                  window.is_pressed(ti_ui.LMB), keys,
                  shift=window.is_pressed(ti_ui.SHIFT),
                  ctrl=window.is_pressed(ti_ui.CTRL),
                  alt=window.is_pressed(ti_ui.ALT))

    # -- чтение --------------------------------------------------------

    def pressed(self, *names):
        """Нажата ли в этом кадре одна из клавиш. Регистр букв не важен."""
        if not self.keys:
            return False
        low = [k.lower() for k in self.keys]
        for n in names:
            n = str(n)
            if n in self.keys or n.lower() in low:
                return True
        return False


# ---------------------------------------------------------------- контекст

class Ui:
    """Контекст кадра: буфер команд, атласы, ввод, масштаб, память деталей."""

    def __init__(self, load=None):
        self.load = load or _font.load
        # Слои: пересекаться можно только между ними, но не внутри одного.
        self.back = gpu.DrawList()      # подложки под детали
        self.dl = gpu.DrawList()        # тела деталей и картинки
        self.over = gpu.DrawList()      # данные и разделители на телах
        self.top = gpu.DrawList()       # указатели поверх данных
        # Буфер глифов на каждое начертание: `Surface.draw` берёт ровно один
        # атлас, а начертаний теперь четыре.
        self.tx = gpu.DrawList()
        self.tx_small = gpu.DrawList()
        self._text = {}                 # id(atlas) -> (atlas, DrawList)
        self.input = Input()
        self.w = self.h = 0
        self.scale = 1.0
        self.font = None            # основной кегль
        self.small = None           # мелкий кегль
        self.hot = None             # ключ детали под курсором
        self.active = None          # ключ детали, захваченной мышью
        self.focus = None           # ключ детали в фокусе: ей идут стрелки
        self._atlas = {}            # кегль -> Atlas, чтобы не читать файл зря
        self._scroll = {}           # ключ -> прокрутка в пикселях
        self._span = {}             # ключ -> прежний предел прокрутки
        self._click_used = False    # щелчок достаётся одной детали за кадр
        self._wheel_used = False

    # -- кадр ----------------------------------------------------------

    def begin(self, w, h, inp=None):
        if inp is not None:
            self.input = inp
        self.w, self.h = int(w), int(h)
        self.scale = scale_for(w, h)
        self._fonts()
        for layer in self._layers():
            layer.clear()
        self.hot = None
        self._click_used = False
        self._wheel_used = False
        return self

    def _layers(self):
        return ((self.back, self.dl, self.over, self.top)
                + tuple(d for _, d in self._text.values()))

    def end(self):
        """Снять захват, если кнопку отпустили, и отдать тела деталей."""
        if not self.input.down:
            self.active = None
        return self.dl

    def flush(self, surf):
        """Выложить кадр.

        Четыре слоя фигур — это удобство выкладки, а не требование ядра:
        порядок внутри одного вызова `draw` теперь соблюдается, поэтому слои
        склеиваются в один список и уходят одним запуском. Раньше каждый слой
        стоил отдельного запуска, а запуск ядра — 195 микросекунд.

        Два буфера текста остаются: `draw` берёт ровно один атлас, а мелкий
        текст, нарисованный атласом основного кегля, расползается — номер
        глифа общий, ширины разные.
        """
        self.end()
        merged = gpu.DrawList()
        for layer in (self.back, self.dl, self.over, self.top):
            merged.extend(layer)
        if merged.cmds or merged.images:
            surf.draw(merged)
        for atlas, buf in self._text.values():
            if buf.gi:
                surf.draw(buf, atlas)

    def commands(self):
        """Сколько команд в кадре: прямоугольники, глифы, картинки."""
        return sum(len(d.rects) + len(d.gi) + len(d.images)
                   for d in self._layers())

    def rect_count(self):
        return sum(len(d.rects) for d in self._layers())

    def glyph_count(self):
        return sum(len(d.gi) for d in self._layers())

    def _fonts(self):
        """Четыре начертания, а не одно.

        Набирать интерфейс целиком моноширинным было ошибкой: без
        пропорционального текста и без разницы в насыщенности окно читается
        как дамп терминала. Текст идёт гуманистической антиквой, заголовки —
        её полужирным, а моноширинное начертание остаётся там, где оно
        несёт смысл: числа, селекторы, имена типов.
        """
        body = max(11, min(20, int(round(theme.SIZE_BODY * self.scale))))
        small = max(10, min(18, int(round(theme.SIZE_SMALL * self.scale))))
        title = max(14, min(30, int(round(theme.SIZE_TITLE * self.scale))))
        self.font = self._at(body)
        self.small = self._at(small, "ui")
        self.strong = self._at(body, "strong")
        self.title = self._at(title, "strong")
        # Моноширинное начертание выше при том же кегле; подбираем так,
        # чтобы строка совпадала по высоте с текстовой.
        self.num = self._at(self._mono_size(body), "num")
        self.num_small = self._at(self._mono_size(small), "num")

    def _mono_size(self, ui_size):
        """Кегль моноширинного начертания под текстовое.

        По высоте строки их равнять нельзя: у моноширинных выносные крупнее,
        и цифры выходят мельче текста, с которым стоят в одной строке.
        """
        key = ("mono", ui_size)
        got = self._atlas.get(key)
        if got is None:
            got = _font.match_size(ui_size)
            self._atlas[key] = got
        return got

    def _at(self, size, role="ui"):
        a = self._atlas.get((role, size))
        if a is None:
            a = (self.load(size, role=role) if self.load is not _font.load
                 else _font.load(size, role=role))
            self._atlas[(role, size)] = a
        return a

    # -- метрика -------------------------------------------------------

    def px(self, v):
        """Любое число из theme — только через этот пересчёт."""
        return max(1, int(round(float(v) * self.scale)))

    @property
    def pad(self):
        return self.px(theme.PAD)

    @property
    def gap(self):
        return self.px(theme.GAP)

    @property
    def row(self):
        return self.px(theme.ROW_H)

    @property
    def foot(self):
        return self.px(theme.FOOT_H)

    @property
    def rail(self):
        return self.px(theme.RAIL_W)

    @property
    def meter(self):
        return self.px(theme.METER_W)

    def line(self, small=False):
        """Высота строки текста с воздухом под неё."""
        a = self.small if small else self.font
        return a.gh + self.px(4)

    def _mid(self, b, atlas):
        """Верх строки, чтобы текст встал посередине коробки."""
        return b.y0 + max(0, (b.h - atlas.gh)) // 2

    # -- текст: каждый кегль в свой буфер ------------------------------

    def _buf(self, atlas):
        got = self._text.get(id(atlas))
        if got is None:
            got = (atlas, gpu.DrawList())
            self._text[id(atlas)] = got
        return got[1]

    def _txt(self, atlas, s, x, y, color, max_px=None):
        return self._buf(atlas).text(atlas, s, x, y, color, max_px=max_px)

    def _txt_right(self, atlas, s, x_right, y, color):
        return self._buf(atlas).text_right(atlas, s, x_right, y, color)

    def _txt_centre(self, atlas, s, x0, x1, y, color, max_px=None):
        if max_px is not None:
            s = atlas.fit(s, max_px)
        return self._buf(atlas).text_centre(atlas, s, x0, x1, y, color)

    # -- общее поведение ----------------------------------------------

    def _hit(self, b, key=None):
        """Курсор над коробкой и ничто другое не захвачено мышью."""
        if self.active is not None and self.active != key:
            return False
        if b.empty() or not b.contains(self.input.x, self.input.y):
            return False
        if key is not None:
            self.hot = key
        return True

    def _press(self, key, hot):
        """Начало захвата: щелчок достаётся ровно одной детали."""
        if hot and self.input.clicked and not self._click_used:
            self._click_used = True
            self.active = key
            self.focus = key
            return True
        return False

    def _fire(self, key, hot):
        """Срабатывание по отпусканию над той же деталью.

        По отпусканию, а не по нажатию: нажал, увёл курсор — ничего не
        случилось. Это одно правило на все кнопки, переключатели, сегменты,
        бирки и строки списков.
        """
        if self.active == key and self.input.released:
            self.active = None
            return hot
        return False

    def _wheel_at(self, b):
        """Колесо достаётся той прокручиваемой детали, что под курсором."""
        if self._wheel_used or self.input.wheel == 0.0:
            return 0.0
        if self.active is not None:
            return 0.0
        if not b.contains(self.input.x, self.input.y):
            return 0.0
        self._wheel_used = True
        return float(self.input.wheel)

    def _cell(self, b, hover=False, press=False, current=False, enabled=True,
              tone=None, rest=None, frame=True, fill=True):
        """Единая отделка ячейки. Возвращает цвет текста для неё.

        Весь вид интерфейса — здесь: ступень фона под курсором (не свечение),
        волосяная рамка в один пиксель, ни одной тени, янтарь только у
        нажатого и текущего. Заливка ложится **внутрь** рамки, а не под неё:
        в одном слое прямоугольники пересекаться не могут.
        """
        ink = theme.INK if rest is None else rest
        if not enabled:
            paint, edge, ink = theme.PANEL, theme.HAIRLINE, theme.FAINT
        elif tone == "primary":
            soft = _mix(theme.ACCENT, theme.PANEL, 0.88)
            paint = (theme.ACCENT_DIM if press
                     else (theme.ACCENT if hover else soft))
            edge, ink = theme.ACCENT, theme.BG
        elif tone == "danger":
            soft = _mix(theme.DANGER, theme.PANEL, 0.88)
            paint = (_mix(theme.DANGER, theme.BG, 0.55) if press
                     else (theme.DANGER if hover else soft))
            edge, ink = theme.DANGER, theme.BG
        elif tone == "well":
            # «Дорожка»: углубление под данные — ползунок, график, журнал
            paint = _mix(theme.BG, theme.PANEL, 0.6) if hover else theme.BG
            edge = theme.ACCENT if press else theme.HAIRLINE
        elif tone == "row":
            # Строка внутри дорожки: обычная не отличается от дна списка
            paint = theme.PANEL_HI if (hover or press or current) else theme.BG
            edge = theme.HAIRLINE
            if hover or press or current:
                ink = theme.INK
        else:
            lift = hover or press or current
            paint = theme.PANEL_HI if lift else theme.PANEL
            # Янтарь на рамке — это и есть «нажато» и «выбрано»: одно правило
            # на бирки, переключатели и кнопки, никаких исключений
            edge = theme.ACCENT if (press or current) else theme.HAIRLINE
            if hover or press or current:
                ink = theme.INK
        if frame:
            if fill:
                self.dl.rect(b.x0 + 1, b.y0 + 1, b.x1 - 1, b.y1 - 1, paint)
            self.dl.frame(b.x0, b.y0, b.x1, b.y1, edge)
        elif fill:
            self.dl.rect(b.x0, b.y0, b.x1, b.y1, paint)
        return ink

    def _mark(self, b, side="left"):
        """Янтарная метка текущего: полоса в две точки, а не заливка."""
        t = self.px(2)
        if side == "left":
            self.top.rect(b.x0, b.y0, b.x0 + t, b.y1, theme.ACCENT)
        else:
            self.top.rect(b.x0, b.y1 - t, b.x1, b.y1, theme.ACCENT)

    # ============================================================== 1
    def label(self, text, x, y, color=theme.INK, small=False, face=None,
              max_px=None):
        """Подпись. `face` — «ui», «strong», «num», иначе текстовое начертание."""
        a = {"strong": self.strong, "num": self.num,
             "num_small": self.num_small, "title": self.title,
             "small": self.small}.get(face)
        if a is None:
            a = self.small if small else self.font
        return self._txt(a, str(text), x, y, color, max_px=max_px)

    # ============================================================== 2
    def value(self, label, value, x, y, w):
        """Подпись слева, число справа. Возвращает y следующей строки.

        Подпись — пропорциональным текстом, число — моноширинным: цифры
        табличные, колонка не дёргается от кадра к кадру, и по начертанию
        сразу видно, где показание, а где название.
        """
        v = value if isinstance(value, str) else fmt_num(value)
        num = self.num
        vw = num.width(v)
        pad = self.px(6)
        base = max(num.gh, self.font.gh)
        self._txt(self.font, str(label), x,
                  y + max(0, (base - self.font.gh) // 2),
                  theme.DIM, max_px=max(0, w - vw - pad))
        self._txt_right(num, v, x + w, y + max(0, (base - num.gh) // 2),
                        theme.INK)
        return y + base + self.px(5)

    # ============================================================== 3
    def button(self, key, rect, text, kind="normal", enabled=True):
        """Кнопка. True в том кадре, когда щелчок завершился на ней."""
        b = as_box(rect)
        hot = enabled and self._hit(b, key)
        press = enabled and self.active == key and self.input.down
        if enabled:
            self._press(key, hot)
        fired = self._fire(key, hot) if enabled else False
        tone = {"primary": "primary", "danger": "danger"}.get(kind)
        ink = self._cell(b, hover=hot and not press, press=press,
                         enabled=enabled, tone=tone)
        self._txt_centre(self.font, str(text), b.x0 + self.px(8),
                         b.x1 - self.px(8), self._mid(b, self.font), ink)
        return bool(fired)

    # ============================================================== 4
    def toggle(self, key, rect, text, state):
        """Переключатель. Возвращает новое состояние."""
        b = as_box(rect)
        hot = self._hit(b, key)
        press = self.active == key and self.input.down
        self._press(key, hot)
        if self._fire(key, hot):
            state = not state
        ink = self._cell(b, hover=hot and not press, press=press,
                         current=bool(state),
                         rest=theme.DIM if not state else theme.INK)
        # Квадрат-указатель слева: янтарь включено, волосяная рамка выключено
        s = min(self.px(10), max(6, b.h - self.px(8)))
        m = Box(b.x0 + self.px(8), b.y0 + (b.h - s) // 2,
                b.x0 + self.px(8) + s, b.y0 + (b.h - s) // 2 + s)
        if state:
            self.over.rect(m.x0, m.y0, m.x1, m.y1, theme.ACCENT)
        else:
            self.over.frame(m.x0, m.y0, m.x1, m.y1, theme.FAINT)
        tx = m.x1 + self.px(8)
        self._txt(self.font, str(text), tx, self._mid(b, self.font), ink,
                  max_px=max(0, b.x1 - self.px(8) - tx))
        return bool(state)

    # ============================================================== 5
    def slider(self, key, rect, label, value, lo, hi, step=None, fmt=None):
        """Ползунок: подпись и число сверху, дорожка снизу. Тащится мышью."""
        b = as_box(rect)
        lo, hi = float(lo), float(hi)
        hot = self._hit(b, key)
        press = self.active == key and self.input.down
        self._press(key, hot)

        # Подпись отмеряется сверху, а дорожка снизу, с запасом на ползунок:
        # иначе в тесной коробке строка подписи не помещалась целиком и
        # молча пропадала, а ползунок вылезал за коробку на соседнюю деталь.
        track_h = max(self.px(5), 4)
        over = self.px(3)
        track = Box(b.x0, b.y1 - over - track_h, b.x1, b.y1 - over)
        top = Box(b.x0, b.y0, b.x1, b.y0 + self.small.gh)
        head = track.y0 - over - b.y0

        knob_w = self.px(4)
        keyed = self.focus == key and self.input.pressed("Left", "Right")
        if press or keyed:
            if press:
                span = max(1, track.w - knob_w)
                t = (self.input.x - track.x0 - knob_w * 0.5) / span
                value = lo + max(0.0, min(1.0, t)) * (hi - lo)
            else:
                d = (hi - lo) * 0.02 if step is None else float(step)
                value = value + (d if self.input.pressed("Right") else -d)
        if step:
            value = lo + round((value - lo) / float(step)) * float(step)
        value = max(lo, min(hi, float(value)))
        t = 0.0 if hi <= lo else (value - lo) / (hi - lo)

        if head >= self.small.gh:
            self._txt(self.small, str(label), top.x0,
                      self._mid(top, self.small), theme.DIM,
                      max_px=max(0, top.w // 2))
            self._txt_right(self.small, _apply_fmt(fmt, value, step), top.x1,
                            self._mid(top, self.small),
                            theme.ACCENT if press else theme.INK)
        self._cell(track, hover=hot and not press, press=press, tone="well")
        fill = int(round(t * (track.w - 2)))
        if fill > 0:
            self.over.rect(track.x0 + 1, track.y0 + 1, track.x0 + 1 + fill,
                           track.y1 - 1, theme.ACCENT_DIM)
        kx = track.x0 + int(round(t * max(1, track.w - knob_w)))
        self.top.rect(kx, track.y0 - over, kx + knob_w, track.y1 + over,
                      theme.ACCENT)
        return value

    # ============================================================== 6
    def segmented(self, key, rect, options, current):
        """Переключатель из нескольких долей. Возвращает индекс выбранной."""
        b = as_box(rect)
        n = max(1, len(options))
        current = int(current)
        self._hit(b, key)
        for i, opt in enumerate(options):
            c = Box(b.x0 + round(i * b.w / n), b.y0,
                    b.x0 + round((i + 1) * b.w / n), b.y1)
            k = key + "/" + str(i)
            hot = self._hit(c, k)
            press = self.active == k and self.input.down
            self._press(k, hot)
            if self._fire(k, hot):
                current = i
            sel = i == current
            ink = self._cell(c, hover=hot and not press, press=press,
                             current=sel, rest=theme.DIM, frame=False)
            if sel:
                self._mark(c, "bottom")
            self._txt_centre(self.font, str(opt), c.x0 + self.px(6),
                             c.x1 - self.px(6), self._mid(c, self.font), ink)
        # Рамка и разделители ложатся поверх долей, поэтому слоем выше
        self.over.frame(b.x0, b.y0, b.x1, b.y1, theme.HAIRLINE)
        for i in range(1, n):
            x = b.x0 + round(i * b.w / n)
            self.over.vline(x, b.y0 + 1, b.y1 - 1, theme.HAIRLINE)
        return current

    # ============================================================== 7
    def list_box(self, key, rect, items, current, filter_text=None):
        """Список: прокрутка колесом, выделение щелчком, стрелки в фокусе.

        Возвращает индекс в исходном списке `items`, а не в отфильтрованном:
        вызывающий код не должен знать про фильтр.
        """
        b = as_box(rect)
        self._cell(b, tone="well", frame=True, fill=False)
        inner = b.inset(1)
        keep = list(range(len(items)))
        if filter_text:
            f = str(filter_text).lower()
            keep = [i for i in keep if f in str(items[i]).lower()]

        rh = self.row
        total = rh * len(keep)
        span = max(0, total - inner.h)
        s = self._scroll.get(key, 0.0)
        s -= self._wheel_at(b) * rh * WHEEL_ROWS
        s = max(0.0, min(float(span), s))

        if self.focus == key and keep and self.input.pressed("Up", "Down"):
            at = keep.index(current) if current in keep else 0
            d = 1 if self.input.pressed("Down") else -1
            at = max(0, min(len(keep) - 1, at + d))
            current = keep[at]
            # Довести выбранную строку до видимой части, иначе стрелки уводят
            # выделение за край и список выглядит мёртвым.
            s = max(min(s, at * rh), (at + 1) * rh - inner.h)
            s = max(0.0, min(float(span), s))
        self._scroll[key] = s

        first = max(0, int(s // rh))
        last = min(len(keep), first + inner.h // rh + 2)
        pad = self.px(8)
        rail = self.px(3) if span > 0 else 0
        # Дно списка закрашивается только там, где строк уже нет: заливка
        # под строками в одном слое с ними даёт неопределённый порядок.
        cover = max(inner.y0, min(inner.y1, inner.y0 + int(total - s)))
        if cover < inner.y1:
            self.dl.rect(inner.x0, cover, inner.x1, inner.y1, theme.BG)
        if rail:
            self.dl.rect(inner.x1 - rail, inner.y0, inner.x1, cover,
                         theme.PANEL)
            vis = max(self.px(14), int(inner.h * inner.h / max(1, total)))
            ty = inner.y0 + int((inner.h - vis) * s / max(1.0, span))
            self.over.rect(inner.x1 - rail, ty, inner.x1, ty + vis,
                           theme.FAINT)
        for j in range(first, last):
            i = keep[j]
            y = inner.y0 + int(round(j * rh - s))
            r = Box(inner.x0, max(inner.y0, y), inner.x1 - rail,
                    min(inner.y1, y + rh))
            if r.h <= 1:
                continue
            k = key + "#" + str(i)
            hot = self._hit(r, k)
            press = self.active == k and self.input.down
            self._press(k, hot)
            if self._fire(k, hot):
                current = i
            sel = i == current
            ink = self._cell(r, hover=hot and not press, press=press,
                             current=sel, rest=theme.DIM, tone="row",
                             frame=False)
            if sel:
                self._mark(r, "left")
            # Строка пишется только целиком: обрезать глифы нечем, и
            # недорисованная строка вылезала бы за рамку списка
            ty = self._mid(Box(r.x0, y, r.x1, y + rh), self.font)
            if ty >= inner.y0 and ty + self.font.gh <= inner.y1:
                self._txt(self.font, str(items[i]), r.x0 + pad, ty, ink,
                          max_px=max(0, r.w - pad * 2))
        if not keep:
            self._txt(self.small, "ничего не нашлось", inner.x0 + pad,
                      inner.y0 + pad, theme.FAINT,
                      max_px=max(0, inner.w - pad * 2))
        return current

    # ============================================================== 8
    def chips(self, key, rect, options, selected):
        """Бирки в несколько строк. Возвращает новое множество выбранных."""
        b = as_box(rect)
        out = set(selected)
        self._hit(b, key)
        h = min(self.px(22), max(self.px(16), b.h))
        gap = self.px(6)
        x, y = b.x0, b.y0
        for i, opt in enumerate(options):
            s = str(opt)
            w = self.font.width(s) + self.px(20)
            w = min(w, b.w)
            if x > b.x0 and x + w > b.x1:
                x, y = b.x0, y + h + gap
            if y + h > b.y1:
                break
            c = Box(x, y, x + w, y + h)
            k = key + "/" + s
            hot = self._hit(c, k)
            press = self.active == k and self.input.down
            self._press(k, hot)
            if self._fire(k, hot):
                if opt in out:
                    out.discard(opt)
                else:
                    out.add(opt)
            on = opt in out
            ink = self._cell(c, hover=hot and not press, press=press,
                             current=on, rest=theme.DIM)
            self._txt_centre(self.font, s, c.x0 + self.px(4),
                             c.x1 - self.px(4), self._mid(c, self.font), ink)
            x += w + gap
        return out

    # ============================================================== 9
    def meter(self, rect, value01, color, label=None, value_text=None):
        """Шкала-мера: дорожка и полоса в ней, ровно до края дорожки."""
        b = as_box(rect)
        v = 0.0 if value01 is None else max(0.0, min(1.0, float(value01)))
        bar = b
        if label is not None or value_text is not None:
            head_h = self.small.gh
            if b.h >= head_h + self.px(7):
                top, bar = b.cut_top(head_h, self.px(2))
                vw = 0
                if value_text is not None:
                    vw = self.small.width(str(value_text)) + self.px(6)
                    self._txt_right(self.small, str(value_text), top.x1,
                                    top.y0, theme.INK)
                if label is not None:
                    self._txt(self.small, str(label), top.x0, top.y0,
                              theme.DIM, max_px=max(0, top.w - vw))
                bar = Box(bar.x0, bar.y0, bar.x1, b.y1)
        self._cell(bar, tone="well")
        inner = bar.inset(1)
        if inner.w > 0 and inner.h > 0:
            fill = int(round(v * inner.w))
            if fill > 0:
                self.over.rect(inner.x0, inner.y0, inner.x0 + fill, inner.y1,
                               color)
            if bar is b and (label is not None or value_text is not None):
                # Места на отдельную строку не хватило: подписи ложатся на
                # дорожку, как в полосах областей.
                pad = self.px(6)
                vw = 0
                if value_text is not None:
                    vw = self.small.width(str(value_text)) + pad
                    self._txt_right(self.small, str(value_text),
                                    inner.x1 - pad,
                                    self._mid(inner, self.small), theme.INK)
                if label is not None:
                    self._txt(self.small, str(label), inner.x0 + pad,
                              self._mid(inner, self.small), theme.INK,
                              max_px=max(0, inner.w - pad * 2 - vw))

    # ============================================================== 10
    def spark(self, rect, series, color, lo=None, hi=None):
        """Искровой график.

        Кривая выкладывается столбиком вертикальных отрезков: отдельного
        примитива для линии в `DrawList` нет, а заводить ядро ради него
        дороже, чем эти отрезки. Колонок не больше, чем точек в серии, и не
        мельче двух точек экрана — иначе широкий график стоил бы по команде на
        каждый пиксель ширины, а выглядел бы так же.
        """
        b = as_box(rect)
        self._cell(b, tone="well")
        inner = b.inset(1)
        s = np.asarray(series, dtype=np.float64).ravel()
        s = s[np.isfinite(s)]
        if inner.w < 2 or inner.h < 2 or s.size == 0:
            return
        a = float(np.min(s)) if lo is None else float(lo)
        z = float(np.max(s)) if hi is None else float(hi)
        if z - a < 1e-12:
            a, z = a - 0.5, z + 0.5
        step = max(1, self.px(2))
        cols = max(2, min(int(inner.w // step), int(s.size)))
        edge = inner.x0 + np.rint(
            np.linspace(0, inner.w, cols + 1)).astype(np.int64)
        if s.size == 1:
            v = np.full(cols, s[0])
        else:
            v = np.interp(np.linspace(0, s.size - 1, cols),
                          np.arange(s.size), s)
        t = np.clip((v - a) / (z - a), 0.0, 1.0)
        yi = np.rint(inner.y1 - 1 - t * (inner.h - 1)).astype(np.int64)
        prev = int(yi[0])
        for i in range(cols):
            y = int(yi[i])
            top, bot = (prev, y) if prev <= y else (y, prev)
            self.over.rect(edge[i], top, max(edge[i + 1], edge[i] + 1),
                           bot + 1, color)
            prev = y

    # ============================================================== 11
    def log_view(self, key, rect, lines):
        """Журнал: прокрутка колесом, прилипание к последней строке."""
        b = as_box(rect)
        self._cell(b, tone="well")
        inner = b.inset(self.px(6))
        lh = self.small.gh + self.px(2)
        n = len(lines)
        total = lh * n
        span = max(0.0, float(total - inner.h))
        s = self._scroll.get(key)
        was = self._span.get(key)
        # Прилипание: если прошлый кадр стоял внизу, остаёмся внизу, иначе
        # новая строка в журнале незаметно уводила бы взгляд.
        s = span if s is None or (was is not None and s >= was - 1.0) else s
        s -= self._wheel_at(b) * lh * WHEEL_ROWS
        s = max(0.0, min(span, s))
        self._scroll[key] = s
        self._span[key] = span
        first = max(0, int(s // lh))
        last = min(n, first + inner.h // lh + 2)
        for j in range(first, last):
            y = inner.y0 + int(round(j * lh - s))
            if y < inner.y0 - 1 or y + self.small.gh > inner.y1 + 1:
                continue
            self._txt(self.small, str(lines[j]), inner.x0, y, theme.DIM,
                      max_px=max(0, inner.w))

    # ============================================================== 12
    def track(self, rect, xy, color, closed=False):
        """Траектория в арене: вписана по пропорциям, голова в янтаре.

        Отрезки тоже выкладываются точками: наклонной линии в `DrawList` нет.
        Путь прореживается по сетке размером в точку пути: час прогона иначе
        превратился бы в десятки тысяч команд, из которых почти все легли бы
        друг на друга в одном пикселе.
        """
        b = as_box(rect)
        self._cell(b, tone="well")
        inner = b.inset(self.px(4))
        p = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        p = p[np.isfinite(p).all(axis=1)]
        if inner.w < 4 or inner.h < 4 or p.shape[0] == 0:
            return
        if p.shape[0] > 3000:
            p = p[:: int(np.ceil(p.shape[0] / 3000.0))]
        mn, mx = p.min(axis=0), p.max(axis=0)
        span = np.maximum(mx - mn, 1e-9)
        k = min((inner.w - 1) / span[0], (inner.h - 1) / span[1])
        cx, cy = 0.5 * (mn + mx)
        sx = inner.x0 + inner.w * 0.5 + (p[:, 0] - cx) * k
        # y данных вверх, y экрана вниз
        sy = inner.y0 + inner.h * 0.5 - (p[:, 1] - cy) * k
        xi = np.rint(sx).astype(np.int64)
        yi = np.rint(sy).astype(np.int64)
        d = max(1, self.px(2))
        if xi.size > 2:
            cell = np.stack([xi // d, yi // d], axis=1)
            sel = np.concatenate(([True], np.any(np.diff(cell, axis=0) != 0,
                                                 axis=1)))
            sel[-1] = True
            xi, yi = xi[sel], yi[sel]
        seq = list(range(len(xi)))
        if closed and len(xi) > 2:
            seq.append(0)
        budget = 6000
        for a, z in zip(seq, seq[1:]):
            x0, y0, x1, y1 = int(xi[a]), int(yi[a]), int(xi[z]), int(yi[z])
            far = max(abs(x1 - x0), abs(y1 - y0))
            n = max(1, min(budget, -(-far // d)))
            for t in range(n + 1):
                f = t / n
                x = int(round(x0 + (x1 - x0) * f))
                y = int(round(y0 + (y1 - y0) * f))
                # Точки пути одного цвета: их наложение друг на друга порядка
                # не требует, поэтому они живут в одном слое
                self.over.rect(x, y, x + d, y + d, color)
            budget -= n
            if budget <= 0:
                break
        hd = max(2, self.px(4))
        self.top.rect(int(xi[-1]) - hd // 2, int(yi[-1]) - hd // 2,
                      int(xi[-1]) + hd - hd // 2, int(yi[-1]) + hd - hd // 2,
                      theme.ACCENT)

    # ============================================================== 13
    def image_box(self, rect, rgb, label=None):
        """Картинка в рамке. Подпись — полосой сверху, а не поверх кадра."""
        b = as_box(rect)
        # Рамка отдельно от заливки: полоса подписи и поле картинки красятся
        # порознь, иначе они легли бы друг на друга в одном слое
        self._cell(b, tone="well", fill=False)
        inner = b.inset(1)
        if label is not None:
            head_h = self.small.gh + self.px(6)
            if inner.h > head_h * 2:
                top, inner = inner.cut_top(head_h)
                self.dl.rect(top.x0, top.y0, top.x1, top.y1 - 1, theme.PANEL)
                self.dl.hline(top.x0, top.x1, top.y1 - 1, theme.HAIRLINE)
                self._txt(self.small, str(label), top.x0 + self.px(6),
                          self._mid(top, self.small), theme.DIM,
                          max_px=max(0, top.w - self.px(12)))
        self.dl.rect(inner.x0, inner.y0, inner.x1, inner.y1, theme.BG)
        if rgb is None or inner.w < 2 or inner.h < 2:
            return
        a = np.asarray(rgb)
        if a.ndim != 3 or a.shape[2] < 3:
            return
        ih, iw = a.shape[0], a.shape[1]
        k = min(inner.w / iw, inner.h / ih)
        w, h = max(1, int(iw * k)), max(1, int(ih * k))
        self.dl.image(np.ascontiguousarray(a[:, :, :3]),
                      inner.x0 + (inner.w - w) // 2,
                      inner.y0 + (inner.h - h) // 2, w, h)
