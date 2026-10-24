"""Система компоновки: прямоугольники, док-зоны, разделители, прокрутка.

Прежний интерфейс считал раскладку долями от окна — `0,21 × ширина` на
рейку и так далее. Отсюда шли все жалобы: при растягивании росло всё
подряд, закрепить ничего было нельзя, а текст вылезал за соседа, потому
что коробки подгонялись под содержимое, а не содержимое укладывалось в
коробку.

Здесь наоборот:

* зоны имеют закреплённый размер в логических единицах, с пределами;
* тянется только центр;
* разделители двигают размер зоны, а не пересчитывают доли;
* при нехватке места зоны отключаются по порядку, а не сжимаются в кашу;
* содержимое обрезается по своей зоне, а длинный текст укладывается
  переносом или усечением — но никогда не вылезает;
* разворот центра — это состояние раскладки, а не другой экран:
  размеры зон сохраняются и возвращаются как были.
"""

from dataclasses import dataclass, field

from .tokens import S

__all__ = ["Rect", "Stack", "DockState", "Shell", "compute", "ScrollState",
           "scroll_begin", "scroll_end", "wrap_text", "ZONES"]

ZONES = ("inspector", "secondary", "timeline")


# ---------------------------------------------------------------- прямоугольник

class Rect:
    """Прямоугольник в пикселях устройства. Левый верхний угол и размер."""

    __slots__ = ("x", "y", "w", "h")

    def __init__(self, x=0, y=0, w=0, h=0):
        self.x, self.y = int(x), int(y)
        self.w, self.h = max(0, int(w)), max(0, int(h))

    # -- чтение ---------------------------------------------------------

    @property
    def x0(self):
        return self.x

    @property
    def y0(self):
        return self.y

    @property
    def x1(self):
        return self.x + self.w

    @property
    def y1(self):
        return self.y + self.h

    @property
    def cx(self):
        return self.x + self.w // 2

    @property
    def cy(self):
        return self.y + self.h // 2

    @property
    def empty(self):
        return self.w <= 0 or self.h <= 0

    def __iter__(self):
        return iter((self.x, self.y, self.w, self.h))

    def __repr__(self):
        return f"Rect({self.x}, {self.y}, {self.w}x{self.h})"

    def contains(self, px, py):
        return self.x <= px < self.x1 and self.y <= py < self.y1

    def overlaps(self, o):
        return not (o.x1 <= self.x or o.x >= self.x1
                    or o.y1 <= self.y or o.y >= self.y1)

    # -- производные ----------------------------------------------------

    def inset(self, dx, dy=None):
        dy = dx if dy is None else dy
        return Rect(self.x + dx, self.y + dy,
                    max(0, self.w - 2 * dx), max(0, self.h - 2 * dy))

    def pad(self, left=0, top=0, right=0, bottom=0):
        return Rect(self.x + left, self.y + top,
                    max(0, self.w - left - right),
                    max(0, self.h - top - bottom))

    def moved(self, dx, dy):
        return Rect(self.x + dx, self.y + dy, self.w, self.h)

    def with_h(self, h):
        return Rect(self.x, self.y, self.w, h)

    def with_w(self, w):
        return Rect(self.x, self.y, w, self.h)

    # Срезы возвращают (отрезанное, остаток) — так читается порядок
    # выкладки: что взяли и что осталось.

    def cut_top(self, h, gap=0):
        h = max(0, min(int(h), self.h))
        # Начало остатка зажато нижним краем: иначе исчерпанный курсор
        # уезжает за границу и отдаёт пустые коробки «ниже дна».
        ry = min(self.y + h + gap, self.y1)
        return (Rect(self.x, self.y, self.w, h),
                Rect(self.x, ry, self.w, max(0, self.y1 - ry)))

    def cut_bottom(self, h, gap=0):
        h = max(0, min(int(h), self.h))
        return (Rect(self.x, self.y1 - h, self.w, h),
                Rect(self.x, self.y, self.w, max(0, self.h - h - gap)))

    def cut_left(self, w, gap=0):
        w = max(0, min(int(w), self.w))
        rx = min(self.x + w + gap, self.x1)
        return (Rect(self.x, self.y, w, self.h),
                Rect(rx, self.y, max(0, self.x1 - rx), self.h))

    def cut_right(self, w, gap=0):
        w = max(0, min(int(w), self.w))
        return (Rect(self.x1 - w, self.y, w, self.h),
                Rect(self.x, self.y, max(0, self.w - w - gap), self.h))

    def cols(self, n, gap=0):
        n = max(1, int(n))
        free = self.w - gap * (n - 1)
        step = free // n
        out = []
        for i in range(n):
            x = self.x + i * (step + gap)
            w = step if i < n - 1 else self.x1 - x
            out.append(Rect(x, self.y, w, self.h))
        return out

    def rows(self, n, gap=0):
        n = max(1, int(n))
        free = self.h - gap * (n - 1)
        step = free // n
        out = []
        for i in range(n):
            y = self.y + i * (step + gap)
            h = step if i < n - 1 else self.y1 - y
            out.append(Rect(self.x, y, self.w, h))
        return out

    def fit_aspect(self, ar):
        """Вписать прямоугольник заданного отношения сторон по центру.

        Нужно виду камеры: кадр не должен растягиваться вслед за панелью.
        """
        if self.w <= 0 or self.h <= 0 or ar <= 0:
            return Rect(self.x, self.y, 0, 0)
        w = self.w
        h = int(round(w / ar))
        if h > self.h:
            h = self.h
            w = int(round(h * ar))
        return Rect(self.x + (self.w - w) // 2, self.y + (self.h - h) // 2,
                    w, h)


# ---------------------------------------------------------------- курсор

class Stack:
    """Вертикальный курсор выкладки внутри прямоугольника.

    Главное свойство: **никогда не отдаёт коробку за нижним краем**. В
    прежнем наборе курсор раздавал их молча, и подпись состояния
    оказывалась на дорожке ползунка.
    """

    __slots__ = ("area", "gap", "_start")

    def __init__(self, area, gap=0):
        self.area = area
        self.gap = int(gap)
        self._start = area.y

    @property
    def left(self):
        return self.area.h

    @property
    def rest(self):
        return self.area

    @property
    def used(self):
        return self.area.y - self._start

    def fits(self, h):
        return self.area.h >= int(h)

    def take(self, h, gap=None):
        g = self.gap if gap is None else int(gap)
        piece, self.area = self.area.cut_top(min(int(h), self.area.h), g)
        return piece

    def skip(self, h):
        _, self.area = self.area.cut_top(min(int(h), self.area.h), 0)

    def take_all(self):
        piece, self.area = self.area, Rect(self.area.x, self.area.y1,
                                           self.area.w, 0)
        return piece


# ---------------------------------------------------------------- док-зоны

@dataclass
class DockState:
    """Что помнит оболочка между кадрами. Размеры — логические."""

    inspector_w: float = S.inspector
    secondary_w: float = S.secondary
    timeline_h: float = S.timeline

    inspector_open: bool = True
    secondary_open: bool = True
    timeline_open: bool = True

    # Разворот центра — состояние, а не отдельный экран. Зоны прячутся,
    # их размеры сохраняются и возвращаются как были.
    maximized: bool = False

    drag: str = ""          # какой разделитель тянут
    drag_from: float = 0.0  # размер зоны на начало перетаскивания
    drag_at: int = 0        # координата курсора на начало

    def toggle(self, zone):
        setattr(self, zone + "_open", not getattr(self, zone + "_open"))

    def is_open(self, zone):
        return bool(getattr(self, zone + "_open"))


@dataclass
class Shell:
    """Посчитанная раскладка одного кадра."""

    w: int
    h: int
    topbar: Rect = None
    inspector: Rect = None
    centre: Rect = None
    secondary: Rect = None
    timeline: Rect = None
    footer: Rect = None
    splitters: dict = field(default_factory=dict)
    dropped: list = field(default_factory=list)

    def has(self, name):
        r = getattr(self, name, None)
        return r is not None and not r.empty


# Сколько места центр обязан получить, прежде чем отключать соседей.
CENTRE_MIN_W = 420
CENTRE_MIN_H = 240


def compute(w, h, sc, st):
    """Разложить окно. `sc` — Scale, `st` — DockState."""
    sh = Shell(w=int(w), h=int(h))
    px = sc.px
    top_h = px(S.topbar)
    foot_h = px(S.footer)
    sp = max(3, px(S.splitter))

    body = Rect(0, 0, w, h)
    sh.topbar, body = body.cut_top(top_h)
    sh.footer, body = body.cut_bottom(foot_h)

    if st.maximized:
        sh.centre = body
        return sh

    # Нижняя полоса: отключается, если центру не остаётся высоты.
    if st.timeline_open:
        t_h = px(max(S.timeline_min, min(S.timeline_max, st.timeline_h)))
        if body.h - t_h >= px(CENTRE_MIN_H):
            sh.timeline, body = body.cut_bottom(t_h)
            sh.splitters["timeline"] = Rect(body.x, body.y1 - sp // 2,
                                            body.w, sp)
        else:
            sh.dropped.append("timeline")

    # Боковые: сперва уходит правая, потом левая — карта важнее показаний.
    want_l = px(max(S.inspector_min, min(S.inspector_max, st.inspector_w))) \
        if st.inspector_open else 0
    want_r = px(max(S.secondary_min, min(S.secondary_max, st.secondary_w))) \
        if st.secondary_open else 0
    cmin = px(CENTRE_MIN_W)

    if want_r and body.w - want_l - want_r < cmin:
        want_r = 0
        sh.dropped.append("secondary")
    if want_l and body.w - want_l < cmin:
        want_l = 0
        sh.dropped.append("inspector")

    if want_l:
        sh.inspector, body = body.cut_left(want_l)
        sh.splitters["inspector"] = Rect(body.x - sp // 2, body.y, sp, body.h)
    if want_r:
        sh.secondary, body = body.cut_right(want_r)
        sh.splitters["secondary"] = Rect(sh.secondary.x - sp // 2, body.y,
                                         sp, body.h)
    sh.centre = body
    return sh


def hit_splitter(sh, px_, py):
    for name, r in sh.splitters.items():
        if r.contains(px_, py):
            return name
    return ""


def drag_splitter(st, sc, name, px_, py):
    """Продолжить перетаскивание разделителя.

    Размер меняется в логических единицах: при смене масштаба зона
    остаётся той же ширины «на глаз», а не прыгает.
    """
    if not name or not st.drag:
        return
    k = max(sc.k, 1e-6)
    if name == "inspector":
        st.inspector_w = max(S.inspector_min, min(
            S.inspector_max, st.drag_from + (px_ - st.drag_at) / k))
    elif name == "secondary":
        st.secondary_w = max(S.secondary_min, min(
            S.secondary_max, st.drag_from - (px_ - st.drag_at) / k))
    elif name == "timeline":
        st.timeline_h = max(S.timeline_min, min(
            S.timeline_max, st.drag_from - (py - st.drag_at) / k))


def start_drag(st, sc, name, px_, py):
    st.drag = name
    st.drag_at = py if name == "timeline" else px_
    st.drag_from = {"inspector": st.inspector_w,
                    "secondary": st.secondary_w,
                    "timeline": st.timeline_h}.get(name, 0.0)


# ---------------------------------------------------------------- прокрутка

@dataclass
class ScrollState:
    """Независимая прокрутка одной области."""

    offset: float = 0.0
    content: int = 0
    view: int = 0

    @property
    def max_offset(self):
        return max(0, self.content - self.view)

    def clamp(self):
        self.offset = max(0.0, min(float(self.max_offset), self.offset))

    def by(self, dy):
        self.offset += dy
        self.clamp()


def scroll_begin(dl, rect, st, wheel=0.0, step=48):
    """Начать прокручиваемую область.

    Возвращает прямоугольник содержимого, сдвинутый на прокрутку, и
    прежнюю обрезку. Содержимое выкладывается в него как обычно, а всё
    лишнее отрежется по границе области.
    """
    if wheel:
        st.by(-wheel * step)
    st.view = rect.h
    st.clamp()
    prev = dl.push_clip(rect.x, rect.y, rect.x1, rect.y1)
    return Rect(rect.x, rect.y - int(st.offset), rect.w, 1 << 20), prev


def scroll_end(dl, rect, st, used, prev, bar_w=4, track=None, thumb=None):
    """Закончить область и нарисовать полосу прокрутки, если нужно."""
    dl.pop_clip(prev)
    st.content = int(used)
    st.clamp()
    if st.content <= st.view or rect.h <= 0:
        return
    frac = st.view / max(st.content, 1)
    th = max(24, int(rect.h * frac))
    ty = rect.y + int((rect.h - th) * (st.offset / max(st.max_offset, 1)))
    x1 = rect.x1 - 2
    if track is not None:
        dl.rect(x1 - bar_w, rect.y, x1, rect.y1, track, 1.0, bar_w / 2)
    if thumb is not None:
        dl.rect(x1 - bar_w, ty, x1, ty + th, thumb, 1.0, bar_w / 2)


# ---------------------------------------------------------------- текст

def wrap_text(atlas, text, max_px):
    """Разбить строку на строки по ширине, по словам.

    В наборе виджетов переноса не было вовсе — только усечение, — и
    длинные названия приходилось резать многоточием даже там, где места
    хватало на две строки.
    """
    if max_px <= 0 or not text:
        return []
    out, line = [], ""
    for word in str(text).split(" "):
        probe = word if not line else line + " " + word
        if atlas.width(probe) <= max_px:
            line = probe
            continue
        if line:
            out.append(line)
        # одно слово длиннее строки — режем по символам
        while atlas.width(word) > max_px and len(word) > 1:
            cut = len(word)
            while cut > 1 and atlas.width(word[:cut]) > max_px:
                cut -= 1
            out.append(word[:cut])
            word = word[cut:]
        line = word
    if line:
        out.append(line)
    return out
